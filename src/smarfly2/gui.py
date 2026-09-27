from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .fly import Fly
from .observers.base import RidgeReadout
from .observers.resident import resident_features
from .observers.sequencing import SequencingControls, SequencingObserver
from .observers.window import window_features
from .records import FrameRecord, HiddenFlyState, VisibleFlyState
from .session import Session, make_targets
from .vision import extract_local_features


class LiveBench:
    """Webcam/artificial-ethology bench with hidden-state fly and visible-only observers."""

    def __init__(
        self,
        camera_index: int = 0,
        model_path: str | None = None,
        *,
        model_world_size: tuple[float, float] | None = None,
        frame_provider: Callable[[], np.ndarray | None] | None = None,
        headless: bool = False,
        seed: int = 7,
        dt: float = 1 / 30,
    ):
        self.camera_index = int(camera_index)
        self.model_path = model_path
        self.frame_provider = frame_provider
        self.headless = bool(headless)
        self.seed = int(seed)
        self.dt = float(dt)
        self.session = Session([], horizon=8)
        self.fly: Fly | None = None
        self.previous_frame: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.last_predictions: dict[str, np.ndarray] = {}
        self.audit = False
        self.show_sequencing_inspector = False
        self.publication_block = False
        self.last_sequencing_state: dict | None = None
        self._models: dict[str, RidgeReadout] = {}
        self._sequencing_model: SequencingObserver | None = None
        self._sequencing_stream = None
        self._camera = None
        self._camera_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._camera_thread: threading.Thread | None = None
        self._running = False
        if model_path is not None:
            self._fit_models_from_session(Session.load(model_path), world_size=model_world_size)

    @staticmethod
    def _visible(fly: Fly) -> VisibleFlyState:
        return VisibleFlyState(fly.x, fly.y, fly.heading, fly.vx, fly.vy)

    @staticmethod
    def _hidden(fly: Fly) -> HiddenFlyState:
        return HiddenFlyState(fly.fast_trace, fly.slow_context, fly.adaptation)

    def _fit_models_from_session(
        self,
        session: Session,
        world_size: tuple[float, float] | None = None,
    ) -> None:
        if world_size is None and (session.world_width is None or session.world_height is None):
            raise ValueError("world dimensions required for legacy model session")
        indices, y = make_targets(session, horizon=8, world_size=world_size)
        X = np.stack([session.records[i].observer_vector() for i in indices])
        dt = np.asarray([session.records[i].dt for i in indices], dtype=float)
        warmup = 11
        if len(X) <= warmup + 3:
            raise ValueError("model session too short")
        train = np.arange(warmup, len(X))
        feature_sets = {
            "present": X,
            "window": window_features(X, 12),
            "resident": resident_features(X, dt),
        }
        self._models = {
            name: RidgeReadout(1e-3).fit(phi[train], y[train])
            for name, phi in feature_sets.items()
        }
        self._sequencing_model = SequencingObserver().fit(session.records, indices, y, train)
        self._sequencing_stream = self._sequencing_model.new_stream()

    def _predict_latest(self) -> dict[str, np.ndarray]:
        if not self._models or not self.session.records:
            return {}
        recent = self.session.records[-240:]
        X = np.stack([r.observer_vector() for r in recent])
        dt = np.asarray([r.dt for r in recent], dtype=float)
        feature_sets = {
            "present": X,
            "window": window_features(X, 12),
            "resident": resident_features(X, dt),
        }
        predictions = {
            name: self._models[name].predict(phi[-1:])[0]
            for name, phi in feature_sets.items()
        }
        self.last_sequencing_state = None
        if self._sequencing_stream is not None:
            controls = SequencingControls(publication_block=self.publication_block)
            pred, step = self._sequencing_stream.step(self.session.records[-1], controls=controls)
            published = bool(step.publish)
            predictions["sequencing_internal"] = pred
            if published:
                predictions["sequencing_published"] = pred
            self.last_sequencing_state = {
                "channels": step.channels.copy(),
                "phase_bin": int(step.phase_bin),
                "active_channel": int(step.active_channel),
                "tuft_gain": step.tuft_gain.copy(),
                "closure": step.closure.copy(),
                "published": published,
                "internal_prediction": pred.copy(),
                "published_prediction": pred.copy() if published else None,
            }
        return predictions

    def get_sequencing_inspector(self) -> dict | None:
        return self.last_sequencing_state

    def process_frame(self, frame: np.ndarray | None) -> bool:
        if frame is None:
            return False
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be BGR image with shape (H,W,3)")
        if self.fly is None:
            h, w = frame.shape[:2]
            self.fly = Fly(w, h, seed=self.seed)
            self.session.world_width = int(w)
            self.session.world_height = int(h)
        visible = self._visible(self.fly)
        features = extract_local_features(frame, visible, self.previous_frame)
        record = FrameRecord(
            t=len(self.session.records) * self.dt,
            dt=self.dt,
            visible=visible,
            features=features,
            hidden=self._hidden(self.fly),
        )
        self.session.records.append(record)
        self.last_predictions = self._predict_latest()
        self.fly.step(features, self.dt)
        self.previous_frame = frame.copy()
        self.last_frame = frame.copy()
        return True

    def step_once(self) -> bool:
        if self.frame_provider is not None:
            return self.process_frame(self.frame_provider())
        try:
            frame = self._camera_queue.get_nowait()
        except queue.Empty:
            return False
        return self.process_frame(frame)

    def _camera_loop(self) -> None:
        assert self._camera is not None
        while self._running:
            ok, frame = self._camera.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self._camera_queue.full():
                try:
                    self._camera_queue.get_nowait()
                except queue.Empty:
                    pass
            self._camera_queue.put_nowait(frame)

    def start_camera(self) -> None:
        if self.frame_provider is not None:
            return
        self._camera = cv2.VideoCapture(self.camera_index)
        if not self._camera.isOpened():
            raise RuntimeError(f"could not open camera {self.camera_index}")
        self._running = True
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()

    def close(self) -> None:
        self._running = False
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=1.0)
        if self._camera is not None:
            self._camera.release()

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        if not self.session.records:
            return out
        record = self.session.records[-1]
        v = record.visible
        x, y = int(round(v.x)), int(round(v.y))
        tip = (int(round(v.x + 16 * math.cos(v.heading))), int(round(v.y + 16 * math.sin(v.heading))))
        cv2.circle(out, (x, y), 5, (255, 255, 255), 1)
        cv2.line(out, (x, y), tip, (255, 255, 255), 2)

        length = max(30, int(0.25 * min(out.shape[:2])))
        p1 = (int(v.x + length * math.cos(v.heading - math.pi/4)), int(v.y + length * math.sin(v.heading - math.pi/4)))
        p2 = (int(v.x + length * math.cos(v.heading + math.pi/4)), int(v.y + length * math.sin(v.heading + math.pi/4)))
        cv2.line(out, (x, y), p1, (120, 120, 120), 1)
        cv2.line(out, (x, y), p2, (120, 120, 120), 1)

        styles = {
            "present": ((200, 200, 255), 1),
            "window": ((120, 255, 120), 2),
            "resident": ((255, 180, 80), 2),
            "sequencing_internal": ((180, 180, 80), 1),
            "sequencing_published": ((80, 220, 255), 2),
        }
        for name, pred in self.last_predictions.items():
            color, thickness = styles.get(name, ((255, 255, 255), 1))
            end = (int(round(v.x + pred[0])), int(round(v.y + pred[1])))
            cv2.line(out, (x, y), end, color, thickness, cv2.LINE_AA)

        if self.show_sequencing_inspector and self.last_sequencing_state is not None:
            st = self.last_sequencing_state
            mode = "PUB" if st["published"] else "silent"
            text = f"SEQ bin={st['phase_bin']} active={st['active_channel']} {mode}"
            cv2.putText(out, text, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 140), 1, cv2.LINE_AA)
            ch = np.asarray(st["channels"])
            cl = np.asarray(st["closure"])
            text2 = "z=" + ",".join(f"{v:+.2f}" for v in ch) + "  sst=" + ",".join(f"{v:.2f}" for v in cl)
            cv2.putText(out, text2, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 140), 1, cv2.LINE_AA)

        if self.audit:
            h = record.hidden
            text = f"AUDIT fast={h.fast_trace:+.3f} slow={h.slow_context:+.3f} adapt={h.adaptation:+.3f}"
            cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def run(self) -> None:
        if self.headless:
            raise RuntimeError("headless LiveBench has no GUI; call step_once()")
        from PIL import Image, ImageTk
        import tkinter as tk
        from tkinter import ttk

        self.start_camera()
        root = tk.Tk()
        root.title("Smarfly2 — Artificial Ethology Bench")
        image_label = ttk.Label(root)
        image_label.pack(fill=tk.BOTH, expand=True)
        audit_var = tk.BooleanVar(value=False)
        sequence_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(root, text="AUDIT hidden fly state", variable=audit_var).pack(anchor=tk.W)
        ttk.Checkbutton(root, text="SEQUENCE inspector", variable=sequence_var).pack(anchor=tk.W)

        def tick():
            self.audit = bool(audit_var.get())
            self.show_sequencing_inspector = bool(sequence_var.get())
            if self.step_once() and self.last_frame is not None:
                overlay = self.render_overlay(self.last_frame)
                rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                image_label.configure(image=photo)
                image_label.image = photo
            if self._running:
                root.after(15, tick)

        def finish():
            self.close()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", finish)
        root.after(0, tick)
        root.mainloop()

    def save_session(self, path: str | Path) -> None:
        self.session.save(path)
