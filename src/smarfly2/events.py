from __future__ import annotations

import math
import numpy as np

from .records import FrameRecord


def _wrap_angle(delta: float) -> float:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def raw_events(records: list[FrameRecord]) -> np.ndarray:
    """Return visible-only temporal innovations.

    Column order is fixed:
    d_brightness_lr, d_motion, d_motion_lr, d_contrast,
    turn_innovation, speed_innovation.
    """
    n = len(records)
    out = np.zeros((n, 6), dtype=float)
    for i in range(1, n):
        a = records[i - 1]
        b = records[i]
        speed_a = math.hypot(a.visible.vx, a.visible.vy)
        speed_b = math.hypot(b.visible.vx, b.visible.vy)
        out[i] = [
            b.features.brightness_lr - a.features.brightness_lr,
            b.features.motion - a.features.motion,
            b.features.motion_lr - a.features.motion_lr,
            b.features.contrast - a.features.contrast,
            _wrap_angle(b.visible.heading - a.visible.heading),
            speed_b - speed_a,
        ]
    return out


class EventScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, events: np.ndarray) -> "EventScaler":
        events = np.asarray(events, dtype=float)
        if events.ndim != 2 or len(events) == 0:
            raise ValueError("events must be a non-empty 2-D matrix")
        if not np.all(np.isfinite(events)):
            raise ValueError("events must be finite")
        self.mean_ = events.mean(axis=0)
        scale = events.std(axis=0)
        self.scale_ = np.where(scale < 1e-12, 1.0, scale)
        return self

    def transform(self, events: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("event scaler is not fitted")
        events = np.asarray(events, dtype=float)
        if events.ndim != 2 or events.shape[1] != self.mean_.shape[0]:
            raise ValueError("events have incompatible shape")
        if not np.all(np.isfinite(events)):
            raise ValueError("events must be finite")
        return (events - self.mean_) / self.scale_
