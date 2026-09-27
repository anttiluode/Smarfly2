from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from ..events import EventScaler


_H8 = np.asarray([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1],
    [1, -1, -1, 1, 1, -1, -1, 1],
    [1, 1, 1, 1, -1, -1, -1, -1],
    [1, -1, 1, -1, -1, 1, -1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1],
    [1, -1, -1, 1, -1, 1, 1, -1],
], dtype=float)
_INHERITED_PROJECTIONS = _H8[:4, :6] / math.sqrt(6.0)


@dataclass(frozen=True)
class SequencingParams:
    fast_tau: float = 0.30
    slow_tau: float = 3.0
    period_s: float = 0.8
    n_channels: int = 4
    recurrence: float = 0.80
    transfer: float = 0.35
    event_gain: float = 0.50
    tuft_gain: float = 0.50
    closure_tau: float = 1.2


@dataclass(frozen=True)
class SequencingControls:
    no_tuft: bool = False
    no_route_closure: bool = False
    publication_block: bool = False
    flatten_phase: bool = False


@dataclass(frozen=True)
class SequencingTrace:
    features: np.ndarray
    events: np.ndarray
    change_residue: np.ndarray
    context_residue: np.ndarray
    channels: np.ndarray
    tuft_gain: np.ndarray
    closure: np.ndarray
    phase_bin: np.ndarray
    active_channel: np.ndarray
    publish_mask: np.ndarray


@dataclass
class SequencingRuntime:
    fast: np.ndarray
    context: np.ndarray
    channels: np.ndarray
    closure: np.ndarray
    phase: float = 0.0
    pending_channel: int = -1
    pending_prediction: float = 0.0


@dataclass(frozen=True)
class SequencingStep:
    feature: np.ndarray
    event: np.ndarray
    change_residue: np.ndarray
    context_residue: np.ndarray
    channels: np.ndarray
    tuft_gain: np.ndarray
    closure: np.ndarray
    phase_bin: int
    active_channel: int
    publish: bool


class SequencingFeatureBuilder:
    def __init__(self, params: SequencingParams | None = None):
        self.params = params or SequencingParams()
        if self.params.n_channels != 4:
            raise ValueError("v1 sequencing core requires exactly four channels")
        self.scaler = EventScaler()

    @property
    def projections(self) -> np.ndarray:
        return _INHERITED_PROJECTIONS.copy()

    def fit(self, event_train: np.ndarray) -> "SequencingFeatureBuilder":
        self.scaler.fit(event_train)
        return self

    def new_runtime(self) -> SequencingRuntime:
        return SequencingRuntime(
            fast=np.zeros(6, dtype=float),
            context=np.zeros(6, dtype=float),
            channels=np.zeros(4, dtype=float),
            closure=np.zeros(4, dtype=float),
        )

    def _advance_scaled(
        self,
        scaled_event: np.ndarray,
        dt: float,
        runtime: SequencingRuntime,
        controls: SequencingControls,
    ) -> SequencingStep:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt values must be finite and positive")
        p = self.params
        runtime.closure *= math.exp(-dt / p.closure_tau)
        if runtime.pending_channel >= 0:
            k_pending = runtime.pending_channel
            target = math.tanh(float(_INHERITED_PROJECTIONS[k_pending] @ scaled_event))
            mismatch = (runtime.pending_prediction - target) ** 2
            desired = float(np.clip(mismatch / 3.0, 0.0, 1.0))
            runtime.closure[k_pending] = max(runtime.closure[k_pending], desired)
            runtime.pending_channel = -1

        af = 1.0 - math.exp(-dt / p.fast_tau)
        a_slow = 1.0 - math.exp(-dt / p.slow_tau)
        runtime.fast += af * (scaled_event - runtime.fast)
        runtime.context += a_slow * (scaled_event - runtime.context)
        change = runtime.fast - runtime.context

        gains = 1.0 + p.tuft_gain * np.tanh(_INHERITED_PROJECTIONS @ runtime.context)
        if controls.no_tuft:
            gains = np.ones(4, dtype=float)

        bin_width = p.period_s / 8.0
        phase_bin = int(math.floor((runtime.phase + 1e-12) / bin_width)) % 8
        old = runtime.channels.copy()
        active_channel = -1
        if controls.flatten_phase:
            update_channels = range(4)
        elif phase_bin % 2 == 0:
            active_channel = phase_bin // 2
            update_channels = (active_channel,)
        else:
            update_channels = ()

        last_updated = -1
        for k in update_channels:
            predecessor = old[(k - 1) % 4]
            event_drive = float(_INHERITED_PROJECTIONS[k] @ scaled_event)
            drive = p.recurrence * old[k] + p.transfer * predecessor + p.event_gain * event_drive
            route_gain = 1.0 if controls.no_route_closure else 1.0 - runtime.closure[k]
            runtime.channels[k] = route_gain * math.tanh(gains[k] * drive)
            last_updated = k

        if last_updated >= 0:
            runtime.pending_channel = last_updated
            runtime.pending_prediction = math.tanh(runtime.channels[last_updated])

        surviving = np.abs(runtime.channels) * (
            1.0 if controls.no_route_closure else (1.0 - runtime.closure)
        )
        total = float(surviving.sum())
        publish = False
        if total > 1e-12:
            shares = np.sort(surviving / total)
            top = float(shares[-1])
            second = float(shares[-2]) if len(shares) > 1 else 0.0
            publish = (top - second) >= 0.05
        if controls.publication_block:
            publish = False

        angle = 2.0 * math.pi * runtime.phase / p.period_s
        feature = np.concatenate([
            change,
            runtime.context,
            runtime.channels,
            gains,
            runtime.closure,
            [math.sin(angle), math.cos(angle)],
        ])
        step = SequencingStep(
            feature=feature.copy(),
            event=np.asarray(scaled_event, dtype=float).copy(),
            change_residue=change.copy(),
            context_residue=runtime.context.copy(),
            channels=runtime.channels.copy(),
            tuft_gain=gains.copy(),
            closure=runtime.closure.copy(),
            phase_bin=phase_bin,
            active_channel=active_channel,
            publish=bool(publish),
        )
        runtime.phase = (runtime.phase + dt) % p.period_s
        return step

    def step(
        self,
        event: np.ndarray,
        dt: float,
        runtime: SequencingRuntime,
        controls: SequencingControls | None = None,
    ) -> SequencingStep:
        event = np.asarray(event, dtype=float)
        if event.shape != (6,):
            raise ValueError("event must have shape (6,)")
        scaled = self.scaler.transform(event[None, :])[0]
        return self._advance_scaled(scaled, float(dt), runtime, controls or SequencingControls())

    def transform(
        self,
        events: np.ndarray,
        dt: np.ndarray,
        controls: SequencingControls | None = None,
    ) -> SequencingTrace:
        controls = controls or SequencingControls()
        events = np.asarray(events, dtype=float)
        if events.ndim != 2 or events.shape[1] != 6:
            raise ValueError("events must have shape (n, 6)")
        dt = np.asarray(dt, dtype=float)
        if dt.shape != (len(events),):
            raise ValueError("dt must have one value per event row")
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
            raise ValueError("dt values must be finite and positive")

        scaled = self.scaler.transform(events)
        runtime = self.new_runtime()
        steps = [
            self._advance_scaled(scaled[i], float(dt[i]), runtime, controls)
            for i in range(len(events))
        ]
        n = len(steps)
        if not n:
            return SequencingTrace(
                features=np.empty((0, 26)), events=scaled, change_residue=np.empty((0, 6)),
                context_residue=np.empty((0, 6)), channels=np.empty((0, 4)), tuft_gain=np.empty((0, 4)),
                closure=np.empty((0, 4)), phase_bin=np.empty(0, dtype=int),
                active_channel=np.empty(0, dtype=int), publish_mask=np.empty(0, dtype=bool),
            )
        return SequencingTrace(
            features=np.stack([x.feature for x in steps]),
            events=np.stack([x.event for x in steps]),
            change_residue=np.stack([x.change_residue for x in steps]),
            context_residue=np.stack([x.context_residue for x in steps]),
            channels=np.stack([x.channels for x in steps]),
            tuft_gain=np.stack([x.tuft_gain for x in steps]),
            closure=np.stack([x.closure for x in steps]),
            phase_bin=np.asarray([x.phase_bin for x in steps], dtype=int),
            active_channel=np.asarray([x.active_channel for x in steps], dtype=int),
            publish_mask=np.asarray([x.publish for x in steps], dtype=bool),
        )


class SequencingObserver:
    def __init__(self, params: SequencingParams | None = None, alpha: float = 1e-3):
        from .base import RidgeReadout

        self.params = params or SequencingParams()
        self.builder = SequencingFeatureBuilder(self.params)
        self.readout = RidgeReadout(alpha)
        self.train_feature_rows_: np.ndarray | None = None

    @staticmethod
    def _present(records) -> np.ndarray:
        if not records:
            return np.empty((0, 12), dtype=float)
        return np.stack([r.observer_vector() for r in records])

    def _trace(self, records, controls: SequencingControls | None = None, events_override: np.ndarray | None = None) -> SequencingTrace:
        from ..events import raw_events

        events = raw_events(records) if events_override is None else np.asarray(events_override, dtype=float)
        if len(events) != len(records):
            raise ValueError("event override must have one row per record")
        dt = np.asarray([r.dt for r in records], dtype=float)
        return self.builder.transform(events, dt, controls=controls)

    def _feature_matrix(self, records, trace: SequencingTrace) -> np.ndarray:
        present = self._present(records)
        if len(present) != len(trace.features):
            raise ValueError("trace length does not match records")
        return np.concatenate([present, trace.features], axis=1)

    def fit(self, records, target_indices: np.ndarray, y: np.ndarray, train_pos: np.ndarray) -> "SequencingObserver":
        from ..events import raw_events

        target_indices = np.asarray(target_indices, dtype=int)
        train_pos = np.asarray(train_pos, dtype=int)
        y = np.asarray(y, dtype=float)
        if len(target_indices) != len(y):
            raise ValueError("target_indices and y must have matching rows")
        if train_pos.size == 0:
            raise ValueError("training positions must be non-empty")
        events = raw_events(records)
        train_record_indices = target_indices[train_pos]
        self.builder.fit(events[train_record_indices])
        trace = self._trace(records, events_override=events)
        phi = self._feature_matrix(records, trace)[target_indices]
        self.train_feature_rows_ = phi[train_pos].copy()
        self.readout.fit(self.train_feature_rows_, y[train_pos])
        return self

    def predict_internal(
        self,
        records,
        target_indices: np.ndarray,
        controls: SequencingControls | None = None,
        *,
        events_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SequencingTrace]:
        target_indices = np.asarray(target_indices, dtype=int)
        trace = self._trace(records, controls=controls, events_override=events_override)
        phi = self._feature_matrix(records, trace)[target_indices]
        return self.readout.predict(phi), trace

    def new_stream(
        self,
        controls: SequencingControls | None = None,
    ) -> "SequencingStream":
        return SequencingStream(self, controls=controls)

    def predict_published(
        self,
        records,
        target_indices: np.ndarray,
        controls: SequencingControls | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pred, trace = self.predict_internal(records, target_indices, controls=controls)
        mask = trace.publish_mask[np.asarray(target_indices, dtype=int)]
        external = pred.copy()
        external[~mask] = np.nan
        return external, mask


class SequencingStream:
    """Incremental online runner equivalent to batch sequencing from a fresh start."""

    def __init__(self, observer: SequencingObserver, controls: SequencingControls | None = None):
        self.observer = observer
        self.controls = controls or SequencingControls()
        self.runtime = observer.builder.new_runtime()
        self.previous_record = None

    def step(
        self,
        record,
        controls: SequencingControls | None = None,
    ) -> tuple[np.ndarray, SequencingStep]:
        from ..events import raw_events

        if self.previous_record is None:
            event = np.zeros(6, dtype=float)
        else:
            event = raw_events([self.previous_record, record])[1]
        step = self.observer.builder.step(
            event, float(record.dt), self.runtime, controls or self.controls
        )
        phi = np.concatenate([record.observer_vector(), step.feature])[None, :]
        prediction = self.observer.readout.predict(phi)[0]
        self.previous_record = record
        return prediction, step
