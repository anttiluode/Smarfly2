from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from .fly import Fly
from .records import FrameRecord, HiddenFlyState, VisibleFlyState
from .session import Session
from .vision import extract_local_features


def _visible(fly: Fly) -> VisibleFlyState:
    return VisibleFlyState(fly.x, fly.y, fly.heading, fly.vx, fly.vy)


def _hidden(fly: Fly) -> HiddenFlyState:
    return HiddenFlyState(fly.fast_trace, fly.slow_context, fly.adaptation)


def replay_frames(frames: Sequence[np.ndarray], seed: int, dt: float, horizon: int = 8) -> Session:
    if not frames:
        return Session([], horizon=horizon)
    h, w = frames[0].shape[:2]
    fly = Fly(w, h, seed=seed)
    records: list[FrameRecord] = []
    previous: np.ndarray | None = None
    for i, frame in enumerate(frames):
        visible = _visible(fly)
        features = extract_local_features(frame, visible, previous)
        records.append(
            FrameRecord(
                t=i * dt,
                dt=dt,
                visible=visible,
                features=features,
                hidden=_hidden(fly),
            )
        )
        fly.step(features, dt)
        previous = frame
    return Session(records, horizon=horizon)
