from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

from .records import FrameRecord, HiddenFlyState, VisibleFeatures, VisibleFlyState


@dataclass
class Session:
    records: list[FrameRecord]
    horizon: int = 8
    world_width: int | None = None
    world_height: int | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        n = len(self.records)
        payload = {
            "t": np.asarray([r.t for r in self.records], dtype=float),
            "dt": np.asarray([r.dt for r in self.records], dtype=float),
            "visible": np.stack([r.visible.as_array() for r in self.records]) if n else np.empty((0, 5)),
            "features": np.stack([r.features.as_array() for r in self.records]) if n else np.empty((0, 5)),
            "hidden": np.stack([r.hidden.as_array() for r in self.records]) if n else np.empty((0, 3)),
            "horizon": np.asarray(self.horizon),
        }
        if self.world_width is not None:
            payload["world_width"] = np.asarray(self.world_width)
        if self.world_height is not None:
            payload["world_height"] = np.asarray(self.world_height)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        with np.load(Path(path), allow_pickle=False) as data:
            t = data["t"]
            dt = data["dt"]
            visible = data["visible"]
            features = data["features"]
            hidden = data["hidden"]
            horizon = int(data["horizon"])
            world_width = int(data["world_width"]) if "world_width" in data.files else None
            world_height = int(data["world_height"]) if "world_height" in data.files else None
        records: list[FrameRecord] = []
        for i in range(len(t)):
            records.append(
                FrameRecord(
                    t=float(t[i]),
                    dt=float(dt[i]),
                    visible=VisibleFlyState(*map(float, visible[i])),
                    features=VisibleFeatures(*map(float, features[i])),
                    hidden=HiddenFlyState(*map(float, hidden[i])),
                )
            )
        return cls(records, horizon=horizon, world_width=world_width, world_height=world_height)


def _wrapped_delta(delta: float, size: float) -> float:
    return (delta + 0.5 * size) % size - 0.5 * size


def make_targets(
    session: Session,
    horizon: int = 8,
    world_size: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(session.records) < horizon + 2:
        raise ValueError(f"session too short for horizon {horizon}: need at least {horizon + 2} records")
    if world_size is None and session.world_width is not None and session.world_height is not None:
        world_size = (float(session.world_width), float(session.world_height))
    if world_size is not None:
        width, height = map(float, world_size)
        if width <= 0 or height <= 0:
            raise ValueError("world dimensions must be positive")
    else:
        width = height = 0.0

    n = len(session.records) - horizon
    indices = np.arange(n, dtype=int)
    y = np.zeros((n, 3), dtype=float)
    for row, i in enumerate(indices):
        a = session.records[i].visible
        b = session.records[i + horizon].visible
        dx = b.x - a.x
        dy = b.y - a.y
        if world_size is not None:
            dx = _wrapped_delta(dx, width)
            dy = _wrapped_delta(dy, height)
        y[row, 0] = dx
        y[row, 1] = dy
        y[row, 2] = (b.heading - a.heading + math.pi) % (2.0 * math.pi) - math.pi
    return indices, y
