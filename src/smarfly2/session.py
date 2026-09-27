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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        n = len(self.records)
        t = np.asarray([r.t for r in self.records], dtype=float)
        dt = np.asarray([r.dt for r in self.records], dtype=float)
        visible = np.stack([r.visible.as_array() for r in self.records]) if n else np.empty((0, 5))
        features = np.stack([r.features.as_array() for r in self.records]) if n else np.empty((0, 5))
        hidden = np.stack([r.hidden.as_array() for r in self.records]) if n else np.empty((0, 3))
        np.savez_compressed(path, t=t, dt=dt, visible=visible, features=features, hidden=hidden, horizon=self.horizon)

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        with np.load(Path(path), allow_pickle=False) as data:
            t = data["t"]
            dt = data["dt"]
            visible = data["visible"]
            features = data["features"]
            hidden = data["hidden"]
            horizon = int(data["horizon"])
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
        return cls(records, horizon=horizon)


def make_targets(session: Session, horizon: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if len(session.records) < horizon + 2:
        raise ValueError(f"session too short for horizon {horizon}: need at least {horizon + 2} records")
    n = len(session.records) - horizon
    indices = np.arange(n, dtype=int)
    y = np.zeros((n, 3), dtype=float)
    for row, i in enumerate(indices):
        a = session.records[i].visible
        b = session.records[i + horizon].visible
        y[row, 0] = b.x - a.x
        y[row, 1] = b.y - a.y
        y[row, 2] = (b.heading - a.heading + math.pi) % (2.0 * math.pi) - math.pi
    return indices, y
