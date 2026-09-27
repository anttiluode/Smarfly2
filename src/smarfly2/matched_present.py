from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .session import Session


@dataclass(frozen=True)
class MatchedPair:
    i: int
    j: int
    visible_distance: float
    hidden_distance: float
    future_divergence: float


def _visible_vector(record) -> np.ndarray:
    v = record.visible
    return np.concatenate([
        np.asarray([v.x, v.y, math.sin(v.heading), math.cos(v.heading), v.vx, v.vy]),
        record.features.as_array(),
    ])


def _angle_delta(a: float, b: float) -> float:
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def _future_delta(session: Session, i: int, horizon: int) -> np.ndarray:
    a = session.records[i].visible
    b = session.records[i + horizon].visible
    return np.asarray([b.x - a.x, b.y - a.y, _angle_delta(a.heading, b.heading)], dtype=float)


def find_matched_present_pairs(
    session: Session,
    max_pairs: int = 20,
    *,
    temporal_separation: int = 30,
    horizon: int = 8,
) -> list[MatchedPair]:
    last = len(session.records) - horizon
    if last <= temporal_separation or max_pairs <= 0:
        return []
    V = np.stack([_visible_vector(r) for r in session.records[:last]])
    mean = V.mean(axis=0)
    std = V.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Z = (V - mean) / std

    candidates: dict[tuple[int, int], float] = {}
    for i in range(last):
        valid = np.flatnonzero(np.abs(np.arange(last) - i) >= temporal_separation)
        if valid.size == 0:
            continue
        distances = np.linalg.norm(Z[valid] - Z[i], axis=1)
        j = int(valid[int(np.argmin(distances))])
        key = (min(i, j), max(i, j))
        candidates[key] = min(candidates.get(key, float("inf")), float(np.min(distances)))

    chosen = sorted(candidates.items(), key=lambda kv: kv[1])[:max_pairs]
    out: list[MatchedPair] = []
    for (i, j), visible_distance in chosen:
        hi = session.records[i].hidden.as_array()
        hj = session.records[j].hidden.as_array()
        hidden_distance = float(np.linalg.norm(hi - hj))
        future_divergence = float(np.linalg.norm(_future_delta(session, i, horizon) - _future_delta(session, j, horizon)))
        out.append(MatchedPair(i, j, visible_distance, hidden_distance, future_divergence))
    return out
