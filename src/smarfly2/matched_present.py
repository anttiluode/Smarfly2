from __future__ import annotations

from dataclasses import dataclass, replace
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
    event_residue_distance: float | None = None
    tuft_gain_distance: float | None = None
    active_channel_i: int | None = None
    active_channel_j: int | None = None
    sequencing_prediction_divergence: float | None = None


def _visible_vector(record) -> np.ndarray:
    v = record.visible
    return np.concatenate([
        np.asarray([v.x, v.y, math.sin(v.heading), math.cos(v.heading), v.vx, v.vy]),
        record.features.as_array(),
    ])


def _angle_delta(a: float, b: float) -> float:
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def _wrapped(delta: float, size: float | None) -> float:
    if size is None:
        return delta
    return (delta + 0.5 * size) % size - 0.5 * size


def _future_delta(session: Session, i: int, horizon: int) -> np.ndarray:
    a = session.records[i].visible
    b = session.records[i + horizon].visible
    return np.asarray([
        _wrapped(b.x - a.x, float(session.world_width) if session.world_width else None),
        _wrapped(b.y - a.y, float(session.world_height) if session.world_height else None),
        _angle_delta(a.heading, b.heading),
    ], dtype=float)


def find_matched_present_pairs(
    session: Session,
    max_pairs: int = 20,
    *,
    temporal_separation: int = 30,
    horizon: int = 8,
) -> list[MatchedPair]:
    """Select pairs using visible present only; audit data is appended elsewhere."""
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


def attach_sequencing_audit(
    pairs: list[MatchedPair],
    trace,
    prediction_by_index: dict[int, np.ndarray] | None = None,
) -> list[MatchedPair]:
    """Attach sequencing diagnostics after visible-only pair selection."""
    audited: list[MatchedPair] = []
    for pair in pairs:
        i, j = pair.i, pair.j
        prediction_divergence = None
        if prediction_by_index is not None and i in prediction_by_index and j in prediction_by_index:
            prediction_divergence = float(
                np.linalg.norm(np.asarray(prediction_by_index[i]) - np.asarray(prediction_by_index[j]))
            )
        audited.append(replace(
            pair,
            event_residue_distance=float(np.linalg.norm(trace.change_residue[i] - trace.change_residue[j])),
            tuft_gain_distance=float(np.linalg.norm(trace.tuft_gain[i] - trace.tuft_gain[j])),
            active_channel_i=int(trace.active_channel[i]),
            active_channel_j=int(trace.active_channel[j]),
            sequencing_prediction_divergence=prediction_divergence,
        ))
    return audited
