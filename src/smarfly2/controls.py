from __future__ import annotations

import numpy as np

from .observers.resident import resident_features


def shuffle_history_preserve_present(
    X: np.ndarray,
    dt: np.ndarray | float,
    seed: int | None = 0,
    *,
    fast_tau: float = 0.30,
    slow_tau: float = 3.0,
    period_s: float = 0.8,
) -> np.ndarray:
    """Destroy temporal order while preserving each row's visible present.

    ``seed=None`` returns the unshuffled reference. For a seeded control, the
    persistent state is driven by a causal surrogate stream: at each row it
    receives one randomly selected earlier visible sample. The current visible
    row is then restored in the public portion of the feature vector. This
    keeps the control O(n), avoids future leakage, and preserves its scientific
    purpose: current evidence is identical while ordered history is destroyed.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n, d = X.shape
    dts = np.asarray(dt, dtype=float)
    if dts.ndim == 0:
        dts = np.full(n, float(dts))
    if dts.shape != (n,):
        raise ValueError("dt must be scalar or length n")
    if seed is None:
        return resident_features(
            X, dts, fast_tau=fast_tau, slow_tau=slow_tau, period_s=period_s
        )
    if n == 0:
        return np.empty((0, d * 3 + 2), dtype=float)
    rng = np.random.default_rng(seed)
    surrogate = np.empty_like(X)
    surrogate[0] = X[0]
    for i in range(1, n):
        surrogate[i] = X[int(rng.integers(0, i))]
    out = resident_features(
        surrogate, dts, fast_tau=fast_tau, slow_tau=slow_tau, period_s=period_s
    )
    out[:, :d] = X
    return out


def reset_history_at(
    X: np.ndarray,
    dt: np.ndarray | float,
    reset_mask: np.ndarray,
    *,
    fast_tau: float = 0.30,
    slow_tau: float = 3.0,
    period_s: float = 0.8,
) -> np.ndarray:
    return resident_features(
        X,
        dt,
        reset_mask=np.asarray(reset_mask, dtype=bool),
        fast_tau=fast_tau,
        slow_tau=slow_tau,
        period_s=period_s,
    )


def shuffle_events_preserve_present(events: np.ndarray, seed: int = 1729) -> np.ndarray:
    """Destroy event order causally without moving the visible present.

    Row zero remains the zero-origin event. At each later row, the surrogate
    history event is sampled only from strictly earlier rows, so no future
    event can leak into an earlier observer state. Present-visible features
    and targets are supplied separately and are never moved.
    """
    events = np.asarray(events, dtype=float)
    if events.ndim != 2:
        raise ValueError("events must be 2-D")
    out = events.copy()
    if len(events) <= 1:
        return out
    rng = np.random.default_rng(seed)
    for i in range(1, len(events)):
        out[i] = events[int(rng.integers(0, i))]
    return out
