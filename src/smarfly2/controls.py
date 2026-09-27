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
    """Build resident features with each row's prefix history permuted.

    The current visible row is never moved. ``seed=None`` is the unshuffled
    reference path and is useful for matched-control tests.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    dts = np.asarray(dt, dtype=float)
    if dts.ndim == 0:
        dts = np.full(n, float(dts))
    if seed is None:
        return resident_features(
            X, dts, fast_tau=fast_tau, slow_tau=slow_tau, period_s=period_s
        )
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if i == 0:
            order = np.array([0], dtype=int)
        else:
            history = rng.permutation(i)
            order = np.concatenate([history, [i]])
        phi = resident_features(
            X[order],
            dts[order],
            fast_tau=fast_tau,
            slow_tau=slow_tau,
            period_s=period_s,
        )
        rows.append(phi[-1])
    return np.stack(rows) if rows else np.empty((0, X.shape[1] * 3 + 2))


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
