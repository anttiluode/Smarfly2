from __future__ import annotations

import math
import numpy as np

from .base import RidgeReadout


def _dt_array(dt: np.ndarray | float, n: int) -> np.ndarray:
    arr = np.asarray(dt, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError("dt must be scalar or length n")
    if np.any(arr <= 0):
        raise ValueError("dt values must be positive")
    return arr


def resident_features(
    X: np.ndarray,
    dt: np.ndarray | float,
    reset_mask: np.ndarray | None = None,
    *,
    fast_tau: float = 0.30,
    slow_tau: float = 3.0,
    period_s: float = 0.8,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n, d = X.shape
    dts = _dt_array(dt, n)
    reset = np.zeros(n, dtype=bool) if reset_mask is None else np.asarray(reset_mask, dtype=bool)
    if reset.shape != (n,):
        raise ValueError("reset_mask must have length n")

    fast = np.zeros(d, dtype=float)
    slow = np.zeros(d, dtype=float)
    phase = 0.0
    out = np.empty((n, 3 * d + 2), dtype=float)
    for i in range(n):
        if reset[i]:
            fast.fill(0.0)
            slow.fill(0.0)
            phase = 0.0
        af = 1.0 - math.exp(-dts[i] / fast_tau)
        a_slow = 1.0 - math.exp(-dts[i] / slow_tau)
        fast += af * (X[i] - fast)
        slow += a_slow * (X[i] - slow)
        phase = (phase + 2.0 * math.pi * dts[i] / period_s) % (2.0 * math.pi)
        out[i] = np.concatenate([X[i], fast - slow, slow, [math.sin(phase), math.cos(phase)]])
    return out


class ResidentObserver:
    def __init__(
        self,
        fast_tau: float = 0.30,
        slow_tau: float = 3.0,
        period_s: float = 0.8,
        alpha: float = 1e-3,
    ):
        self.fast_tau = float(fast_tau)
        self.slow_tau = float(slow_tau)
        self.period_s = float(period_s)
        self.readout = RidgeReadout(alpha)

    def transform(
        self,
        X: np.ndarray,
        dt: np.ndarray | float | None = None,
        reset_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if dt is None:
            dt = 1.0 / 30.0
        return resident_features(
            X,
            dt,
            reset_mask=reset_mask,
            fast_tau=self.fast_tau,
            slow_tau=self.slow_tau,
            period_s=self.period_s,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dt: np.ndarray | float | None = None,
        reset_mask: np.ndarray | None = None,
    ) -> "ResidentObserver":
        self.readout.fit(self.transform(X, dt, reset_mask), y)
        return self

    def predict(
        self,
        X: np.ndarray,
        dt: np.ndarray | float | None = None,
        reset_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.readout.predict(self.transform(X, dt, reset_mask))
