from __future__ import annotations

import numpy as np
from .base import RidgeReadout


def window_features(X: np.ndarray, window: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if window < 1:
        raise ValueError("window must be positive")
    if len(X) == 0:
        return np.empty((0, X.shape[1] * window), dtype=float)
    out = np.empty((len(X), X.shape[1] * window), dtype=float)
    for i in range(len(X)):
        rows = []
        for k in range(window - 1, -1, -1):
            rows.append(X[max(0, i - k)])
        out[i] = np.concatenate(rows)
    return out


class WindowObserver:
    def __init__(self, window: int = 12, alpha: float = 1e-3):
        self.window = int(window)
        self.readout = RidgeReadout(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WindowObserver":
        self.readout.fit(window_features(X, self.window), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.readout.predict(window_features(X, self.window))
