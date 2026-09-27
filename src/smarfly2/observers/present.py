from __future__ import annotations

import numpy as np
from .base import RidgeReadout


class PresentObserver:
    def __init__(self, alpha: float = 1e-3):
        self.readout = RidgeReadout(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PresentObserver":
        self.readout.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.readout.predict(X)
