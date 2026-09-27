from __future__ import annotations

from typing import Protocol
import numpy as np


class Observer(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray): ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class RidgeReadout:
    def __init__(self, alpha: float = 1e-3):
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeReadout":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2-D matrix")
        if y.ndim == 1:
            y = y[:, None]
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale < 1e-12, 1.0, scale)
        Z = (X - self.mean_) / self.scale_
        A = np.concatenate([np.ones((len(Z), 1)), Z], axis=1)
        penalty = np.eye(A.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(A.T @ A + penalty, A.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("observer is not fitted")
        X = np.asarray(X, dtype=float)
        Z = (X - self.mean_) / self.scale_
        A = np.concatenate([np.ones((len(Z), 1)), Z], axis=1)
        return A @ self.coef_
