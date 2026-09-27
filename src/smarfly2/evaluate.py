from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .controls import reset_history_at, shuffle_history_preserve_present
from .metrics import angular_mae, trajectory_rmse
from .observers.base import RidgeReadout
from .observers.resident import resident_features
from .observers.window import window_features
from .session import Session, make_targets


@dataclass
class EvaluationResult:
    train_indices: np.ndarray
    test_indices: np.ndarray
    targets: np.ndarray
    predictions: dict[str, np.ndarray]
    metrics: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "test_indices": self.test_indices.tolist(),
            "n_test": int(len(self.test_indices)),
            "metrics": self.metrics,
        }


def _fit_predict(phi: np.ndarray, y: np.ndarray, train_pos: np.ndarray, test_pos: np.ndarray) -> np.ndarray:
    model = RidgeReadout(alpha=1e-3).fit(phi[train_pos], y[train_pos])
    return model.predict(phi[test_pos])


def evaluate_session(
    session: Session,
    train_fraction: float = 0.6,
    horizon: int = 8,
) -> EvaluationResult:
    if not 0.2 <= train_fraction <= 0.8:
        raise ValueError("train_fraction must be between 0.2 and 0.8")
    indices, y = make_targets(session, horizon=horizon)
    if len(indices) < 40:
        raise ValueError("session too short for observer fitting; need at least 40 valid targets")

    X = np.stack([session.records[i].observer_vector() for i in indices])
    dt = np.asarray([session.records[i].dt for i in indices], dtype=float)
    n = len(indices)
    split = int(n * train_fraction)
    warmup = 11
    if split <= warmup or n - split < 2:
        raise ValueError("session too short after contiguous split and window warm-up")
    train_stop = split - horizon
    if train_stop <= warmup:
        raise ValueError("session too short to keep training targets out of held-out period")
    train_pos = np.arange(warmup, train_stop, dtype=int)
    test_pos = np.arange(split, n, dtype=int)

    reset_mask = np.ones(n, dtype=bool)
    feature_sets = {
        "present": X,
        "window": window_features(X, 12),
        "resident": resident_features(X, dt),
        "resident_reset": reset_history_at(X, dt, reset_mask),
        "resident_shuffle": shuffle_history_preserve_present(X, dt, seed=1729),
    }

    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    truth = y[test_pos]
    for name, phi in feature_sets.items():
        pred = _fit_predict(phi, y, train_pos, test_pos)
        predictions[name] = pred
        metrics[name] = {
            "endpoint_rmse": trajectory_rmse(pred, truth),
            "angular_mae": angular_mae(pred[:, 2], truth[:, 2]),
        }

    return EvaluationResult(
        train_indices=indices[train_pos],
        test_indices=indices[test_pos],
        targets=truth,
        predictions=predictions,
        metrics=metrics,
    )
