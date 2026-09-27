from __future__ import annotations

import math
import numpy as np


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + math.pi) % (2.0 * math.pi) - math.pi


def trajectory_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] < 2:
        raise ValueError("pred and truth must have matching shape (n, >=2)")
    sq = np.sum((pred[:, :2] - truth[:, :2]) ** 2, axis=1)
    return float(np.sqrt(np.mean(sq)))


def angular_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if pred.shape != truth.shape:
        raise ValueError("pred and truth must have matching shape")
    diff = wrap_angle(pred - truth)
    return float(np.mean(np.abs(diff)))
