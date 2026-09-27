import math
import numpy as np
import pytest

from smarfly2.metrics import angular_mae, trajectory_rmse


def test_angular_mae_wraps_across_pi():
    truth = np.array([[-0.01]])
    pred = np.array([[2*math.pi - 0.01]])
    assert angular_mae(pred, truth) < 1e-9


def test_trajectory_rmse_uses_xy_endpoint_distance():
    pred = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    truth = np.zeros_like(pred)
    assert trajectory_rmse(pred, truth) == pytest.approx(math.sqrt((25.0 + 0.0) / 2.0))
