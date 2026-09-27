import math
import numpy as np
import pytest

from smarfly2.events import EventScaler, raw_events
from smarfly2.records import FrameRecord, HiddenFlyState, VisibleFeatures, VisibleFlyState


def rec(i, *, x=10.0, y=20.0, heading=0.2, vx=3.0, vy=4.0,
        brightness_lr=0.1, motion=0.2, motion_lr=-0.1, contrast=0.3):
    return FrameRecord(
        t=i / 30,
        dt=1 / 30,
        visible=VisibleFlyState(x, y, heading, vx, vy),
        features=VisibleFeatures(0.5, brightness_lr, motion, motion_lr, contrast),
        hidden=HiddenFlyState(999.0 + i, -999.0 - i, 123.0 + i),
    )


def test_constant_stream_has_zero_innovation_after_first_row():
    e = raw_events([rec(i) for i in range(5)])
    assert e.shape == (5, 6)
    assert np.array_equal(e[0], np.zeros(6))
    assert np.allclose(e[1:], 0.0)


def test_signed_feature_changes_retain_sign():
    a = rec(0, brightness_lr=-0.4, motion_lr=0.3)
    b = rec(1, brightness_lr=0.2, motion_lr=-0.1)
    e = raw_events([a, b])
    assert e[1, 0] > 0
    assert e[1, 2] < 0


def test_turn_innovation_wraps_across_pi():
    a = rec(0, heading=math.pi - 0.02)
    b = rec(1, heading=-math.pi + 0.03)
    e = raw_events([a, b])
    assert e[1, 4] == pytest.approx(0.05)


def test_absolute_position_change_alone_is_not_an_event():
    a = rec(0, x=1.0, y=2.0)
    b = rec(1, x=600.0, y=400.0)
    e = raw_events([a, b])
    assert np.allclose(e[1], 0.0)


def test_speed_innovation_uses_visible_speed_magnitude():
    a = rec(0, vx=3.0, vy=4.0)
    b = rec(1, vx=0.0, vy=13.0)
    e = raw_events([a, b])
    assert e[1, 5] == pytest.approx(8.0)


def test_event_scaler_uses_only_rows_passed_to_fit():
    train = np.arange(30, dtype=float).reshape(5, 6)
    heldout = np.full((5, 6), 1e9)
    all_events = np.vstack([train, heldout])
    a = EventScaler().fit(all_events[:5])
    all_events[5:] += 1e12
    b = EventScaler().fit(all_events[:5])
    assert np.array_equal(a.mean_, b.mean_)
    assert np.array_equal(a.scale_, b.scale_)


def test_event_scaler_rejects_empty_or_nonfinite_training_data():
    with pytest.raises(ValueError):
        EventScaler().fit(np.empty((0, 6)))
    bad = np.zeros((2, 6))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        EventScaler().fit(bad)
