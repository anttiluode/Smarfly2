import numpy as np
import pytest

from smarfly2.observers.sequencing import (
    SequencingControls,
    SequencingFeatureBuilder,
    SequencingParams,
)


def fitted_builder():
    train = np.array([
        [-1.0, 0.0, 0.2, -0.1, 0.0, 0.1],
        [0.0, 0.4, -0.3, 0.2, 0.1, -0.2],
        [1.0, -0.2, 0.1, 0.0, -0.1, 0.3],
        [0.5, 0.1, 0.0, -0.2, 0.2, 0.0],
    ])
    return SequencingFeatureBuilder().fit(train)


def test_residues_follow_only_event_matrix_and_stay_finite_with_irregular_dt():
    builder = fitted_builder()
    events = np.zeros((6, 6))
    events[1, 0] = 1.0
    events[3, 4] = -0.5
    dt = np.array([0.03, 0.07, 0.02, 0.31, 0.11, 0.05])
    trace = builder.transform(events, dt)
    assert trace.change_residue.shape == events.shape
    assert trace.context_residue.shape == events.shape
    assert np.all(np.isfinite(trace.change_residue))
    assert np.all(np.isfinite(trace.context_residue))
    assert np.any(np.abs(trace.change_residue[1:]) > 0)


def test_nonpositive_dt_is_rejected():
    builder = fitted_builder()
    events = np.zeros((3, 6))
    with pytest.raises(ValueError, match="positive"):
        builder.transform(events, np.array([0.1, 0.0, 0.1]))


def test_scheduler_has_four_active_bins_separated_by_dead_bins():
    builder = fitted_builder()
    events = np.tile(np.array([[1.0, 0.5, -0.25, 0.1, 0.2, -0.1]]), (8, 1))
    trace = builder.transform(events, np.full(8, 0.1))
    assert trace.phase_bin.tolist() == list(range(8))
    assert trace.active_channel.tolist() == [0, -1, 1, -1, 2, -1, 3, -1]
    for i in [1, 3, 5, 7]:
        assert np.allclose(trace.channels[i], trace.channels[i - 1])


def test_only_scheduled_channel_changes_in_active_bins():
    builder = fitted_builder()
    events = np.tile(np.array([[0.6, -0.2, 0.4, 0.1, 0.3, -0.1]]), (7, 1))
    trace = builder.transform(events, np.full(7, 0.1))
    previous = np.zeros(4)
    for i, active in enumerate(trace.active_channel):
        changed = np.flatnonzero(np.abs(trace.channels[i] - previous) > 1e-12)
        if active == -1:
            assert changed.size == 0
        else:
            assert changed.tolist() == [active]
        previous = trace.channels[i].copy()


def test_flatten_phase_updates_multiple_channels_at_once():
    builder = fitted_builder()
    events = np.tile(np.array([[0.8, -0.3, 0.2, 0.1, 0.4, -0.2]]), (2, 1))
    trace = builder.transform(events, np.full(2, 0.1), SequencingControls(flatten_phase=True))
    changed = np.flatnonzero(np.abs(trace.channels[0]) > 1e-12)
    assert changed.size > 1


def test_same_present_different_history_changes_tuft_susceptibility():
    builder = fitted_builder()
    a = np.zeros((16, 6))
    b = np.zeros((16, 6))
    a[:12, 0] = 1.0
    b[:12, 0] = -1.0
    a[-1] = 0.0
    b[-1] = 0.0
    dt = np.full(16, 0.1)
    ta = builder.transform(a, dt)
    tb = builder.transform(b, dt)
    assert not np.allclose(ta.tuft_gain[-1], tb.tuft_gain[-1])

    ca = builder.transform(a, dt, SequencingControls(no_tuft=True))
    cb = builder.transform(b, dt, SequencingControls(no_tuft=True))
    assert np.array_equal(ca.tuft_gain, np.ones_like(ca.tuft_gain))
    assert np.array_equal(cb.tuft_gain, np.ones_like(cb.tuft_gain))
