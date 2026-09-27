import numpy as np

from smarfly2.observers.sequencing import SequencingControls, SequencingFeatureBuilder


def builder_unit_scale():
    train = np.vstack([np.ones((1, 6)), -np.ones((1, 6))])
    return SequencingFeatureBuilder().fit(train)


def test_conflict_closes_only_pending_route_and_then_decays():
    b = builder_unit_scale()
    events = np.zeros((4, 6))
    events[0] = 4.0
    events[1] = -4.0
    dt = np.full(4, 0.1)
    trace = b.transform(events, dt)

    assert trace.active_channel[0] == 0
    assert trace.active_channel[1] == -1
    assert trace.closure[1, 0] > 0.5
    assert np.allclose(trace.closure[1, 1:], 0.0)
    assert np.allclose(trace.channels[1, 1:], trace.channels[0, 1:])
    assert trace.closure[2, 0] < trace.closure[1, 0]


def test_no_route_closure_preserves_measured_closure_but_bypasses_its_gain():
    b = builder_unit_scale()
    events = np.zeros((10, 6))
    events[0] = 4.0
    events[1] = -4.0
    events[8] = 3.0
    dt = np.full(10, 0.1)
    normal = b.transform(events, dt)
    bypass = b.transform(events, dt, SequencingControls(no_route_closure=True))
    assert np.allclose(normal.closure, bypass.closure)
    assert not np.allclose(normal.channels, bypass.channels)


def test_publication_block_changes_only_external_publish_mask():
    b = builder_unit_scale()
    events = np.zeros((12, 6))
    events[:, 0] = np.linspace(-2.0, 2.0, len(events))
    events[:, 1] = 0.3
    dt = np.full(len(events), 0.1)

    normal = b.transform(events, dt)
    blocked = b.transform(events, dt, SequencingControls(publication_block=True))

    assert np.array_equal(normal.events, blocked.events)
    assert np.array_equal(normal.change_residue, blocked.change_residue)
    assert np.array_equal(normal.context_residue, blocked.context_residue)
    assert np.array_equal(normal.channels, blocked.channels)
    assert np.array_equal(normal.tuft_gain, blocked.tuft_gain)
    assert np.array_equal(normal.closure, blocked.closure)
    assert np.array_equal(normal.features, blocked.features)
    assert np.any(normal.publish_mask)
    assert not np.any(blocked.publish_mask)
