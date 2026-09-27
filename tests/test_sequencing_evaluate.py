from dataclasses import replace
import numpy as np

from smarfly2.controls import shuffle_events_preserve_present
from smarfly2.evaluate import evaluate_session
from smarfly2.observers.sequencing import SequencingObserver
from smarfly2.replay import replay_frames
from smarfly2.session import make_targets
from smarfly2.world import SyntheticWorld
from smarfly2.records import HiddenFlyState


def synthetic_session(n=220, seed=7):
    world = SyntheticWorld(120, 90, seed=seed)
    frames = [world.frame(i) for i in range(n)]
    return replay_frames(frames, seed=seed, dt=1/30)


def test_fit_uses_training_events_only_and_hidden_audit_is_inert():
    session = synthetic_session()
    indices, y = make_targets(session, horizon=8)
    train_pos = np.arange(11, 90)

    original = SequencingObserver().fit(session.records, indices, y, train_pos)

    mutated = list(session.records)
    for i in range(120, len(mutated)):
        r = mutated[i]
        mutated[i] = replace(
            r,
            features=replace(
                r.features,
                brightness_lr=r.features.brightness_lr + 1000.0,
                motion=r.features.motion + 1000.0,
            ),
        )
    changed_test = SequencingObserver().fit(mutated, indices, y, train_pos)

    assert np.array_equal(original.builder.scaler.mean_, changed_test.builder.scaler.mean_)
    assert np.array_equal(original.builder.scaler.scale_, changed_test.builder.scaler.scale_)
    assert np.array_equal(original.readout.coef_, changed_test.readout.coef_)

    hidden_changed = [replace(r, hidden=HiddenFlyState(1e9, -1e9, 5e8)) for r in session.records]
    p1, t1 = original.predict_internal(session.records, indices)
    p2, t2 = original.predict_internal(hidden_changed, indices)
    assert np.array_equal(p1, p2)
    assert np.array_equal(t1.features, t2.features)


def test_shuffle_events_uses_only_strictly_prior_events():
    events = np.arange(60, dtype=float).reshape(10, 6)
    events[0] = 0.0
    shuffled = shuffle_events_preserve_present(events, seed=4)
    assert shuffled.shape == events.shape
    assert np.array_equal(shuffled[0], events[0])
    for i in range(1, len(events)):
        assert any(np.array_equal(shuffled[i], events[j]) for j in range(i))


def test_shuffle_events_is_causal_with_respect_to_future_mutations():
    events = np.arange(72, dtype=float).reshape(12, 6)
    events[0] = 0.0
    before = shuffle_events_preserve_present(events, seed=17)
    mutated = events.copy()
    mutated[7:] += 100000.0
    after = shuffle_events_preserve_present(mutated, seed=17)
    assert np.array_equal(before[:7], after[:7])


def test_evaluation_adds_sequencing_controls_on_same_targets():
    session = synthetic_session(n=500, seed=5)
    result = evaluate_session(session, train_fraction=0.6, horizon=8)
    expected = {
        "present", "window", "resident", "resident_reset", "resident_shuffle",
        "sequencing", "sequencing_event_shuffle", "sequencing_no_tuft",
        "sequencing_no_route_closure",
    }
    assert set(result.predictions) == expected
    n = len(result.test_indices)
    for name in expected:
        assert result.predictions[name].shape == (n, 3)
        assert np.all(np.isfinite(result.predictions[name]))
    assert 0.0 <= result.publication_rate["sequencing"] <= 1.0
    assert result.invariance_checks["sequencing_publication_block"] is True


def test_streaming_prediction_matches_batch_sequence_exactly():
    session = synthetic_session(n=180, seed=9)
    indices, y = make_targets(session, horizon=8)
    train = np.arange(11, 90)
    observer = SequencingObserver().fit(session.records, indices, y, train)

    records = session.records[100:150]
    batch_indices = np.arange(len(records), dtype=int)
    batch, trace = observer.predict_internal(records, batch_indices)

    stream = observer.new_stream()
    streamed = []
    for record in records:
        pred, step = stream.step(record)
        streamed.append(pred)
    streamed = np.asarray(streamed)
    assert np.allclose(streamed, batch, rtol=0.0, atol=1e-12)
    assert np.array_equal(stream.runtime.channels, trace.channels[-1])
    assert np.array_equal(stream.runtime.context, trace.context_residue[-1])
