import numpy as np

from smarfly2.evaluate import evaluate_session
from smarfly2.replay import replay_frames
from smarfly2.world import SyntheticWorld


def test_all_observer_arms_score_identical_heldout_indices():
    world = SyntheticWorld(160, 120, seed=5)
    frames = [world.frame(i) for i in range(500)]
    session = replay_frames(frames, seed=7, dt=1/30)
    result = evaluate_session(session, train_fraction=0.6, horizon=8)
    expected = {
        "present", "window", "resident", "resident_reset", "resident_shuffle",
        "sequencing", "sequencing_event_shuffle", "sequencing_no_tuft",
        "sequencing_no_route_closure",
    }
    assert set(result.predictions) == expected
    n = len(result.test_indices)
    assert n > 0
    for name in expected:
        pred = result.predictions[name]
        assert pred.shape == (n, 3)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(list(result.metrics[name].values())))
    assert result.targets.shape == (n, 3)


def test_training_targets_do_not_reach_into_heldout_period():
    world = SyntheticWorld(120, 90, seed=3)
    frames = [world.frame(i) for i in range(220)]
    session = replay_frames(frames, seed=9, dt=1/30)
    result = evaluate_session(session, train_fraction=0.6, horizon=8)
    assert result.train_indices[-1] + 8 < result.test_indices[0]
