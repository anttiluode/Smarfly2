import numpy as np

from smarfly2.gui import LiveBench


class Provider:
    def __init__(self, frames):
        self.frames = list(frames)

    def __call__(self):
        return self.frames.pop(0) if self.frames else None


def test_headless_bench_skips_missing_frame_without_advancing_state():
    frame = np.full((80, 120, 3), 80, dtype=np.uint8)
    bench = LiveBench(frame_provider=Provider([None, frame]), headless=True, seed=7)
    assert len(bench.session.records) == 0
    assert bench.step_once() is False
    assert len(bench.session.records) == 0
    assert bench.step_once() is True
    assert len(bench.session.records) == 1


def test_absent_model_weights_do_not_prevent_collection():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    bench = LiveBench(frame_provider=Provider([frame]), headless=True, model_path=None)
    assert bench.step_once() is True
    assert len(bench.session.records) == 1
    assert bench.last_predictions == {}


def test_live_bench_sets_world_dimensions_from_first_valid_frame():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    bench = LiveBench(frame_provider=Provider([frame]), headless=True, model_path=None)
    assert bench.session.world_width is None
    assert bench.session.world_height is None
    assert bench.step_once() is True
    assert bench.session.world_width == 120
    assert bench.session.world_height == 80


def test_headless_sequencing_inspector_exposes_internal_state_when_publication_blocked(tmp_path):
    from smarfly2.replay import replay_frames
    from smarfly2.world import SyntheticWorld

    world = SyntheticWorld(120, 80, seed=3)
    training = replay_frames([world.frame(i) for i in range(120)], seed=4, dt=1/30)
    path = tmp_path / "train.npz"
    training.save(path)

    frame = world.frame(121)
    bench = LiveBench(frame_provider=Provider([frame]), headless=True, model_path=str(path))
    bench.publication_block = True
    assert bench.step_once() is True
    state = bench.get_sequencing_inspector()
    assert state is not None
    assert len(state["channels"]) == 4
    assert len(state["tuft_gain"]) == 4
    assert len(state["closure"]) == 4
    assert state["internal_prediction"] is not None
    assert state["published_prediction"] is None


def test_live_bench_uses_incremental_sequencing_stream_not_full_history_replay(tmp_path, monkeypatch):
    from smarfly2.replay import replay_frames
    from smarfly2.world import SyntheticWorld

    world = SyntheticWorld(120, 80, seed=8)
    training = replay_frames([world.frame(i) for i in range(120)], seed=2, dt=1/30)
    path = tmp_path / "train-stream.npz"
    training.save(path)
    bench = LiveBench(frame_provider=Provider([world.frame(121)]), headless=True, model_path=str(path))

    def forbidden(*args, **kwargs):
        raise AssertionError("live path replayed full sequencing history")

    monkeypatch.setattr(bench._sequencing_model, "predict_internal", forbidden)
    assert bench.step_once() is True
    assert bench.get_sequencing_inspector() is not None


def test_legacy_model_session_requires_explicit_world_size(tmp_path):
    from smarfly2.replay import replay_frames
    from smarfly2.world import SyntheticWorld
    from smarfly2.session import Session
    import pytest

    world = SyntheticWorld(120, 80, seed=10)
    training = replay_frames([world.frame(i) for i in range(120)], seed=2, dt=1/30)
    legacy = Session(training.records, horizon=training.horizon)
    path = tmp_path / "legacy-model.npz"
    legacy.save(path)

    with pytest.raises(ValueError, match="world dimensions"):
        LiveBench(frame_provider=Provider([]), headless=True, model_path=str(path))

    bench = LiveBench(
        frame_provider=Provider([]),
        headless=True,
        model_path=str(path),
        model_world_size=(120, 80),
    )
    assert bench._sequencing_model is not None
