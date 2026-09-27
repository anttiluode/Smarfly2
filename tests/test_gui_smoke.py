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
