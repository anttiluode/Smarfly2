import math
import numpy as np
import pytest

from smarfly2.records import FrameRecord, HiddenFlyState, VisibleFeatures, VisibleFlyState
from smarfly2.replay import replay_frames
from smarfly2.session import Session, make_targets
from smarfly2.world import SyntheticWorld


def make_record(i, heading=0.0):
    return FrameRecord(
        t=i/30,
        dt=1/30,
        visible=VisibleFlyState(i*1.0, i*2.0, heading, 1.0, 2.0),
        features=VisibleFeatures(0.1, 0.2, 0.3, 0.4, 0.5),
        hidden=HiddenFlyState(0.01*i, -0.01*i, 0.02*i),
    )


def test_npz_round_trip_preserves_records(tmp_path):
    session = Session([make_record(i) for i in range(12)], horizon=8)
    path = tmp_path / "s.npz"
    session.save(path)
    loaded = Session.load(path)
    assert loaded.horizon == 8
    assert loaded.records == session.records


def test_make_targets_uses_eight_frame_delta_and_circular_heading():
    records = [make_record(i, heading=0.0) for i in range(10)]
    records[0] = make_record(0, heading=math.pi - 0.05)
    records[8] = make_record(8, heading=-math.pi + 0.05)
    session = Session(records, horizon=8)
    idx, y = make_targets(session, horizon=8)
    assert idx.tolist() == [0, 1]
    assert y[0, 0] == pytest.approx(8.0)
    assert y[0, 1] == pytest.approx(16.0)
    assert y[0, 2] == pytest.approx(0.1)


def test_replay_is_exact_for_same_frames_and_seed():
    world = SyntheticWorld(96, 64, seed=4)
    frames = [world.frame(i) for i in range(40)]
    a = replay_frames(frames, seed=11, dt=1/30)
    b = replay_frames(frames, seed=11, dt=1/30)
    assert a.records == b.records


def test_short_session_rejected_for_targets():
    session = Session([make_record(i) for i in range(9)], horizon=8)
    with pytest.raises(ValueError, match="too short"):
        make_targets(session, horizon=8)
