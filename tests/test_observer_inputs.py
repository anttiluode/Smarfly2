import numpy as np

from smarfly2.records import FrameRecord, HiddenFlyState, VisibleFeatures, VisibleFlyState


def test_observer_vector_excludes_hidden_state():
    record = FrameRecord(
        t=1.25,
        dt=1/30,
        visible=VisibleFlyState(x=10, y=20, heading=0.3, vx=1.2, vy=-0.4),
        features=VisibleFeatures(0.1, 0.2, 0.3, 0.4, 0.5),
        hidden=HiddenFlyState(91.1, 92.2, 93.3),
    )
    v = record.observer_vector()
    assert v.ndim == 1
    assert np.all(np.isfinite(v))
    assert not np.any(np.isin(v, [91.1, 92.2, 93.3]))
    assert np.isclose(v[-2], record.t)
    assert np.isclose(v[-1], record.dt)
