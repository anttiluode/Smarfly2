import numpy as np

from smarfly2.world import SyntheticWorld


def test_synthetic_world_is_seeded_and_time_varying():
    a = SyntheticWorld(96, 64, seed=3)
    b = SyntheticWorld(96, 64, seed=3)
    f10a = a.frame(10)
    f10b = b.frame(10)
    f11 = a.frame(11)
    assert f10a.shape == (64, 96, 3)
    assert f10a.dtype == np.uint8
    assert np.array_equal(f10a, f10b)
    assert not np.array_equal(f10a, f11)
