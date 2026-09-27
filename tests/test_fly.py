import numpy as np

from smarfly2.fly import Fly
from smarfly2.records import VisibleFeatures


def tape():
    out = []
    for i in range(40):
        out.append(VisibleFeatures(
            brightness=0.4 + 0.1*np.sin(i/5),
            brightness_lr=0.6*np.sin(i/7),
            motion=0.2 + 0.1*np.cos(i/4),
            motion_lr=0.5*np.cos(i/6),
            contrast=0.3 + 0.05*np.sin(i/3),
        ))
    return out


def test_fly_is_deterministic_under_same_seed_and_tape():
    a = Fly(640, 480, seed=7)
    b = Fly(640, 480, seed=7)
    for f in tape():
        va, ha = a.step(f, 1/30)
        vb, hb = b.step(f, 1/30)
        assert va == vb
        assert ha == hb


def test_hidden_states_stay_finite_and_bounded():
    fly = Fly(640, 480, seed=7)
    for f in tape() * 10:
        _, h = fly.step(f, 1/30)
        values = np.array([h.fast_trace, h.slow_context, h.adaptation])
        assert np.all(np.isfinite(values))
        assert np.all(np.abs(values) <= 1.0 + 1e-12)
