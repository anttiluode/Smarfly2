import numpy as np

from smarfly2.observers.resident import ResidentObserver, resident_features


def test_same_present_different_history_produces_different_resident_state():
    a = np.array([[2.0], [2.0], [2.0], [0.25]])
    b = np.array([[-2.0], [-2.0], [-2.0], [0.25]])
    dt = np.full(4, 0.1)
    pa = resident_features(a, dt)
    pb = resident_features(b, dt)
    assert np.allclose(pa[-1, :1], pb[-1, :1])
    assert not np.allclose(pa[-1, 1:], pb[-1, 1:])


def test_reset_before_same_present_erases_prior_residue():
    a = np.array([[2.0], [2.0], [2.0], [0.25]])
    b = np.array([[-2.0], [-2.0], [-2.0], [0.25]])
    dt = np.full(4, 0.1)
    reset = np.array([False, False, False, True])
    pa = resident_features(a, dt, reset_mask=reset)
    pb = resident_features(b, dt, reset_mask=reset)
    assert np.allclose(pa[-1], pb[-1])


def test_resident_observer_fits_and_predicts_three_targets():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = np.column_stack([X[:, 0], X[:, 1], X[:, 0] - X[:, 1]])
    model = ResidentObserver().fit(X[:40], y[:40], dt=np.full(40, 1/30))
    pred = model.predict(X[40:], dt=np.full(20, 1/30))
    assert pred.shape == (20, 3)
    assert np.all(np.isfinite(pred))
