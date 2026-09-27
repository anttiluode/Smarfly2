import numpy as np

from smarfly2.controls import reset_history_at, shuffle_history_preserve_present


def test_shuffle_preserves_current_visible_rows_but_changes_history_features():
    X = np.arange(30, dtype=float).reshape(10, 3)
    dt = np.full(10, 0.05)
    normal = shuffle_history_preserve_present(X, dt, seed=None)
    shuffled = shuffle_history_preserve_present(X, dt, seed=4)
    assert np.array_equal(normal[:, :3], X)
    assert np.array_equal(shuffled[:, :3], X)
    assert not np.allclose(normal[5:, 3:], shuffled[5:, 3:])


def test_reset_control_preserves_current_rows_and_erases_persistent_state_at_mask():
    X = np.arange(18, dtype=float).reshape(6, 3)
    dt = np.full(6, 0.1)
    mask = np.array([False, False, False, True, False, False])
    controlled = reset_history_at(X, dt, mask)
    fresh = reset_history_at(X[3:], dt[3:], np.array([True, False, False]))
    assert np.array_equal(controlled[:, :3], X)
    assert np.allclose(controlled[3], fresh[0])


def test_shuffle_history_control_uses_single_pass_resident_transform(monkeypatch):
    import smarfly2.controls as controls

    calls = {"n": 0}
    original = controls.resident_features

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(controls, "resident_features", counted)
    X = np.arange(600, dtype=float).reshape(100, 6)
    dt = np.full(100, 1/30)
    out = controls.shuffle_history_preserve_present(X, dt, seed=7)
    assert np.array_equal(out[:, :6], X)
    assert calls["n"] <= 2
