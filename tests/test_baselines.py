import numpy as np

from smarfly2.observers.present import PresentObserver
from smarfly2.observers.window import WindowObserver


def history_task(n=80):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(n, 1))
    y = np.zeros((n, 1))
    y[1:, 0] = x[:-1, 0]
    return x, y


def test_present_observer_prediction_depends_only_on_current_row():
    X, y = history_task()
    model = PresentObserver(alpha=1e-3)
    model.fit(X, y)
    a = np.array([[10.0], [0.25]])
    b = np.array([[-10.0], [0.25]])
    assert np.allclose(model.predict(a)[-1], model.predict(b)[-1])


def test_window_observer_uses_prior_rows_when_current_row_is_fixed():
    X, y = history_task()
    model = WindowObserver(window=4, alpha=1e-3)
    model.fit(X, y)
    a = np.array([[0.0], [0.0], [2.0], [0.25]])
    b = np.array([[0.0], [0.0], [-2.0], [0.25]])
    pa = model.predict(a)[-1, 0]
    pb = model.predict(b)[-1, 0]
    assert abs(pa - pb) > 1.0


def test_observer_arms_return_same_number_of_rows():
    X, y = history_task(50)
    p = PresentObserver().fit(X[:30], y[:30])
    w = WindowObserver(window=12).fit(X[:30], y[:30])
    idx = np.arange(30, 50)
    pp = p.predict(X[idx])
    pw = w.predict(X[idx])
    assert len(pp) == len(pw) == len(idx)
