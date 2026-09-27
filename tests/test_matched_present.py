from smarfly2.matched_present import find_matched_present_pairs
from smarfly2.records import FrameRecord, HiddenFlyState, VisibleFeatures, VisibleFlyState
from smarfly2.session import Session


def rec(i, x=None, hidden=0.0):
    x = float(i) if x is None else float(x)
    return FrameRecord(
        t=i/30,
        dt=1/30,
        visible=VisibleFlyState(x=x, y=20.0, heading=0.2, vx=1.0, vy=0.0),
        features=VisibleFeatures(0.3, 0.1, 0.2, -0.1, 0.4),
        hidden=HiddenFlyState(hidden, -hidden, hidden/2),
    )


def test_matcher_finds_same_visible_present_with_different_hidden_history_without_hidden_search():
    records = [rec(i) for i in range(70)]
    records[5] = rec(5, x=42.0, hidden=0.9)
    records[45] = rec(45, x=42.0, hidden=-0.9)
    for k in range(1, 9):
        records[5+k] = rec(5+k, x=42.0 + k, hidden=0.9)
        records[45+k] = rec(45+k, x=42.0 - k, hidden=-0.9)
    pairs = find_matched_present_pairs(Session(records), max_pairs=20)
    keys = {(min(p.i, p.j), max(p.i, p.j)) for p in pairs}
    assert (5, 45) in keys
    pair = next(p for p in pairs if {p.i, p.j} == {5, 45})
    assert pair.visible_distance < 1e-9
    assert pair.hidden_distance > 1.0
    assert pair.future_divergence > 5.0
