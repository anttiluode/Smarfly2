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


def test_sequencing_audit_is_attached_only_after_visible_pair_selection():
    import numpy as np
    from smarfly2.events import raw_events
    from smarfly2.matched_present import attach_sequencing_audit
    from smarfly2.observers.sequencing import SequencingFeatureBuilder

    records = [rec(i) for i in range(80)]
    for k in range(0, 8):
        r = records[k]
        records[k] = FrameRecord(r.t, r.dt, r.visible,
            VisibleFeatures(0.3, 0.8, 0.2, -0.1, 0.4), r.hidden)
    for k in range(40, 48):
        r = records[k]
        records[k] = FrameRecord(r.t, r.dt, r.visible,
            VisibleFeatures(0.3, -0.8, 0.2, -0.1, 0.4), r.hidden)
    records[10] = rec(10, x=42.0, hidden=0.7)
    records[50] = rec(50, x=42.0, hidden=-0.7)
    for k in range(1, 9):
        records[10+k] = rec(10+k, x=42.0 + k, hidden=0.7)
        records[50+k] = rec(50+k, x=42.0 - k, hidden=-0.7)
    session = Session(records)

    plain = find_matched_present_pairs(session, max_pairs=20)
    events = raw_events(records)
    builder = SequencingFeatureBuilder().fit(events[:40])
    trace = builder.transform(events, np.full(len(events), 1/30))
    pred_map = {i: np.array([float(i), 0.0, 0.0]) for i in range(len(records))}
    audited = attach_sequencing_audit(plain, trace, pred_map)

    assert [(p.i, p.j) for p in audited] == [(p.i, p.j) for p in plain]
    pair = next(p for p in audited if {p.i, p.j} == {10, 50})
    assert pair.event_residue_distance is not None and pair.event_residue_distance > 0
    assert pair.tuft_gain_distance is not None and pair.tuft_gain_distance > 0
    assert pair.sequencing_prediction_divergence is not None
