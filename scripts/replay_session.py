#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cv2
import numpy as np

from smarfly2.observers.sequencing import SequencingObserver
from smarfly2.session import Session, make_targets


def _sequencing(session: Session, world_size: tuple[float, float] | None = None):
    if world_size is None and (session.world_width is None or session.world_height is None):
        raise ValueError("world dimensions required for legacy session sequencing")
    indices, y = make_targets(session, horizon=session.horizon, world_size=world_size)
    if len(indices) <= 12:
        return {}, None
    train = np.arange(11, len(indices), dtype=int)
    observer = SequencingObserver().fit(session.records, indices, y, train)
    pred, trace = observer.predict_internal(session.records, indices)
    return {int(idx): pred[pos] for pos, idx in enumerate(indices)}, trace


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--delay-ms", type=int, default=25)
    p.add_argument("--inspect-sequencing", action="store_true")
    p.add_argument("--world-width", type=float)
    p.add_argument("--world-height", type=float)
    args = p.parse_args()
    session = Session.load(args.session)
    world_size = None
    if args.world_width is not None or args.world_height is not None:
        if args.world_width is None or args.world_height is None:
            p.error("--world-width and --world-height must be supplied together")
        world_size = (args.world_width, args.world_height)
        session.world_width = int(args.world_width)
        session.world_height = int(args.world_height)
    pred_map, trace = _sequencing(session, world_size=world_size) if args.inspect_sequencing and session.records else ({}, None)
    if args.headless:
        text = f"records={len(session.records)} horizon={session.horizon}"
        if trace is not None and len(trace.channels):
            text += f" phase_bin={int(trace.phase_bin[-1])} active={int(trace.active_channel[-1])} published={bool(trace.publish_mask[-1])}"
        print(text)
        return
    if not session.records:
        print("empty session")
        return
    width = session.world_width or max(320, int(max(r.visible.x for r in session.records) + 30))
    height = session.world_height or max(240, int(max(r.visible.y for r in session.records) + 30))
    trail = []
    for i, record in enumerate(session.records):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        trail.append((int(record.visible.x), int(record.visible.y)))
        if len(trail) > 120:
            trail.pop(0)
        for a, b in zip(trail[:-1], trail[1:]):
            cv2.line(frame, a, b, (100, 100, 100), 1)
        p0 = trail[-1]
        cv2.circle(frame, p0, 5, (255, 255, 255), -1)
        if i in pred_map:
            pred = pred_map[i]
            p1 = (int(round(record.visible.x + pred[0])), int(round(record.visible.y + pred[1])))
            cv2.line(frame, p0, p1, (80, 220, 255), 2)
        cv2.putText(frame, f"t={record.t:.2f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        if trace is not None and i < len(trace.channels):
            msg = f"SEQ bin={int(trace.phase_bin[i])} active={int(trace.active_channel[i])} pub={bool(trace.publish_mask[i])}"
            cv2.putText(frame, msg, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 140), 1)
        cv2.imshow("Smarfly2 replay", frame)
        if cv2.waitKey(args.delay_ms) & 0xFF in (27, ord('q')):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
