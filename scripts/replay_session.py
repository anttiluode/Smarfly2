#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cv2
import numpy as np

from smarfly2.session import Session


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--delay-ms", type=int, default=25)
    args = p.parse_args()
    session = Session.load(args.session)
    if args.headless:
        print(f"records={len(session.records)} horizon={session.horizon}")
        return
    if not session.records:
        print("empty session")
        return
    width = max(320, int(max(r.visible.x for r in session.records) + 30))
    height = max(240, int(max(r.visible.y for r in session.records) + 30))
    trail = []
    for record in session.records:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        trail.append((int(record.visible.x), int(record.visible.y)))
        if len(trail) > 120:
            trail.pop(0)
        for a, b in zip(trail[:-1], trail[1:]):
            cv2.line(frame, a, b, (100, 100, 100), 1)
        p0 = trail[-1]
        cv2.circle(frame, p0, 5, (255, 255, 255), -1)
        cv2.putText(frame, f"t={record.t:.2f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Smarfly2 replay", frame)
        if cv2.waitKey(args.delay_ms) & 0xFF in (27, ord('q')):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
