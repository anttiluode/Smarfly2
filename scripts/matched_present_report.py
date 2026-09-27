#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import numpy as np

from smarfly2.matched_present import attach_sequencing_audit, find_matched_present_pairs
from smarfly2.observers.sequencing import SequencingObserver
from smarfly2.session import Session, make_targets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--max-pairs", type=int, default=20)
    p.add_argument("--sequencing", action="store_true")
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
    if args.sequencing and world_size is None and (session.world_width is None or session.world_height is None):
        raise ValueError("world dimensions required for legacy session sequencing")
    pairs = find_matched_present_pairs(session, max_pairs=args.max_pairs)
    if args.sequencing and pairs:
        indices, y = make_targets(session, horizon=session.horizon, world_size=world_size)
        train = np.arange(min(11, max(0, len(indices) - 1)), len(indices), dtype=int)
        if len(train):
            observer = SequencingObserver().fit(session.records, indices, y, train)
            pred, trace = observer.predict_internal(session.records, indices)
            pred_map = {int(idx): pred[pos] for pos, idx in enumerate(indices)}
            pairs = attach_sequencing_audit(pairs, trace, pred_map)
    print(json.dumps([asdict(x) for x in pairs], indent=2))


if __name__ == "__main__":
    main()
