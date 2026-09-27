#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from smarfly2.evaluate import evaluate_session
from smarfly2.observers.sequencing import SequencingParams
from smarfly2.replay import replay_frames
from smarfly2.session import Session
from smarfly2.world import SyntheticWorld


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("session", nargs="?", help="saved .npz session")
    p.add_argument("--synthetic", type=int, metavar="FRAMES")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path)
    p.add_argument("--world-width", type=float)
    p.add_argument("--world-height", type=float)
    args = p.parse_args()

    if args.synthetic is not None:
        world = SyntheticWorld(320, 240, seed=args.seed)
        frames = [world.frame(i) for i in range(args.synthetic)]
        session = replay_frames(frames, seed=args.seed, dt=1/30)
    elif args.session:
        session = Session.load(args.session)
    else:
        p.error("provide SESSION or --synthetic FRAMES")

    world_size = None
    if args.world_width is not None or args.world_height is not None:
        if args.world_width is None or args.world_height is None:
            p.error("--world-width and --world-height must be supplied together")
        world_size = (args.world_width, args.world_height)
    result = evaluate_session(session, world_size=world_size)
    payload = result.to_dict()
    effective_width = world_size[0] if world_size is not None else session.world_width
    effective_height = world_size[1] if world_size is not None else session.world_height
    payload.update({
        "n_records": len(session.records),
        "horizon": session.horizon,
        "world_width": effective_width,
        "world_height": effective_height,
        "sequencing_params": asdict(SequencingParams()),
    })
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
