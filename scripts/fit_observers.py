#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from smarfly2.evaluate import evaluate_session
from smarfly2.replay import replay_frames
from smarfly2.session import Session
from smarfly2.world import SyntheticWorld


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("session", nargs="?", help="saved .npz session")
    p.add_argument("--synthetic", type=int, metavar="FRAMES")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    if args.synthetic is not None:
        world = SyntheticWorld(320, 240, seed=args.seed)
        frames = [world.frame(i) for i in range(args.synthetic)]
        session = replay_frames(frames, seed=args.seed, dt=1/30)
    elif args.session:
        session = Session.load(args.session)
    else:
        p.error("provide SESSION or --synthetic FRAMES")

    result = evaluate_session(session)
    payload = result.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
