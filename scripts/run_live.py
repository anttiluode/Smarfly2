#!/usr/bin/env python3
from __future__ import annotations

import argparse

from smarfly2.gui import LiveBench


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model-session", help="optional prior session used to fit live prediction overlays")
    p.add_argument("--save", default="session.npz")
    p.add_argument("--world-width", type=float)
    p.add_argument("--world-height", type=float)
    args = p.parse_args()
    model_world_size = None
    if args.world_width is not None or args.world_height is not None:
        if args.world_width is None or args.world_height is None:
            p.error("--world-width and --world-height must be supplied together")
        model_world_size = (args.world_width, args.world_height)
    bench = LiveBench(
        camera_index=args.camera,
        model_path=args.model_session,
        model_world_size=model_world_size,
        seed=args.seed,
    )
    try:
        bench.run()
    finally:
        bench.close()
        if bench.session.records:
            bench.save_session(args.save)
            print(f"saved {len(bench.session.records)} records to {args.save}")


if __name__ == "__main__":
    main()
