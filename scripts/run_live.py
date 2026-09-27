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
    args = p.parse_args()
    bench = LiveBench(camera_index=args.camera, model_path=args.model_session, seed=args.seed)
    try:
        bench.run()
    finally:
        bench.close()
        if bench.session.records:
            bench.save_session(args.save)
            print(f"saved {len(bench.session.records)} records to {args.save}")


if __name__ == "__main__":
    main()
