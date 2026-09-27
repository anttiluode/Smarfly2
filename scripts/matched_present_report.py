#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from smarfly2.matched_present import find_matched_present_pairs
from smarfly2.session import Session


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--max-pairs", type=int, default=20)
    args = p.parse_args()
    session = Session.load(args.session)
    pairs = find_matched_present_pairs(session, max_pairs=args.max_pairs)
    print(json.dumps([asdict(x) for x in pairs], indent=2))


if __name__ == "__main__":
    main()
