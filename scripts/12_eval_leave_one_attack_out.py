#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from src.data.splits import build_split
from src.utils.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", required=True)
    parser.add_argument("--leave_label", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["classification", "anomaly"], default="classification")
    args = parser.parse_args()
    flows = read_jsonl(args.flows)
    payload = build_split(
        flows,
        "leave_one_attack_out",
        seed=args.seed,
        leave_label=args.leave_label,
        leave_one_mode=args.mode,
    )
    write_json(payload, args.out)
    print(json.dumps({"out": args.out, "leave_label": payload.get("leave_label"), "counts": payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
