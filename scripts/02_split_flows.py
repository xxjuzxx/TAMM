#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.data.splits import attach_split_to_flows, build_split
from src.utils.io import read_jsonl, read_yaml, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", required=True)
    parser.add_argument(
        "--split",
        choices=[
            "small_debug",
            "random",
            "random_stratified",
            "stratified_random",
            "temporal",
            "temporal_chronological",
            "temporal_stratified",
            "day_wise",
            "leave_one_attack_out",
            "few_label",
        ],
        required=True,
    )
    parser.add_argument("--config", default="configs/cicids2017.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--flows_out", default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--leave_label", default=None)
    parser.add_argument("--leave_one_mode", choices=["classification", "anomaly"], default="classification")
    parser.add_argument("--train_per_label", type=int, default=8)
    parser.add_argument("--train_fraction", type=float, default=None)
    parser.add_argument("--max_per_label", type=int, default=1000)
    parser.add_argument("--train_days", nargs="+", default=None)
    parser.add_argument("--val_days", nargs="+", default=None)
    parser.add_argument("--test_days", nargs="+", default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    split_cfg = cfg.get("split", {})
    flows = read_jsonl(args.flows)
    payload = build_split(
        flows,
        args.split,
        val_ratio=float(args.val_ratio if args.val_ratio is not None else split_cfg.get("val_ratio", 0.1)),
        test_ratio=float(args.test_ratio if args.test_ratio is not None else split_cfg.get("test_ratio", 0.2)),
        seed=int(args.seed if args.seed is not None else cfg.get("seed", 42)),
        leave_label=args.leave_label,
        leave_one_mode=args.leave_one_mode,
        train_per_label=int(args.train_per_label),
        train_fraction=args.train_fraction,
        max_per_label=int(args.max_per_label),
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
    )
    payload["flows"] = args.flows
    payload["config"] = args.config
    write_json(payload, args.out)
    if args.flows_out:
        write_jsonl(attach_split_to_flows(flows, payload), args.flows_out)
    print({"out": args.out, "counts": payload["counts"], "label_counts": payload["label_counts"]})


if __name__ == "__main__":
    main()
