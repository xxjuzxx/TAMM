#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.features.profile_primitives import extract_all_profile_primitives
from src.utils.io import read_jsonl, read_yaml, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FlowPrim profile primitives from canonical flow JSONL records.")
    parser.add_argument("--flows", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats_out", required=True)
    parser.add_argument("--config", default="configs/cicids2017.yaml")
    args = parser.parse_args()

    config = read_yaml(args.config)
    profile_cfg = dict(config.get("profile_primitives") or {})
    flows = read_jsonl(args.flows)
    rows, stats = extract_all_profile_primitives(flows, profile_cfg)
    for row in rows:
        row["profile_primitive_provenance"] = "fixed_config"
        row["raw_ip_used_as_token"] = False
        row["absolute_time_used_as_token"] = False
        row["five_tuple_used_as_token"] = False
    write_jsonl(rows, args.out)
    write_json(
        {
            **stats,
            "profile_primitive_config": profile_cfg,
            "train_only": False,
            "note": "Standalone extraction over the provided flow file. Leave-one experiments build train-only profile peer primitives separately.",
            "raw_ip_used_as_token": False,
            "absolute_time_used_as_token": False,
            "five_tuple_used_as_token": False,
        },
        args.stats_out,
    )
    print(stats)


if __name__ == "__main__":
    main()
