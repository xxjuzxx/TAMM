#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.features.behavior_tokens import build_behavior_token_dataset
from src.utils.io import read_jsonl, read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--profile_primitives", required=True)
    parser.add_argument("--config", default="configs/cicids2017.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--profile_mode", dest="profile_mode", choices=["none", "packet", "summary", "full"], default=None)
    parser.add_argument("--disable_flow_summary", action="store_true")
    parser.add_argument("--disable_packet_tokens", action="store_true")
    parser.add_argument("--disable_burst_tokens", action="store_true")
    parser.add_argument("--disable_rhythm_tokens", action="store_true")
    parser.add_argument("--use_burst_shape_tokens", action="store_true")
    parser.add_argument("--use_transition_profile_tokens", dest="use_transition_profile_tokens", action="store_true")
    parser.add_argument("--use_raw_profile_id_tokens", dest="use_raw_profile_id_tokens", action="store_true")
    parser.add_argument("--use_service_tokens", action="store_true")
    parser.add_argument("--label_field", choices=["label", "attack_family"], default=None)
    parser.add_argument("--manifest_out", default=None)
    parser.add_argument("--stats_out", default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config).get("tokenizer", {})
    cfg["use_service_tokens"] = False
    if args.profile_mode is not None:
        cfg["profile_mode"] = args.profile_mode
    if args.max_len is not None:
        cfg["max_len"] = int(args.max_len)
    if args.disable_flow_summary:
        cfg["include_flow_summary"] = False
    if args.disable_packet_tokens:
        cfg["include_packet_tokens"] = False
    if args.disable_burst_tokens:
        cfg["include_burst_tokens"] = False
    if args.disable_rhythm_tokens:
        cfg["include_rhythm_tokens"] = False
    if args.use_burst_shape_tokens:
        cfg["use_burst_shape_tokens"] = True
    if args.use_transition_profile_tokens:
        cfg["use_transition_profile_tokens"] = True
    if args.use_raw_profile_id_tokens:
        cfg["use_raw_profile_id_tokens"] = True
    if args.use_service_tokens:
        cfg["use_service_tokens"] = True
    if args.label_field is not None:
        cfg["label_field"] = args.label_field
    flows = read_jsonl(args.flows)
    profile_rows = read_jsonl(args.profile_primitives)
    with open(args.splits, "r", encoding="utf-8") as handle:
        split_payload = json.load(handle)
    token_data, stats = build_behavior_token_dataset(flows, profile_rows, split_payload, cfg, max_len=args.max_len)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(token_data, args.out)
    write_json(token_data["vocab"], args.vocab)
    stats_out = args.stats_out or args.out.rsplit(".", 1)[0] + "_stats.json"
    manifest_out = args.manifest_out or args.out.rsplit(".", 1)[0] + "_manifest.json"
    write_json(stats, stats_out)
    write_json(
        {
            "flows": args.flows,
            "splits": args.splits,
            "profile_primitives": args.profile_primitives,
            "out": args.out,
            "vocab": args.vocab,
            "stats_out": stats_out,
            "config": args.config,
            "tokenizer_config": cfg,
            "label_field": cfg.get("label_field", "label"),
            "ablation_features": stats.get("ablation_features"),
            "vocab_provenance": "train_only",
            "provenance": "train_only",
            "train_only": True,
            "threshold_tuning_split": "val",
            "has_port_token": bool(args.use_service_tokens),
            "allow_port_tokens": bool(args.use_service_tokens),
            "has_ip_token": False,
            "has_abs_time_token": False,
        },
        manifest_out,
    )
    print(stats)


if __name__ == "__main__":
    main()
