#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import torch

from src.data.splits import split_lookup
from src.features.behavior_tokens import build_behavior_token_dataset
from src.features.profile_primitives import (
    extract_all_profile_primitives,
    extract_duplicate_profile_primitives,
    extract_repeat_profile_primitives,
    extract_same_len_profile_primitive,
    extract_short_profile_primitive,
)
from src.utils.io import read_jsonl, read_yaml, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOWS = ROOT / "outputs" / "processed" / "ccfa" / "cicids2017_interim_labeled_flows.jsonl"
DEFAULT_SPLIT_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown"
DEFAULT_OUTPUT = DEFAULT_SPLIT_DIR / "tokens_category"

ATTACK_SLUG = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "DoS": "dos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
    "Infiltration": "infiltration",
}


def _read_split(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _profile_rows_for_split(flows: list[dict[str, Any]], split_payload: dict[str, Any], profile_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = split_lookup(split_payload)
    train_flows = [flow for flow in flows if lookup.get(str(flow.get("flow_id"))) == "train"]
    train_rows, _stats = extract_all_profile_primitives(train_flows, profile_cfg)
    rows_by_id = {str(row["flow_id"]): {**row, "profile_primitive_provenance": "train_only"} for row in train_rows}
    rows: list[dict[str, Any]] = []
    len_bin_max = int(profile_cfg.get("length_bin_max", 15))
    count_bin_max = int(profile_cfg.get("count_bin_max", 15))
    for flow in flows:
        flow_id = str(flow.get("flow_id"))
        row = rows_by_id.get(flow_id)
        if row is None:
            primitive = {
                "short": extract_short_profile_primitive(flow, threshold=int(profile_cfg.get("short_flow_packet_threshold", 6))),
                "same": extract_same_len_profile_primitive(flow, len_bin_max=len_bin_max, count_bin_max=count_bin_max),
                "packet": [],
                "local": [],
                "repeat": extract_repeat_profile_primitives(
                    flow,
                    zero_len_min_repeat=int(profile_cfg.get("zero_len_min_repeat", 6)),
                    normal_min_repeat=int(profile_cfg.get("normal_min_repeat", 2)),
                    len_bin_max=len_bin_max,
                    count_bin_max=count_bin_max,
                ),
                "duplicate": extract_duplicate_profile_primitives(
                    flow,
                    duplicate_min_repeat=int(profile_cfg.get("duplicate_min_repeat", 2)),
                    len_bin_max=len_bin_max,
                    count_bin_max=count_bin_max,
                ),
            }
            row = {
                "flow_id": flow_id,
                "service_key": flow.get("service_key"),
                "label": flow.get("label"),
                "profile": primitive,
                "profile_primitive_provenance": "fixed_rules_only_peer_primitives_train_only",
            }
        rows.append(row)
    return rows


def _profile_mode(config: dict[str, Any], default: str = "full") -> str:
    tokenizer_cfg = config.get("tokenizer", {})
    return str(tokenizer_cfg.get("profile_mode", default))


def _tokenizer_config(config: dict[str, Any], max_len: int) -> dict[str, Any]:
    cfg = dict(config.get("tokenizer", {}))
    cfg.update(
        {
            "max_len": int(max_len),
            "profile_mode": _profile_mode(config, "full"),
            "include_flow_summary": True,
            "include_packet_tokens": True,
            "include_burst_tokens": True,
            "include_rhythm_tokens": True,
            "use_burst_shape_tokens": True,
            "use_transition_profile_tokens": bool(config.get("tokenizer", {}).get("use_transition_profile_tokens", True)),
            "label_field": "attack_family",
            "record_service_context": False,
            "use_service_context": False,
            "use_service_tokens": False,
        }
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild canonical FlowPrim category-token corpora from source flows and leave-one splits.")
    parser.add_argument("--flows", default=str(DEFAULT_FLOWS))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default=str(ROOT / "configs" / "cicids2017.yaml"))
    parser.add_argument("--attacks", nargs="+", default=list(ATTACK_SLUG))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--write-profile-rows", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    flows = read_jsonl(args.flows)
    config = read_yaml(args.config)
    profile_cfg = dict(config.get("profile_primitives") or config.get("profile") or {})
    tokenizer_cfg = _tokenizer_config(config, args.max_len)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)

    manifest_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for attack in args.attacks:
            slug = ATTACK_SLUG[attack]
            split_path = split_dir / f"splits_leave_one_{slug}_anomaly_seed{seed}.json"
            if not split_path.exists():
                manifest_rows.append({"attack": attack, "seed": seed, "status": "missing_split", "split_path": str(split_path)})
                continue
            split_payload = _read_split(split_path)
            profile_rows = _profile_rows_for_split(flows, split_payload, profile_cfg)
            token_data, stats = build_behavior_token_dataset(flows, profile_rows, split_payload, tokenizer_cfg, max_len=args.max_len)
            out_pt = out_dir / f"cicids2017_leave_one_{slug}_anomaly_seed{seed}_a3_full_rhythm.pt"
            torch.save(token_data, out_pt)
            write_json(token_data["vocab"], str(out_pt).replace(".pt", "_vocab.json"))
            write_json(stats, str(out_pt).replace(".pt", "_stats.json"))
            if args.write_profile_rows:
                write_jsonl(profile_rows, str(out_pt).replace(".pt", "_profile_primitives.jsonl"))
            row = {
                "attack": attack,
                "seed": seed,
                "status": "ok",
                "token_path": str(out_pt),
                "split_path": str(split_path),
                "num_rows": int(stats.get("num_rows") or stats.get("num_flows") or 0),
                "vocab_size": int(stats.get("vocab_size") or 0),
                "train_only_vocab": True,
                "profile_primitives_train_only": True,
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
            }
            manifest_rows.append(row)
            print(json.dumps(row, sort_keys=True))

    write_json(
        {
            "flows": str(args.flows),
            "config": str(args.config),
            "output": str(out_dir),
            "attacks": args.attacks,
            "seeds": args.seeds,
            "tokenizer_config": tokenizer_cfg,
            "profile_primitive_config": profile_cfg,
            "rows": manifest_rows,
        },
        out_dir / "manifest.json",
    )


if __name__ == "__main__":
    main()
