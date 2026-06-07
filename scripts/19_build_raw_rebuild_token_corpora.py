#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import _bootstrap  # noqa: F401
import torch

import rebuild_category_token_corpora as category_builder
from src.features.behavior_tokens import build_behavior_token_dataset
from src.utils.io import read_yaml, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOW_GLOB = str(ROOT / "data" / "interim" / "flows" / "cicids2017" / "*" / "raw_cicids2017_*_labeled_flows.jsonl")
DEFAULT_SPLIT_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "raw_rebuild" / "unknown"
DEFAULT_OUTPUT = DEFAULT_SPLIT_DIR / "tokens_category"
ATTACK_SLUG = category_builder.ATTACK_SLUG


def _glob_paths(pattern: str) -> list[Path]:
    if Path(pattern).is_absolute():
        return sorted(Path("/").glob(pattern.lstrip("/")))
    return sorted(Path().glob(pattern))


def _display_path(path: Path) -> str:
    """Return a repository-relative path when possible for portable manifests."""

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def _canonical_family(row: dict[str, Any]) -> str:
    raw = str(row.get("attack_family") or row.get("label") or row.get("binary_label") or "").strip()
    norm = raw.lower().replace("-", "").replace("_", "").replace(" ", "")
    if norm in {"benign", "normal"}:
        return "BENIGN"
    if "bot" in norm:
        return "Botnet"
    if "ddos" in norm:
        return "DDoS"
    if "probe" in norm or "portscan" in norm or "scan" in norm:
        return "Probe"
    if "web" in norm or "sql" in norm or "xss" in norm:
        return "WebAttack"
    if "brute" in norm or "patator" in norm:
        return "BruteForce"
    return raw or "UNKNOWN"


def _read_split(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_ids(split_payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for ids in split_payload.get("splits", {}).values():
        out.update(str(flow_id) for flow_id in ids)
    return out


def _load_selected_flows(paths: list[Path], wanted: set[str]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _jsonl_rows(path):
            flow_id = str(row.get("flow_id") or "")
            if flow_id not in wanted or flow_id in selected:
                continue
            flow = dict(row)
            family = _canonical_family(flow)
            flow["attack_family"] = family
            flow["label"] = family
            flow["binary_label"] = "BENIGN" if family == "BENIGN" else "ATTACK"
            flow["raw_ip_used_as_token"] = False
            flow["absolute_time_used_as_token"] = False
            flow["five_tuple_used_as_token"] = False
            selected[flow_id] = flow
    missing = sorted(wanted.difference(selected))
    if missing:
        raise FileNotFoundError(f"{len(missing)} split flow_ids were not found in raw flow files; first missing={missing[:5]}")
    return sorted(selected.values(), key=lambda x: (str(x.get("day") or ""), float(x.get("start_ts") or 0.0), str(x.get("flow_id") or "")))


def _artifact_name(prefix: str, slug: str, seed: int) -> str:
    return f"{prefix}_leave_one_{slug}_anomaly_seed{seed}_a3_full_rhythm.pt"


def build_corpora(args: argparse.Namespace) -> dict[str, Any]:
    flow_paths = _glob_paths(args.flow_glob)
    if not flow_paths:
        raise FileNotFoundError(f"No raw labeled flow files matched: {args.flow_glob}")
    split_dir = Path(args.split_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = split_dir / "selected_flows"
    if args.write_selected_flows:
        selected_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(args.config)
    profile_cfg = dict(config.get("profile_primitives") or config.get("profile") or {})
    tokenizer_cfg = category_builder._tokenizer_config(config, args.max_len)
    manifest_rows: list[dict[str, Any]] = []
    jobs: list[tuple[int, str, str, Path, dict[str, Any]]] = []
    wanted_all: set[str] = set()

    for seed in args.seeds:
        for attack in args.attacks:
            slug = ATTACK_SLUG[attack]
            split_path = split_dir / f"splits_leave_one_{slug}_anomaly_seed{seed}.json"
            if not split_path.exists():
                manifest_rows.append({"seed": seed, "attack": attack, "status": "missing_split", "split_path": str(split_path)})
                continue
            split_payload = _read_split(split_path)
            ids = _split_ids(split_payload)
            wanted_all.update(ids)
            jobs.append((seed, attack, slug, split_path, split_payload))

    if jobs:
        selected_all = {str(flow.get("flow_id")): flow for flow in _load_selected_flows(flow_paths, wanted_all)}
    else:
        selected_all = {}

    for seed, attack, slug, split_path, split_payload in jobs:
            ids = _split_ids(split_payload)
            missing = sorted(ids.difference(selected_all))
            if missing:
                raise FileNotFoundError(f"{len(missing)} split flow_ids were not found after raw scan; first missing={missing[:5]}")
            flows = sorted(
                (selected_all[flow_id] for flow_id in ids),
                key=lambda x: (str(x.get("day") or ""), float(x.get("start_ts") or 0.0), str(x.get("flow_id") or "")),
            )
            profile_rows = category_builder._profile_rows_for_split(flows, split_payload, profile_cfg)
            token_data, stats = build_behavior_token_dataset(flows, profile_rows, split_payload, tokenizer_cfg, max_len=args.max_len)

            out_pt = out_dir / _artifact_name(args.artifact_prefix, slug, seed)
            torch.save(token_data, out_pt)
            write_json(token_data["vocab"], str(out_pt).replace(".pt", "_vocab.json"))
            write_json(stats, str(out_pt).replace(".pt", "_stats.json"))
            if args.write_profile_rows:
                write_jsonl(profile_rows, str(out_pt).replace(".pt", "_profile_primitives.jsonl"))
            if args.write_selected_flows:
                write_jsonl(flows, selected_dir / f"{args.artifact_prefix}_leave_one_{slug}_anomaly_seed{seed}_selected_flows.jsonl")

            train_labels = [token_data["binary_labels"][idx].item() for idx, meta in enumerate(token_data["meta"]) if meta.get("split") == "train"]
            val_labels = [token_data["binary_labels"][idx].item() for idx, meta in enumerate(token_data["meta"]) if meta.get("split") == "val"]
            if any(label != 0 for label in train_labels):
                raise ValueError(f"{out_pt} train split is not benign-only")
            if any(label != 0 for label in val_labels):
                raise ValueError(f"{out_pt} val split is not benign-only")

            row = {
                "seed": seed,
                "attack": attack,
                "status": "ok",
                "token_path": _display_path(out_pt),
                "split_path": _display_path(split_path),
                "num_rows": int(stats.get("num_rows") or 0),
                "split_counts": json.dumps(stats.get("split_counts", {}), sort_keys=True),
                "vocab_size": int(stats.get("vocab_size") or 0),
                "vocab_provenance": "train_only",
                "profile_primitives_train_only": True,
                "threshold_calibration_split": "val_benign_only",
                "attack_labels_used_for_threshold": False,
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
            }
            manifest_rows.append(row)
            print(json.dumps(row, sort_keys=True))

    write_json(
        {
            "flow_glob": _display_path(Path(args.flow_glob.replace("*", ""))) if "*" not in args.flow_glob else args.flow_glob.replace(str(ROOT) + "/", ""),
            "flow_paths": [_display_path(path) for path in flow_paths],
            "dataset_name": args.dataset_name,
            "artifact_prefix": args.artifact_prefix,
            "split_dir": _display_path(split_dir),
            "output": _display_path(out_dir),
            "config": _display_path(Path(args.config)),
            "attacks": args.attacks,
            "seeds": args.seeds,
            "max_len": int(args.max_len),
            "tokenizer_config": tokenizer_cfg,
            "profile_primitive_config": profile_cfg,
            "rows": manifest_rows,
            "notes": [
                "Token vocabulary is built from the split-first train split only.",
                "Profile primitive peer mining is fit from the split-first train split only; fixed-rule primitives are applied to val/test for evidence extraction.",
                "Raw IP addresses, absolute timestamps, and complete five-tuples are retained only as metadata and are not behavior tokens.",
            ],
        },
        out_dir / "manifest.json",
    )
    return {"rows": manifest_rows, "output": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw-rebuild split-first FlowPrim token corpora.")
    parser.add_argument("--flow-glob", default=DEFAULT_FLOW_GLOB)
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default=str(ROOT / "configs" / "cicids2017.yaml"))
    parser.add_argument("--attacks", nargs="+", default=list(ATTACK_SLUG))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--dataset-name", default="CICIDS2017")
    parser.add_argument("--artifact-prefix", default="cicids2017")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--write-profile-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-selected-flows", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    summary = build_corpora(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
