#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.data.cicids2017_labeler import label_flows, read_cicids_flow_csvs
from src.data.label_policy import ATTEMPTED_POLICIES
from src.data.zeek_parser import aggregate_zeek_logs
from src.utils.io import write_json, write_jsonl


def _expand_inputs(items: list[str], suffixes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if any(char in item for char in "*?[]"):
            paths.extend(Path(match) for match in sorted(glob.glob(item)))
        elif path.is_dir():
            paths.extend(sorted(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in suffixes))
        else:
            paths.append(path)
    return paths


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one Zeek log set against multiple CICIDS label CSVs.")
    parser.add_argument("--zeek_logs", nargs="+", required=True)
    parser.add_argument("--label_csv", nargs="+", required=True)
    parser.add_argument("--attempted_policy", choices=ATTEMPTED_POLICIES, default="drop")
    parser.add_argument("--tolerance_seconds", type=float, default=2.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--write_matches_dir", default=None)
    args = parser.parse_args()

    zeek_paths = _expand_inputs(args.zeek_logs, {".log", ".tsv", ".json", ".jsonl", ".ndjson"})
    label_paths = _expand_inputs(args.label_csv, {".csv"})
    flows = aggregate_zeek_logs(zeek_paths)

    audits: list[dict[str, Any]] = []
    for label_path in label_paths:
        label_rows = read_cicids_flow_csvs([label_path], attempted_policy=args.attempted_policy)
        labeled, unmatched, stats = label_flows(flows, label_rows, tolerance_seconds=args.tolerance_seconds)
        row = {
            "label_csv": str(label_path),
            "attempted_policy": args.attempted_policy,
            "tolerance_seconds": float(args.tolerance_seconds),
            **stats,
        }
        audits.append(row)
        if args.write_matches_dir:
            out_dir = Path(args.write_matches_dir)
            stem = _safe_stem(label_path)
            write_jsonl(labeled, out_dir / f"{stem}_labeled_flows.jsonl")
            write_jsonl(unmatched, out_dir / f"{stem}_unmatched_flows.jsonl")
            write_json(row, out_dir / f"{stem}_label_stats.json")

    result = {
        "zeek_logs": [str(path) for path in zeek_paths],
        "num_flows": len(flows),
        "attempted_policy": args.attempted_policy,
        "tolerance_seconds": float(args.tolerance_seconds),
        "audits": sorted(audits, key=lambda item: (-float(item.get("match_rate", 0.0)), item["label_csv"])),
    }
    write_json(result, args.out)
    print(result)


if __name__ == "__main__":
    main()
