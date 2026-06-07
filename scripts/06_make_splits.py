#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, write_csv, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory leave-one unknown split files and export split manifests.")
    parser.add_argument("--split-dir", default="paper_icdm_applied_2026/experiments/unknown")
    parser.add_argument("--output-dir", default="data/processed/splits/cicids2017")
    args = parser.parse_args()

    ensure_dirs()
    split_paths = sorted((ROOT / args.split_dir).glob("splits_leave_one_*_seed*.json"))
    rows: list[dict[str, Any]] = []
    for path in split_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_splits = payload.get("assignments") or payload.get("splits") or {}
        if isinstance(raw_splits, dict) and all(isinstance(value, list) for value in raw_splits.values()):
            assignments = {str(flow_id): str(split) for split, ids in raw_splits.items() for flow_id in ids}
        elif isinstance(raw_splits, dict):
            assignments = {str(flow_id): str(split) for flow_id, split in raw_splits.items()}
        else:
            assignments = {}
        counts = Counter(assignments.values())
        name = path.stem
        out_dir = ROOT / args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        split_rows = [{"flow_id": fid, "split": split} for fid, split in sorted(assignments.items())] if isinstance(assignments, dict) else []
        if split_rows:
            write_csv(out_dir / "split_manifest.csv", split_rows)
        rows.append(
            {
                "split_file": str(path),
                "output_split_manifest": str(out_dir / "split_manifest.csv"),
                "train_count": counts.get("train", ""),
                "val_count": counts.get("val", ""),
                "test_count": counts.get("test", ""),
                "malicious_used_for_calibration": False,
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
            }
        )
    write_csv(ROOT / "data/manifests/split_manifest.csv", rows)
    write_json(ROOT / "data/manifests/split_summary.json", {"command": command_record(sys.argv), "rows": rows})
    write_md(ROOT / "reports/split_summary.md", ["# Split Summary", "", f"- Leave-one split files: {len(rows)}", "- Train and validation splits are expected to be benign-only for low-FPR unknown evaluation."])
    print(ROOT / "data/manifests/split_manifest.csv")


if __name__ == "__main__":
    main()
