#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import pandas as pd

from src.pipeline.common import ROOT, command_record, ensure_dirs, run_command, write_csv, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or inventory FlowPrim behavior token corpora.")
    parser.add_argument("--mode", choices=["inventory", "rebuild"], default="inventory")
    parser.add_argument("--flows", default="outputs/processed/ccfa/cicids2017_interim_labeled_flows.jsonl")
    parser.add_argument("--split-dir", default="paper_icdm_applied_2026/experiments/unknown")
    parser.add_argument("--output", default="paper_icdm_applied_2026/experiments/unknown/tokens_category")
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    ensure_dirs()
    out_dir = ROOT / args.output
    cmd = [
        "python",
        "scripts/rebuild_category_token_corpora.py",
        "--flows",
        args.flows,
        "--split-dir",
        args.split_dir,
        "--output",
        args.output,
        "--attacks",
        *args.attacks,
        "--seeds",
        *[str(seed) for seed in args.seeds],
    ]
    if args.mode == "rebuild":
        run_command(cmd, log_path=ROOT / "logs/experiments/03_tokenize_rebuild.log")

    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.pt")):
        stem = path.stem
        rows.append(
            {
                "token_corpus": str(path),
                "size_bytes": path.stat().st_size,
                "vocab_path": str(path).replace(".pt", "_vocab.json"),
                "stats_path": str(path).replace(".pt", "_stats.json"),
                "profile_primitives_path": str(path).replace(".pt", "_profile_primitives.jsonl"),
                "train_only_vocab": True,
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
            }
        )
    write_csv(ROOT / "data/manifests/token_manifest.csv", rows)
    write_json(ROOT / "data/manifests/tokenization_manifest.json", {"command": command_record(sys.argv), "mode": args.mode, "rows": rows})
    write_md(ROOT / "reports/tokenization_summary.md", ["# Tokenization Summary", "", f"- Mode: `{args.mode}`", f"- Token corpora found: {len(rows)}", f"- Output directory: `{out_dir}`"])
    print(ROOT / "data/manifests/token_manifest.csv")


if __name__ == "__main__":
    main()

