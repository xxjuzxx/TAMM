#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import pandas as pd

from src.pipeline.common import ROOT, command_record, ensure_dirs, run_command, write_csv, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract or inventory FlowPrim profile/structural primitive events.")
    parser.add_argument("--mode", choices=["inventory", "rebuild"], default="inventory")
    parser.add_argument("--token-dir", default="paper_icdm_applied_2026/experiments/unknown/tokens_category")
    parser.add_argument("--results-dir", default="results/primitive_categories")
    parser.add_argument("--output", default="data/interim/primitive_events/primitive_event_manifest.csv")
    args = parser.parse_args()

    ensure_dirs()
    if args.mode == "rebuild":
        run_command(["bash", "scripts/run_primitive_category_experiments.sh", "--mode", "full"], log_path=ROOT / "logs/experiments/04_primitives_rebuild.log")

    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / args.results_dir / "tokens").glob("structural_*_seed*.json")):
        vocab_path = path.with_name(path.stem + "_vocab.json")
        rows.append(
            {
                "primitive_event_artifact": str(path),
                "vocab_path": str(vocab_path),
                "size_bytes": path.stat().st_size,
                "primitive_categories": "profile,structural",
                "structural_vocab_train_only": True,
                "supports_trigger_positions": True,
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
            }
        )
    out = ROOT / args.output
    write_csv(out, rows)
    write_json(ROOT / "data/manifests/primitive_manifest.json", {"command": command_record(sys.argv), "mode": args.mode, "rows": rows})
    write_md(ROOT / "reports/primitive_extraction_summary.md", ["# Primitive Extraction Summary", "", f"- Mode: `{args.mode}`", f"- Primitive event artifacts: {len(rows)}", "- Profile and structural primitives are parallel categories, not sequential versioned stages."])
    print(out)


if __name__ == "__main__":
    main()
