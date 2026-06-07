#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, run_command, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified FlowPrim ICDM Applied Track pipeline.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-heavy-pcap", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ensure_dirs()
    steps: list[tuple[str, list[str]]] = [
        ("inventory", ["python", "scripts/00_inventory_data.py"]),
        ("extract_flows", ["python", "scripts/01_extract_flows_from_pcap.py", "--mode", "dry_run" if args.skip_heavy_pcap else "full"]),
        ("normalize_flows", ["python", "scripts/02_normalize_flows.py"] + (["--max-rows", "2000"] if args.mode == "quick" else [])),
        ("tokenize", ["python", "scripts/03_tokenize_flows.py", "--mode", "inventory"]),
        ("extract_primitives", ["python", "scripts/04_extract_primitives.py", "--mode", "inventory"]),
        ("feature_matrices", ["python", "scripts/05_build_feature_matrices.py", "--max-corpora", "1" if args.mode == "quick" else "3"]),
        ("splits", ["python", "scripts/06_make_splits.py"]),
        ("benign_memory", ["python", "scripts/07_build_benign_memory.py", "--max-corpora", "1" if args.mode == "quick" else "3"]),
        ("main_detection", ["python", "scripts/08_run_main_detection.py"]),
        ("ablation", ["python", "scripts/09_run_ablation.py"]),
        ("primitive_analysis", ["python", "scripts/10_run_primitive_analysis.py"]),
        ("calibration", ["python", "scripts/11_run_calibration_robustness.py"]),
        ("diagnosis", ["python", "scripts/12_run_diagnosis_cases.py"]),
        ("efficiency", ["python", "scripts/13_run_efficiency.py"]),
        ("baselines", ["python", "scripts/run_baselines.py"]),
        ("tables_figures", ["python", "scripts/14_generate_figures_tables.py"]),
        ("validate", ["python", "scripts/15_validate_results.py"]),
    ]
    records: list[dict[str, Any]] = []
    for name, cmd in steps:
        try:
            record = run_command(cmd, log_path=ROOT / "logs/experiments" / f"run_all_{name}.log", check=False)
            status = "ok" if int(record.get("returncode", 1)) == 0 else "failed"
        except Exception as exc:
            record = {"command": " ".join(cmd), "returncode": -1, "error": str(exc)}
            status = "failed"
        records.append({"step": name, "status": status, **record})
        if status == "failed" and name in {"inventory", "validate"}:
            break
    write_json(ROOT / "results/summaries/run_all_experiments_manifest.json", {"command": command_record(sys.argv), "mode": args.mode, "steps": records})
    write_md(
        ROOT / "reports/run_all_experiments_summary.md",
        [
            "# Run-All Experiment Summary",
            "",
            f"Mode: `{args.mode}`",
            f"Heavy PCAP parsing skipped: `{args.skip_heavy_pcap}`",
            "",
            *[f"- {row['step']}: {row['status']} (`{row.get('command', '')}`)" for row in records],
        ],
    )
    print(ROOT / "reports/run_all_experiments_summary.md")
    if any(row["status"] == "failed" for row in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
