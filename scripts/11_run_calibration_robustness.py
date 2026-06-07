#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect calibration robustness artifacts.")
    parser.add_argument("--revision-dir", default="paper_icdm_applied_2026/experiments/revision")
    parser.add_argument("--output-dir", default="results/calibration_robustness")
    args = parser.parse_args()

    ensure_dirs()
    rev = ROOT / args.revision_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    names = [
        "target_realized_fpr_summary.csv",
        "target_realized_fpr_runs.csv",
        "validation_size_stability_summary.csv",
        "alert_budget_summary.csv",
        "prevalence_alert_budget.csv",
        "primitive_threshold_sensitivity.csv",
        "primitive_sensitivity.csv",
    ]
    manifest = []
    for name in names:
        src = rev / name
        dst = out / name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            status = "copied"
        else:
            status = "missing"
        manifest.append({"source": str(src), "output": str(dst), "status": status, "attack_labels_used_for_threshold": False})
    write_json(out / "calibration_manifest.json", {"command": command_record(sys.argv), "sources": manifest})
    write_md(ROOT / "reports/calibration_robustness_summary.md", ["# Calibration Robustness Summary", "", "- Deployable thresholds are benign-validation percentile thresholds.", *[f"- {m['source']}: {m['status']}" for m in manifest]])
    print(out / "calibration_manifest.json")


if __name__ == "__main__":
    main()

