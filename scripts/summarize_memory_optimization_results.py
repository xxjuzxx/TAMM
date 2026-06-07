#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from run_memory_optimization_experiments import _write_final_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "memory_optimization"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate FlowPrim memory optimization summary report from saved CSV artifacts.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--write-root-aliases", action="store_true", help="Copy requested report/CSV names to project root and results/ root.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_rows = _read_csv(results_dir / "summary_table.csv")
    retrieval_rows = _read_csv(results_dir / "retrieval_metrics.csv")
    calibration_rows = _read_csv(results_dir / "calibration_metrics.csv")
    manifest_path = results_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    _write_final_report(
        results_dir,
        summary_rows,
        retrieval_rows,
        calibration_rows,
        manifest.get("missing", []),
        manifest.get("unsupported", []),
    )

    if args.write_root_aliases:
        copies = [
            (results_dir / "analysis_notes.md", ROOT / "analysis_notes.md"),
            (results_dir / "optimization_plan.md", ROOT / "optimization_plan.md"),
            (results_dir / "final_report.md", ROOT / "final_report.md"),
            (results_dir / "summary_table.csv", ROOT / "results" / "summary_table.csv"),
            (results_dir / "retrieval_metrics.csv", ROOT / "results" / "retrieval_metrics.csv"),
            (results_dir / "calibration_metrics.csv", ROOT / "results" / "calibration_metrics.csv"),
            (results_dir / "latency_metrics.csv", ROOT / "results" / "latency_metrics.csv"),
        ]
        for src, dst in copies:
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    print(json.dumps({"results_dir": str(results_dir), "summary_rows": len(summary_rows), "retrieval_rows": len(retrieval_rows), "calibration_rows": len(calibration_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
