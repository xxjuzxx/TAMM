#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, read_csv, write_csv, write_json, write_md


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect FlowPrim main detection results from existing reproducible artifacts.")
    parser.add_argument("--revision-dir", default="paper_icdm_applied_2026/experiments/revision")
    parser.add_argument("--output-dir", default="results/main_detection")
    args = parser.parse_args()

    ensure_dirs()
    rev = ROOT / args.revision_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        rev / "unknown_best_settings_3seed_runs.csv",
        rev / "target_realized_fpr_summary.csv",
        rev / "alert_budget_summary.csv",
        ROOT / "results/primitive_categories/primitive_category_feature_attribution.csv",
    ]
    manifest: list[dict[str, Any]] = []
    for src in sources:
        dst = out / src.name
        copied = _copy_if_exists(src, dst)
        manifest.append({"source": str(src), "output": str(dst), "status": "copied" if copied else "missing"})

    rows = read_csv(rev / "unknown_best_settings_3seed_runs.csv") if (rev / "unknown_best_settings_3seed_runs.csv").exists() else []
    metric_rows: list[dict[str, Any]] = []
    if rows:
        vals = [float(row["auroc"]) for row in rows if row.get("auroc")]
        r1 = [float(row["recall_at_1pct_fpr"]) for row in rows if row.get("recall_at_1pct_fpr")]
        p99 = [float(row["val_p99_0_false_positive_rate"]) for row in rows if row.get("val_p99_0_false_positive_rate")]
        metric_rows.append(
            {
                "setting": "leave_one_unknown_locked_best_settings",
                "runs": len(rows),
                "auroc_mean": sum(vals) / max(len(vals), 1),
                "recall_at_1pct_fpr_mean": sum(r1) / max(len(r1), 1),
                "p99_realized_fpr_mean": sum(p99) / max(len(p99), 1),
                "calibration_uses_attack_labels": False,
                "selection_note": "Seed-42 sweep is diagnostic; deployable P99 thresholds use benign validation only.",
            }
        )
    write_csv(out / "main_detection_summary.csv", metric_rows)
    write_json(out / "main_detection_manifest.json", {"command": command_record(sys.argv), "sources": manifest, "summary_rows": metric_rows})
    write_md(
        ROOT / "reports/main_detection_summary.md",
        [
            "# Main Detection Summary",
            "",
            "This script collects existing real artifacts; it does not fabricate closed-set or low-FPR numbers.",
            "",
            *[f"- {row['source']}: {row['status']}" for row in manifest],
            "",
            *[
                f"- {row['setting']}: AUROC {row['auroc_mean']:.4f}, R@1%FPR {row['recall_at_1pct_fpr_mean']:.4f}, P99 FPR {row['p99_realized_fpr_mean']:.4f} over {row['runs']} runs."
                for row in metric_rows
            ],
        ],
    )
    print(out / "main_detection_summary.csv")


if __name__ == "__main__":
    main()

