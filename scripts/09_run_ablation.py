#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, read_csv, write_csv, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect representation and primitive ablation results.")
    parser.add_argument("--revision-dir", default="paper_icdm_applied_2026/experiments/revision")
    parser.add_argument("--primitive-dir", default="results/primitive_categories")
    parser.add_argument("--output-dir", default="results/ablation")
    args = parser.parse_args()

    ensure_dirs()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / args.revision_dir / "behavior_feature_attribution.csv",
        ROOT / args.revision_dir / "knn_feature_baselines_summary.csv",
        ROOT / args.revision_dir / "primitive_sensitivity.csv",
        ROOT / args.primitive_dir / "primitive_category_feature_attribution.csv",
    ]
    manifest = []
    for src in sources:
        dst = out / src.name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            status = "copied"
        else:
            status = "missing"
        manifest.append({"source": str(src), "output": str(dst), "status": status})
    cat_rows = read_csv(ROOT / args.primitive_dir / "primitive_category_feature_attribution.csv") if (ROOT / args.primitive_dir / "primitive_category_feature_attribution.csv").exists() else []
    summary = []
    for row in cat_rows:
        summary.append(
            {
                "feature_view": row.get("feature_view"),
                "primitive_category": row.get("primitive_category"),
                "runs": row.get("runs"),
                "auroc": row.get("auroc"),
                "recall_at_1pct_fpr": row.get("recall_at_1pct_fpr"),
                "p99_fpr": row.get("val_p99_realized_fpr"),
            }
        )
    write_csv(out / "ablation_summary.csv", summary)
    write_json(out / "ablation_manifest.json", {"command": command_record(sys.argv), "sources": manifest})
    write_md(ROOT / "reports/ablation_summary.md", ["# Ablation Summary", "", *[f"- {m['source']}: {m['status']}" for m in manifest]])
    print(out / "ablation_summary.csv")


if __name__ == "__main__":
    main()
