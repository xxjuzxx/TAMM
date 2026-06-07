#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect baseline comparison artifacts and deployment-assumption notes.")
    parser.add_argument("--revision-dir", default="paper_icdm_applied_2026/experiments/revision")
    parser.add_argument("--output-dir", default="results/baselines")
    args = parser.parse_args()

    ensure_dirs()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / args.revision_dir / "knn_feature_baselines_summary.csv",
        ROOT / args.revision_dir / "memory_scope_audit_summary.csv",
        ROOT / args.revision_dir / "external_diagnostics_interpreted.csv",
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
    write_json(out / "baseline_manifest.json", {"command": command_record(sys.argv), "sources": manifest, "supervised_and_benign_only_separated": True})
    write_md(ROOT / "reports/baseline_summary.md", ["# Baseline Summary", "", "- Benign-only KNN/open-set baselines and supervised baselines are reported under separate assumptions.", *[f"- {m['source']}: {m['status']}" for m in manifest]])
    print(out / "baseline_manifest.json")


if __name__ == "__main__":
    main()
