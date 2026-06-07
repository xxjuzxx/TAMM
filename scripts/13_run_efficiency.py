#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect efficiency and deployability artifacts.")
    parser.add_argument("--revision-dir", default="paper_icdm_applied_2026/experiments/revision")
    parser.add_argument("--primitive-dir", default="results/primitive_categories")
    parser.add_argument("--output-dir", default="results/efficiency")
    args = parser.parse_args()

    ensure_dirs()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / args.revision_dir / "ann_knn_scalability.csv",
        ROOT / args.revision_dir / "e2e_throughput_smoke.csv",
        ROOT / args.revision_dir / "extra_benign_e2e_throughput.csv",
        ROOT / args.primitive_dir / "primitive_category_runtime.csv",
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
    write_json(out / "efficiency_manifest.json", {"command": command_record(sys.argv), "sources": manifest, "cpu_only_path_supported": True})
    write_md(ROOT / "reports/efficiency_summary.md", ["# Efficiency Summary", "", "- Flow-ready KNN scoring and offline throughput artifacts are collected from existing real runs.", "- GPU is not required for the transparent KNN scorer.", *[f"- {m['source']}: {m['status']}" for m in manifest]])
    print(out / "efficiency_manifest.json")


if __name__ == "__main__":
    main()

