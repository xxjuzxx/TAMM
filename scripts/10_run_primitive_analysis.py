#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, read_csv, write_csv, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect primitive trigger, lift, coverage, and scale analyses.")
    parser.add_argument("--primitive-dir", default="results/primitive_categories")
    parser.add_argument("--output-dir", default="results/primitive_analysis")
    args = parser.parse_args()

    ensure_dirs()
    src_dir = ROOT / args.primitive_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    files = [
        "primitive_category_scale_summary.csv",
        "primitive_category_explanation_coverage.csv",
        "top_structural_primitives_by_lift.csv",
        "primitive_category_runtime.csv",
    ]
    manifest = []
    for name in files:
        src = src_dir / name
        dst = out / name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            status = "copied"
        else:
            status = "missing"
        manifest.append({"source": str(src), "output": str(dst), "status": status})
    lift_rows = read_csv(src_dir / "top_structural_primitives_by_lift.csv") if (src_dir / "top_structural_primitives_by_lift.csv").exists() else []
    write_csv(out / "top_primitive_examples.csv", lift_rows[:50])
    write_json(out / "primitive_analysis_manifest.json", {"command": command_record(sys.argv), "sources": manifest})
    write_md(ROOT / "reports/primitive_analysis_summary.md", ["# Primitive Analysis Summary", "", f"- Top structural primitive rows available: {len(lift_rows)}", *[f"- {m['source']}: {m['status']}" for m in manifest]])
    print(out / "top_primitive_examples.csv")


if __name__ == "__main__":
    main()

