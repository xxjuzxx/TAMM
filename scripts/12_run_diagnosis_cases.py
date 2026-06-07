#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, read_csv, write_csv, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect diagnosis case-study artifacts.")
    parser.add_argument("--revision-dir", default="paper_icdm_applied_2026/experiments/revision")
    parser.add_argument("--output-dir", default="results/diagnosis_cases")
    args = parser.parse_args()

    ensure_dirs()
    rev = ROOT / args.revision_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    names = ["diagnosis_audit_cases.csv", "diagnosis_audit_simplified.csv"]
    manifest = []
    for name in names:
        src = rev / name
        dst = out / name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            status = "copied"
        else:
            status = "missing"
        manifest.append({"source": str(src), "output": str(dst), "status": status})
    cases = read_csv(rev / "diagnosis_audit_simplified.csv") if (rev / "diagnosis_audit_simplified.csv").exists() else []
    write_json(out / "diagnosis_manifest.json", {"command": command_record(sys.argv), "sources": manifest, "case_count": len(cases)})
    write_md(ROOT / "reports/diagnosis_case_summary.md", ["# Diagnosis Case Summary", "", f"- Simplified diagnosis cases: {len(cases)}", "- These are analyst-facing audit examples, not proof that primitives fully explain every alert."])
    print(out / "diagnosis_manifest.json")


if __name__ == "__main__":
    main()

