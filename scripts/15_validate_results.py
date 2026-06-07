#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.pipeline.common import PAPER_ROOT, ROOT, command_record, ensure_dirs, read_csv, write_json, write_md


RISKY_PATTERNS = [
    "additional benign solves calibration",
    "guarantees low FPR",
    "fully clean benign",
    "generalizes across datasets",
    "fully explains",
    "state-of-the-art",
    "solves 0.1% FPR",
    "primitive alone",
]


OLD_PREFIX_PATTERNS = [r"\bMIC(?:_[A-Z0-9]+|\b)", r"\bPRIM1\b", r"\bPRIM2\b", r"\bP2_", r"primitive_v2", r"Primitive-v2", r"max_packet_micro_tokens"]


def _exists(path: Path) -> str:
    return "pass" if path.exists() and path.stat().st_size > 0 else "fail"


def _grep(paths: list[Path], patterns: list[str], *, ignore_case: bool = True) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    flags = re.IGNORECASE if ignore_case else 0
    compiled = [(pat, re.compile(pat, flags)) for pat in patterns]
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for raw, pattern in compiled:
                if pattern.search(line):
                    hits.append({"path": str(path), "line": lineno, "pattern": raw, "text": line.strip()[:240]})
    return hits


def _csv_has_false_flags(path: Path, fields: list[str]) -> list[str]:
    if not path.exists():
        return [f"missing {path}"]
    rows = read_csv(path)
    issues: list[str] = []
    for idx, row in enumerate(rows, start=2):
        for field in fields:
            value = str(row.get(field, "")).strip().lower()
            if value and value not in {"false", "no", "0"}:
                issues.append(f"{path}:{idx} {field}={row.get(field)}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FlowPrim result artifacts, leakage flags, paper links, and risky claims.")
    parser.add_argument("--paper", default=str(PAPER_ROOT / "main.tex"))
    args = parser.parse_args()

    ensure_dirs()
    required = [
        ROOT / "data/manifests/pcap_manifest.csv",
        ROOT / "data/manifests/token_manifest.csv",
        ROOT / "data/manifests/split_manifest.csv",
        ROOT / "results/primitive_categories/primitive_category_feature_attribution.csv",
        ROOT / "results/main_detection/main_detection_summary.csv",
        PAPER_ROOT / "tables/table_primitive_category_attribution.tex",
        PAPER_ROOT / "main.tex",
    ]
    checks = [{"artifact": str(path), "status": _exists(path)} for path in required]
    leakage_issues: list[str] = []
    leakage_issues.extend(
        _csv_has_false_flags(
            ROOT / "results/primitive_categories/primitive_category_unknown_metrics.csv",
            ["attack_labels_used_for_threshold", "raw_ip_used_as_token", "absolute_time_used_as_token", "five_tuple_used_as_token"],
        )
    )
    leakage_issues.extend(
        _csv_has_false_flags(
            ROOT / "data/manifests/pcap_manifest.csv",
            ["ready_ids2017_used", "raw_ip_used_as_token", "absolute_time_used_as_token", "five_tuple_used_as_token"],
        )
    )
    paper_paths = [Path(args.paper), PAPER_ROOT / "reproducibility_checklist.md", ROOT / "README.md", ROOT / "reports/experiment_summary.md"]
    risky_hits = _grep(paper_paths, RISKY_PATTERNS, ignore_case=True)
    checked_code_paths = list((ROOT / "src/features").glob("*.py")) + [
        path for path in (ROOT / "scripts").glob("*.py") if path.name != "15_validate_results.py"
    ]
    old_prefix_hits = _grep(paper_paths + checked_code_paths, OLD_PREFIX_PATTERNS, ignore_case=False)

    status = "pass"
    if any(row["status"] == "fail" for row in checks) or leakage_issues or risky_hits or old_prefix_hits:
        status = "partial" if not leakage_issues else "fail"

    payload = {
        "command": command_record(sys.argv),
        "status": status,
        "artifact_checks": checks,
        "leakage_issues": leakage_issues,
        "risky_claim_hits": risky_hits,
        "old_prefix_hits": old_prefix_hits,
    }
    write_json(ROOT / "reports/result_consistency_check.json", payload)
    write_md(
        ROOT / "reports/reproducibility_check.md",
        [
            "# Reproducibility Check",
            "",
            f"Status: `{status}`",
            "",
            "## Required Artifacts",
            "",
            *[f"- {row['status']}: `{row['artifact']}`" for row in checks],
            "",
            "## Leakage Controls",
            "",
            *(["- pass: no non-false leakage flags found in checked CSVs."] if not leakage_issues else [f"- {item}" for item in leakage_issues]),
            "",
            "## Risky Claims",
            "",
            *(["- pass: no risky claim phrases found in checked paper/report files."] if not risky_hits else [f"- {hit['path']}:{hit['line']} `{hit['pattern']}`: {hit['text']}" for hit in risky_hits]),
            "",
            "## Old Prefix / Terminology Hits",
            "",
            *(["- pass: no old prefix hits found in active checked files."] if not old_prefix_hits else [f"- {hit['path']}:{hit['line']} `{hit['pattern']}`: {hit['text']}" for hit in old_prefix_hits]),
        ],
    )
    print(ROOT / "reports/reproducibility_check.md")
    if status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
