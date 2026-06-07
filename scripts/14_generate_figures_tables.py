#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.pipeline.common import PAPER_ROOT, ROOT, command_record, ensure_dirs, read_csv, run_command, write_csv, write_json, write_md


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _fmt(value: Any, digits: int = 4) -> str:
    val = _num(value)
    return "-" if val is None else f"{val:.{digits}f}"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_Missing result rows._"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def _tex_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], caption_note: str = "") -> str:
    aligns = "l" + "r" * (len(columns) - 1)
    lines = [rf"\begin{{tabular}}{{{aligns}}}", r"\toprule", " & ".join(label for _key, label in columns) + r" \\", r"\midrule"]
    for row in rows:
        vals = []
        for key, _label in columns:
            val = row.get(key, "")
            if isinstance(val, float):
                vals.append(_fmt(val))
            else:
                vals.append(str(val).replace("_", r"\_"))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if caption_note:
        lines.extend(["", rf"% {caption_note}"])
    return "\n".join(lines) + "\n"


def _copy_paper_tables() -> list[dict[str, Any]]:
    table_dir = ROOT / "tables"
    generated: list[dict[str, Any]] = []
    mappings = [
        (ROOT / "results/primitive_categories/primitive_category_feature_attribution.csv", table_dir / "ablation/primitive_category_attribution.md"),
        (ROOT / "results/main_detection/main_detection_summary.csv", table_dir / "main_results/main_detection_summary.md"),
        (ROOT / "results/calibration_robustness/prevalence_alert_budget.csv", table_dir / "main_results/prevalence_alert_budget.md"),
        (ROOT / "results/efficiency/ann_knn_scalability.csv", table_dir / "efficiency/ann_knn_scalability.md"),
        (ROOT / "results/diagnosis_cases/diagnosis_audit_simplified.csv", table_dir / "diagnosis_cases/diagnosis_audit_simplified.md"),
    ]
    for src, dst in mappings:
        rows = read_csv(src) if src.exists() else []
        if rows:
            columns = list(rows[0].keys())[: min(8, len(rows[0]))]
            write_md(dst, [f"# {src.stem}", "", *_markdown_table(rows[:20], columns)])
            generated.append({"source": str(src), "output": str(dst), "status": "generated", "rows": len(rows)})
        else:
            write_md(dst, [f"# {src.stem}", "", "_Missing source CSV._"])
            generated.append({"source": str(src), "output": str(dst), "status": "missing_source", "rows": 0})
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FlowPrim paper tables/figures from CSV/JSON artifacts.")
    parser.add_argument("--build-existing-paper-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-primitive-category-table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-low-fpr-table-suite", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ensure_dirs()
    generated = _copy_paper_tables()
    commands = []
    if args.build_existing_paper_figures and (ROOT / "scripts/59_build_icdm_figures.py").exists():
        try:
            commands.append(run_command(["python", "scripts/59_build_icdm_figures.py"], log_path=ROOT / "logs/paper_sync/14_build_icdm_figures.log", check=False))
        except Exception as exc:
            commands.append({"command": "python scripts/59_build_icdm_figures.py", "returncode": -1, "error": str(exc)})
    if args.build_primitive_category_table and (ROOT / "scripts/build_primitive_category_table.py").exists():
        commands.append(run_command(["python", "scripts/build_primitive_category_table.py"], log_path=ROOT / "logs/paper_sync/14_build_primitive_category_table.log", check=False))
    if args.build_low_fpr_table_suite and (ROOT / "scripts/build_low_fpr_table_suite.py").exists():
        commands.append(run_command(["python", "scripts/build_low_fpr_table_suite.py"], log_path=ROOT / "logs/paper_sync/14_build_low_fpr_table_suite.log", check=False))
    write_json(ROOT / "results/summaries/figure_table_manifest.json", {"command": command_record(sys.argv), "generated": generated, "commands": commands})
    write_md(
        ROOT / "reports/figure_table_generation_summary.md",
        [
            "# Figure and Table Generation Summary",
            "",
            *[f"- {row['output']}: {row['status']} ({row['rows']} rows)" for row in generated],
            "",
            "Existing paper LaTeX tables remain under `../paper/tables/`; generated markdown mirrors are under `tables/`.",
        ],
    )
    print(ROOT / "results/summaries/figure_table_manifest.json")


if __name__ == "__main__":
    main()
