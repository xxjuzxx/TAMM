#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "raw_rebuild_split_first" / "primitive_categories"
DEFAULT_OUT = ROOT / "results" / "raw_rebuild_split_first"
DEFAULT_TABLE_DIR = ROOT / "tables" / "main_results"
DEFAULT_PAPER_TABLE = ROOT.parents[0] / "paper" / "tables" / "table_raw_rebuild_split_first.tex"


def _display_path(path: Path) -> str:
    """Return a path relative to the workspace root when possible."""

    for base in (ROOT, ROOT.parent):
        try:
            return str(path.resolve().relative_to(base))
        except ValueError:
            continue
    return str(path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _num(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    val = _num(value)
    if math.isnan(val):
        return "-"
    return f"{val:.{digits}f}"


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[str, str]:
    vals = [_num(row.get(key)) for row in rows]
    vals = [val for val in vals if not math.isnan(val)]
    if not vals:
        return "", ""
    return str(float(sum(vals) / len(vals))), str(float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_behavior_only_copy(src: Path, dst: Path) -> None:
    rows = _behavior_only_rows(_read_csv(src))
    if rows:
        _write_csv(dst, rows)


def _behavior_only_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep reported behavior-only global-memory rows."""

    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("memory_scope", "global")) != "global":
            continue
        out = dict(row)
        out["memory_scope"] = "global"
        cleaned.append(out)
    return cleaned


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    result_dir = Path(args.results)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "primitive_category_unknown_metrics.csv"
    attribution_path = result_dir / "primitive_category_feature_attribution.csv"
    metrics = _behavior_only_rows(_read_csv(metrics_path))
    attribution = _behavior_only_rows(_read_csv(attribution_path))

    _write_csv(out_dir / "raw_rebuild_unknown_metrics.csv", metrics)
    _write_csv(out_dir / "raw_rebuild_feature_attribution.csv", attribution)
    _copy_if_exists(result_dir / "primitive_category_scale_summary.csv", out_dir / "primitive_category_scale_summary.csv")
    _write_behavior_only_copy(
        result_dir / "primitive_category_explanation_coverage.csv",
        out_dir / "primitive_category_explanation_coverage.csv",
    )
    _write_behavior_only_copy(result_dir / "primitive_category_runtime.csv", out_dir / "primitive_category_runtime.csv")
    _copy_if_exists(result_dir / "top_structural_primitives_by_lift.csv", out_dir / "top_structural_primitives_by_lift.csv")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Raw Rebuild Split-First Artifacts",
                "",
                "These artifacts summarize the capped raw-PCAP split-first low-FPR sanity check.",
                "Only behavior-only global-memory views are exported here.",
                "Protocol/service grouping, raw IP addresses, absolute timestamps, and complete five-tuples are not used as behavior tokens or memory-grouping keys.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[str(row.get("feature_view"))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for view, rows in sorted(grouped.items()):
        auroc_mean, auroc_std = _mean_std(rows, "auroc")
        r01_mean, r01_std = _mean_std(rows, "recall_at_0_1pct_fpr")
        r1_mean, r1_std = _mean_std(rows, "recall_at_1pct_fpr")
        p99_mean, p99_std = _mean_std(rows, "val_p99_realized_fpr")
        fpr95_mean, fpr95_std = _mean_std(rows, "fpr95")
        fa_mean, fa_std = _mean_std(rows, "false_alerts_per_10k_benign")
        summary_rows.append(
            {
                "feature_view": view,
                "runs": len(rows),
                "attacks": ",".join(sorted({str(row.get("heldout_attack")) for row in rows})),
                "seeds": ",".join(sorted({str(row.get("seed")) for row in rows})),
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "fpr95_mean": fpr95_mean,
                "fpr95_std": fpr95_std,
                "recall_at_0_1pct_fpr_mean": r01_mean,
                "recall_at_0_1pct_fpr_std": r01_std,
                "recall_at_1pct_fpr_mean": r1_mean,
                "recall_at_1pct_fpr_std": r1_std,
                "val_p99_realized_fpr_mean": p99_mean,
                "val_p99_realized_fpr_std": p99_std,
                "false_alerts_per_10k_benign_mean": fa_mean,
                "false_alerts_per_10k_benign_std": fa_std,
            }
        )
    _write_csv(out_dir / "raw_rebuild_split_first_summary.csv", summary_rows)

    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    headers = ["Feature view", "Runs", "AUROC", "FPR95", "R@0.1%FPR", "R@1%FPR", "P99 FPR", "False alerts / 10k"]
    md_lines = [
        "# Raw Rebuild Split-First Low-FPR Check",
        "",
        "Splits are created from PCAP-derived raw labeled flows before token vocabulary fitting, structural primitive min-support filtering, benign memory construction, and benign-validation calibration.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    preferred = [
        "profile_only",
        "structural_only",
        "packet_burst_only",
        "packet_burst_plus_structural",
        "packet_burst_plus_profile_structural",
    ]
    by_view = {row["feature_view"]: row for row in summary_rows}
    selected_rows = [by_view[view] for view in preferred if view in by_view]
    if not selected_rows:
        selected_rows = summary_rows
    for row in selected_rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["feature_view"]),
                    str(row["runs"]),
                    f"{_fmt(row.get('auroc_mean'))}±{_fmt(row.get('auroc_std'))}",
                    f"{_fmt(row.get('fpr95_mean'))}±{_fmt(row.get('fpr95_std'))}",
                    f"{_fmt(row.get('recall_at_0_1pct_fpr_mean'))}±{_fmt(row.get('recall_at_0_1pct_fpr_std'))}",
                    f"{_fmt(row.get('recall_at_1pct_fpr_mean'))}±{_fmt(row.get('recall_at_1pct_fpr_std'))}",
                    f"{_fmt(row.get('val_p99_realized_fpr_mean'))}±{_fmt(row.get('val_p99_realized_fpr_std'))}",
                    f"{_fmt(row.get('false_alerts_per_10k_benign_mean'), 1)}±{_fmt(row.get('false_alerts_per_10k_benign_std'), 1)}",
                ]
            )
            + " |"
        )
    md_lines.extend(
        [
            "",
            "Missing or skipped runs are not filled. WebAttack remains low-support in the corrected raw labels.",
        ]
    )
    (table_dir / "raw_rebuild_split_first.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Raw-PCAP split-first low-FPR rebuild check.}",
        "\\label{tab:raw_rebuild_split_first}",
        "\\FlowPrimTableStyle",
        "\\FlowPrimTableFit{\\columnwidth}{",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "View & Runs & AUROC & R@1\\%FPR & P99 FPR & Alerts/10k \\\\",
        "\\midrule",
    ]
    for row in selected_rows:
        tex_lines.append(
            f"{_latex_escape(row['feature_view'])} & {row['runs']} & "
            f"{_fmt(row.get('auroc_mean'))} & {_fmt(row.get('recall_at_1pct_fpr_mean'))} & "
            f"{_fmt(row.get('val_p99_realized_fpr_mean'))} & {_fmt(row.get('false_alerts_per_10k_benign_mean'), 1)} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])
    paper_table = Path(args.paper_table)
    paper_table.parent.mkdir(parents=True, exist_ok=True)
    paper_table.write_text("\n".join(tex_lines), encoding="utf-8")

    summary = {
        "attacks": sorted({str(row.get("heldout_attack")) for row in metrics}),
        "feature_views": sorted({str(row.get("feature_view")) for row in metrics}),
        "metrics_rows": len(metrics),
        "summary_rows": len(summary_rows),
        "memory_scope": "global",
        "behavior_only_export": True,
        "result_dir": _display_path(result_dir),
        "output_dir": _display_path(out_dir),
        "table_md": _display_path(table_dir / "raw_rebuild_split_first.md"),
        "paper_table": _display_path(paper_table),
    }
    (out_dir / "primitive_category_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "raw_rebuild_split_first_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize raw-rebuild split-first low-FPR results.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--table-dir", default=str(DEFAULT_TABLE_DIR))
    parser.add_argument("--paper-table", default=str(DEFAULT_PAPER_TABLE))
    args = parser.parse_args()
    print(json.dumps(summarize(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
