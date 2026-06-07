from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import defaultdict

from extra_benign_common import PAPER_TABLE_DIR, fmt, fmt_pm, read_csv


def _esc(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def write_extra_benign_data_summary(csv_path: str | Path, out_path: str | Path | None = None) -> None:
    rows = read_csv(csv_path)
    out = Path(out_path) if out_path else PAPER_TABLE_DIR / "table_extra_benign_data_summary.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lllrrrrrl}",
        "\\toprule",
        "Source & Input type & Capability & Raw & Dedup & Pass & Quarantine & Tail & Use \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _esc(row.get("Source", "")),
                    _esc(row.get("Input type", "")),
                    _esc(row.get("Capability level", "")),
                    _esc(row.get("Raw flows", "")),
                    _esc(row.get("Dedup flows", "")),
                    _esc(row.get("Admission pass", "")),
                    _esc(row.get("Quarantine", "")),
                    _esc(row.get("Tail-test candidates", "")),
                    _esc(row.get("Use in paper", "")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_attribution_table(csv_path: str | Path, out_path: str | Path | None = None) -> None:
    rows = read_csv(csv_path)
    out = Path(out_path) if out_path else PAPER_TABLE_DIR / "table_extra_benign_memory_calibration_attribution.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Setting & Memory & Calib. & AUROC & FPR95 & R@1\\%FPR & P99 FPR & Alerts/10k & ms/flow \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _esc(row.get("setting", "")),
                    _esc(row.get("memory_size_mean", "")),
                    _esc(row.get("calibration_size_mean", "")),
                    fmt_pm(row.get("auroc_mean"), row.get("auroc_std")),
                    fmt_pm(row.get("fpr95_mean"), row.get("fpr95_std")),
                    fmt_pm(row.get("recall_at_1pct_fpr_mean"), row.get("recall_at_1pct_fpr_std")),
                    fmt_pm(row.get("p99_realized_fpr_mean"), row.get("p99_realized_fpr_std")),
                    fmt_pm(row.get("false_alerts_per_10k_benign_mean"), row.get("false_alerts_per_10k_benign_std"), 1),
                    fmt_pm(row.get("query_ms_per_flow_mean"), row.get("query_ms_per_flow_std"), 3),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_strategy_table(csv_path: str | Path, out_path: str | Path | None = None) -> None:
    behavior_strategies = {"old_memory_only", "random", "low_risk_only", "token_coreset"}
    rows = [row for row in read_csv(csv_path) if row.get("strategy") in behavior_strategies]
    out = Path(out_path) if out_path else PAPER_TABLE_DIR / "table_extra_benign_memory_strategies.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Strategy & Memory & AUROC & FPR95 & R@0.1\\%FPR & R@1\\%FPR & P99 FPR & ms/flow \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _esc(row.get("strategy", "")),
                    _esc(row.get("memory_size_mean", "")),
                    fmt_pm(row.get("auroc_mean"), row.get("auroc_std")),
                    fmt_pm(row.get("fpr95_mean"), row.get("fpr95_std")),
                    fmt_pm(row.get("recall_at_0_1pct_fpr_mean"), row.get("recall_at_0_1pct_fpr_std")),
                    fmt_pm(row.get("recall_at_1pct_fpr_mean"), row.get("recall_at_1pct_fpr_std")),
                    fmt_pm(row.get("p99_realized_fpr_mean"), row.get("p99_realized_fpr_std")),
                    fmt_pm(row.get("query_ms_per_flow_mean"), row.get("query_ms_per_flow_std"), 3),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_table(csv_path: str | Path, out_path: str | Path | None = None) -> None:
    rows = read_csv(csv_path)
    out = Path(out_path) if out_path else PAPER_TABLE_DIR / "table_extra_benign_calibration_scaling.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("threshold_type") not in {"p99", "p99_5"}:
            continue
        grouped[(row.get("calibration_source", ""), row.get("threshold_type", ""), int(float(row.get("calibration_size") or 0)))].append(row)
    summaries = []
    for (source, threshold_type, size), items in sorted(grouped.items()):
        def avg(key: str) -> float:
            vals = [float(item[key]) for item in items if item.get(key) not in {"", None}]
            return sum(vals) / len(vals) if vals else 0.0
        summaries.append(
            {
                "calibration_source": source,
                "threshold_type": threshold_type,
                "calibration_size": size,
                "threshold_std": avg("threshold_std"),
                "realized_test_fpr": avg("realized_test_fpr"),
                "recall": avg("recall"),
            }
        )
    selected = []
    for source in sorted({row["calibration_source"] for row in summaries}):
        for threshold_type in ["p99", "p99_5"]:
            source_rows = [row for row in summaries if row["calibration_source"] == source and row["threshold_type"] == threshold_type]
            sizes = sorted({row["calibration_size"] for row in source_rows})
            keep = set()
            if sizes:
                keep.add(sizes[0])
                keep.add(sizes[-1])
                for candidate in [100, 200, 500, 1000]:
                    if candidate in sizes:
                        keep.add(candidate)
                        break
            selected.extend(row for row in source_rows if row["calibration_size"] in keep)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Source & Threshold & Calib. size & Thr. std & FPR & Recall \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            " & ".join(
                [
                    _esc(row.get("calibration_source", "")),
                    _esc(row.get("threshold_type", "")),
                    _esc(row.get("calibration_size", "")),
                    fmt(row.get("threshold_std"), 4),
                    fmt(row.get("realized_test_fpr"), 4),
                    fmt(row.get("recall"), 4),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_throughput_table(csv_path: str | Path, out_path: str | Path | None = None) -> None:
    rows = read_csv(csv_path)
    out = Path(out_path) if out_path else PAPER_TABLE_DIR / "table_extra_benign_e2e_throughput.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Stage & Input & Output & Wall s & Throughput/s & Notes \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _esc(row.get("stage", "")),
                    _esc(row.get("input_count", "")),
                    _esc(row.get("output_count", "")),
                    fmt(row.get("wall_time_seconds"), 4),
                    fmt(row.get("throughput_per_second"), 1),
                    _esc(row.get("notes", "")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
