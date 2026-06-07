#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER_TABLE_DIR = ROOT.parent / "paper" / "tables"

DEFAULT_CATEGORY = ROOT / "results" / "primitive_categories" / "primitive_category_feature_attribution.csv"
DEFAULT_CATEGORY_RUNS = ROOT / "results" / "primitive_categories" / "primitive_category_unknown_metrics.csv"

MAIN_FIXED_OUT = PAPER_TABLE_DIR / "table_main_low_fpr_fixed.tex"
FIXED_PER_ATTACK_OUT = PAPER_TABLE_DIR / "table_unknown_behavior_only.tex"
FIXED_CALIBRATION_OUT = PAPER_TABLE_DIR / "table_target_realized_fpr_behavior_only.tex"
FIXED_ALERT_OUT = PAPER_TABLE_DIR / "table_alert_budget_behavior_only.tex"
FIXED_PREVALENCE_OUT = PAPER_TABLE_DIR / "table_prevalence_alert_budget_behavior_only.tex"

MAIN_FIXED_ORDER = [
    "profile_only",
    "structural_only",
    "packet_burst_only",
    "packet_burst_plus_profile",
    "packet_burst_plus_structural",
    "packet_burst_plus_profile_structural",
]

MAIN_FIXED_LABELS = {
    "profile_only": ("Profile primitives only", r"\texttt{PRIM\_PROFILE\_*}"),
    "structural_only": ("Structural primitives only", r"\texttt{PRIM\_STRUCT\_*}"),
    "packet_burst_only": ("Packet+burst tokens", "packet/burst"),
    "packet_burst_plus_profile": ("Packet+burst + profile", r"packet/burst + \texttt{PRIM\_PROFILE\_*}"),
    "packet_burst_plus_structural": ("Packet+burst + structural", r"packet/burst + \texttt{PRIM\_STRUCT\_*}"),
    "packet_burst_plus_profile_structural": (
        r"\textbf{FlowPrim fixed}",
        "packet/burst + both primitive categories",
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _fmt(value: Any, std: Any | None = None, digits: int = 4, bold: bool = False) -> str:
    mean = _num(value)
    if mean is None:
        return "-"
    sigma = _num(std)
    body = f"{mean:.{digits}f}" if sigma is None else rf"{mean:.{digits}f}\pm{sigma:.{digits}f}"
    return rf"$\mathbf{{{body}}}$" if bold else rf"${body}$"


def _fmt_plain(value: Any, digits: int = 1) -> str:
    val = _num(value)
    return "-" if val is None else f"{val:.{digits}f}"


def build_main_fixed(rows: list[dict[str, str]]) -> str:
    by_view = {row["feature_view"]: row for row in rows}
    lines = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"View & Evidence & AUROC & FPR95 & R@0.1\%FPR & R@1\%FPR & P99 FPR & Alerts/10k \\",
        r"\midrule",
    ]
    for view in MAIN_FIXED_ORDER:
        row = by_view.get(view)
        if row is None:
            continue
        label, evidence = MAIN_FIXED_LABELS[view]
        bold = view == "packet_burst_plus_profile_structural"
        lines.append(
            " & ".join(
                [
                    label,
                    evidence,
                    _fmt(row.get("auroc"), row.get("auroc_std"), bold=bold),
                    _fmt(row.get("fpr95"), row.get("fpr95_std"), bold=bold),
                    _fmt(row.get("recall_at_0_1pct_fpr"), row.get("recall_at_0_1pct_fpr_std"), bold=bold),
                    _fmt(row.get("recall_at_1pct_fpr"), row.get("recall_at_1pct_fpr_std"), bold=bold),
                    _fmt(row.get("val_p99_realized_fpr"), row.get("val_p99_realized_fpr_std"), bold=bold),
                    _fmt_plain(row.get("false_alerts_per_10k_benign")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _group_stats(rows: list[dict[str, str]], keys: list[str]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key in keys:
        vals = [_num(row.get(key)) for row in rows]
        clean = [float(v) for v in vals if v is not None]
        out[key] = (_mean(clean), _std(clean)) if clean else (float("nan"), float("nan"))
    return out


def _fixed_flowprim_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("feature_view") == "packet_burst_plus_profile_structural"
        and row.get("memory_scope", "global") == "global"
    ]


def build_fixed_per_attack(rows: list[dict[str, str]]) -> str:
    metrics = [
        "best_macro_f1",
        "auroc",
        "auprc",
        "fpr95",
        "recall_at_0_1pct_fpr",
        "recall_at_1pct_fpr",
        "val_p99_realized_fpr",
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _fixed_flowprim_runs(rows):
        grouped[row["heldout_attack"]].append(row)
    attack_order = ["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"]
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Unknown & Macro-F1 & AUROC & AUPRC & FPR95 & R@0.1\%FPR & R@1\%FPR & P99 FPR \\",
        r"\midrule",
    ]
    all_rows: list[dict[str, str]] = []
    for attack in attack_order:
        sub = grouped.get(attack, [])
        if not sub:
            continue
        all_rows.extend(sub)
        stats = _group_stats(sub, metrics)
        label = attack + (r"$^\ddagger$" if attack == "WebAttack" else "")
        lines.append(
            " & ".join(
                [
                    label,
                    _fmt(*stats["best_macro_f1"]),
                    _fmt(*stats["auroc"]),
                    _fmt(*stats["auprc"]),
                    _fmt(*stats["fpr95"]),
                    _fmt(*stats["recall_at_0_1pct_fpr"]),
                    _fmt(*stats["recall_at_1pct_fpr"]),
                    _fmt(*stats["val_p99_realized_fpr"]),
                ]
            )
            + r" \\"
        )
    stats = _group_stats(all_rows, metrics)
    lines.extend(
        [
            r"\midrule",
            " & ".join(
                [
                    "Aggregate",
                    _fmt(*stats["best_macro_f1"], bold=True),
                    _fmt(*stats["auroc"], bold=True),
                    _fmt(*stats["auprc"], bold=True),
                    _fmt(*stats["fpr95"], bold=True),
                    _fmt(*stats["recall_at_0_1pct_fpr"], bold=True),
                    _fmt(*stats["recall_at_1pct_fpr"], bold=True),
                    _fmt(*stats["val_p99_realized_fpr"], bold=True),
                ]
            )
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_fixed_calibration(rows: list[dict[str, str]]) -> str:
    sub = _fixed_flowprim_runs(rows)
    metrics = {
        "macro_f1_oracle_1pct": "Macro-F1",
        "val_p99_attack_recall": "Attack recall",
        "val_p99_realized_fpr": "Realized FPR",
        "val_p99_threshold": "Threshold",
    }
    stats = _group_stats(sub, list(metrics))
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"BENIGN-val threshold & Macro-F1 & Attack recall & Realized FPR & Threshold \\",
        r"\midrule",
        " & ".join(
            [
                "P99",
                _fmt(*stats["macro_f1_oracle_1pct"]),
                _fmt(*stats["val_p99_attack_recall"]),
                _fmt(*stats["val_p99_realized_fpr"]),
                _fmt(*stats["val_p99_threshold"]),
            ]
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def build_fixed_alert_budget(rows: list[dict[str, str]]) -> str:
    sub = _fixed_flowprim_runs(rows)
    stats = _group_stats(
        sub,
        [
            "macro_f1_oracle_1pct",
            "recall_at_1pct_fpr",
            "actual_fpr_at_1pct_fpr",
            "val_p99_macro_f1",
            "val_p99_attack_recall",
            "val_p99_realized_fpr",
            "false_alerts_per_10k_benign",
        ],
    )
    oracle_false_alerts = (stats["actual_fpr_at_1pct_fpr"][0] * 10000, stats["actual_fpr_at_1pct_fpr"][1] * 10000)
    rows_out = [
        (
            "Oracle 1\\%FPR",
            oracle_false_alerts,
            (stats["recall_at_1pct_fpr"][0] * 10000, stats["recall_at_1pct_fpr"][1] * 10000),
            stats["recall_at_1pct_fpr"],
            stats["macro_f1_oracle_1pct"],
        ),
        (
            "BENIGN-val P99",
            stats["false_alerts_per_10k_benign"],
            (stats["val_p99_attack_recall"][0] * 10000, stats["val_p99_attack_recall"][1] * 10000),
            stats["val_p99_attack_recall"],
            stats["val_p99_macro_f1"],
        ),
    ]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Threshold & False alerts / 10k benign & Detected / 10k attacks & Recall & Macro-F1 \\",
        r"\midrule",
    ]
    for label, false_alerts, detected, recall, macro in rows_out:
        lines.append(
            " & ".join(
                [
                    label,
                    _fmt(*false_alerts, digits=1),
                    _fmt(*detected, digits=1),
                    _fmt(*recall),
                    _fmt(*macro),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _prevalence_row(label: str, fpr: tuple[float, float], recall: tuple[float, float], prevalence: float) -> tuple[str, str, float, float, float, float, float]:
    fpr_mean, fpr_std = fpr
    rec_mean, rec_std = recall
    false_alerts = (1.0 - prevalence) * fpr_mean * 10000.0
    true_alerts = prevalence * rec_mean * 10000.0
    total_alerts = false_alerts + true_alerts
    precision = true_alerts / total_alerts if total_alerts > 0 else float("nan")
    return (label, f"{prevalence * 100:.1f}\\%", false_alerts, true_alerts, total_alerts, precision, rec_mean)


def build_fixed_prevalence(rows: list[dict[str, str]]) -> str:
    sub = _fixed_flowprim_runs(rows)
    stats = _group_stats(
        sub,
        [
            "actual_fpr_at_1pct_fpr",
            "recall_at_1pct_fpr",
            "val_p99_realized_fpr",
            "val_p99_attack_recall",
        ],
    )
    body = []
    for label, fpr_key, recall_key in [
        ("Oracle 1\\%FPR", "actual_fpr_at_1pct_fpr", "recall_at_1pct_fpr"),
        ("BENIGN-val P99", "val_p99_realized_fpr", "val_p99_attack_recall"),
    ]:
        for prevalence in [0.001, 0.01, 0.05]:
            body.append(_prevalence_row(label, stats[fpr_key], stats[recall_key], prevalence))
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Threshold & Prevalence & False alerts/10k & True alerts/10k & Total alerts/10k & Precision & Recall \\",
        r"\midrule",
    ]
    for label, prevalence, false_alerts, true_alerts, total_alerts, precision, recall in body:
        lines.append(
            " & ".join(
                [
                    label,
                    prevalence,
                    f"${false_alerts:.1f}$",
                    f"${true_alerts:.1f}$",
                    f"${total_alerts:.1f}$",
                    f"${precision:.4f}$",
                    f"${recall:.4f}$",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed behavior-only low-FPR paper tables.")
    parser.add_argument("--category-input", default=str(DEFAULT_CATEGORY))
    parser.add_argument("--category-runs-input", default=str(DEFAULT_CATEGORY_RUNS))
    parser.add_argument("--paper-table-dir", default=str(PAPER_TABLE_DIR))
    args = parser.parse_args()

    table_dir = Path(args.paper_table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    category_rows = _read_csv(Path(args.category_input))
    category_run_rows = _read_csv(Path(args.category_runs_input))

    main_out = table_dir / MAIN_FIXED_OUT.name
    fixed_per_attack_out = table_dir / FIXED_PER_ATTACK_OUT.name
    fixed_calibration_out = table_dir / FIXED_CALIBRATION_OUT.name
    fixed_alert_out = table_dir / FIXED_ALERT_OUT.name
    fixed_prevalence_out = table_dir / FIXED_PREVALENCE_OUT.name
    main_out.write_text(build_main_fixed(category_rows), encoding="utf-8")
    fixed_per_attack_out.write_text(build_fixed_per_attack(category_run_rows), encoding="utf-8")
    fixed_calibration_out.write_text(build_fixed_calibration(category_run_rows), encoding="utf-8")
    fixed_alert_out.write_text(build_fixed_alert_budget(category_run_rows), encoding="utf-8")
    fixed_prevalence_out.write_text(build_fixed_prevalence(category_run_rows), encoding="utf-8")
    print(main_out)
    print(fixed_per_attack_out)
    print(fixed_calibration_out)
    print(fixed_alert_out)
    print(fixed_prevalence_out)


if __name__ == "__main__":
    main()
