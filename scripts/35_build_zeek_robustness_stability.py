#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import write_json


CONDITIONS = [
    "clean",
    "packet_delete_010",
    "packet_insert_010",
    "direction_flip_010",
    "length_align_050",
    "length_padding_050",
    "low_rate_c2_070",
]

ROBUST_EVAL_DIRS = {
    42: "outputs/results/eval_zeek_dedup_{condition}_rawlabel_fullmix6_pad025_checkpoint",
    43: "outputs/results/eval_zeek_dedup_{condition}_rawlabel_fullmix6_pad025_seed43_checkpoint",
    44: "outputs/results/eval_zeek_dedup_{condition}_rawlabel_fullmix6_pad025_seed44_checkpoint",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _report_item(report: dict[str, Any], name: str, field: str) -> float | None:
    item = report.get(name)
    if not isinstance(item, dict):
        return None
    value = item.get(field)
    return None if value is None else float(value)


def _row(seed: int, condition: str, result_dir: str) -> dict[str, Any]:
    root = Path(result_dir)
    metrics = _load_json(root / "metrics.json")
    report = _load_json(root / "classification_report.json")
    return {
        "seed": seed,
        "condition": condition,
        "result_dir": result_dir,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "auroc_ovr": None if metrics.get("auroc_ovr") is None else float(metrics["auroc_ovr"]),
        "num_test": int(metrics["num_test"]),
        "webattack_precision": _report_item(report, "WebAttack", "precision"),
        "webattack_recall": _report_item(report, "WebAttack", "recall"),
        "webattack_f1": _report_item(report, "WebAttack", "f1-score"),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "auroc_ovr",
        "webattack_precision",
        "webattack_recall",
        "webattack_f1",
    ]
    aggregate_rows: list[dict[str, Any]] = []
    by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    clean_stats = _mean_std([float(row["macro_f1"]) for row in by_condition["clean"]])
    clean_macro_mean = clean_stats["mean"]
    for condition in CONDITIONS:
        condition_rows = by_condition[condition]
        item: dict[str, Any] = {
            "condition": condition,
            "num_runs": len(condition_rows),
            "seeds": [int(row["seed"]) for row in condition_rows],
            "num_test": condition_rows[0]["num_test"] if condition_rows else None,
        }
        for field in fields:
            values = [float(row[field]) for row in condition_rows if row.get(field) is not None]
            item[field] = _mean_std(values) if values else None
        if item["macro_f1"] is None or clean_macro_mean == 0.0:
            item["macro_f1_drop_from_clean_mean"] = None
        else:
            item["macro_f1_drop_from_clean_mean"] = float(clean_macro_mean - item["macro_f1"]["mean"])
        aggregate_rows.append(item)
    return aggregate_rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else str(value)


def _fmt_pm(stats: dict[str, float] | None, digits: int = 4) -> str:
    if stats is None:
        return "-"
    return f"{stats['mean']:.{digits}f} +/- {stats['std']:.{digits}f}"


def _markdown(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Zeek-first robust augmentation 3-seed stability",
        "",
        "## Aggregate by condition",
        "",
        "| Condition | Macro-F1 mean +/- std | Drop from clean mean | Accuracy mean +/- std | Weighted-F1 mean +/- std | AUROC-OVR mean +/- std | WebAttack F1 mean +/- std | Test N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {condition} | {macro} | {drop} | {acc} | {weighted} | {auroc} | {web_f1} | {n} |".format(
                condition=row["condition"],
                macro=_fmt_pm(row["macro_f1"]),
                drop=_fmt(row["macro_f1_drop_from_clean_mean"]),
                acc=_fmt_pm(row["accuracy"]),
                weighted=_fmt_pm(row["weighted_f1"]),
                auroc=_fmt_pm(row["auroc_ovr"]),
                web_f1=_fmt_pm(row["webattack_f1"]),
                n=row["num_test"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-seed rows",
            "",
            "| Seed | Condition | Accuracy | Macro-F1 | Weighted-F1 | AUROC-OVR | WebAttack P | WebAttack R | WebAttack F1 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["condition"], item["seed"])):
        lines.append(
            "| {seed} | {condition} | {acc} | {macro} | {weighted} | {auroc} | {wp} | {wr} | {wf1} |".format(
                seed=row["seed"],
                condition=row["condition"],
                acc=_fmt(row["accuracy"]),
                macro=_fmt(row["macro_f1"]),
                weighted=_fmt(row["weighted_f1"]),
                auroc=_fmt(row["auroc_ovr"]),
                wp=_fmt(row["webattack_precision"]),
                wr=_fmt(row["webattack_recall"]),
                wf1=_fmt(row["webattack_f1"]),
            )
        )
    lines.extend(
        [
            "",
            "说明：三次运行均使用同一 Zeek-first dedup token、同一 `temporal_stratified_raw_label` 测试索引和 fullmix6-pad025 增强策略，仅改变训练随机种子。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3-seed robustness stability report for Zeek-first robust checkpoint.")
    parser.add_argument("--out_json", default="outputs/results/zeek_dedup_robustness_stability.json")
    parser.add_argument("--out_md", default="outputs/results/zeek_dedup_robustness_stability.md")
    args = parser.parse_args()

    rows = [
        _row(seed, condition, pattern.format(condition=condition))
        for seed, pattern in ROBUST_EVAL_DIRS.items()
        for condition in CONDITIONS
    ]
    aggregate_rows = _aggregate(rows)
    payload = {
        "protocol": {
            "base": "Zeek-first dedup balanced PCAP subset",
            "split": "temporal_stratified_raw_label",
            "split_source": "clean token dataset",
            "task": "multiclass_merged",
            "robust_training": "fullmix6-pad025",
            "varying_factor": "train_seed",
        },
        "rows": rows,
        "aggregate_rows": aggregate_rows,
    }
    write_json(payload, args.out_json)
    markdown = _markdown(rows, aggregate_rows)
    Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
