#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import write_json


DEFAULT_RUNS = [
    ("seed42", "outputs/results/zeek_purebenign_expanded_dosweb_balanced3000_dedup_pcap_multiclass_merged_weighted_temporal_rawlabel"),
    ("seed43", "outputs/results/zeek_purebenign_expanded_dosweb_balanced3000_dedup_pcap_multiclass_merged_weighted_temporal_rawlabel_seed43"),
    ("seed44", "outputs/results/zeek_purebenign_expanded_dosweb_balanced3000_dedup_pcap_multiclass_merged_weighted_temporal_rawlabel_seed44"),
]


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


def _row(name: str, result_dir: str) -> dict[str, Any]:
    root = Path(result_dir)
    metrics = _load_json(root / "metrics.json")
    report = _load_json(root / "classification_report.json")
    return {
        "run": name,
        "result_dir": result_dir,
        "train_seed": int(metrics.get("train_seed", metrics.get("seed", -1))),
        "split": metrics.get("split"),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "auroc_ovr": float(metrics["auroc_ovr"]),
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


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "auroc_ovr",
        "webattack_precision",
        "webattack_recall",
        "webattack_f1",
    ]
    aggregate: dict[str, Any] = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        aggregate[field] = _mean_std(values) if values else None
    aggregate["num_runs"] = len(rows)
    aggregate["train_seeds"] = [row["train_seed"] for row in rows]
    return aggregate


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else str(value)


def _markdown(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        "# Zeek-first 3-seed stability",
        "",
        "| Run | Train seed | Accuracy | Macro-F1 | Weighted-F1 | AUROC-OVR | WebAttack P | WebAttack R | WebAttack F1 | Test N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {seed} | {acc} | {macro} | {weighted} | {auroc} | {wp} | {wr} | {wf1} | {n} |".format(
                run=row["run"],
                seed=row["train_seed"],
                acc=_fmt(row["accuracy"]),
                macro=_fmt(row["macro_f1"]),
                weighted=_fmt(row["weighted_f1"]),
                auroc=_fmt(row["auroc_ovr"]),
                wp=_fmt(row["webattack_precision"]),
                wr=_fmt(row["webattack_recall"]),
                wf1=_fmt(row["webattack_f1"]),
                n=row["num_test"],
            )
        )
    lines.extend(
        [
            "",
            "| Metric | Mean | Std | Min | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for field, label in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted-F1"),
        ("auroc_ovr", "AUROC-OVR"),
        ("webattack_precision", "WebAttack P"),
        ("webattack_recall", "WebAttack R"),
        ("webattack_f1", "WebAttack F1"),
    ]:
        stats = aggregate[field]
        lines.append(
            f"| {label} | {_fmt(stats['mean'])} | {_fmt(stats['std'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} |"
        )
    lines.extend(
        [
            "",
            "说明：三次运行使用相同 clean token、相同 `temporal_stratified_raw_label` split 和相同 weighted CE 配置，仅改变训练随机种子。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3-seed stability report for Zeek-first main classifier.")
    parser.add_argument("--out_json", default="outputs/results/zeek_dedup_rawlabel_seed_stability.json")
    parser.add_argument("--out_md", default="outputs/results/zeek_dedup_rawlabel_seed_stability.md")
    args = parser.parse_args()

    rows = [_row(name, path) for name, path in DEFAULT_RUNS]
    aggregate = _aggregate(rows)
    write_json(
        {
            "protocol": {
                "base": "Zeek-first dedup balanced PCAP subset",
                "split": "temporal_stratified_raw_label",
                "task": "multiclass_merged",
                "loss": "weighted_ce",
                "varying_factor": "train_seed",
            },
            "rows": rows,
            "aggregate": aggregate,
        },
        args.out_json,
    )
    Path(args.out_md).write_text(_markdown(rows, aggregate), encoding="utf-8")
    print(_markdown(rows, aggregate))


if __name__ == "__main__":
    main()
