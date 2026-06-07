#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import write_json


DEFAULT_ROWS = [
    ("clean", "outputs/results/eval_zeek_dedup_clean_rawlabel_checkpoint"),
    ("packet_delete_010", "outputs/results/eval_zeek_dedup_packet_delete_010_rawlabel_checkpoint"),
    ("packet_insert_010", "outputs/results/eval_zeek_dedup_packet_insert_010_rawlabel_checkpoint"),
    ("direction_flip_010", "outputs/results/eval_zeek_dedup_direction_flip_010_rawlabel_checkpoint"),
    ("length_align_050", "outputs/results/eval_zeek_dedup_length_align_050_rawlabel_checkpoint"),
    ("length_padding_050", "outputs/results/eval_zeek_dedup_length_padding_050_rawlabel_checkpoint"),
    ("low_rate_c2_070", "outputs/results/eval_zeek_dedup_low_rate_c2_070_rawlabel_checkpoint"),
]

FULLMIX6_PAD025_ROWS = [
    ("clean", "outputs/results/eval_zeek_dedup_clean_rawlabel_fullmix6_pad025_checkpoint"),
    ("packet_delete_010", "outputs/results/eval_zeek_dedup_packet_delete_010_rawlabel_fullmix6_pad025_checkpoint"),
    ("packet_insert_010", "outputs/results/eval_zeek_dedup_packet_insert_010_rawlabel_fullmix6_pad025_checkpoint"),
    ("direction_flip_010", "outputs/results/eval_zeek_dedup_direction_flip_010_rawlabel_fullmix6_pad025_checkpoint"),
    ("length_align_050", "outputs/results/eval_zeek_dedup_length_align_050_rawlabel_fullmix6_pad025_checkpoint"),
    ("length_padding_050", "outputs/results/eval_zeek_dedup_length_padding_050_rawlabel_fullmix6_pad025_checkpoint"),
    ("low_rate_c2_070", "outputs/results/eval_zeek_dedup_low_rate_c2_070_rawlabel_fullmix6_pad025_checkpoint"),
]


def _load_metrics(result_dir: str) -> dict[str, Any]:
    path = Path(result_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _row(name: str, result_dir: str, clean_macro_f1: float | None) -> dict[str, Any]:
    metrics = _load_metrics(result_dir)
    macro_f1 = float(metrics["macro_f1"])
    row = {
        "condition": name,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": macro_f1,
        "weighted_f1": float(metrics["weighted_f1"]),
        "auroc_ovr": metrics.get("auroc_ovr"),
        "num_test": int(metrics["num_test"]),
        "result_dir": result_dir,
    }
    if clean_macro_f1 is None or clean_macro_f1 == 0.0:
        row["macro_f1_drop_abs"] = 0.0
        row["macro_f1_drop_rel_pct"] = 0.0
    else:
        row["macro_f1_drop_abs"] = float(clean_macro_f1 - macro_f1)
        row["macro_f1_drop_rel_pct"] = float((clean_macro_f1 - macro_f1) / clean_macro_f1 * 100.0)
    return row


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Zeek-first robustness diagnostics",
        "",
        "| Condition | Accuracy | Macro-F1 | Drop abs | Drop % | Weighted-F1 | AUROC-OVR | Test N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        auroc = row.get("auroc_ovr")
        auroc_text = "" if auroc is None else f"{float(auroc):.4f}"
        lines.append(
            "| {condition} | {accuracy:.4f} | {macro_f1:.4f} | {drop_abs:.4f} | {drop_pct:.2f} | "
            "{weighted_f1:.4f} | {auroc} | {num_test} |".format(
                condition=row["condition"],
                accuracy=row["accuracy"],
                macro_f1=row["macro_f1"],
                drop_abs=row["macro_f1_drop_abs"],
                drop_pct=row["macro_f1_drop_rel_pct"],
                weighted_f1=row["weighted_f1"],
                auroc=auroc_text,
                num_test=row["num_test"],
            )
        )
    lines.extend(
        [
            "",
            "说明：所有扰动评估均使用 clean token 的 `temporal_stratified_raw_label` 测试索引，checkpoint 为当前 Zeek-first exact-label temporal 主结果模型。",
        ]
    )
    return "\n".join(lines) + "\n"


def _comparison_markdown(clean_rows: list[dict[str, Any]], robust_rows: list[dict[str, Any]]) -> str:
    clean_by_condition = {row["condition"]: row for row in clean_rows}
    lines = [
        "# Zeek-first robustness: clean vs fullmix6-pad025",
        "",
        "| Condition | Clean Macro-F1 | Robust Macro-F1 | Gain | Clean Acc. | Robust Acc. | Robust AUROC-OVR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for robust in robust_rows:
        clean = clean_by_condition[robust["condition"]]
        gain = robust["macro_f1"] - clean["macro_f1"]
        auroc = robust.get("auroc_ovr")
        lines.append(
            "| {condition} | {clean_macro:.4f} | {robust_macro:.4f} | {gain:.4f} | "
            "{clean_acc:.4f} | {robust_acc:.4f} | {auroc:.4f} |".format(
                condition=robust["condition"],
                clean_macro=clean["macro_f1"],
                robust_macro=robust["macro_f1"],
                gain=gain,
                clean_acc=clean["accuracy"],
                robust_acc=robust["accuracy"],
                auroc=float(auroc) if auroc is not None else 0.0,
            )
        )
    lines.extend(
        [
            "",
            "说明：fullmix6-pad025 使用 packet delete/insert、direction flip、length align、low-rate C2 全量增强，length padding 使用 0.25 训练比例；评估仍使用 clean token 的 `temporal_stratified_raw_label` 测试索引。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", default="outputs/results/zeek_dedup_robustness_results.json")
    parser.add_argument("--out_md", default="outputs/results/zeek_dedup_robustness_results.md")
    args = parser.parse_args()

    clean_metrics = _load_metrics(DEFAULT_ROWS[0][1])
    clean_macro_f1 = float(clean_metrics["macro_f1"])
    rows = [_row(name, path, clean_macro_f1) for name, path in DEFAULT_ROWS]
    robust_metrics = _load_metrics(FULLMIX6_PAD025_ROWS[0][1])
    robust_clean_macro_f1 = float(robust_metrics["macro_f1"])
    robust_rows = [_row(name, path, robust_clean_macro_f1) for name, path in FULLMIX6_PAD025_ROWS]
    payload = {
        "protocol": {
            "base": "Zeek-first dedup balanced PCAP subset",
            "split": "temporal_stratified_raw_label",
            "split_source": "clean token dataset",
            "task": "multiclass_merged",
        },
        "rows": rows,
        "fullmix6_pad025": {
            "training": {
                "augment_all_labels": True,
                "augment_fractions": {
                    "packet_delete_010": 1.0,
                    "packet_insert_010": 1.0,
                    "direction_flip_010": 1.0,
                    "length_align_050": 1.0,
                    "length_padding_050": 0.25,
                    "low_rate_c2_070": 1.0,
                },
            },
            "rows": robust_rows,
        },
    }
    write_json(payload, args.out_json)
    markdown = _markdown(rows) + "\n" + _comparison_markdown(rows, robust_rows)
    Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
