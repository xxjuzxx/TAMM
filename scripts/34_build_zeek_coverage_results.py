#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import write_json


MAIN_RESULT = {
    "name": "dedup_6k",
    "dataset": "Zeek-first dedup balanced PCAP subset",
    "flows": "outputs/processed/zeek_purebenign_expanded_dosweb_binary_balanced3000_dedup_pcap_labeled_flows.jsonl",
    "sample_stats": "outputs/processed/zeek_purebenign_expanded_dosweb_binary_balanced3000_dedup_pcap_sample_stats.json",
    "token_stats": "outputs/tokens/zeek_purebenign_expanded_dosweb_binary_balanced3000_dedup_pcap_tokens_behavior_stats.json",
    "result": "outputs/results/zeek_purebenign_expanded_dosweb_balanced3000_dedup_pcap_multiclass_merged_weighted_temporal_rawlabel",
    "note": "当前论文主结果；不含 Botnet/Infiltration。",
}

V2_RESULT = {
    "name": "botnet_v2_6p5k",
    "dataset": "Zeek-first dedup PCAP subset with Botnet",
    "flows": "outputs/processed/zeek_purebenign_expanded_bot_dosweb_binary_balanced3500_dedup_pcap_labeled_flows.jsonl",
    "sample_stats": "outputs/processed/zeek_purebenign_expanded_bot_dosweb_binary_balanced3500_dedup_pcap_sample_stats.json",
    "token_stats": "outputs/tokens/zeek_purebenign_expanded_bot_dosweb_binary_balanced3500_dedup_pcap_tokens_behavior_stats.json",
    "result": "outputs/results/zeek_purebenign_expanded_bot_dosweb_balanced3500_dedup_pcap_multiclass_merged_weighted_temporal_rawlabel",
    "note": "覆盖扩展诊断；加入 Botnet，类别集合变更，不替换主表。",
}

THURSDAY_INFILTRATION_DROP_RESULT = {
    "name": "thursday_infiltration_drop",
    "dataset": "Zeek-first Thursday Infiltration window subset",
    "policy": "drop",
    "flows": "outputs/processed/zeek_thursday_infiltration_drop_windows_pcap_labeled_flows.jsonl",
    "sample_stats": "outputs/processed/thursday_infiltration_drop_slice_plan.json",
    "label_stats": "outputs/processed/zeek_thursday_infiltration_drop_windows_pcap_label_stats.json",
    "token_stats": "outputs/tokens/zeek_thursday_infiltration_drop_windows_pcap_tokens_behavior_stats.json",
    "result": "outputs/results/eval_zeek_thursday_infiltration_drop_windows_pcap_binary_checkpoint",
    "note": "full-day Thursday 诊断；drop-policy 口径，二分类 checkpoint 评估。",
}

THURSDAY_INFILTRATION_ATTACK_RESULT = {
    "name": "thursday_infiltration_attack",
    "dataset": "Zeek-first Thursday Infiltration window subset",
    "policy": "attack",
    "flows": "outputs/processed/zeek_thursday_infiltration_attack_windows_pcap_labeled_flows.jsonl",
    "sample_stats": "outputs/processed/thursday_infiltration_attack_slice_plan.json",
    "label_stats": "outputs/processed/zeek_thursday_infiltration_attack_windows_pcap_label_stats.json",
    "token_stats": "outputs/tokens/zeek_thursday_infiltration_attack_windows_pcap_tokens_behavior_stats.json",
    "result": "outputs/results/eval_zeek_thursday_infiltration_attack_windows_pcap_binary_checkpoint",
    "note": "full-day Thursday 诊断；attack-policy 反事实口径，二分类 checkpoint 评估。",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _report_item(report: dict[str, Any], name: str, field: str) -> Any:
    item = report.get(name)
    if not isinstance(item, dict):
        return None
    return item.get(field)


def _metric_item(metrics: dict[str, Any], *fields: str) -> Any:
    for field in fields:
        value = metrics.get(field)
        if value is not None:
            return value
    return None


def _row(spec: dict[str, str]) -> dict[str, Any]:
    required = [
        spec["sample_stats"],
        spec["token_stats"],
        str(Path(spec["result"]) / "metrics.json"),
        str(Path(spec["result"]) / "classification_report.json"),
    ]
    if spec.get("label_stats"):
        required.append(str(spec["label_stats"]))
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        return {
            **spec,
            "artifact_status": "archived",
            "missing_paths": missing,
            "selected_rows": None,
            "selected_binary_counts": None,
            "selected_label_counts": {},
            "semantic_duplicate_rows_dropped": None,
            "vocab_size": None,
            "num_test": None,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "auroc_ovr": None,
            "auprc": None,
            "botnet_f1": None,
            "botnet_recall": None,
            "webattack_f1": None,
            "webattack_recall": None,
            "attack_f1": None,
            "attack_recall": None,
            "benign_f1": None,
            "benign_recall": None,
        }
    sample = _load_json(spec["sample_stats"])
    token_stats = _load_json(spec["token_stats"])
    result_dir = Path(spec["result"])
    metrics = _load_json(result_dir / "metrics.json")
    report = _load_json(result_dir / "classification_report.json")
    label_stats = _load_json(spec["label_stats"]) if spec.get("label_stats") else None
    row = {
        **spec,
        "selected_rows": sample.get("selected_rows"),
        "selected_binary_counts": sample.get("selected_binary_counts"),
        "selected_label_counts": sample.get("selected_label_counts"),
        "semantic_duplicate_rows_dropped": sample.get("semantic_duplicate_rows_dropped"),
        "vocab_size": token_stats.get("vocab_size"),
        "num_test": metrics.get("num_test"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "auroc_ovr": _metric_item(metrics, "auroc_ovr", "auroc"),
        "auprc": metrics.get("auprc"),
        "botnet_f1": _report_item(report, "Botnet", "f1-score"),
        "botnet_recall": _report_item(report, "Botnet", "recall"),
        "webattack_f1": _report_item(report, "WebAttack", "f1-score"),
        "webattack_recall": _report_item(report, "WebAttack", "recall"),
        "attack_f1": _report_item(report, "ATTACK", "f1-score"),
        "attack_recall": _report_item(report, "ATTACK", "recall"),
        "benign_f1": _report_item(report, "BENIGN", "f1-score"),
        "benign_recall": _report_item(report, "BENIGN", "recall"),
    }
    if label_stats is not None:
        row.update(
            {
                "match_rate": label_stats.get("match_rate"),
                "matched": label_stats.get("matched"),
                "unmatched": label_stats.get("unmatched"),
                "label_counts": label_stats.get("label_counts"),
                "label_timestamp_status_counts": label_stats.get("label_timestamp_status_counts"),
                "attempted_policy": label_stats.get("attempted_policy"),
            }
        )
    return row


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown(rows: list[dict[str, Any]], infiltration: dict[str, Any]) -> str:
    lines = [
        "# Zeek-first coverage expansion diagnostics",
        "",
        "| Dataset | Flows | Exact labels | Vocab | Test N | Accuracy | Macro-F1 | Weighted-F1 | AUROC-OVR | Botnet R/F1 | WebAttack R/F1 | Note |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        classes = ", ".join(sorted(row.get("selected_label_counts") or {}))
        botnet = "-" if row.get("botnet_f1") is None else f"{_fmt(row['botnet_recall'])}/{_fmt(row['botnet_f1'])}"
        webattack = "-" if row.get("webattack_f1") is None else f"{_fmt(row['webattack_recall'])}/{_fmt(row['webattack_f1'])}"
        lines.append(
            "| {dataset} | {flows} | {classes} | {vocab} | {test_n} | {acc} | {macro} | {weighted} | {auroc} | {botnet} | {webattack} | {note} |".format(
                dataset=row["dataset"],
                flows=row["selected_rows"],
                classes=classes,
                vocab=row["vocab_size"],
                test_n=row["num_test"],
                acc=_fmt(row["accuracy"]),
                macro=_fmt(row["macro_f1"]),
                weighted=_fmt(row["weighted_f1"]),
                auroc=_fmt(row["auroc_ovr"]),
                botnet=botnet,
                webattack=webattack,
                note=f"{row['note']} ({row.get('artifact_status', 'available')})",
            )
        )
    infiltration_rows = infiltration.get("rows", [])
    if infiltration_rows:
        lines.extend(
            [
                "",
                "## Thursday Infiltration binary diagnostics",
                "",
                "| Dataset | Policy | Window Flows | Selected Labels | Match Rate | Test N | Accuracy | Macro-F1 | Weighted-F1 | AUROC-OVR | AUPRC | ATTACK R/F1 | BENIGN R/F1 | Note |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for row in infiltration_rows:
            labels = ", ".join(f"{label} {count}" for label, count in sorted((row.get("selected_label_counts") or {}).items()))
            attack = "-" if row.get("attack_f1") is None else f"{_fmt(row['attack_recall'])}/{_fmt(row['attack_f1'])}"
            benign = "-" if row.get("benign_f1") is None else f"{_fmt(row['benign_recall'])}/{_fmt(row['benign_f1'])}"
            lines.append(
                "| {dataset} | {policy} | {flows} | {labels} | {match_rate} | {test_n} | {acc} | {macro} | {weighted} | {auroc} | {auprc} | {attack} | {benign} | {note} |".format(
                    dataset=row["dataset"],
                    policy=row.get("policy", "-"),
                    flows=row["selected_rows"],
                    labels=labels,
                    match_rate=_fmt(row.get("match_rate")),
                    test_n=row["num_test"],
                    acc=_fmt(row["accuracy"]),
                    macro=_fmt(row["macro_f1"]),
                    weighted=_fmt(row["weighted_f1"]),
                    auroc=_fmt(row["auroc_ovr"]),
                    auprc=_fmt(row.get("auprc")),
                    attack=attack,
                    benign=benign,
                    note=f"{row['note']} ({row.get('artifact_status', 'available')})",
                )
            )
    lines.extend(
        [
            "",
            "## Coverage notes",
            "",
            f"- Botnet can be added under `attempted_policy=drop`: the local Bot PCAP produced 729 matched Botnet flows, and the expanded coverage sample retained 378 Botnet flows after class-balanced binary capping.",
            "- Thursday Infiltration binary diagnostics are reported when artifacts are available; archived rows are marked explicitly.",
            f"- The binary checkpoint reaches macro-F1 {_fmt(infiltration['drop_macro_f1'])} on the drop-policy window subset and {_fmt(infiltration['attack_macro_f1'])} on the attack-policy window subset. This is useful for coverage diagnosis, but it still does not replace the 6K multiclass main table.",
            "- The expanded coverage sample is therefore a coverage diagnostic, not a replacement for the current 6K paper main table.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Zeek-first coverage expansion diagnostics.")
    parser.add_argument("--out_json", default="outputs/results/zeek_coverage_expansion_results.json")
    parser.add_argument("--out_md", default="outputs/results/zeek_coverage_expansion_results.md")
    args = parser.parse_args()

    rows = [_row(MAIN_RESULT), _row(V2_RESULT)]
    infiltration_rows = [_row(THURSDAY_INFILTRATION_DROP_RESULT), _row(THURSDAY_INFILTRATION_ATTACK_RESULT)]
    infiltration = {
        "rows": infiltration_rows,
        "drop_selected_rows": infiltration_rows[0].get("selected_rows"),
        "drop_matched": infiltration_rows[0].get("matched"),
        "drop_match_rate": infiltration_rows[0].get("match_rate"),
        "drop_macro_f1": infiltration_rows[0].get("macro_f1"),
        "attack_selected_rows": infiltration_rows[1].get("selected_rows"),
        "attack_matched": infiltration_rows[1].get("matched"),
        "attack_match_rate": infiltration_rows[1].get("match_rate"),
        "attack_macro_f1": infiltration_rows[1].get("macro_f1"),
    }
    payload = {
        "protocol": {
            "main_label_policy": "attempted_policy=drop",
            "split": "temporal_stratified_raw_label",
            "task": "multiclass_merged",
            "note": "Coverage rows use different label sets; compare as diagnostics, not a single leaderboard.",
        },
        "rows": rows,
        "infiltration_rows": infiltration_rows,
        "infiltration_policy_note": infiltration,
    }
    write_json(payload, args.out_json)
    Path(args.out_md).write_text(_markdown(rows, infiltration), encoding="utf-8")
    print(_markdown(rows, infiltration))


if __name__ == "__main__":
    main()
