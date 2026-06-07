#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import write_json


def _read_json(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _name_from_path(path: str | Path) -> str:
    parent = Path(path).parent.name
    return parent or Path(path).stem


def _metric(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = summary.get(key)
    return default if value is None else float(value)


def flatten_policy(path: str | Path, name: str | None = None) -> dict[str, Any]:
    payload = _read_json(path)
    summary = payload.get("summary", {})
    stateful = payload.get("stateful_service") or {}
    row = {
        "name": name or _name_from_path(path),
        "path": str(path),
        "tokens": payload.get("tokens"),
        "metadata_tokens": payload.get("metadata_tokens"),
        "checkpoint": payload.get("checkpoint"),
        "split": payload.get("split"),
        "scope": payload.get("scope"),
        "threshold": payload.get("threshold"),
        "threshold_source": payload.get("threshold_source"),
        "calibration_scope": payload.get("calibration_scope"),
        "warmup_flows": int(payload.get("warmup_flows") or 0),
        "num_eval_flows": int(payload.get("num_eval_flows") or summary.get("num_flows") or 0),
        "macro_f1": _metric(summary, "macro_f1"),
        "weighted_f1": _metric(summary, "weighted_f1"),
        "accuracy": _metric(summary, "accuracy"),
        "false_positive_rate": _metric(summary, "false_positive_rate"),
        "attack_recall_online": _metric(summary, "attack_recall_online"),
        "false_positives": int(summary.get("false_positives") or 0),
        "true_positives": int(summary.get("true_positives") or 0),
        "confusion_matrix": summary.get("confusion_matrix"),
        "time_to_first_false_positive_seconds": summary.get("time_to_first_false_positive_seconds"),
        "mean_delay_seconds": summary.get("mean_delay_seconds"),
        "mean_delay_flows": summary.get("mean_delay_flows"),
        "stateful_enabled": bool(stateful.get("enabled", False)),
        "stateful_min_model_score": stateful.get("min_model_score"),
        "stateful_only_alerts": int(stateful.get("stateful_only_alerts") or 0),
        "stateful_missing_key_flows": int(stateful.get("stateful_missing_key_flows") or 0),
        "stateful_updated_flows": int(stateful.get("stateful_updated_flows") or 0),
    }
    if stateful.get("model_only_summary"):
        model_only = stateful["model_only_summary"]
        row["model_only_macro_f1"] = _metric(model_only, "macro_f1")
        row["model_only_false_positive_rate"] = _metric(model_only, "false_positive_rate")
        row["model_only_attack_recall_online"] = _metric(model_only, "attack_recall_online")
        row["model_only_confusion_matrix"] = model_only.get("confusion_matrix")
    return row


def select_policy(rows: list[dict[str, Any]], max_fpr: float, min_attack_recall: float) -> dict[str, Any] | None:
    feasible = [
        row
        for row in rows
        if float(row["false_positive_rate"]) <= max_fpr and float(row["attack_recall_online"]) >= min_attack_recall
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["attack_recall_online"]),
            -float(row["false_positive_rate"]),
            -int(bool(row.get("stateful_enabled", False))),
            -int(row.get("stateful_only_alerts") or 0),
            -int(row.get("warmup_flows") or 0),
            str(row["name"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="Path to online_replay_summary.json")
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--max_fpr", type=float, default=0.01)
    parser.add_argument("--min_attack_recall", type=float, default=0.995)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.names is not None and len(args.names) != len(args.summary):
        raise ValueError("--names length must match --summary length")

    rows = [
        flatten_policy(path, name=args.names[idx] if args.names is not None else None)
        for idx, path in enumerate(args.summary)
    ]
    rows = sorted(rows, key=lambda row: (-float(row["macro_f1"]), float(row["false_positive_rate"]), str(row["name"])))
    selected = select_policy(rows, max_fpr=float(args.max_fpr), min_attack_recall=float(args.min_attack_recall))
    payload = {
        "constraints": {
            "max_fpr": float(args.max_fpr),
            "min_attack_recall": float(args.min_attack_recall),
        },
        "selected": selected,
        "policies": rows,
    }
    write_json(payload, args.out)
    print({"selected": selected["name"] if selected else None, "num_policies": len(rows)})


if __name__ == "__main__":
    main()
