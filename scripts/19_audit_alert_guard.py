#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import iter_jsonl, write_json


def _service(row: dict[str, Any]) -> str:
    protocol = row.get("alert_service_protocol") or row.get("stateful_service_protocol") or "UNKNOWN"
    port = row.get("alert_service_port") or row.get("stateful_service_port") or "UNKNOWN"
    return f"{protocol}|{port}"


def _top(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": int(count)} for key, count in counter.most_common(limit)]


def audit_scores(path: str | Path, top_k: int = 20) -> dict[str, Any]:
    rows = list(iter_jsonl(path))
    blocked = [row for row in rows if row.get("alert_guard_blocked_prediction")]
    blocked_tp = [row for row in blocked if int(row.get("binary_label_id", 0)) == 1]
    blocked_fp = [row for row in blocked if int(row.get("binary_label_id", 0)) == 0]
    passed_tp = [row for row in rows if int(row.get("binary_label_id", 0)) == 1 and int(row.get("prediction", 0)) == 1]
    missed = [row for row in rows if int(row.get("binary_label_id", 0)) == 1 and int(row.get("prediction", 0)) == 0]
    benign_alerts = [row for row in rows if int(row.get("binary_label_id", 0)) == 0 and int(row.get("prediction", 0)) == 1]
    return {
        "scores": str(path),
        "num_rows": len(rows),
        "blocked_predictions": len(blocked),
        "blocked_true_positives": len(blocked_tp),
        "blocked_false_positives": len(blocked_fp),
        "passed_true_positives": len(passed_tp),
        "missed_attacks": len(missed),
        "benign_alerts_after_guard": len(benign_alerts),
        "blocked_true_positive_labels": _top(Counter(str(row.get("label")) for row in blocked_tp), top_k),
        "blocked_true_positive_services": _top(Counter(_service(row) for row in blocked_tp), top_k),
        "blocked_false_positive_services": _top(Counter(_service(row) for row in blocked_fp), top_k),
        "passed_true_positive_labels": _top(Counter(str(row.get("label")) for row in passed_tp), top_k),
        "missed_attack_labels": _top(Counter(str(row.get("label")) for row in missed), top_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = audit_scores(args.scores, top_k=args.top_k)
    write_json(payload, args.out)
    print(
        {
            "blocked_true_positives": payload["blocked_true_positives"],
            "blocked_false_positives": payload["blocked_false_positives"],
            "benign_alerts_after_guard": payload["benign_alerts_after_guard"],
        }
    )


if __name__ == "__main__":
    main()
