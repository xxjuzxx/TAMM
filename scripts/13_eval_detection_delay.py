#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import read_jsonl, write_json


def _is_attack(row: dict[str, Any]) -> bool:
    if int(row.get("binary_label", 0)) == 1:
        return True
    label = str(row.get("label", "")).lower()
    return label not in {"", "benign", "normal"}


def _is_alert(row: dict[str, Any], threshold: float | None) -> bool:
    if threshold is None:
        return int(row.get("prediction", 0)) == 1
    return float(row.get("anomaly_score", 0.0)) >= threshold


def _safe_ts(row: dict[str, Any], fallback: int) -> float:
    value = row.get("start_ts")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _first_consecutive_alert(items: list[tuple[int, dict[str, Any]]], threshold: float | None, consecutive_alerts: int) -> tuple[int, dict[str, Any]] | None:
    needed = max(1, int(consecutive_alerts))
    run: list[tuple[int, dict[str, Any]]] = []
    for pos, row in items:
        if _is_alert(row, threshold):
            run.append((pos, row))
            if len(run) >= needed:
                return run[0]
        else:
            run = []
    return None


def _delay_rows(rows: list[dict[str, Any]], threshold: float | None, consecutive_alerts: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(enumerate(rows), key=lambda item: (_safe_ts(item[1], item[0]), int(item[1].get("index", item[0]))))
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for stream_pos, (_, row) in enumerate(ordered):
        if not _is_attack(row):
            continue
        label = str(row.get("label") or "ATTACK")
        groups[label].append((stream_pos, row))

    out: list[dict[str, Any]] = []
    detected_delays_sec: list[float] = []
    detected_delays_flows: list[int] = []
    for label, items in sorted(groups.items()):
        first_pos, first_row = items[0]
        first_ts = _safe_ts(first_row, first_pos)
        alert_item = _first_consecutive_alert(items, threshold, consecutive_alerts)
        detected = alert_item is not None
        if alert_item is None:
            alert_pos = None
            alert_ts = None
            delay_sec = None
            delay_flows = None
        else:
            alert_pos, alert_row = alert_item
            alert_ts = _safe_ts(alert_row, alert_pos)
            delay_sec = float(max(0.0, alert_ts - first_ts))
            delay_flows = int(max(0, alert_pos - first_pos))
            detected_delays_sec.append(delay_sec)
            detected_delays_flows.append(delay_flows)
        out.append(
            {
                "label": label,
                "attack_flows": len(items),
                "first_attack_ts": first_ts,
                "first_alert_ts": alert_ts,
                "detected": detected,
                "delay_seconds": delay_sec,
                "delay_flows": delay_flows,
            }
        )

    summary = {
        "attack_labels": len(out),
        "detected_labels": int(sum(1 for row in out if row["detected"])),
        "missed_labels": int(sum(1 for row in out if not row["detected"])),
        "mean_delay_seconds": float(sum(detected_delays_sec) / len(detected_delays_sec)) if detected_delays_sec else None,
        "median_delay_seconds": float(sorted(detected_delays_sec)[len(detected_delays_sec) // 2]) if detected_delays_sec else None,
        "mean_delay_flows": float(sum(detected_delays_flows) / len(detected_delays_flows)) if detected_delays_flows else None,
        "median_delay_flows": float(sorted(detected_delays_flows)[len(detected_delays_flows) // 2]) if detected_delays_flows else None,
        "consecutive_alerts": int(max(1, consecutive_alerts)),
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--consecutive_alerts", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    delays, summary = _delay_rows(rows, threshold=args.threshold, consecutive_alerts=args.consecutive_alerts)
    payload = {
        "scores": args.scores,
        "threshold": args.threshold,
        "summary": summary,
        "per_label": delays,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, out)
    print(payload)


if __name__ == "__main__":
    main()
