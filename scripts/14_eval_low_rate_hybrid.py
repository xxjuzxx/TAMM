#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.utils.io import read_jsonl, write_json, write_jsonl


def _ts(row: dict[str, Any], fallback: int) -> float:
    try:
        return float(row.get("start_ts"))
    except (TypeError, ValueError):
        return float(fallback)


def _score(row: dict[str, Any]) -> float:
    if "attack_probability" in row:
        return float(row["attack_probability"])
    if "anomaly_score" in row:
        return float(row["anomaly_score"])
    return 0.0


def _hybrid(
    rows: list[dict[str, Any]],
    threshold: float,
    window_seconds: float,
    min_count: int,
    max_packets: int,
    min_episode_score: float,
) -> list[dict[str, Any]]:
    indexed = sorted(enumerate(rows), key=lambda item: (_ts(item[1], item[0]), int(item[1].get("index", item[0]))))
    recent: deque[tuple[float, int]] = deque()
    out_by_original: dict[int, dict[str, Any]] = {}
    for original_pos, row in indexed:
        cur_ts = _ts(row, original_pos)
        pkt_n = int(row.get("packet_count") or 0)
        is_short = pkt_n <= max_packets
        while recent and cur_ts - recent[0][0] > window_seconds:
            recent.popleft()
        if is_short:
            recent.append((cur_ts, original_pos))
        episode_count = len(recent)
        model_alert = _score(row) >= threshold
        episode_alert = is_short and episode_count >= min_count and _score(row) >= min_episode_score
        merged = dict(row)
        merged["model_prediction"] = int(model_alert)
        merged["episode_count"] = int(episode_count)
        merged["episode_prediction"] = int(episode_alert)
        merged["hybrid_prediction"] = int(model_alert or episode_alert)
        out_by_original[original_pos] = merged
    return [out_by_original[idx] for idx in range(len(rows))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--window_seconds", type=float, default=1.0)
    parser.add_argument("--min_count", type=int, default=20)
    parser.add_argument("--max_packets", type=int, default=4)
    parser.add_argument("--min_episode_score", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    scored = _hybrid(
        rows,
        threshold=args.threshold,
        window_seconds=args.window_seconds,
        min_count=args.min_count,
        max_packets=args.max_packets,
        min_episode_score=args.min_episode_score,
    )
    y_true = [int(row.get("binary_label", 0)) for row in scored]
    y_pred = [int(row["hybrid_prediction"]) for row in scored]
    probs = np.array([[1.0 - _score(row), _score(row)] for row in scored], dtype=np.float32)
    metrics = classification_metrics(y_true, y_pred, probs)
    metrics.update(
        {
            "threshold": args.threshold,
            "window_seconds": args.window_seconds,
            "min_count": args.min_count,
            "max_packets": args.max_packets,
            "min_episode_score": args.min_episode_score,
            "num_test": len(rows),
            "model_alerts": int(sum(int(row["model_prediction"]) for row in scored)),
            "episode_alerts": int(sum(int(row["episode_prediction"]) for row in scored)),
            "hybrid_alerts": int(sum(y_pred)),
        }
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics, out_dir / "metrics.json")
    write_json(report_dict(y_true, y_pred, target_names=["BENIGN", "ATTACK"]), out_dir / "classification_report.json")
    write_json(confusion(y_true, y_pred), out_dir / "confusion_matrix.json")
    write_jsonl(scored, out_dir / "scores.jsonl")
    print(metrics)


if __name__ == "__main__":
    main()
