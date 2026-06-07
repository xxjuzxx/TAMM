#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

from src.evaluation.metrics import classification_metrics, confusion
from src.utils.io import write_json


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _binary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    benign = y_true == 0
    attack = y_true == 1
    fp = int(np.sum((y_pred == 1) & benign))
    tn = int(np.sum((y_pred == 0) & benign))
    tp = int(np.sum((y_pred == 1) & attack))
    fn = int(np.sum((y_pred == 0) & attack))
    return {
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "attack_recall": float(tp / max(tp + fn, 1)),
        "attack_precision": float(tp / max(tp + fp, 1)),
        "false_positives": fp,
        "true_negatives": tn,
        "true_positives": tp,
        "false_negatives": fn,
    }


def _metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (scores >= float(threshold)).astype(np.int64)
    score_matrix = np.column_stack([1.0 - scores, scores])
    metrics = classification_metrics(y_true.astype(int).tolist(), y_pred.astype(int).tolist(), score_matrix)
    metrics.update(_binary_rates(y_true, y_pred))
    metrics["threshold"] = float(threshold)
    metrics["confusion_matrix"] = confusion(y_true.astype(int).tolist(), y_pred.astype(int).tolist())
    return metrics


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores.astype(np.float64))
    if unique_scores.size == 0:
        return np.array([0.5], dtype=np.float64)
    eps = 1e-12
    thresholds = [float(unique_scores[0] - eps), float(unique_scores[-1] + eps)]
    thresholds.extend(float(value) for value in unique_scores)
    return np.asarray(sorted(set(thresholds)), dtype=np.float64)


def best_metric_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    best_value = -1.0
    for threshold in _candidate_thresholds(scores):
        metrics = _metrics_at_threshold(y_true, scores, float(threshold))
        value = float(metrics.get(metric, -1.0))
        if value > best_value:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_value = value
    assert best_metrics is not None
    return best_threshold, best_metrics


def best_recall_under_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> dict[str, Any]:
    best_metrics: dict[str, Any] | None = None
    best_recall = -1.0
    best_threshold = 0.5
    for threshold in _candidate_thresholds(scores):
        metrics = _metrics_at_threshold(y_true, scores, float(threshold))
        fpr = float(metrics["false_positive_rate"])
        recall = float(metrics["attack_recall"])
        if fpr <= max_fpr and (recall > best_recall or (recall == best_recall and threshold < best_threshold)):
            best_recall = recall
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        best_threshold = float(np.max(scores) + 1e-12)
        best_metrics = _metrics_at_threshold(y_true, scores, best_threshold)
    out = dict(best_metrics)
    out["max_fpr"] = float(max_fpr)
    return out


def summarize_scores(rows: list[dict[str, Any]], optimize_metric: str = "macro_f1") -> dict[str, Any]:
    if not rows:
        raise ValueError("scores file is empty")
    y_true = np.asarray([int(row["binary_label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["anomaly_score"]) for row in rows], dtype=np.float64)
    if sorted(set(y_true.tolist())) != [0, 1]:
        raise ValueError("anomaly threshold sweep requires binary labels with both BENIGN=0 and ATTACK=1")
    best_threshold, best_metrics = best_metric_threshold(y_true, scores, optimize_metric)
    return {
        "num_scores": int(len(rows)),
        "num_benign": int(np.sum(y_true == 0)),
        "num_attack": int(np.sum(y_true == 1)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "optimize_metric": optimize_metric,
        "best_threshold": float(best_threshold),
        f"best_{optimize_metric}": float(best_metrics[optimize_metric]),
        "best_metric_metrics": best_metrics,
        "recall_at_1pct_fpr": best_recall_under_fpr(y_true, scores, 0.01),
        "recall_at_5pct_fpr": best_recall_under_fpr(y_true, scores, 0.05),
        "recall_at_10pct_fpr": best_recall_under_fpr(y_true, scores, 0.10),
        "rank_metrics": _metrics_at_threshold(y_true, scores, float(best_threshold)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--optimize_metric", choices=["macro_f1", "weighted_f1", "accuracy", "attack_recall"], default="macro_f1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = _read_jsonl(args.scores)
    payload = summarize_scores(rows, optimize_metric=args.optimize_metric)
    payload["scores"] = args.scores
    if args.metrics:
        with Path(args.metrics).open("r", encoding="utf-8") as handle:
            original_metrics = json.load(handle)
        payload["source_metrics"] = {
            key: original_metrics.get(key)
            for key in ("baseline", "macro_f1", "weighted_f1", "auroc", "auprc", "fpr95", "threshold")
            if key in original_metrics
        }
    payload["command"] = shlex.join(sys.argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, args.out)
    print(
        {
            "best_threshold": payload["best_threshold"],
            "best_macro_f1": payload["best_metric_metrics"]["macro_f1"],
            "recall_at_1pct_fpr": payload["recall_at_1pct_fpr"]["attack_recall"],
            "recall_at_5pct_fpr": payload["recall_at_5pct_fpr"]["attack_recall"],
        }
    )


if __name__ == "__main__":
    main()
