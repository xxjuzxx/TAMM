#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.utils.io import read_jsonl, write_json


LEAKY_GROUP_KEYS = {"label", "true_label", "app", "app_id", "dataset_file", "capture_file"}


def _load_predictions(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"Prediction file must contain a list of rows: {path}")
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _merge_flow_metadata(predictions: list[dict[str, Any]], metadata_flows: str | Path | None) -> list[dict[str, Any]]:
    if metadata_flows is None:
        return predictions
    flow_rows = {str(row.get("flow_id")): row for row in read_jsonl(metadata_flows)}
    merged: list[dict[str, Any]] = []
    for row in predictions:
        out = dict(row)
        flow = flow_rows.get(str(row.get("flow_id")))
        if flow:
            for key in ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "dataset_file", "start_ts", "service_key"]:
                if out.get(key) is None and flow.get(key) is not None:
                    out[key] = flow[key]
        merged.append(out)
    return merged


def _target_names(rows: list[dict[str, Any]]) -> list[str]:
    scores = rows[0].get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("Prediction rows must contain a non-empty scores mapping.")
    return [str(label) for label in scores.keys()]


def _label_ids(labels: list[str], target_names: list[str]) -> list[int]:
    mapping = {label: idx for idx, label in enumerate(target_names)}
    unknown = sorted({label for label in labels if label not in mapping})
    if unknown:
        raise ValueError(f"Labels missing from score keys: {unknown[:5]}")
    return [mapping[label] for label in labels]


def _score_matrix(rows: list[dict[str, Any]], target_names: list[str]) -> np.ndarray:
    return np.asarray([[float(row["scores"].get(label, 0.0)) for label in target_names] for row in rows], dtype=np.float32)


def _apply_temperature_to_rows(rows: list[dict[str, Any]], temperature: float) -> list[dict[str, Any]]:
    if abs(float(temperature) - 1.0) < 1e-12:
        return rows
    if temperature <= 0:
        raise ValueError("--temperature must be > 0")
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        scores = row.get("scores")
        if not isinstance(scores, dict) or not scores:
            out_rows.append(row)
            continue
        labels = list(scores.keys())
        probs = np.asarray([max(float(scores[label]), 1e-12) for label in labels], dtype=np.float64)
        logits = np.log(probs) / float(temperature)
        logits -= np.max(logits)
        adjusted = np.exp(logits)
        adjusted /= adjusted.sum()
        adjusted_scores = {label: float(adjusted[idx]) for idx, label in enumerate(labels)}
        pred_label = labels[int(np.argmax(adjusted))]
        new_row = dict(row)
        new_row["scores"] = adjusted_scores
        new_row["pred_label"] = pred_label
        new_row["pred_confidence"] = float(np.max(adjusted))
        new_row["temperature"] = float(temperature)
        out_rows.append(new_row)
    return out_rows


def _safe_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if value is None:
        return ("NONE",)
    return (str(value),)


def _group_key(row: dict[str, Any], group_by: str) -> tuple[str, ...]:
    service_key = _safe_tuple(row.get("service_key"))
    if group_by == "service_key":
        return ("service_key", *service_key)
    if group_by == "service_host_proto":
        host = service_key[0] if len(service_key) >= 1 else str(row.get("dst_ip") or row.get("src_ip") or "NONE")
        proto = service_key[2] if len(service_key) >= 3 else str(row.get("protocol") or "NONE")
        return ("service_host_proto", host, proto)
    if group_by == "src_dst_proto":
        return ("src_dst_proto", str(row.get("src_ip") or "NONE"), str(row.get("dst_ip") or "NONE"), str(row.get("protocol") or "NONE"))
    if group_by == "src_dst_pair":
        return ("src_dst_pair", str(row.get("src_ip") or "NONE"), str(row.get("dst_ip") or "NONE"))
    if group_by in {"dataset_file", "capture_file"}:
        return (group_by, str(row.get(group_by) or "NONE"))
    if group_by == "global_time":
        return ("global_time",)
    raise ValueError(f"Unsupported group_by: {group_by}")


def _sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    try:
        ts = float(row.get("start_ts"))
    except (TypeError, ValueError):
        ts = math.inf
    try:
        index = int(row.get("index"))
    except (TypeError, ValueError):
        index = 0
    return ts, index, str(row.get("flow_id") or "")


def _windows_for_group(rows: list[dict[str, Any]], *, window_n: int | None, time_window: float | None) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=_sort_key)
    if time_window is not None and time_window > 0:
        windows: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        start_ts: float | None = None
        for row in ordered:
            ts = _sort_key(row)[0]
            if start_ts is None or not math.isfinite(ts) or ts - start_ts <= time_window:
                if start_ts is None and math.isfinite(ts):
                    start_ts = ts
                current.append(row)
            else:
                windows.append(current)
                current = [row]
                start_ts = ts if math.isfinite(ts) else None
        if current:
            windows.append(current)
        return windows
    size = max(1, int(window_n or 1))
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


def _build_windows(
    rows: list[dict[str, Any]],
    *,
    group_by: str,
    window_n: int | None,
    time_window: float | None,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, group_by)].append(row)
    windows: list[list[dict[str, Any]]] = []
    group_sizes = []
    for group_rows in groups.values():
        group_sizes.append(len(group_rows))
        windows.extend(_windows_for_group(group_rows, window_n=window_n, time_window=time_window))
    singleton_flow_count = sum(len(window) for window in windows if len(window) == 1)
    stats = {
        "num_groups": len(groups),
        "avg_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
        "num_windows": len(windows),
        "avg_flows_per_window": float(np.mean([len(window) for window in windows])) if windows else 0.0,
        "max_flows_per_window": int(max([len(window) for window in windows], default=0)),
        "unaggregated_flow_ratio": float(singleton_flow_count / len(rows)) if rows else 0.0,
    }
    return windows, stats


def _hard_majority_scores(rows: list[dict[str, Any]], scores: np.ndarray, target_names: list[str]) -> np.ndarray:
    pred_counts = Counter(str(row.get("pred_label")) for row in rows)
    confidence_by_label: dict[str, float] = defaultdict(float)
    for row in rows:
        pred = str(row.get("pred_label"))
        confidence_by_label[pred] += float(row.get("pred_confidence") or max(row.get("scores", {}).values()))
    label_order = {label: idx for idx, label in enumerate(target_names)}
    winner = sorted(pred_counts, key=lambda label: (-pred_counts[label], -confidence_by_label[label], label_order.get(label, 10**9)))[0]
    out = np.zeros((len(target_names),), dtype=np.float32)
    out[label_order[winner]] = 1.0
    return out


def _aggregate_scores(rows: list[dict[str, Any]], target_names: list[str], *, method: str, top_k: int) -> np.ndarray:
    scores = _score_matrix(rows, target_names)
    if method == "mean_softmax":
        return scores.mean(axis=0)
    if method == "majority":
        return _hard_majority_scores(rows, scores, target_names)
    if method == "max_confidence":
        confidences = scores.max(axis=1)
        return scores[int(np.argmax(confidences))]
    if method == "topk_confidence":
        confidences = scores.max(axis=1)
        k = min(max(1, int(top_k)), len(rows))
        selected = np.argsort(-confidences)[:k]
        return scores[selected].mean(axis=0)
    raise ValueError(f"Unsupported aggregation method: {method}")


def _score_entropy(scores: np.ndarray) -> float:
    clipped = np.clip(scores.astype(np.float64), 1e-12, 1.0)
    entropy = -float(np.sum(clipped * np.log(clipped)))
    normalizer = math.log(len(clipped)) if len(clipped) > 1 else 1.0
    return float(entropy / normalizer)


def _score_margin(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 1.0
    ordered = np.sort(scores)
    return float(ordered[-1] - ordered[-2])


def _passes_aggregation_gate(
    scores: np.ndarray,
    *,
    min_confidence: float | None,
    max_entropy: float | None,
    min_margin: float | None,
) -> bool:
    if min_confidence is not None and float(np.max(scores)) < min_confidence:
        return False
    if max_entropy is not None and _score_entropy(scores) > max_entropy:
        return False
    if min_margin is not None and _score_margin(scores) < min_margin:
        return False
    return True


def _fallback_scores(rows: list[dict[str, Any]], target_names: list[str], *, fallback: str) -> np.ndarray:
    scores = _score_matrix(rows, target_names)
    if fallback == "top_confidence":
        return scores[int(np.argmax(scores.max(axis=1)))]
    if fallback == "majority":
        return _hard_majority_scores(rows, scores, target_names)
    raise ValueError(f"Unsupported gated fallback score mode: {fallback}")


def _majority_label(labels: list[str], target_names: list[str]) -> tuple[str, float, bool]:
    counts = Counter(labels)
    label_order = {label: idx for idx, label in enumerate(target_names)}
    winner = sorted(counts, key=lambda label: (-counts[label], label_order.get(label, 10**9)))[0]
    purity = counts[winner] / len(labels)
    return winner, float(purity), len(counts) > 1


def _metrics_for_rows(rows: list[dict[str, Any]], target_names: list[str]) -> tuple[dict[str, Any], dict[str, Any], list[list[int]]]:
    y_true = _label_ids([str(row["true_label"]) for row in rows], target_names)
    y_pred = _label_ids([str(row["pred_label"]) for row in rows], target_names)
    y_score = _score_matrix(rows, target_names)
    return classification_metrics(y_true, y_pred, y_score), report_dict(y_true, y_pred, target_names), confusion(y_true, y_pred)


def evaluate_aggregation(
    rows: list[dict[str, Any]],
    *,
    method: str,
    group_by: str,
    window_n: int | None,
    time_window: float | None,
    top_k: int,
    gate_min_confidence: float | None = None,
    gate_max_entropy: float | None = None,
    gate_min_margin: float | None = None,
    gate_fallback: str = "top_confidence",
) -> dict[str, Any]:
    target_names = _target_names(rows)
    flow_metrics, flow_report, flow_confusion = _metrics_for_rows(rows, target_names)
    windows, window_stats = _build_windows(rows, group_by=group_by, window_n=window_n, time_window=time_window)

    aggregated_flow_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    purities: list[float] = []
    mixed_count = 0
    gated_accept_count = 0
    gated_reject_count = 0
    gate_active = any(value is not None for value in [gate_min_confidence, gate_max_entropy, gate_min_margin])
    for window_idx, window in enumerate(windows):
        aggregate_scores = _aggregate_scores(window, target_names, method=method, top_k=top_k)
        aggregate_confidence = float(np.max(aggregate_scores))
        aggregate_entropy = _score_entropy(aggregate_scores)
        aggregate_margin = _score_margin(aggregate_scores)
        gate_accept = True
        if gate_active:
            gate_accept = _passes_aggregation_gate(
                aggregate_scores,
                min_confidence=gate_min_confidence,
                max_entropy=gate_max_entropy,
                min_margin=gate_min_margin,
            )
        gated_accept_count += int(gate_accept)
        gated_reject_count += int(not gate_accept)
        scores = aggregate_scores if gate_accept else _fallback_scores(window, target_names, fallback=gate_fallback)
        pred_id = int(np.argmax(scores))
        pred_label = target_names[pred_id]
        true_label, purity, mixed = _majority_label([str(row["true_label"]) for row in window], target_names)
        purities.append(purity)
        mixed_count += int(mixed)
        score_map = {label: float(scores[idx]) for idx, label in enumerate(target_names)}
        for row in window:
            out = dict(row)
            out["flow_pred_label"] = row.get("pred_label")
            out["flow_pred_confidence"] = row.get("pred_confidence")
            out["pred_label"] = pred_label
            out["pred_confidence"] = float(scores[pred_id])
            out["scores"] = score_map
            out["aggregation_window_id"] = window_idx
            out["aggregation_window_size"] = len(window)
            out["aggregation_window_true_majority"] = true_label
            out["aggregation_window_purity"] = purity
            out["aggregation_gate_active"] = gate_active
            out["aggregation_gate_accept"] = gate_accept
            out["aggregation_confidence"] = aggregate_confidence
            out["aggregation_entropy"] = aggregate_entropy
            out["aggregation_margin"] = aggregate_margin
            aggregated_flow_rows.append(out)
        window_rows.append(
            {
                "window_id": window_idx,
                "window_size": len(window),
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_confidence": float(scores[pred_id]),
                "scores": score_map,
                "purity": purity,
                "mixed_true_labels": mixed,
                "aggregation_gate_active": gate_active,
                "aggregation_gate_accept": gate_accept,
                "aggregation_confidence": aggregate_confidence,
                "aggregation_entropy": aggregate_entropy,
                "aggregation_margin": aggregate_margin,
                "flow_indices": [row.get("index") for row in window],
                "flow_ids": [row.get("flow_id") for row in window],
                "true_label_counts": dict(Counter(str(row["true_label"]) for row in window)),
            }
        )

    agg_metrics, agg_report, agg_confusion = _metrics_for_rows(aggregated_flow_rows, target_names)
    win_metrics, win_report, win_confusion = _metrics_for_rows(window_rows, target_names)
    return {
        "target_names": target_names,
        "method": method,
        "group_by": group_by,
        "window_n": window_n,
        "time_window": time_window,
        "top_k": top_k,
        "flow_level": flow_metrics,
        "aggregated_flow_level": agg_metrics,
        "window_level": win_metrics,
        "window_stats": {
            **window_stats,
            "mixed_window_ratio": float(mixed_count / len(windows)) if windows else 0.0,
            "mean_window_purity": float(np.mean(purities)) if purities else 0.0,
            "gate_active": gate_active,
            "gate_min_confidence": gate_min_confidence,
            "gate_max_entropy": gate_max_entropy,
            "gate_min_margin": gate_min_margin,
            "gate_fallback": gate_fallback,
            "gate_accept_ratio": float(gated_accept_count / len(windows)) if windows else 0.0,
            "gate_reject_ratio": float(gated_reject_count / len(windows)) if windows else 0.0,
        },
        "reports": {
            "flow_level": flow_report,
            "aggregated_flow_level": agg_report,
            "window_level": win_report,
        },
        "confusion_matrices": {
            "flow_level": flow_confusion,
            "aggregated_flow_level": agg_confusion,
            "window_level": win_confusion,
        },
        "aggregated_flow_predictions": aggregated_flow_rows,
        "window_predictions": window_rows,
    }


def _leakage_warnings(rows: list[dict[str, Any]], group_by: str, allow_leaky_group: bool) -> list[str]:
    warnings: list[str] = []
    if group_by in LEAKY_GROUP_KEYS and not allow_leaky_group:
        raise ValueError(
            f"group_by={group_by} is disabled because it can be equivalent to the app label. "
            "Use --allow_leaky_group only for audits, not for reported experiments."
        )
    dataset_values = [str(row.get("dataset_file") or "") for row in rows]
    true_labels = {str(row.get("true_label")) for row in rows}
    if dataset_values and any(label and f"/{label}/" in value or value.endswith(f"/{label}.csv") for value in dataset_values for label in true_labels):
        warnings.append("dataset_file appears to encode the app label in CrossNet; it was not used as the grouping key.")
    if group_by in {"service_key", "service_host_proto", "src_dst_pair", "src_dst_proto"}:
        warnings.append("Grouping uses endpoint/context metadata only at evaluation time; it is not added to Behavior Composer tokens or encoder input.")
    return warnings


def _write_outputs(result: dict[str, Any], out_dir: Path, command: str, warnings: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in result.items() if key not in {"aggregated_flow_predictions", "window_predictions"}}
    serializable["command"] = command
    serializable["warnings"] = warnings
    write_json(serializable, out_dir / "metrics.json")
    write_json(result["reports"]["flow_level"], out_dir / "classification_report_flow_level.json")
    write_json(result["reports"]["aggregated_flow_level"], out_dir / "classification_report_aggregated_flow_level.json")
    write_json(result["reports"]["window_level"], out_dir / "classification_report_window_level.json")
    write_json(result["confusion_matrices"]["flow_level"], out_dir / "confusion_matrix_flow_level.json")
    write_json(result["confusion_matrices"]["aggregated_flow_level"], out_dir / "confusion_matrix_aggregated_flow_level.json")
    write_json(result["confusion_matrices"]["window_level"], out_dir / "confusion_matrix_window_level.json")
    write_json(result["aggregated_flow_predictions"], out_dir / "aggregated_flow_predictions.json")
    write_json(result["window_predictions"], out_dir / "window_predictions.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Flow-level test_predictions.json from classifier evaluation.")
    parser.add_argument("--metadata_flows", default=None, help="Optional flow JSONL used only to fill missing src/dst metadata by flow_id.")
    parser.add_argument(
        "--method",
        choices=["mean_softmax", "majority", "max_confidence", "topk_confidence"],
        default="mean_softmax",
    )
    parser.add_argument(
        "--group_by",
        choices=["service_key", "service_host_proto", "src_dst_proto", "src_dst_pair", "global_time", "dataset_file", "capture_file"],
        default="service_host_proto",
    )
    parser.add_argument("--window_n", type=int, default=5)
    parser.add_argument("--time_window", type=float, default=None, help="Seconds. If set, overrides --window_n within each group.")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0, help="Optional probability temperature before aggregation.")
    parser.add_argument("--gate_min_confidence", type=float, default=None)
    parser.add_argument("--gate_max_entropy", type=float, default=None)
    parser.add_argument("--gate_min_margin", type=float, default=None)
    parser.add_argument(
        "--gate_fallback",
        choices=["top_confidence", "majority"],
        default="top_confidence",
        help="Prediction strategy used when a window fails the aggregation gate.",
    )
    parser.add_argument("--allow_leaky_group", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = _merge_flow_metadata(_load_predictions(args.predictions), args.metadata_flows)
    rows = _apply_temperature_to_rows(rows, args.temperature)
    warnings = _leakage_warnings(rows, args.group_by, args.allow_leaky_group)
    result = evaluate_aggregation(
        rows,
        method=args.method,
        group_by=args.group_by,
        window_n=args.window_n,
        time_window=args.time_window,
        top_k=args.top_k,
        gate_min_confidence=args.gate_min_confidence,
        gate_max_entropy=args.gate_max_entropy,
        gate_min_margin=args.gate_min_margin,
        gate_fallback=args.gate_fallback,
    )
    result["prediction_path"] = args.predictions
    result["metadata_flows"] = args.metadata_flows
    result["temperature"] = float(args.temperature)
    _write_outputs(result, Path(args.out), shlex.join(sys.argv), warnings)
    compact = {
        "flow_level": result["flow_level"],
        "aggregated_flow_level": result["aggregated_flow_level"],
        "window_level": result["window_level"],
        "window_stats": result["window_stats"],
        "warnings": warnings,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
