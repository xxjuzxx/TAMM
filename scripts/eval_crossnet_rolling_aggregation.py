#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from . import _bootstrap  # type: ignore  # noqa: F401
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


def _merge_flow_metadata(predictions: list[dict[str, Any]], metadata_flows: str | Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if metadata_flows is None:
        return predictions, {"metadata_flows": None, "matched_metadata_rows": 0, "missing_metadata_rows": 0}
    flow_rows = {str(row.get("flow_id")): row for row in read_jsonl(metadata_flows)}
    matched = 0
    missing = 0
    merged: list[dict[str, Any]] = []
    fill_keys = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "dataset_file", "start_ts", "service_key"]
    for row in predictions:
        out = dict(row)
        flow = flow_rows.get(str(row.get("flow_id")))
        if flow:
            matched += 1
            for key in fill_keys:
                if out.get(key) is None and flow.get(key) is not None:
                    out[key] = flow[key]
        else:
            missing += 1
        merged.append(out)
    return merged, {
        "metadata_flows": str(metadata_flows),
        "metadata_flow_rows": int(len(flow_rows)),
        "matched_metadata_rows": int(matched),
        "missing_metadata_rows": int(missing),
        "metadata_match_rate": float(matched / len(predictions)) if predictions else 0.0,
    }


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
    adjusted_rows: list[dict[str, Any]] = []
    for row in rows:
        scores = row.get("scores")
        if not isinstance(scores, dict) or not scores:
            adjusted_rows.append(row)
            continue
        labels = list(scores.keys())
        probs = np.asarray([max(float(scores[label]), 1e-12) for label in labels], dtype=np.float64)
        logits = np.log(probs) / float(temperature)
        logits -= np.max(logits)
        adjusted = np.exp(logits)
        adjusted /= adjusted.sum()
        new_row = dict(row)
        new_row["scores"] = {label: float(adjusted[idx]) for idx, label in enumerate(labels)}
        new_row["pred_label"] = labels[int(np.argmax(adjusted))]
        new_row["pred_confidence"] = float(np.max(adjusted))
        new_row["temperature"] = float(temperature)
        adjusted_rows.append(new_row)
    return adjusted_rows


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


def _score_entropy(scores: np.ndarray) -> float:
    clipped = np.clip(scores.astype(np.float64), 1e-12, 1.0)
    entropy = -float(np.sum(clipped * np.log(clipped)))
    normalizer = math.log(len(clipped)) if len(clipped) > 1 else 1.0
    return float(entropy / normalizer)


def _neighbors(rows: list[dict[str, Any]], position: int, *, mode: str, window_n: int) -> list[dict[str, Any]]:
    size = max(1, int(window_n))
    if mode == "causal":
        start = max(0, position - size + 1)
        return rows[start : position + 1]
    if mode == "forward":
        return rows[position : position + size]
    if mode == "centered":
        left = (size - 1) // 2
        right = size - left
        start = max(0, position - left)
        end = min(len(rows), position + right)
        return rows[start:end]
    raise ValueError(f"Unsupported rolling mode: {mode}")


def _aggregate_scores(rows: list[dict[str, Any]], target_names: list[str], *, method: str, top_k: int) -> np.ndarray:
    scores = _score_matrix(rows, target_names)
    if method == "mean_softmax":
        return scores.mean(axis=0)
    confidences = scores.max(axis=1)
    if method == "max_confidence":
        return scores[int(np.argmax(confidences))]
    if method == "topk_confidence":
        k = min(max(1, int(top_k)), len(rows))
        selected = np.argsort(-confidences)[:k]
        return scores[selected].mean(axis=0)
    if method == "confidence_weighted":
        weights = confidences + 1e-6
        return np.average(scores, axis=0, weights=weights)
    if method == "entropy_weighted":
        weights = np.asarray([1.0 - _score_entropy(row_scores) for row_scores in scores], dtype=np.float32) + 1e-6
        return np.average(scores, axis=0, weights=weights)
    raise ValueError(f"Unsupported rolling aggregation method: {method}")


def _metrics_for_rows(rows: list[dict[str, Any]], target_names: list[str]) -> tuple[dict[str, Any], dict[str, Any], list[list[int]]]:
    y_true = _label_ids([str(row["true_label"]) for row in rows], target_names)
    y_pred = _label_ids([str(row["pred_label"]) for row in rows], target_names)
    y_score = _score_matrix(rows, target_names)
    return classification_metrics(y_true, y_pred, y_score), report_dict(y_true, y_pred, target_names), confusion(y_true, y_pred)


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


def evaluate_rolling(
    rows: list[dict[str, Any]],
    *,
    method: str,
    group_by: str,
    window_n: int,
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    target_names = _target_names(rows)
    flow_metrics, flow_report, flow_confusion = _metrics_for_rows(rows, target_names)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, group_by)].append(row)

    smoothed_rows_by_index: dict[int, dict[str, Any]] = {}
    group_sizes: list[int] = []
    neighbor_sizes: list[int] = []
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=_sort_key)
        group_sizes.append(len(ordered))
        for position, row in enumerate(ordered):
            window = _neighbors(ordered, position, mode=mode, window_n=window_n)
            neighbor_sizes.append(len(window))
            scores = _aggregate_scores(window, target_names, method=method, top_k=top_k)
            pred_id = int(np.argmax(scores))
            out = dict(row)
            out["flow_pred_label"] = row.get("pred_label")
            out["flow_pred_confidence"] = row.get("pred_confidence")
            out["pred_label"] = target_names[pred_id]
            out["pred_confidence"] = float(scores[pred_id])
            out["scores"] = {label: float(scores[idx]) for idx, label in enumerate(target_names)}
            out["rolling_group_size"] = len(ordered)
            out["rolling_neighbor_size"] = len(window)
            out["rolling_mode"] = mode
            out["rolling_window_n"] = int(window_n)
            smoothed_rows_by_index[int(row.get("index", len(smoothed_rows_by_index)))] = out

    smoothed_rows = [smoothed_rows_by_index[key] for key in sorted(smoothed_rows_by_index)]
    rolling_metrics, rolling_report, rolling_confusion = _metrics_for_rows(smoothed_rows, target_names)
    return {
        "target_names": target_names,
        "method": method,
        "group_by": group_by,
        "window_n": int(window_n),
        "mode": mode,
        "top_k": int(top_k),
        "flow_level": flow_metrics,
        "rolling_flow_level": rolling_metrics,
        "rolling_stats": {
            "num_groups": len(groups),
            "avg_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
            "max_group_size": int(max(group_sizes)) if group_sizes else 0,
            "avg_neighbors": float(np.mean(neighbor_sizes)) if neighbor_sizes else 0.0,
            "max_neighbors": int(max(neighbor_sizes)) if neighbor_sizes else 0,
            "singleton_flow_ratio": float(sum(1 for item in neighbor_sizes if item == 1) / len(rows)) if rows else 0.0,
        },
        "reports": {
            "flow_level": flow_report,
            "rolling_flow_level": rolling_report,
        },
        "confusion_matrices": {
            "flow_level": flow_confusion,
            "rolling_flow_level": rolling_confusion,
        },
        "rolling_predictions": smoothed_rows,
    }


def _write_outputs(result: dict[str, Any], out_dir: Path, command: str, warnings: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in result.items() if key != "rolling_predictions"}
    serializable["command"] = command
    serializable["warnings"] = warnings
    write_json(serializable, out_dir / "metrics.json")
    write_json(result["reports"]["flow_level"], out_dir / "classification_report_flow_level.json")
    write_json(result["reports"]["rolling_flow_level"], out_dir / "classification_report_rolling_flow_level.json")
    write_json(result["confusion_matrices"]["flow_level"], out_dir / "confusion_matrix_flow_level.json")
    write_json(result["confusion_matrices"]["rolling_flow_level"], out_dir / "confusion_matrix_rolling_flow_level.json")
    write_json(result["rolling_predictions"], out_dir / "rolling_predictions.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Flow-level prediction JSON from classifier evaluation.")
    parser.add_argument("--metadata_flows", default=None, help="Optional flow JSONL used only to fill missing src/dst metadata by flow_id.")
    parser.add_argument(
        "--method",
        choices=["mean_softmax", "confidence_weighted", "entropy_weighted", "max_confidence", "topk_confidence"],
        default="mean_softmax",
    )
    parser.add_argument(
        "--group_by",
        choices=["service_key", "service_host_proto", "src_dst_proto", "src_dst_pair", "global_time", "dataset_file", "capture_file"],
        default="service_host_proto",
    )
    parser.add_argument("--window_n", type=int, default=5)
    parser.add_argument("--mode", choices=["causal", "centered", "forward"], default="causal")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--allow_leaky_group", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    loaded_rows, metadata_summary = _merge_flow_metadata(_load_predictions(args.predictions), args.metadata_flows)
    rows = _apply_temperature_to_rows(loaded_rows, args.temperature)
    warnings = _leakage_warnings(rows, args.group_by, args.allow_leaky_group)
    result = evaluate_rolling(
        rows,
        method=args.method,
        group_by=args.group_by,
        window_n=args.window_n,
        mode=args.mode,
        top_k=args.top_k,
    )
    result["prediction_path"] = args.predictions
    result["metadata_flows"] = args.metadata_flows
    result["metadata_summary"] = metadata_summary
    result["temperature"] = float(args.temperature)
    _write_outputs(result, Path(args.out), shlex.join(sys.argv), warnings)
    compact = {
        "flow_level": result["flow_level"],
        "rolling_flow_level": result["rolling_flow_level"],
        "rolling_stats": result["rolling_stats"],
        "metadata_summary": metadata_summary,
        "warnings": warnings,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
