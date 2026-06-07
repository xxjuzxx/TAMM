#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import classification_metrics
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.training.classifier_trainer import (
    _raw_label_group_ids,
    _split_indices,
    _task_labels,
    _temporal_stratified_by_group_indices,
)
from src.utils.io import read_yaml, write_json


DEFAULT_CONDITIONS = {
    "clean": "outputs/tokens/zeek_purebenign_expanded_dosweb_binary_balanced3000_dedup_pcap_tokens_behavior.pt",
    "packet_delete_010": "outputs/tokens/zeek_dedup_packet_delete_010_tokens_behavior.pt",
    "packet_insert_010": "outputs/tokens/zeek_dedup_packet_insert_010_tokens_behavior.pt",
    "direction_flip_010": "outputs/tokens/zeek_dedup_direction_flip_010_tokens_behavior.pt",
    "length_align_050": "outputs/tokens/zeek_dedup_length_align_050_tokens_behavior.pt",
    "length_padding_050": "outputs/tokens/zeek_dedup_length_padding_050_tokens_behavior.pt",
    "low_rate_c2_070": "outputs/tokens/zeek_dedup_low_rate_c2_070_tokens_behavior.pt",
}

DEFAULT_CALIBRATION_CONDITIONS = ["clean", "packet_insert_010", "length_padding_050", "low_rate_c2_070"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _state_dict_from_checkpoint(checkpoint: str, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint format: {type(payload)!r}")
    return payload


def _safe_ts(meta: dict[str, Any], fallback: int) -> float:
    try:
        return float(meta.get("start_ts"))
    except (TypeError, ValueError):
        return float(fallback)


def _split_by_scope(
    token_data: dict[str, Any],
    labels_np: np.ndarray,
    cfg: dict[str, Any],
    split: str,
) -> dict[str, np.ndarray]:
    train_cfg = cfg.get("training", {})
    order_values = np.array([_safe_ts(meta, idx) for idx, meta in enumerate(token_data.get("meta", []))])
    if split == "temporal_stratified_raw_label":
        train_idx, val_idx, test_idx = _temporal_stratified_by_group_indices(
            labels_np,
            _raw_label_group_ids(token_data),
            val_ratio=float(train_cfg.get("val_ratio", 0.1)),
            test_ratio=float(train_cfg.get("test_ratio", 0.2)),
            order_values=order_values,
        )
    else:
        train_idx, val_idx, test_idx = _split_indices(
            labels_np,
            val_ratio=float(train_cfg.get("val_ratio", 0.1)),
            test_ratio=float(train_cfg.get("test_ratio", 0.2)),
            seed=int(cfg.get("split_seed", cfg.get("seed", 42))),
            split=split,
            order_values=order_values,
        )
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "val_test": np.concatenate([val_idx, test_idx]),
        "all": np.arange(len(labels_np), dtype=np.int64),
    }


def _build_model(
    token_data: dict[str, Any],
    cfg: dict[str, Any],
    checkpoint: str,
    num_classes: int,
    device: torch.device,
) -> BehaviorComposer:
    model_cfg = cfg.get("model", {})
    model = BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=num_classes,
        max_seq_len=int(model_cfg.get("max_seq_len", token_data.get("max_len", 256))),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        intermediate_size=int(model_cfg.get("intermediate_size", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        **resolve_pooling_config(model_cfg),
    ).to(device)
    model.load_state_dict(_state_dict_from_checkpoint(checkpoint, device))
    model.eval()
    return model


def _predict_all(model: BehaviorComposer, token_data: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> np.ndarray:
    indices = torch.arange(len(token_data["input_ids"]), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(
            token_data["input_ids"][indices],
            token_data["attention_mask"][indices],
            token_data["token_type_ids"][indices],
        ),
        batch_size=int(cfg.get("training", {}).get("batch_size", 64)),
        shuffle=False,
    )
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in loader:
            logits = model(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
            scores.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(scores, axis=0)


def _binary_labels(labels_np: np.ndarray, benign_id: int) -> np.ndarray:
    return (labels_np != int(benign_id)).astype(np.int64)


def _binary_metrics(y_true: np.ndarray, alert_scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (alert_scores >= float(threshold)).astype(np.int64)
    score_matrix = np.column_stack([1.0 - alert_scores, alert_scores])
    out = classification_metrics(y_true.tolist(), y_pred.tolist(), score_matrix)
    cm = confusion_matrix(y_true.tolist(), y_pred.tolist(), labels=[0, 1]).tolist()
    tn, fp = cm[0]
    fn, tp = cm[1]
    out.update(
        {
            "threshold": float(threshold),
            "confusion_matrix": cm,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "attack_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
            "num_flows": int(len(y_true)),
            "num_benign": int(tn + fp),
            "num_attack": int(tp + fn),
        }
    )
    return out


def _metric_value(metrics: dict[str, Any], metric: str) -> float:
    if metric == "attack_recall":
        return float(metrics["attack_recall"])
    if metric == "false_positive_rate":
        return -float(metrics["false_positive_rate"])
    return float(metrics[metric])


def _best_threshold(y_true: np.ndarray, alert_scores: np.ndarray, metric: str) -> tuple[float, float]:
    best_threshold = 0.5
    best_value = -1e9
    for threshold in np.linspace(0.001, 0.999, 999):
        metrics = _binary_metrics(y_true, alert_scores, float(threshold))
        value = _metric_value(metrics, metric)
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold, best_value


def _best_pooled_threshold(
    calibration_items: list[tuple[np.ndarray, np.ndarray]],
    metric: str,
    policy: str,
) -> tuple[float, float, float]:
    best_threshold = 0.5
    best_value = -1e9
    best_mean = -1e9
    if policy == "pooled_samples":
        y_true = np.concatenate([item[0] for item in calibration_items])
        alert_scores = np.concatenate([item[1] for item in calibration_items])
        threshold, value = _best_threshold(y_true, alert_scores, metric)
        return threshold, value, value
    if policy != "maximin":
        raise ValueError(f"unsupported pooled policy: {policy}")
    for threshold in np.linspace(0.001, 0.999, 999):
        values = [
            _metric_value(_binary_metrics(y_true, alert_scores, float(threshold)), metric)
            for y_true, alert_scores in calibration_items
        ]
        min_value = float(min(values))
        mean_value = float(np.mean(values))
        if min_value > best_value or (min_value == best_value and mean_value > best_mean):
            best_threshold = float(threshold)
            best_value = min_value
            best_mean = mean_value
    return best_threshold, best_value, best_mean


def _attack_label(meta: dict[str, Any]) -> str:
    return str(meta.get("label") or "ATTACK")


def _sliding_false_positive_rates(rows: list[dict[str, Any]], windows: list[float]) -> dict[str, float]:
    benign_rows = [row for row in rows if int(row["binary_label_id"]) == 0]
    if not benign_rows:
        return {str(window): 0.0 for window in windows}
    out: dict[str, float] = {}
    for window in windows:
        max_rate = 0.0
        left = 0
        false_count = 0
        for right, row in enumerate(benign_rows):
            if row["is_false_positive"]:
                false_count += 1
            while row["start_ts"] - benign_rows[left]["start_ts"] > float(window):
                if benign_rows[left]["is_false_positive"]:
                    false_count -= 1
                left += 1
            width = max(1, right - left + 1)
            max_rate = max(max_rate, false_count / width)
        out[str(window)] = float(max_rate)
    return out


def _first_consecutive_alert(items: list[dict[str, Any]], consecutive_alerts: int) -> dict[str, Any] | None:
    needed = max(1, int(consecutive_alerts))
    run: list[dict[str, Any]] = []
    for row in items:
        if int(row["prediction"]) == 1:
            run.append(row)
            if len(run) >= needed:
                return run[0]
        else:
            run = []
    return None


def _online_replay_summary(
    token_data: dict[str, Any],
    indices: np.ndarray,
    y_true_binary: np.ndarray,
    alert_scores: np.ndarray,
    threshold: float,
    consecutive_alerts: int,
    fpr_windows: list[float],
) -> dict[str, Any]:
    meta_rows = token_data.get("meta", [])
    ordered_indices = sorted(
        [int(idx) for idx in indices.tolist()],
        key=lambda idx: (_safe_ts(meta_rows[idx] if idx < len(meta_rows) else {}, idx), idx),
    )
    rows: list[dict[str, Any]] = []
    for stream_pos, idx in enumerate(ordered_indices):
        meta = meta_rows[idx] if idx < len(meta_rows) else {}
        label_id = int(y_true_binary[idx])
        score = float(alert_scores[idx])
        prediction = int(score >= float(threshold))
        rows.append(
            {
                "stream_pos": int(stream_pos),
                "index": idx,
                "flow_id": meta.get("flow_id"),
                "label": _attack_label(meta),
                "binary_label_id": label_id,
                "attack_probability": score,
                "prediction": prediction,
                "is_false_positive": bool(label_id == 0 and prediction == 1),
                "is_true_positive": bool(label_id == 1 and prediction == 1),
                "start_ts": _safe_ts(meta, stream_pos),
            }
        )

    if not rows:
        return {"summary": {"num_flows": 0}, "per_label": []}

    metrics = _binary_metrics(
        np.array([row["binary_label_id"] for row in rows], dtype=np.int64),
        np.array([row["attack_probability"] for row in rows], dtype=np.float64),
        threshold,
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["binary_label_id"]) == 1:
            groups[row["label"]].append(row)

    per_label: list[dict[str, Any]] = []
    delays_sec: list[float] = []
    delays_flows: list[int] = []
    for label, items in sorted(groups.items()):
        first = items[0]
        alert = _first_consecutive_alert(items, consecutive_alerts=consecutive_alerts)
        if alert is None:
            per_label.append(
                {
                    "label": label,
                    "attack_flows": len(items),
                    "first_attack_ts": float(first["start_ts"]),
                    "first_alert_ts": None,
                    "detected": False,
                    "delay_seconds": None,
                    "delay_flows": None,
                }
            )
            continue
        delay_sec = float(max(0.0, alert["start_ts"] - first["start_ts"]))
        delay_flows = int(max(0, alert["stream_pos"] - first["stream_pos"]))
        delays_sec.append(delay_sec)
        delays_flows.append(delay_flows)
        per_label.append(
            {
                "label": label,
                "attack_flows": len(items),
                "first_attack_ts": float(first["start_ts"]),
                "first_alert_ts": float(alert["start_ts"]),
                "detected": True,
                "delay_seconds": delay_sec,
                "delay_flows": delay_flows,
            }
        )

    fp_rows = [row for row in rows if row["is_false_positive"]]
    first_ts = float(rows[0]["start_ts"])
    metrics.update(
        {
            "threshold": float(threshold),
            "consecutive_alerts": int(max(1, consecutive_alerts)),
            "attack_recall_online": float(metrics["attack_recall"]),
            "time_to_first_false_positive_seconds": (
                float(max(0.0, fp_rows[0]["start_ts"] - first_ts)) if fp_rows else None
            ),
            "max_sliding_false_positive_rate": _sliding_false_positive_rates(rows, fpr_windows),
            "attack_labels": len(per_label),
            "detected_labels": int(sum(1 for row in per_label if row["detected"])),
            "missed_labels": int(sum(1 for row in per_label if not row["detected"])),
            "mean_delay_seconds": float(statistics.fmean(delays_sec)) if delays_sec else None,
            "median_delay_seconds": float(statistics.median(delays_sec)) if delays_sec else None,
            "mean_delay_flows": float(statistics.fmean(delays_flows)) if delays_flows else None,
            "median_delay_flows": float(statistics.median(delays_flows)) if delays_flows else None,
        }
    )
    return {"summary": metrics, "per_label": per_label}


def _parse_conditions(items: list[str] | None) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_CONDITIONS)
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--condition must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name] = path
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Zeek-first multiclass alert threshold and online replay",
        "",
        "## Calibration",
        "",
        f"- Threshold: `{payload['calibration']['threshold']:.4f}`",
        f"- Metric: `{payload['calibration']['metric']}`",
        f"- Policy: `{payload['calibration']['policy']}`",
        f"- Conditions: `{', '.join(payload['calibration']['conditions'])}`",
        "",
        "## Threshold Sweep",
        "",
        "| Condition | Own threshold | Own val Macro-F1 | Calibrated val Macro-F1 | Calibrated test Macro-F1 | Test FPR | Test attack recall | Test AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["threshold_rows"]:
        lines.append(
            "| {condition} | {own_threshold} | {own_val_macro} | {cal_val_macro} | {test_macro} | {test_fpr} | {test_recall} | {test_auroc} |".format(
                condition=row["condition"],
                own_threshold=_fmt(row["own_threshold"]),
                own_val_macro=_fmt(row["own_val_metrics"]["macro_f1"]),
                cal_val_macro=_fmt(row["calibrated_val_metrics"]["macro_f1"]),
                test_macro=_fmt(row["calibrated_test_metrics"]["macro_f1"]),
                test_fpr=_fmt(row["calibrated_test_metrics"]["false_positive_rate"]),
                test_recall=_fmt(row["calibrated_test_metrics"]["attack_recall"]),
                test_auroc=_fmt(row["calibrated_test_metrics"].get("auroc")),
            )
        )
    lines.extend(
        [
            "",
            "## Online Replay",
            "",
            "| Condition | Scope | Flows | Macro-F1 | FPR | Attack recall | FP | TP | Detected labels | Median delay flows | Median delay seconds |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["replay_rows"]:
        summary = row["summary"]
        lines.append(
            "| {condition} | {scope} | {flows} | {macro} | {fpr} | {recall} | {fp} | {tp} | {detected}/{labels} | {delay_flows} | {delay_sec} |".format(
                condition=row["condition"],
                scope=payload["replay_scope"],
                flows=summary["num_flows"],
                macro=_fmt(summary["macro_f1"]),
                fpr=_fmt(summary["false_positive_rate"]),
                recall=_fmt(summary["attack_recall_online"]),
                fp=summary["false_positives"],
                tp=summary["true_positives"],
                detected=summary["detected_labels"],
                labels=summary["attack_labels"],
                delay_flows=_fmt(summary["median_delay_flows"]),
                delay_sec=_fmt(summary["median_delay_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "说明：多类 checkpoint 的告警分数定义为 `1 - P(BENIGN)`；阈值只用验证集校准，在线 replay 按 flow `start_ts` 排序后评估。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Zeek-first multiclass checkpoint as binary alert scorer.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_tokens", default=DEFAULT_CONDITIONS["clean"])
    parser.add_argument("--condition", action="append", default=None, help="Condition override as NAME=PATH; repeatable.")
    parser.add_argument("--config", default="configs/model_behavior_composer_weighted.yaml")
    parser.add_argument("--task", choices=["multiclass", "multiclass_merged"], default="multiclass_merged")
    parser.add_argument(
        "--split",
        choices=["stratified", "chronological", "temporal_stratified", "temporal_stratified_raw_label"],
        default="temporal_stratified_raw_label",
    )
    parser.add_argument("--calibration_conditions", nargs="*", default=DEFAULT_CALIBRATION_CONDITIONS)
    parser.add_argument("--metric", choices=["macro_f1", "weighted_f1", "accuracy", "attack_recall", "false_positive_rate"], default="macro_f1")
    parser.add_argument("--policy", choices=["pooled_samples", "maximin"], default="maximin")
    parser.add_argument("--replay_scope", choices=["train", "val", "test", "val_test", "all"], default="all")
    parser.add_argument("--consecutive_alerts", type=int, default=1)
    parser.add_argument("--fpr_windows", nargs="*", type=float, default=[60.0, 300.0, 3600.0])
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    conditions = _parse_conditions(args.condition)
    unknown = sorted(set(args.calibration_conditions) - set(conditions))
    if unknown:
        raise ValueError(f"unknown calibration conditions: {unknown}")

    split_token_data = torch.load(args.split_tokens, map_location="cpu", weights_only=False)
    labels, label_to_id = _task_labels(split_token_data, args.task)
    if label_to_id is None or "BENIGN" not in label_to_id:
        raise ValueError(f"{args.task} requires a BENIGN label mapping")
    labels_np = labels.numpy()
    benign_id = int(label_to_id["BENIGN"])
    y_binary = _binary_labels(labels_np, benign_id)
    split_indices = _split_by_scope(split_token_data, labels_np, cfg, args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(
        split_token_data,
        cfg,
        checkpoint=args.checkpoint,
        num_classes=int(labels.max().item()) + 1,
        device=device,
    )

    scored: dict[str, dict[str, Any]] = {}
    for condition, token_path in conditions.items():
        token_data = torch.load(token_path, map_location="cpu", weights_only=False)
        if len(token_data["input_ids"]) != len(split_token_data["input_ids"]):
            raise ValueError(f"{condition} row count differs from split tokens")
        if token_data.get("vocab") != split_token_data.get("vocab"):
            raise ValueError(f"{condition} vocab differs from split tokens")
        probs = _predict_all(model, token_data, cfg, device)
        alert_scores = 1.0 - probs[:, benign_id]
        scored[condition] = {
            "path": token_path,
            "token_data": token_data,
            "alert_scores": alert_scores,
        }

    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    calibration_items = [
        (y_binary[val_idx], scored[name]["alert_scores"][val_idx]) for name in args.calibration_conditions
    ]
    calibrated_threshold, calibrated_value, calibrated_mean = _best_pooled_threshold(
        calibration_items,
        metric=args.metric,
        policy=args.policy,
    )

    threshold_rows: list[dict[str, Any]] = []
    for condition, item in scored.items():
        scores = item["alert_scores"]
        own_threshold, own_value = _best_threshold(y_binary[val_idx], scores[val_idx], metric=args.metric)
        threshold_rows.append(
            {
                "condition": condition,
                "path": item["path"],
                "own_threshold": own_threshold,
                f"own_val_{args.metric}": own_value,
                "own_val_metrics": _binary_metrics(y_binary[val_idx], scores[val_idx], own_threshold),
                "own_test_metrics": _binary_metrics(y_binary[test_idx], scores[test_idx], own_threshold),
                "calibrated_val_metrics": _binary_metrics(y_binary[val_idx], scores[val_idx], calibrated_threshold),
                "calibrated_test_metrics": _binary_metrics(y_binary[test_idx], scores[test_idx], calibrated_threshold),
            }
        )

    replay_idx = split_indices[args.replay_scope]
    replay_rows: list[dict[str, Any]] = []
    for condition, item in scored.items():
        replay = _online_replay_summary(
            item["token_data"],
            replay_idx,
            y_binary,
            item["alert_scores"],
            threshold=calibrated_threshold,
            consecutive_alerts=args.consecutive_alerts,
            fpr_windows=args.fpr_windows,
        )
        replay_rows.append({"condition": condition, "path": item["path"], **replay})

    payload = {
        "protocol": {
            "base": "Zeek-first dedup balanced PCAP subset",
            "task": args.task,
            "split": args.split,
            "split_source": args.split_tokens,
            "alert_score": "1 - P(BENIGN)",
            "checkpoint": args.checkpoint,
        },
        "calibration": {
            "threshold": calibrated_threshold,
            f"pooled_val_{args.metric}": calibrated_value,
            f"pooled_val_{args.metric}_mean": calibrated_mean,
            "metric": args.metric,
            "policy": args.policy,
            "conditions": args.calibration_conditions,
        },
        "label_to_id": label_to_id,
        "benign_id": benign_id,
        "conditions": conditions,
        "threshold_rows": threshold_rows,
        "replay_scope": args.replay_scope,
        "replay_rows": replay_rows,
    }
    write_json(payload, args.out_json)
    Path(args.out_md).write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))


if __name__ == "__main__":
    main()
