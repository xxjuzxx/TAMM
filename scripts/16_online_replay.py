#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import classification_metrics, confusion
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.training.classifier_trainer import _split_indices
from src.utils.io import read_yaml, write_json, write_jsonl


def _safe_ts(meta: dict[str, Any], fallback: int) -> float:
    try:
        return float(meta.get("start_ts"))
    except (TypeError, ValueError):
        return float(fallback)


def _binary_label(meta: dict[str, Any], label_id: int) -> str:
    value = meta.get("binary_label")
    if value is not None:
        return str(value)
    return "ATTACK" if int(label_id) == 1 else "BENIGN"


def _attack_label(meta: dict[str, Any]) -> str:
    return str(meta.get("label") or "ATTACK")


def _service_key(meta: dict[str, Any]) -> tuple[str, ...] | None:
    key = meta.get("service_key")
    if isinstance(key, (list, tuple)):
        return tuple(str(item) for item in key)
    if key is not None:
        return (str(key),)
    return None


def _service_port_protocol(service_key: tuple[str, ...] | None) -> tuple[str | None, str | None]:
    if not service_key:
        return None, None
    protocol = service_key[-1].upper() if service_key[-1].upper() in {"TCP", "UDP", "ICMP"} else None
    if protocol is not None and len(service_key) >= 2:
        return service_key[-2], protocol
    if len(service_key) >= 2:
        return service_key[1], None
    return None, None


def _stateful_service_allowed(
    service_key: tuple[str, ...] | None,
    allowed_protocols: set[str] | None,
    allowed_ports: set[str] | None,
    excluded_ports: set[str],
) -> bool:
    if service_key is None:
        return False
    port, protocol = _service_port_protocol(service_key)
    if allowed_protocols is not None and (protocol is None or protocol.upper() not in allowed_protocols):
        return False
    if allowed_ports is not None and (port is None or str(port) not in allowed_ports):
        return False
    if port is not None and str(port) in excluded_ports:
        return False
    return True


def _service_guard_configured(
    allowed_protocols: set[str] | None,
    allowed_ports: set[str] | None,
    excluded_ports: set[str],
) -> bool:
    return allowed_protocols is not None or allowed_ports is not None or bool(excluded_ports)


def _packet_count(meta: dict[str, Any]) -> int:
    value = meta.get("packet_count")
    if value is None:
        value = len(meta.get("lens") or [])
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _merged_metadata_rows(token_data: dict[str, Any], metadata_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    primary = list(token_data.get("meta", []))
    if metadata_data is None:
        return primary
    supplemental = list(metadata_data.get("meta", []))
    if len(supplemental) < len(primary):
        raise ValueError(
            f"--metadata_tokens has fewer metadata rows ({len(supplemental)}) than --tokens ({len(primary)})"
        )
    merged: list[dict[str, Any]] = []
    for idx, base_row in enumerate(primary):
        base = dict(base_row)
        extra = supplemental[idx]
        base_flow = base.get("flow_id")
        extra_flow = extra.get("flow_id")
        if base_flow is not None and extra_flow is not None and str(base_flow) != str(extra_flow):
            raise ValueError(
                "--metadata_tokens row order does not match --tokens: "
                f"index={idx}, tokens flow_id={base_flow}, metadata flow_id={extra_flow}"
            )
        for key in ("service_key", "service_context"):
            if base.get(key) is None and extra.get(key) is not None:
                base[key] = extra[key]
        merged.append(base)
    return merged


def _split_eval_indices(token_data: dict[str, Any], cfg: dict[str, Any], split: str, scope: str) -> np.ndarray:
    labels = token_data["binary_labels"].numpy()
    order_values = np.array([_safe_ts(meta, idx) for idx, meta in enumerate(token_data.get("meta", []))])
    train_idx, val_idx, test_idx = _split_indices(
        labels,
        val_ratio=float(cfg.get("training", {}).get("val_ratio", 0.1)),
        test_ratio=float(cfg.get("training", {}).get("test_ratio", 0.2)),
        seed=int(cfg.get("split_seed", cfg.get("seed", 42))),
        split=split,
        order_values=order_values,
    )
    if scope == "train":
        return train_idx
    if scope == "val":
        return val_idx
    if scope == "test":
        return test_idx
    if scope == "val_test":
        return np.concatenate([val_idx, test_idx])
    if scope == "all":
        return np.arange(len(labels), dtype=np.int64)
    raise ValueError(f"unsupported scope: {scope}")


def _build_model(token_data: dict[str, Any], cfg: dict[str, Any], checkpoint: str, device: torch.device) -> BehaviorComposer:
    model_cfg = cfg.get("model", {})
    model = BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=2,
        max_seq_len=int(model_cfg.get("max_seq_len", token_data.get("max_len", 256))),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        intermediate_size=int(model_cfg.get("intermediate_size", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        **resolve_pooling_config(model_cfg),
    ).to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict_scores(
    token_data: dict[str, Any],
    indices: np.ndarray,
    cfg: dict[str, Any],
    checkpoint: str,
) -> tuple[np.ndarray, np.ndarray]:
    idx = torch.tensor(indices, dtype=torch.long)
    train_cfg = cfg.get("training", {})
    loader = DataLoader(
        TensorDataset(
            token_data["input_ids"][idx],
            token_data["attention_mask"][idx],
            token_data["token_type_ids"][idx],
        ),
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(token_data, cfg, checkpoint, device)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in loader:
            logits = model(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
            scores.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(scores, axis=0), np.array(indices, dtype=np.int64)


def _best_threshold(y_true: list[int], attack_scores: list[float], metric: str) -> tuple[float, float]:
    if not y_true:
        return 0.5, 0.0
    scores = np.array(attack_scores, dtype=np.float64)
    best_threshold = 0.5
    best_value = -1.0
    for threshold in np.linspace(0.001, 0.999, 999):
        pred = (scores >= threshold).astype(int).tolist()
        value = float(classification_metrics(y_true, pred)[metric])
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold, best_value


def _label_counts(labels: list[int]) -> dict[str, int]:
    benign = int(sum(1 for label in labels if int(label) == 0))
    attack = int(sum(1 for label in labels if int(label) == 1))
    return {"benign": benign, "attack": attack, "num_classes": int((benign > 0) + (attack > 0))}


def _calibrate_threshold(
    labels: np.ndarray,
    items: list[tuple[int, list[float]]],
    metric: str,
    min_classes: int,
) -> tuple[float, float, dict[str, int]]:
    y_true = [int(labels[int(idx)]) for idx, _ in items]
    attack_scores = [float(prob[1]) for _, prob in items]
    counts = _label_counts(y_true)
    if counts["num_classes"] < int(min_classes):
        raise ValueError(
            "calibration data has "
            f"{counts['num_classes']} class(es), but --min_calibration_classes requires {int(min_classes)}; "
            f"counts={counts}"
        )
    threshold, value = _best_threshold(y_true, attack_scores, metric)
    return threshold, value, counts


def _time_to_false_positive(rows: list[dict[str, Any]]) -> float | None:
    first_ts = rows[0]["start_ts"] if rows else None
    if first_ts is None:
        return None
    for row in rows:
        if row["is_false_positive"]:
            return float(max(0.0, row["start_ts"] - first_ts))
    return None


def _sliding_false_positive_rates(rows: list[dict[str, Any]], windows: list[float]) -> dict[str, float]:
    benign_rows = [row for row in rows if row["binary_label_id"] == 0]
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
            while row["start_ts"] - benign_rows[left]["start_ts"] > window:
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
        if row["prediction"] == 1:
            run.append(row)
            if len(run) >= needed:
                return run[0]
        else:
            run = []
    return None


def summarize_replay(rows: list[dict[str, Any]], consecutive_alerts: int, fpr_windows: list[float]) -> dict[str, Any]:
    y_true = [int(row["binary_label_id"]) for row in rows]
    y_pred = [int(row["prediction"]) for row in rows]
    scores = np.array([[1.0 - row["attack_probability"], row["attack_probability"]] for row in rows], dtype=np.float64)
    metrics = classification_metrics(y_true, y_pred, scores) if rows else {}
    fp = sum(1 for row in rows if row["is_false_positive"])
    benign = sum(1 for row in rows if row["binary_label_id"] == 0)
    tp = sum(1 for row in rows if row["is_true_positive"])
    attacks = sum(1 for row in rows if row["binary_label_id"] == 1)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["binary_label_id"] == 1:
            groups[row["label"]].append(row)

    per_label: list[dict[str, Any]] = []
    delays_sec: list[float] = []
    delays_flows: list[int] = []
    for label, items in sorted(groups.items()):
        first = items[0]
        alert = _first_consecutive_alert(items, consecutive_alerts=consecutive_alerts)
        if alert is None:
            detected = False
            delay_sec = None
            delay_flows = None
            first_alert_ts = None
        else:
            detected = True
            delay_sec = float(max(0.0, alert["start_ts"] - first["start_ts"]))
            delay_flows = int(max(0, alert["stream_pos"] - first["stream_pos"]))
            first_alert_ts = float(alert["start_ts"])
            delays_sec.append(delay_sec)
            delays_flows.append(delay_flows)
        per_label.append(
            {
                "label": label,
                "attack_flows": len(items),
                "first_attack_ts": float(first["start_ts"]),
                "first_alert_ts": first_alert_ts,
                "detected": detected,
                "delay_seconds": delay_sec,
                "delay_flows": delay_flows,
            }
        )

    summary = {
        **metrics,
        "num_flows": len(rows),
        "num_benign": int(benign),
        "num_attack": int(attacks),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "false_positive_rate": float(fp / benign) if benign else 0.0,
        "attack_recall_online": float(tp / attacks) if attacks else 0.0,
        "time_to_first_false_positive_seconds": _time_to_false_positive(rows),
        "max_sliding_false_positive_rate": _sliding_false_positive_rates(rows, fpr_windows),
        "attack_labels": len(per_label),
        "detected_labels": int(sum(1 for row in per_label if row["detected"])),
        "missed_labels": int(sum(1 for row in per_label if not row["detected"])),
        "mean_delay_seconds": float(np.mean(delays_sec)) if delays_sec else None,
        "median_delay_seconds": float(np.median(delays_sec)) if delays_sec else None,
        "mean_delay_flows": float(np.mean(delays_flows)) if delays_flows else None,
        "median_delay_flows": float(np.median(delays_flows)) if delays_flows else None,
        "consecutive_alerts": int(max(1, consecutive_alerts)),
    }
    if rows:
        summary["confusion_matrix"] = confusion(y_true, y_pred)
    return {"summary": summary, "per_label": per_label}


def _rows_with_prediction(rows: list[dict[str, Any]], prediction_field: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        prediction = int(item.get(prediction_field, item["prediction"]))
        label_id = int(item["binary_label_id"])
        item["prediction"] = prediction
        item["is_false_positive"] = bool(label_id == 0 and prediction == 1)
        item["is_true_positive"] = bool(label_id == 1 and prediction == 1)
        out.append(item)
    return out


def _stateful_alert_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "model_alerts": int(sum(int(row.get("model_prediction", row["prediction"])) for row in rows)),
        "stateful_alerts": int(sum(int(row.get("stateful_prediction", 0)) for row in rows)),
        "merged_alerts": int(sum(int(row["prediction"]) for row in rows)),
        "stateful_missing_key_flows": int(sum(1 for row in rows if row.get("stateful_missing_key"))),
        "stateful_guard_blocked_flows": int(sum(1 for row in rows if row.get("stateful_guard_allowed") is False)),
        "stateful_updated_flows": int(sum(1 for row in rows if row.get("stateful_missing_key") is False)),
        "stateful_only_alerts": int(
            sum(1 for row in rows if int(row.get("stateful_prediction", 0)) == 1 and int(row.get("model_prediction", 0)) == 0)
        ),
        "model_only_alerts": int(
            sum(1 for row in rows if int(row.get("model_prediction", 0)) == 1 and int(row.get("stateful_prediction", 0)) == 0)
        ),
    }


def _alert_guard_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "alert_guard_blocked_predictions": int(sum(1 for row in rows if row.get("alert_guard_blocked_prediction"))),
        "alert_guard_blocked_false_positives": int(
            sum(1 for row in rows if row.get("alert_guard_blocked_prediction") and int(row["binary_label_id"]) == 0)
        ),
        "alert_guard_blocked_true_positives": int(
            sum(1 for row in rows if row.get("alert_guard_blocked_prediction") and int(row["binary_label_id"]) == 1)
        ),
    }


def _replay_rows(
    token_data: dict[str, Any],
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    warmup_count: int,
    meta_rows: list[dict[str, Any]] | None = None,
    enable_stateful_service: bool = False,
    stateful_window_seconds: float = 2.0,
    stateful_min_count: int = 40,
    stateful_max_packets: int = 4,
    stateful_min_short_ratio: float = 0.75,
    stateful_min_model_score: float = 0.0,
    stateful_merge: str = "or",
    stateful_allowed_protocols: set[str] | None = None,
    stateful_allowed_ports: set[str] | None = None,
    stateful_excluded_ports: set[str] | None = None,
    alert_allowed_protocols: set[str] | None = None,
    alert_allowed_ports: set[str] | None = None,
    alert_excluded_ports: set[str] | None = None,
) -> list[dict[str, Any]]:
    meta_rows = meta_rows if meta_rows is not None else token_data.get("meta", [])
    labels = token_data["binary_labels"].numpy()
    ordered_items = sorted(
        zip(indices.tolist(), scores.tolist()),
        key=lambda item: (_safe_ts(meta_rows[int(item[0])] if int(item[0]) < len(meta_rows) else {}, int(item[0])), int(item[0])),
    )
    rows: list[dict[str, Any]] = []
    service_state: dict[tuple[str, ...], deque[dict[str, Any]]] = defaultdict(deque)
    window_seconds = max(0.0, float(stateful_window_seconds))
    min_count = max(1, int(stateful_min_count))
    max_packets = max(0, int(stateful_max_packets))
    min_short_ratio = float(stateful_min_short_ratio)
    min_model_score = float(stateful_min_model_score)
    excluded_ports = set(stateful_excluded_ports or set())
    alert_excluded = set(alert_excluded_ports or set())
    alert_guard_enabled = _service_guard_configured(alert_allowed_protocols, alert_allowed_ports, alert_excluded)
    for stream_pos, (data_idx, probs) in enumerate(ordered_items):
        idx = int(data_idx)
        meta = meta_rows[idx] if idx < len(meta_rows) else {}
        label_id = int(labels[idx])
        attack_probability = float(probs[1])
        model_prediction = int(attack_probability >= threshold)
        prediction = model_prediction
        start_ts = _safe_ts(meta, stream_pos)
        packet_count = _packet_count(meta)
        service_key_for_alert = _service_key(meta)
        service_port, service_protocol = _service_port_protocol(service_key_for_alert)
        alert_guard_allowed = True
        alert_guard_blocked = False
        row = {
            "stream_pos": int(stream_pos),
            "index": idx,
            "flow_id": meta.get("flow_id"),
            "label": _attack_label(meta),
            "binary_label": _binary_label(meta, label_id),
            "binary_label_id": label_id,
            "prediction": prediction,
            "attack_probability": attack_probability,
            "threshold": float(threshold),
            "is_warmup": bool(stream_pos < warmup_count),
            "is_false_positive": bool(label_id == 0 and prediction == 1),
            "is_true_positive": bool(label_id == 1 and prediction == 1),
            "packet_count": meta.get("packet_count"),
            "token_count": meta.get("token_count"),
            "start_ts": start_ts,
            "dataset_file": meta.get("dataset_file"),
        }
        if enable_stateful_service:
            service_key = service_key_for_alert
            guard_allowed = _stateful_service_allowed(
                service_key,
                allowed_protocols=stateful_allowed_protocols,
                allowed_ports=stateful_allowed_ports,
                excluded_ports=excluded_ports,
            )
            current_is_short = bool(0 < packet_count <= max_packets)
            if service_key is None:
                recent_count = 0
                recent_short = 0
                recent_packets = 0
                recent_short_ratio = 0.0
                last_gap = None
                episode_count = 0
                episode_short_ratio = 0.0
                stateful_prediction = 0
            else:
                history = service_state[service_key]
                while history and start_ts - float(history[0]["start_ts"]) > window_seconds:
                    history.popleft()
                recent_count = len(history)
                recent_short = int(sum(1 for item in history if item["is_short"]))
                recent_packets = int(sum(int(item["packet_count"]) for item in history))
                recent_short_ratio = float(recent_short / recent_count) if recent_count else 0.0
                last_gap = float(max(0.0, start_ts - float(history[-1]["start_ts"]))) if history else None
                episode_count = recent_short + int(current_is_short)
                episode_total = recent_count + 1
                episode_short_ratio = float(episode_count / episode_total) if episode_total else 0.0
                stateful_prediction = int(
                    guard_allowed
                    and
                    current_is_short
                    and episode_count >= min_count
                    and episode_short_ratio >= min_short_ratio
                    and attack_probability >= min_model_score
                )
            if stateful_merge == "or":
                prediction = int(model_prediction or stateful_prediction)
            elif stateful_merge == "model_only":
                prediction = model_prediction
            else:
                raise ValueError(f"unsupported stateful merge policy: {stateful_merge}")
            row.update(
                {
                    "model_prediction": model_prediction,
                    "stateful_service_key": "|".join(service_key) if service_key is not None else None,
                    "stateful_service_port": service_port,
                    "stateful_service_protocol": service_protocol,
                    "stateful_missing_key": service_key is None,
                    "stateful_guard_allowed": guard_allowed,
                    "stateful_current_is_short": current_is_short,
                    "stateful_recent_count": int(recent_count),
                    "stateful_recent_short": int(recent_short),
                    "stateful_recent_packets": int(recent_packets),
                    "stateful_short_ratio": recent_short_ratio,
                    "stateful_last_gap": last_gap,
                    "stateful_episode_count": int(episode_count),
                    "stateful_episode_short_ratio": episode_short_ratio,
                    "stateful_prediction": stateful_prediction,
                    "prediction": prediction,
                    "is_false_positive": bool(label_id == 0 and prediction == 1),
                    "is_true_positive": bool(label_id == 1 and prediction == 1),
                }
            )
            if service_key is not None:
                history.append({"start_ts": start_ts, "packet_count": packet_count, "is_short": current_is_short})
        if alert_guard_enabled:
            pre_guard_prediction = prediction
            alert_guard_allowed = _stateful_service_allowed(
                service_key_for_alert,
                allowed_protocols=alert_allowed_protocols,
                allowed_ports=alert_allowed_ports,
                excluded_ports=alert_excluded,
            )
            alert_guard_blocked = bool(prediction == 1 and not alert_guard_allowed)
            if alert_guard_blocked:
                prediction = 0
            row.update(
                {
                    "pre_guard_prediction": pre_guard_prediction,
                    "alert_guard_allowed": alert_guard_allowed,
                    "alert_guard_blocked_prediction": alert_guard_blocked,
                    "alert_service_key": "|".join(service_key_for_alert) if service_key_for_alert is not None else None,
                    "alert_service_port": service_port,
                    "alert_service_protocol": service_protocol,
                    "prediction": prediction,
                    "is_false_positive": bool(label_id == 0 and prediction == 1),
                    "is_true_positive": bool(label_id == 1 and prediction == 1),
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--metadata_tokens", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--split", choices=["stratified", "chronological", "temporal_stratified"], default="temporal_stratified")
    parser.add_argument("--scope", choices=["train", "val", "test", "val_test", "all"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibration_scope", choices=["train", "val", "test", "val_test", "all"], default=None)
    parser.add_argument("--warmup_flows", type=int, default=0)
    parser.add_argument("--calibrate_metric", choices=["macro_f1", "weighted_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--min_calibration_classes", type=int, default=2)
    parser.add_argument("--consecutive_alerts", type=int, default=1)
    parser.add_argument("--fpr_windows", nargs="*", type=float, default=[60.0, 300.0])
    parser.add_argument("--enable_stateful_service", action="store_true")
    parser.add_argument("--stateful_window_seconds", type=float, default=2.0)
    parser.add_argument("--stateful_min_count", type=int, default=40)
    parser.add_argument("--stateful_max_packets", type=int, default=4)
    parser.add_argument("--stateful_min_short_ratio", type=float, default=0.75)
    parser.add_argument("--stateful_min_model_score", type=float, default=0.0)
    parser.add_argument("--stateful_merge", choices=["or", "model_only"], default="or")
    parser.add_argument("--stateful_allowed_protocols", nargs="*", default=None)
    parser.add_argument("--stateful_allowed_ports", nargs="*", default=None)
    parser.add_argument("--stateful_excluded_ports", nargs="*", default=None)
    parser.add_argument("--alert_allowed_protocols", nargs="*", default=None)
    parser.add_argument("--alert_allowed_ports", nargs="*", default=None)
    parser.add_argument("--alert_excluded_ports", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    metadata_data = torch.load(args.metadata_tokens, map_location="cpu", weights_only=False) if args.metadata_tokens else None
    replay_meta_rows = _merged_metadata_rows(token_data, metadata_data)
    stateful_allowed_protocols = {str(item).upper() for item in args.stateful_allowed_protocols} if args.stateful_allowed_protocols is not None else None
    stateful_allowed_ports = {str(item) for item in args.stateful_allowed_ports} if args.stateful_allowed_ports is not None else None
    stateful_excluded_ports = {str(item) for item in args.stateful_excluded_ports} if args.stateful_excluded_ports is not None else set()
    alert_allowed_protocols = {str(item).upper() for item in args.alert_allowed_protocols} if args.alert_allowed_protocols is not None else None
    alert_allowed_ports = {str(item) for item in args.alert_allowed_ports} if args.alert_allowed_ports is not None else None
    alert_excluded_ports = {str(item) for item in args.alert_excluded_ports} if args.alert_excluded_ports is not None else set()
    indices = _split_eval_indices(token_data, cfg, args.split, args.scope)
    scores, indices = _predict_scores(token_data, indices, cfg, args.checkpoint)

    warmup_count = min(max(0, int(args.warmup_flows)), int(len(indices)))
    labels = token_data["binary_labels"].numpy()
    ordered = sorted(
        zip(indices.tolist(), scores.tolist()),
        key=lambda item: (_safe_ts(token_data.get("meta", [])[int(item[0])] if int(item[0]) < len(token_data.get("meta", [])) else {}, int(item[0])), int(item[0])),
    )
    if args.threshold is None:
        if args.calibration_scope:
            calibration_indices = _split_eval_indices(token_data, cfg, args.split, args.calibration_scope)
            calibration_scores, calibration_indices = _predict_scores(token_data, calibration_indices, cfg, args.checkpoint)
            calibration_items = sorted(
                zip(calibration_indices.tolist(), calibration_scores.tolist()),
                key=lambda item: (
                    _safe_ts(token_data.get("meta", [])[int(item[0])] if int(item[0]) < len(token_data.get("meta", [])) else {}, int(item[0])),
                    int(item[0]),
                ),
            )
            threshold, calibration_value, calibration_counts = _calibrate_threshold(
                labels,
                calibration_items,
                metric=args.calibrate_metric,
                min_classes=args.min_calibration_classes,
            )
            threshold_source = f"scope:{args.calibration_scope}"
        else:
            if warmup_count <= 0:
                raise ValueError("--threshold, --calibration_scope, or --warmup_flows is required")
            warmup_items = ordered[:warmup_count]
            threshold, calibration_value, calibration_counts = _calibrate_threshold(
                labels,
                warmup_items,
                metric=args.calibrate_metric,
                min_classes=args.min_calibration_classes,
            )
            threshold_source = "warmup_calibrated"
    else:
        threshold = float(args.threshold)
        calibration_value = None
        calibration_counts = None
        threshold_source = "fixed"

    rows = _replay_rows(
        token_data,
        indices,
        scores,
        threshold=threshold,
        warmup_count=warmup_count,
        meta_rows=replay_meta_rows,
        enable_stateful_service=args.enable_stateful_service,
        stateful_window_seconds=args.stateful_window_seconds,
        stateful_min_count=args.stateful_min_count,
        stateful_max_packets=args.stateful_max_packets,
        stateful_min_short_ratio=args.stateful_min_short_ratio,
        stateful_min_model_score=args.stateful_min_model_score,
        stateful_merge=args.stateful_merge,
        stateful_allowed_protocols=stateful_allowed_protocols,
        stateful_allowed_ports=stateful_allowed_ports,
        stateful_excluded_ports=stateful_excluded_ports,
        alert_allowed_protocols=alert_allowed_protocols,
        alert_allowed_ports=alert_allowed_ports,
        alert_excluded_ports=alert_excluded_ports,
    )
    eval_rows = [row for row in rows if not row["is_warmup"]]
    replay = summarize_replay(eval_rows, consecutive_alerts=args.consecutive_alerts, fpr_windows=args.fpr_windows)
    payload = {
        "tokens": args.tokens,
        "metadata_tokens": args.metadata_tokens,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "scope": args.scope,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "calibration_scope": args.calibration_scope,
        "calibrate_metric": args.calibrate_metric,
        "calibration_value": calibration_value,
        "calibration_counts": calibration_counts,
        "calibration_single_class": bool(calibration_counts and calibration_counts["num_classes"] < 2),
        "min_calibration_classes": int(args.min_calibration_classes),
        "warmup_flows": warmup_count,
        "num_scored_flows": int(len(rows)),
        "num_eval_flows": int(len(eval_rows)),
        **replay,
    }
    if args.enable_stateful_service:
        model_only = summarize_replay(
            _rows_with_prediction(eval_rows, "model_prediction"),
            consecutive_alerts=args.consecutive_alerts,
            fpr_windows=args.fpr_windows,
        )
        stateful_only = summarize_replay(
            _rows_with_prediction(eval_rows, "stateful_prediction"),
            consecutive_alerts=args.consecutive_alerts,
            fpr_windows=args.fpr_windows,
        )
        payload["stateful_service"] = {
            "enabled": True,
            "window_seconds": float(args.stateful_window_seconds),
            "min_count": int(args.stateful_min_count),
            "max_packets": int(args.stateful_max_packets),
            "min_short_ratio": float(args.stateful_min_short_ratio),
            "min_model_score": float(args.stateful_min_model_score),
            "merge": args.stateful_merge,
            "allowed_protocols": sorted(stateful_allowed_protocols) if stateful_allowed_protocols is not None else None,
            "allowed_ports": sorted(stateful_allowed_ports) if stateful_allowed_ports is not None else None,
            "excluded_ports": sorted(stateful_excluded_ports),
            **_stateful_alert_counts(eval_rows),
            "model_only_summary": model_only["summary"],
            "stateful_only_summary": stateful_only["summary"],
        }
    if _service_guard_configured(alert_allowed_protocols, alert_allowed_ports, alert_excluded_ports):
        pre_guard = summarize_replay(
            _rows_with_prediction(eval_rows, "pre_guard_prediction"),
            consecutive_alerts=args.consecutive_alerts,
            fpr_windows=args.fpr_windows,
        )
        payload["alert_guard"] = {
            "enabled": True,
            "allowed_protocols": sorted(alert_allowed_protocols) if alert_allowed_protocols is not None else None,
            "allowed_ports": sorted(alert_allowed_ports) if alert_allowed_ports is not None else None,
            "excluded_ports": sorted(alert_excluded_ports),
            **_alert_guard_counts(eval_rows),
            "pre_guard_summary": pre_guard["summary"],
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(payload, out_dir / "online_replay_summary.json")
    write_jsonl(rows, out_dir / "online_replay_scores.jsonl")
    print(payload["summary"])


if __name__ == "__main__":
    main()
