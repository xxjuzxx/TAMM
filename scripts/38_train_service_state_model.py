#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.training.classifier_trainer import _split_indices
from src.utils.io import write_json, write_jsonl


TARGET_NAMES = ["BENIGN", "ATTACK"]
DEFAULT_WINDOWS = [0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_COMMON_PORTS = ["21", "22", "53", "80", "443", "8080"]


def _safe_ts(meta: dict[str, Any], fallback: int) -> float:
    try:
        value = meta.get("start_ts")
        if value is None:
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _packet_count(meta: dict[str, Any]) -> int:
    value = meta.get("packet_count")
    if value is None:
        value = len(meta.get("lens") or [])
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _token_count(meta: dict[str, Any]) -> int:
    value = meta.get("token_count")
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


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
        return str(service_key[-2]), protocol
    if len(service_key) >= 2:
        return str(service_key[1]), None
    return None, None


def _log1p(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(np.log1p(max(0.0, float(value))))


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _merge_metadata_rows(token_data: dict[str, Any], metadata_data: dict[str, Any] | None) -> list[dict[str, Any]]:
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


def load_token_data(path: str | Path, metadata_tokens: str | Path | None = None) -> dict[str, Any]:
    token_data = torch.load(path, map_location="cpu", weights_only=False)
    metadata_data = torch.load(metadata_tokens, map_location="cpu", weights_only=False) if metadata_tokens else None
    token_data = dict(token_data)
    token_data["meta"] = _merge_metadata_rows(token_data, metadata_data)
    return token_data


def _identity_feature_names(common_ports: list[str]) -> list[str]:
    names = ["proto_TCP", "proto_UDP", "proto_ICMP", "proto_OTHER"]
    names.extend(f"port_{port}" for port in common_ports)
    names.extend(["port_common_other", "port_privileged"])
    return names


def service_state_feature_names(
    windows: list[float],
    include_identity: bool = False,
    common_ports: list[str] | None = None,
) -> list[str]:
    common_ports = common_ports or DEFAULT_COMMON_PORTS
    names = [
        "current_log_packet_count",
        "current_log_token_count",
        "current_short_le4",
        "current_short_le6",
        "missing_service_key",
        "prior_total_log_count",
        "prior_total_short4_ratio",
        "prior_total_short6_ratio",
        "prior_total_log_packets",
        "prior_last_gap_log",
    ]
    for window in windows:
        prefix = f"w{window:g}"
        names.extend(
            [
                f"{prefix}_log_recent_count",
                f"{prefix}_short4_ratio",
                f"{prefix}_short6_ratio",
                f"{prefix}_log_recent_packets",
                f"{prefix}_avg_packets_log",
                f"{prefix}_last_gap_log",
                f"{prefix}_episode_short4_count_log",
                f"{prefix}_episode_short4_ratio",
                f"{prefix}_episode_short6_count_log",
                f"{prefix}_episode_short6_ratio",
            ]
        )
    if include_identity:
        names.extend(_identity_feature_names(common_ports))
    return names


def _identity_features(
    service_key: tuple[str, ...] | None,
    common_ports: list[str],
) -> list[float]:
    port, protocol = _service_port_protocol(service_key)
    proto = protocol if protocol in {"TCP", "UDP", "ICMP"} else "OTHER"
    values = [
        float(proto == "TCP"),
        float(proto == "UDP"),
        float(proto == "ICMP"),
        float(proto == "OTHER"),
    ]
    values.extend(float(port == common_port) for common_port in common_ports)
    values.append(float(port is not None and port not in set(common_ports)))
    try:
        values.append(float(port is not None and int(port) <= 1024))
    except (TypeError, ValueError):
        values.append(0.0)
    return values


def build_service_state_matrix(
    meta_rows: list[dict[str, Any]],
    windows: list[float] | None = None,
    include_identity: bool = False,
    common_ports: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    windows = [float(item) for item in (windows or DEFAULT_WINDOWS)]
    common_ports = common_ports or DEFAULT_COMMON_PORTS
    feature_names = service_state_feature_names(windows, include_identity=include_identity, common_ports=common_ports)
    features = np.zeros((len(meta_rows), len(feature_names)), dtype=np.float32)
    order = sorted(range(len(meta_rows)), key=lambda idx: (_safe_ts(meta_rows[idx], idx), idx))
    window_history: dict[float, dict[tuple[str, ...], deque[dict[str, float]]]] = {
        window: defaultdict(deque) for window in windows
    }
    lifetime: dict[tuple[str, ...], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "short4": 0.0, "short6": 0.0, "packets": 0.0, "last_ts": None}
    )

    for idx in order:
        meta = meta_rows[idx]
        start_ts = _safe_ts(meta, idx)
        packet_count = _packet_count(meta)
        token_count = _token_count(meta)
        current_short4 = float(0 < packet_count <= 4)
        current_short6 = float(0 < packet_count <= 6)
        key = _service_key(meta)
        row: list[float] = [
            _log1p(packet_count),
            _log1p(token_count),
            current_short4,
            current_short6,
            float(key is None),
        ]
        if key is None:
            stats = {"count": 0.0, "short4": 0.0, "short6": 0.0, "packets": 0.0, "last_ts": None}
        else:
            stats = lifetime[key]
        total_count = float(stats["count"])
        row.extend(
            [
                _log1p(total_count),
                _ratio(float(stats["short4"]), total_count),
                _ratio(float(stats["short6"]), total_count),
                _log1p(float(stats["packets"])),
                _log1p(max(0.0, start_ts - float(stats["last_ts"]))) if stats["last_ts"] is not None else 0.0,
            ]
        )
        for window in windows:
            if key is None:
                history: deque[dict[str, float]] = deque()
            else:
                history = window_history[window][key]
                while history and start_ts - float(history[0]["start_ts"]) > window:
                    history.popleft()
            recent_count = float(len(history))
            recent_short4 = float(sum(item["short4"] for item in history))
            recent_short6 = float(sum(item["short6"] for item in history))
            recent_packets = float(sum(item["packet_count"] for item in history))
            last_gap = max(0.0, start_ts - float(history[-1]["start_ts"])) if history else None
            episode_short4 = recent_short4 + current_short4
            episode_short6 = recent_short6 + current_short6
            episode_total = recent_count + 1.0
            row.extend(
                [
                    _log1p(recent_count),
                    _ratio(recent_short4, recent_count),
                    _ratio(recent_short6, recent_count),
                    _log1p(recent_packets),
                    _log1p(_ratio(recent_packets, recent_count)),
                    _log1p(last_gap),
                    _log1p(episode_short4),
                    _ratio(episode_short4, episode_total),
                    _log1p(episode_short6),
                    _ratio(episode_short6, episode_total),
                ]
            )
        if include_identity:
            row.extend(_identity_features(key, common_ports))
        features[idx] = np.array(row, dtype=np.float32)

        if key is not None:
            item = {
                "start_ts": float(start_ts),
                "packet_count": float(packet_count),
                "short4": current_short4,
                "short6": current_short6,
            }
            for history_by_key in window_history.values():
                history_by_key[key].append(item)
            stats["count"] = float(stats["count"]) + 1.0
            stats["short4"] = float(stats["short4"]) + current_short4
            stats["short6"] = float(stats["short6"]) + current_short6
            stats["packets"] = float(stats["packets"]) + float(packet_count)
            stats["last_ts"] = float(start_ts)
    return features, feature_names


def _feature_indices(feature_names: list[str], feature_set: str) -> list[int]:
    if feature_set == "all":
        return list(range(len(feature_names)))

    def is_current(name: str) -> bool:
        return name.startswith("current_") or name == "missing_service_key"

    def is_identity(name: str) -> bool:
        return name.startswith("proto_") or name.startswith("port_") or name in {"port_common_other", "port_privileged"}

    if feature_set == "history":
        return [idx for idx, name in enumerate(feature_names) if not is_current(name) and not is_identity(name)]
    if feature_set == "current":
        return [idx for idx, name in enumerate(feature_names) if is_current(name) or is_identity(name)]
    raise ValueError(f"unsupported feature_set: {feature_set}")


def _select_features(matrix: np.ndarray, feature_names: list[str], feature_set: str) -> tuple[np.ndarray, list[str]]:
    indices = _feature_indices(feature_names, feature_set)
    if not indices:
        raise ValueError(f"feature_set={feature_set} selected no features")
    return matrix[:, indices], [feature_names[idx] for idx in indices]


def _order_values(meta_rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([_safe_ts(meta, idx) for idx, meta in enumerate(meta_rows)], dtype=float)


def _labels(token_data: dict[str, Any]) -> np.ndarray:
    return token_data["binary_labels"].detach().cpu().numpy().astype(np.int64)


def _align_proba(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(features)
    if proba.shape[1] == 2:
        return proba
    aligned = np.zeros((features.shape[0], 2), dtype=float)
    for idx, cls in enumerate(model.classes_):
        aligned[:, int(cls)] = proba[:, idx]
    return aligned


def _metrics_for_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    return classification_metrics(y_true.tolist(), pred.tolist(), np.column_stack([1.0 - scores, scores]))


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    if len(set(y_true.tolist())) < 2:
        return 0.5, _metrics_for_threshold(y_true, scores, 0.5)
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    for threshold in np.linspace(0.0, 1.0, 201):
        metrics = _metrics_for_threshold(y_true, scores, float(threshold))
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics or {}


def _evaluate(
    model: RandomForestClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], list[int], np.ndarray]:
    selected = np.array(indices, dtype=np.int64)
    proba = _align_proba(model, features[selected])
    pred = (proba[:, 1] >= threshold).astype(int)
    metrics = classification_metrics(labels[selected].tolist(), pred.tolist(), proba)
    return metrics, pred.tolist(), proba


def _report_row(
    name: str,
    token_path: str,
    metrics: dict[str, Any],
    report: dict[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "name": name,
        "tokens": token_path,
        "num_eval": int(len(indices)),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "attack_recall": (report.get("ATTACK") or {}).get("recall"),
        "attack_f1": (report.get("ATTACK") or {}).get("f1-score"),
        "benign_recall": (report.get("BENIGN") or {}).get("recall"),
        "benign_f1": (report.get("BENIGN") or {}).get("f1-score"),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Trainable service-state baseline",
        "",
        "This is a metadata-only state-feature baseline. It uses the current flow summary and prior same-service history only; it is diagnostic and does not replace the Zeek-first BehaviorFlow-BERT main table.",
        "",
        "| Dataset | Eval N | Accuracy | Macro-F1 | Weighted-F1 | AUROC | AUPRC | ATTACK R/F1 | BENIGN R/F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        attack = f"{row['attack_recall']:.4f}/{row['attack_f1']:.4f}" if row.get("attack_f1") is not None else "-"
        benign = f"{row['benign_recall']:.4f}/{row['benign_f1']:.4f}" if row.get("benign_f1") is not None else "-"
        lines.append(
            "| {name} | {num_eval} | {acc:.4f} | {macro:.4f} | {weighted:.4f} | {auroc:.4f} | {auprc:.4f} | {attack} | {benign} |".format(
                name=row["name"],
                num_eval=row["num_eval"],
                acc=float(row["accuracy"]),
                macro=float(row["macro_f1"]),
                weighted=float(row["weighted_f1"]),
                auroc=float(row.get("auroc") or 0.0),
                auprc=float(row.get("auprc") or 0.0),
                attack=attack,
                benign=benign,
            )
        )
    lines.extend(
        [
            "",
            f"Threshold selected on validation: `{payload['threshold']:.4f}`.",
            "",
            "Top feature importances:",
            "",
            "| Feature | Importance |",
            "|---|---:|",
        ]
    )
    for item in payload["top_features"]:
        lines.append(f"| {item['feature']} | {item['importance']:.6f} |")
    lines.append("")
    return "\n".join(lines)


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    token_data = load_token_data(args.tokens, args.metadata_tokens)
    labels = _labels(token_data)
    features, all_feature_names = build_service_state_matrix(
        token_data.get("meta", []),
        windows=args.windows,
        include_identity=args.include_service_identity,
        common_ports=args.common_ports,
    )
    features, feature_names = _select_features(features, all_feature_names, args.feature_set)
    train_idx, val_idx, test_idx = _split_indices(
        labels,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split=args.split,
        order_values=_order_values(token_data.get("meta", [])),
    )
    train_features = [features[train_idx]]
    train_labels = [labels[train_idx]]
    augment_info: list[dict[str, Any]] = []
    augment_metadata = args.augment_metadata_tokens or []
    if augment_metadata and len(augment_metadata) != len(args.augment_tokens):
        raise ValueError("--augment_metadata_tokens must have the same length as --augment_tokens")
    for pos, augment_path in enumerate(args.augment_tokens):
        metadata_path = augment_metadata[pos] if augment_metadata else None
        augment_data = load_token_data(augment_path, metadata_path)
        augment_labels = _labels(augment_data)
        if len(augment_labels) != len(labels):
            raise ValueError(f"augment token row count does not match base tokens: {augment_path}")
        augment_features, augment_all_names = build_service_state_matrix(
            augment_data.get("meta", []),
            windows=args.windows,
            include_identity=args.include_service_identity,
            common_ports=args.common_ports,
        )
        augment_features, augment_names = _select_features(augment_features, augment_all_names, args.feature_set)
        if augment_names != feature_names:
            raise ValueError(f"augment feature names do not match base features: {augment_path}")
        if args.augment_all_labels:
            selected = train_idx
        else:
            selected = train_idx[labels[train_idx] == 1]
        train_features.append(augment_features[selected])
        train_labels.append(augment_labels[selected])
        augment_info.append(
            {
                "tokens": augment_path,
                "metadata_tokens": metadata_path,
                "selected_train_rows": int(len(selected)),
                "attack_only": not bool(args.augment_all_labels),
            }
        )
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(np.concatenate(train_features, axis=0), np.concatenate(train_labels, axis=0))
    val_proba = _align_proba(model, features[val_idx])
    threshold, val_metrics = _best_threshold(labels[val_idx], val_proba[:, 1])

    eval_tokens = args.eval_tokens or [args.tokens]
    eval_metadata = args.eval_metadata_tokens or []
    eval_names = args.eval_names or [Path(item).stem for item in eval_tokens]
    if len(eval_names) != len(eval_tokens):
        raise ValueError("--eval_names must have the same length as --eval_tokens")
    if eval_metadata and len(eval_metadata) != len(eval_tokens):
        raise ValueError("--eval_metadata_tokens must have the same length as --eval_tokens")

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for pos, eval_path in enumerate(eval_tokens):
        metadata_path = eval_metadata[pos] if eval_metadata else None
        eval_data = token_data if eval_path == args.tokens and metadata_path == args.metadata_tokens else load_token_data(eval_path, metadata_path)
        eval_labels = _labels(eval_data)
        eval_features, eval_all_feature_names = build_service_state_matrix(
            eval_data.get("meta", []),
            windows=args.windows,
            include_identity=args.include_service_identity,
            common_ports=args.common_ports,
        )
        eval_features, eval_feature_names = _select_features(eval_features, eval_all_feature_names, args.feature_set)
        if eval_feature_names != feature_names:
            raise ValueError(f"eval feature names do not match base features: {eval_path}")
        if len(eval_labels) == len(labels):
            indices = test_idx
        else:
            _, _, indices = _split_indices(
                eval_labels,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                split=args.split,
                order_values=_order_values(eval_data.get("meta", [])),
            )
        metrics, pred, proba = _evaluate(model, eval_features, eval_labels, indices, threshold)
        report = report_dict(eval_labels[indices].tolist(), pred, TARGET_NAMES)
        name = eval_names[pos]
        rows.append(_report_row(name, eval_path, metrics, report, indices))
        details[name] = {
            "metrics": metrics,
            "classification_report": report,
            "confusion_matrix": confusion(eval_labels[indices].tolist(), pred),
            "indices": indices.tolist(),
        }
        if args.write_scores:
            score_rows = []
            meta_rows = eval_data.get("meta", [])
            for out_pos, data_idx in enumerate(indices.tolist()):
                meta = meta_rows[int(data_idx)] if int(data_idx) < len(meta_rows) else {}
                score_rows.append(
                    {
                        "index": int(data_idx),
                        "flow_id": meta.get("flow_id"),
                        "label": meta.get("label"),
                        "binary_label_id": int(eval_labels[int(data_idx)]),
                        "prediction": int(pred[out_pos]),
                        "attack_probability": float(proba[out_pos, 1]),
                        "start_ts": meta.get("start_ts"),
                        "packet_count": meta.get("packet_count"),
                        "service_key": meta.get("service_key"),
                    }
                )
            write_jsonl(score_rows, Path(args.out) / f"{name}_scores.jsonl")

    importances = getattr(model, "feature_importances_", np.zeros(len(feature_names), dtype=float))
    ranked = sorted(
        [{"feature": feature, "importance": float(importance)} for feature, importance in zip(feature_names, importances)],
        key=lambda item: item["importance"],
        reverse=True,
    )
    payload = {
        "protocol": {
            "model": "RandomForestClassifier",
            "scope": "trainable service-state diagnostic",
            "tokens": args.tokens,
            "metadata_tokens": args.metadata_tokens,
            "split": args.split,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "windows": args.windows,
            "include_service_identity": bool(args.include_service_identity),
            "common_ports": args.common_ports,
            "feature_set": args.feature_set,
            "augment": augment_info,
        },
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "feature_names": feature_names,
        "top_features": ranked[: args.top_k_features],
        "rows": rows,
        "details": details,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight service-state feature baseline from token metadata.")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--metadata_tokens", default=None)
    parser.add_argument("--eval_tokens", nargs="*", default=None)
    parser.add_argument("--eval_metadata_tokens", nargs="*", default=None)
    parser.add_argument("--eval_names", nargs="*", default=None)
    parser.add_argument("--augment_tokens", nargs="*", default=[])
    parser.add_argument("--augment_metadata_tokens", nargs="*", default=[])
    parser.add_argument("--augment_all_labels", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["stratified", "chronological", "temporal_stratified"], default="temporal_stratified")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", nargs="+", type=float, default=DEFAULT_WINDOWS)
    parser.add_argument("--include_service_identity", action="store_true")
    parser.add_argument("--feature_set", choices=["all", "history", "current"], default="all")
    parser.add_argument("--common_ports", nargs="+", default=DEFAULT_COMMON_PORTS)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--top_k_features", type=int, default=20)
    parser.add_argument("--write_scores", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = train_and_evaluate(args)
    write_json(payload, out_dir / "metrics.json")
    (out_dir / "results.md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))


if __name__ == "__main__":
    main()
