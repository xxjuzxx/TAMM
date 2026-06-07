from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import classification_metrics
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.training.classifier_trainer import _split_indices


@dataclass
class AnomalyResult:
    metrics: dict[str, Any]
    report: dict[str, Any]
    confusion_matrix: list[list[int]]
    split_summary: dict[str, Any]
    score_rows: list[dict[str, Any]]


def _order_values(token_data: dict[str, Any]) -> np.ndarray:
    return np.array([float(meta.get("start_ts") or idx) for idx, meta in enumerate(token_data.get("meta", []))])


def _split_summary(labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
    def counts(indices: np.ndarray) -> dict[str, int]:
        selected = labels[indices]
        return {
            "total": int(len(indices)),
            "benign": int(np.sum(selected == 0)),
            "attack": int(np.sum(selected == 1)),
        }

    return {
        "train": counts(train_idx),
        "val": counts(val_idx),
        "test": counts(test_idx),
    }


def _skip_token_ids(vocab: dict[str, int], include_special: bool) -> set[int]:
    if include_special:
        return set()
    return {
        token_id
        for token in ("[PAD]", "[CLS]", "[SEP]", "[MASK]")
        if (token_id := vocab.get(token)) is not None
    }


def _token_histograms(
    token_data: dict[str, Any],
    indices: np.ndarray,
    *,
    include_special: bool,
    normalize: str,
) -> np.ndarray:
    input_ids = token_data["input_ids"].cpu().numpy()
    attention_mask = token_data["attention_mask"].cpu().numpy()
    vocab_size = len(token_data["vocab"])
    skip_ids = _skip_token_ids(token_data["vocab"], include_special)
    features = np.zeros((len(indices), vocab_size), dtype=np.float32)

    for row_idx, data_idx in enumerate(indices):
        active_ids = input_ids[data_idx][attention_mask[data_idx] > 0]
        if skip_ids:
            active_ids = np.array([token_id for token_id in active_ids if int(token_id) not in skip_ids], dtype=np.int64)
        if active_ids.size:
            features[row_idx] = np.bincount(active_ids, minlength=vocab_size)[:vocab_size]

    if normalize == "none":
        return features
    if normalize == "l1":
        denom = np.sum(np.abs(features), axis=1, keepdims=True)
    elif normalize == "l2":
        denom = np.linalg.norm(features, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported normalization: {normalize}")
    return np.divide(features, denom, out=np.zeros_like(features), where=denom > 0)


def _encoder_embeddings(
    token_data: dict[str, Any],
    indices: np.ndarray,
    config: dict[str, Any],
    checkpoint: str,
    *,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    model_cfg = config.get("model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model_state = model.state_dict()
    compatible = {}
    for key, value in state_dict.items():
        target_key = key
        if target_key not in model_state and target_key.startswith("encoder."):
            target_key = target_key.removeprefix("encoder.")
        if target_key in model_state and tuple(model_state[target_key].shape) == tuple(value.shape):
            compatible[target_key] = value
    model_state.update(compatible)
    model.load_state_dict(model_state)
    model.eval()
    idx = torch.tensor(indices, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(
            token_data["input_ids"][idx],
            token_data["attention_mask"][idx],
            token_data["token_type_ids"][idx],
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in loader:
            embedding = model.encode(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
            rows.append(embedding.cpu().numpy())
    return np.concatenate(rows, axis=0), str(device)


def _score(features: np.ndarray, prototype: np.ndarray, score_method: str) -> np.ndarray:
    if score_method == "cosine":
        proto_norm = np.linalg.norm(prototype)
        if proto_norm == 0:
            return np.ones(features.shape[0], dtype=np.float32)
        proto = prototype / proto_norm
        feature_norms = np.linalg.norm(features, axis=1)
        similarities = np.divide(features @ proto, feature_norms, out=np.zeros(features.shape[0], dtype=np.float32), where=feature_norms > 0)
        return 1.0 - similarities
    if score_method == "euclidean":
        return np.linalg.norm(features - prototype.reshape(1, -1), axis=1)
    raise ValueError(f"Unsupported score_method: {score_method}")


def _service_key(meta: dict[str, Any]) -> str:
    key = meta.get("service_key")
    if isinstance(key, (list, tuple)):
        return "|".join(str(item) for item in key)
    if key:
        return str(key)
    return "GLOBAL"


def _normalize_features(features: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "none":
        return features
    if normalize == "l1":
        denom = np.sum(np.abs(features), axis=1, keepdims=True)
    elif normalize == "l2":
        denom = np.linalg.norm(features, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported normalization: {normalize}")
    return np.divide(features, denom, out=np.zeros_like(features), where=denom > 0)


def _score_one(feature: np.ndarray, reference: np.ndarray, score_method: str) -> float:
    return float(_score(feature.reshape(1, -1), reference, score_method=score_method)[0])


def _dynamic_baseline_scores(
    token_data: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    train_features_all: np.ndarray,
    val_features_all: np.ndarray,
    test_features_all: np.ndarray,
    *,
    score_method: str,
    threshold: float | None,
    memory_size: int,
    min_reputation_to_update: float,
    update_alpha: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    meta_rows = token_data.get("meta", [])
    benign_train_positions = np.where(labels[train_idx] == 0)[0]
    global_bank = train_features_all[benign_train_positions]
    global_prototype = global_bank.mean(axis=0)
    memory: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=memory_size))
    reputation: dict[str, float] = defaultdict(lambda: 1.0)
    for pos in benign_train_positions.tolist():
        data_idx = int(train_idx[pos])
        memory[_service_key(meta_rows[data_idx])].append(train_features_all[pos])

    def bank_score(feature: np.ndarray, bank: list[np.ndarray] | deque[np.ndarray]) -> float:
        if not bank:
            return _score_one(feature, global_prototype, score_method)
        refs = np.stack(list(bank), axis=0)
        if score_method == "cosine":
            feature_norm = np.linalg.norm(feature)
            ref_norms = np.linalg.norm(refs, axis=1)
            similarities = np.divide(refs @ feature, ref_norms * feature_norm, out=np.zeros(refs.shape[0], dtype=np.float32), where=(ref_norms * feature_norm) > 0)
            distances = 1.0 - similarities
        elif score_method == "euclidean":
            distances = np.linalg.norm(refs - feature.reshape(1, -1), axis=1)
        else:
            raise ValueError(f"Unsupported score_method: {score_method}")
        k = min(top_k, distances.shape[0])
        return float(np.mean(np.partition(distances, k - 1)[:k]))

    def run(indices: np.ndarray, features: np.ndarray, allow_updates: bool) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros(len(indices), dtype=np.float32)
        predictions = np.zeros(len(indices), dtype=np.int64)
        order = np.argsort([float(meta_rows[int(idx)].get("start_ts") or pos) for pos, idx in enumerate(indices.tolist())])
        for out_pos in order.tolist():
            data_idx = int(indices[out_pos])
            key = _service_key(meta_rows[data_idx])
            score = bank_score(features[out_pos], memory[key])
            scores[out_pos] = score
            pred = int(score >= threshold) if threshold is not None else 0
            predictions[out_pos] = pred
            if allow_updates and labels[data_idx] == 0:
                normal_enough = threshold is None or score < threshold
                if normal_enough and reputation[key] >= min_reputation_to_update:
                    memory[key].append(features[out_pos])
                    reputation[key] = update_alpha * reputation[key] + (1.0 - update_alpha) * 1.0
                elif not normal_enough:
                    reputation[key] = update_alpha * reputation[key]
        return scores, predictions

    val_scores, _ = run(val_idx, val_features_all, allow_updates=False)
    summary_threshold = threshold
    if summary_threshold is None:
        summary_threshold, _ = _best_threshold(labels[val_idx], val_scores)
    test_scores, test_pred = run(test_idx, test_features_all, allow_updates=True)
    summary = {
        "num_services": len(memory),
        "memory_size": memory_size,
        "min_reputation_to_update": min_reputation_to_update,
        "update_alpha": update_alpha,
        "top_k": top_k,
        "threshold": float(summary_threshold),
    }
    return val_scores, test_scores, test_pred, summary


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    unique_scores = np.unique(scores)
    if unique_scores.size > 400:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 401))
    else:
        thresholds = unique_scores
    thresholds = np.unique(np.concatenate(([float(scores.min()) - 1e-6], thresholds, [float(scores.max()) + 1e-6])))
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        cur_f1 = float(f1_score(y_true, pred, average="macro", zero_division=0))
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float | None:
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
    except ValueError:
        return None
    eligible = fpr[tpr >= target_tpr]
    if eligible.size == 0:
        return None
    return float(np.min(eligible))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(y_true.tolist(), y_pred.tolist(), np.column_stack([1.0 - scores, scores]))
    metrics["fpr_at_95_tpr"] = metrics.get("fpr95")
    return metrics


def _score_rows(
    token_data: dict[str, Any],
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta_rows = token_data.get("meta", [])
    for data_idx, label, pred, score in zip(indices.tolist(), y_true.tolist(), y_pred.tolist(), scores.tolist()):
        meta = meta_rows[data_idx] if data_idx < len(meta_rows) else {}
        rows.append(
            {
                "index": int(data_idx),
                "flow_id": meta.get("flow_id"),
                "label": meta.get("label"),
                "binary_label": int(label),
                "prediction": int(pred),
                "anomaly_score": float(score),
                "packet_count": meta.get("packet_count"),
                "start_ts": meta.get("start_ts"),
                "dataset_file": meta.get("dataset_file"),
            }
        )
    return rows


def train_anomaly_detector(
    token_data: dict[str, Any],
    config: dict[str, Any],
    *,
    split: str = "temporal_stratified",
    score_method: str = "cosine",
    normalize: str = "l2",
    include_special: bool = False,
    feature: str = "token_histogram",
    checkpoint: str | None = None,
    baseline: str = "global_prototype",
) -> AnomalyResult:
    train_cfg = config.get("training", {})
    labels = token_data["binary_labels"].cpu().numpy()
    if split == "predefined":
        meta_rows = token_data.get("meta", [])
        train_idx = np.array([idx for idx, meta in enumerate(meta_rows) if meta.get("split") == "train"], dtype=np.int64)
        val_idx = np.array([idx for idx, meta in enumerate(meta_rows) if meta.get("split") == "val"], dtype=np.int64)
        test_idx = np.array([idx for idx, meta in enumerate(meta_rows) if meta.get("split") == "test"], dtype=np.int64)
    else:
        train_idx, val_idx, test_idx = _split_indices(
            labels,
            val_ratio=float(train_cfg.get("val_ratio", 0.1)),
            test_ratio=float(train_cfg.get("test_ratio", 0.2)),
            seed=int(config.get("seed", 42)),
            split=split,
            order_values=_order_values(token_data),
        )
    benign_train_idx = train_idx[labels[train_idx] == 0]
    if benign_train_idx.size == 0:
        raise ValueError("No benign samples in the train split; cannot fit a normal prototype.")

    device = None
    if feature == "token_histogram":
        all_train_features = _token_histograms(token_data, train_idx, include_special=include_special, normalize=normalize)
        train_features = all_train_features[labels[train_idx] == 0]
        val_features = _token_histograms(token_data, val_idx, include_special=include_special, normalize=normalize)
        test_features = _token_histograms(token_data, test_idx, include_special=include_special, normalize=normalize)
    elif feature == "encoder_embedding":
        if checkpoint is None:
            raise ValueError("--checkpoint is required when feature=encoder_embedding")
        batch_size = int(train_cfg.get("batch_size", 64))
        all_train_features, device = _encoder_embeddings(token_data, train_idx, config, checkpoint, batch_size=batch_size)
        train_features = all_train_features[labels[train_idx] == 0]
        val_features, _ = _encoder_embeddings(token_data, val_idx, config, checkpoint, batch_size=batch_size)
        test_features, _ = _encoder_embeddings(token_data, test_idx, config, checkpoint, batch_size=batch_size)
        if normalize != "none":
            all_train_features = _normalize_features(all_train_features, normalize)
            train_features = all_train_features[labels[train_idx] == 0]
            val_features = _normalize_features(val_features, normalize)
            test_features = _normalize_features(test_features, normalize)
    else:
        raise ValueError(f"Unsupported feature: {feature}")

    y_true = labels[test_idx]
    baseline_summary = None
    if baseline == "global_prototype":
        prototype = train_features.mean(axis=0)
        val_scores = _score(val_features, prototype, score_method=score_method)
        test_scores = _score(test_features, prototype, score_method=score_method)
        threshold, val_macro_f1 = _best_threshold(labels[val_idx], val_scores)
        y_pred = (test_scores >= threshold).astype(int)
    elif baseline == "service_memory":
        if "all_train_features" not in locals():
            all_train_features = _token_histograms(token_data, train_idx, include_special=include_special, normalize=normalize)
        val_scores_for_threshold, _, _, _ = _dynamic_baseline_scores(
            token_data,
            train_idx,
            val_idx,
            test_idx,
            labels,
            all_train_features,
            val_features,
            test_features,
            score_method=score_method,
            threshold=None,
            memory_size=int(config.get("baseline", {}).get("memory_size", 128)),
            min_reputation_to_update=float(config.get("baseline", {}).get("min_reputation_to_update", 0.7)),
            update_alpha=float(config.get("baseline", {}).get("update_alpha", 0.95)),
            top_k=int(config.get("baseline", {}).get("top_k", 5)),
        )
        threshold, val_macro_f1 = _best_threshold(labels[val_idx], val_scores_for_threshold)
        _, test_scores, y_pred, baseline_summary = _dynamic_baseline_scores(
            token_data,
            train_idx,
            val_idx,
            test_idx,
            labels,
            all_train_features,
            val_features,
            test_features,
            score_method=score_method,
            threshold=threshold,
            memory_size=int(config.get("baseline", {}).get("memory_size", 128)),
            min_reputation_to_update=float(config.get("baseline", {}).get("min_reputation_to_update", 0.7)),
            update_alpha=float(config.get("baseline", {}).get("update_alpha", 0.95)),
            top_k=int(config.get("baseline", {}).get("top_k", 5)),
        )
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    metrics = _metrics(y_true, y_pred, test_scores)
    metrics.update(
        {
            "threshold": float(threshold),
            "val_macro_f1_at_threshold": float(val_macro_f1),
            "split": split,
            "score_method": score_method,
            "feature": feature,
            "baseline": baseline,
            "normalize": normalize,
            "include_special": bool(include_special),
            "profile_mode": token_data.get("profile_mode"),
            "num_train": int(len(train_idx)),
            "num_train_benign_used": int(len(benign_train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "seed": int(config.get("seed", 42)),
        }
    )
    if checkpoint is not None:
        metrics["checkpoint"] = checkpoint
    if device is not None:
        metrics["device"] = device
    if baseline_summary is not None:
        metrics["baseline_summary"] = baseline_summary
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )
    return AnomalyResult(
        metrics=metrics,
        report=report,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        split_summary=_split_summary(labels, train_idx, val_idx, test_idx),
        score_rows=_score_rows(token_data, test_idx, y_true, y_pred, test_scores),
    )
