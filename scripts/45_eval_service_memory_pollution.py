#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.evaluation.metrics import classification_metrics, confusion
from src.training.anomaly_trainer import _score
from src.utils.io import read_yaml, write_json, write_jsonl


def _indices(token_data: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray([idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == split], dtype=np.int64)


def _token_histograms(token_data: dict[str, Any], indices: np.ndarray, normalize: str) -> np.ndarray:
    input_ids = token_data["input_ids"].cpu().numpy()
    attention_mask = token_data["attention_mask"].cpu().numpy()
    vocab = token_data["vocab"]
    skip = {vocab[token] for token in ("[PAD]", "[CLS]", "[SEP]", "[MASK]") if token in vocab}
    rows = np.zeros((len(indices), len(vocab)), dtype=np.float32)
    for row_idx, data_idx in enumerate(indices.tolist()):
        active = input_ids[data_idx][attention_mask[data_idx] > 0]
        active = np.asarray([token_id for token_id in active if int(token_id) not in skip], dtype=np.int64)
        if active.size:
            rows[row_idx] = np.bincount(active, minlength=len(vocab))[: len(vocab)]
    if normalize == "none":
        return rows
    if normalize == "l1":
        denom = np.sum(np.abs(rows), axis=1, keepdims=True)
    elif normalize == "l2":
        denom = np.linalg.norm(rows, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported normalize: {normalize}")
    return np.divide(rows, denom, out=np.zeros_like(rows), where=denom > 0)


def _service_key(meta: dict[str, Any]) -> str:
    key = meta.get("service_key")
    if isinstance(key, (list, tuple)):
        return "|".join(str(item) for item in key)
    if key:
        return str(key)
    return "GLOBAL"


def _score_against_bank(feature: np.ndarray, bank: deque[np.ndarray], global_prototype: np.ndarray, score_method: str, top_k: int) -> float:
    if not bank:
        return float(_score(feature.reshape(1, -1), global_prototype, score_method=score_method)[0])
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
    k = min(int(top_k), distances.shape[0])
    return float(np.mean(np.partition(distances, k - 1)[:k]))


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.concatenate(([float(scores.min()) - 1e-6], scores, [float(scores.max()) + 1e-6])))
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        f1 = float(classification_metrics(y_true.tolist(), pred.tolist(), np.column_stack([1.0 - scores, scores]))["macro_f1"])
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    benign = y_true == 0
    attack = y_true == 1
    fp = int(np.sum((y_pred == 1) & benign))
    tn = int(np.sum((y_pred == 0) & benign))
    tp = int(np.sum((y_pred == 1) & attack))
    fn = int(np.sum((y_pred == 0) & attack))
    return {
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "attack_recall": float(tp / max(tp + fn, 1)),
        "false_positives": fp,
        "true_negatives": tn,
        "true_positives": tp,
        "false_negatives": fn,
    }


def _evaluate_pollution(
    token_data: dict[str, Any],
    *,
    normalize: str,
    score_method: str,
    memory_size: int,
    top_k: int,
    threshold: float,
    pollution_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = token_data["binary_labels"].cpu().numpy()
    train_idx = _indices(token_data, "train")
    val_idx = _indices(token_data, "val")
    test_idx = _indices(token_data, "test")
    meta_rows = token_data.get("meta", [])
    train_features = _token_histograms(token_data, train_idx, normalize)
    val_features = _token_histograms(token_data, val_idx, normalize)
    test_features = _token_histograms(token_data, test_idx, normalize)
    benign_train_positions = np.where(labels[train_idx] == 0)[0]
    global_prototype = train_features[benign_train_positions].mean(axis=0)
    memory: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=memory_size))
    for pos in benign_train_positions.tolist():
        data_idx = int(train_idx[pos])
        memory[_service_key(meta_rows[data_idx])].append(train_features[pos])

    val_scores = np.asarray(
        [_score_against_bank(feature, memory[_service_key(meta_rows[int(data_idx)])], global_prototype, score_method, top_k) for feature, data_idx in zip(val_features, val_idx.tolist())],
        dtype=np.float32,
    )
    if threshold < 0:
        threshold, val_macro_f1 = _best_threshold(labels[val_idx], val_scores)
    else:
        val_macro_f1 = float(classification_metrics(labels[val_idx].tolist(), (val_scores >= threshold).astype(int).tolist(), np.column_stack([1.0 - val_scores, val_scores]))["macro_f1"])

    rng = np.random.default_rng(seed)
    attack_positions = np.where(labels[test_idx] == 1)[0]
    poison_count = int(round(len(attack_positions) * float(pollution_fraction)))
    poison_positions = set(rng.choice(attack_positions, size=min(poison_count, len(attack_positions)), replace=False).astype(int).tolist()) if poison_count > 0 else set()
    order = np.argsort([float(meta_rows[int(idx)].get("start_ts") or pos) for pos, idx in enumerate(test_idx.tolist())])
    scores = np.zeros(len(test_idx), dtype=np.float32)
    pred = np.zeros(len(test_idx), dtype=np.int64)
    poisoned_updates = 0
    clean_updates = 0
    for out_pos in order.tolist():
        data_idx = int(test_idx[out_pos])
        key = _service_key(meta_rows[data_idx])
        score = _score_against_bank(test_features[out_pos], memory[key], global_prototype, score_method, top_k)
        scores[out_pos] = score
        pred[out_pos] = int(score >= threshold)
        if out_pos in poison_positions:
            memory[key].append(test_features[out_pos])
            poisoned_updates += 1
        elif labels[data_idx] == 0 and pred[out_pos] == 0:
            memory[key].append(test_features[out_pos])
            clean_updates += 1
    y_true = labels[test_idx]
    metrics = classification_metrics(y_true.tolist(), pred.tolist(), np.column_stack([1.0 - scores, scores]))
    metrics.update(_rates(y_true, pred))
    metrics.update(
        {
            "threshold": float(threshold),
            "val_macro_f1_at_threshold": float(val_macro_f1),
            "pollution_fraction": float(pollution_fraction),
            "poisoned_updates": int(poisoned_updates),
            "clean_updates": int(clean_updates),
            "memory_size": int(memory_size),
            "top_k": int(top_k),
            "normalize": normalize,
            "score_method": score_method,
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
        }
    )
    rows = []
    for data_idx, label, item_pred, score in zip(test_idx.tolist(), y_true.tolist(), pred.tolist(), scores.tolist()):
        meta = meta_rows[int(data_idx)]
        rows.append(
            {
                "index": int(data_idx),
                "flow_id": meta.get("flow_id"),
                "label": meta.get("label"),
                "binary_label": int(label),
                "prediction": int(item_pred),
                "anomaly_score": float(score),
            }
        )
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate service-memory anomaly sensitivity to poisoned memory updates.")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--normalize", choices=["none", "l1", "l2"], default="l2")
    parser.add_argument("--score_method", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--threshold", type=float, default=-1.0, help="Use a fixed threshold; negative means calibrate on val.")
    parser.add_argument("--pollution_fraction", type=float, action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    baseline_cfg = cfg.get("baseline", {})
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    payload = {
        "command": shlex.join(sys.argv),
        "tokens": args.tokens,
        "config": args.config,
        "rows": [],
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fraction in args.pollution_fraction:
        metrics, rows = _evaluate_pollution(
            token_data,
            normalize=args.normalize,
            score_method=args.score_method,
            memory_size=int(baseline_cfg.get("memory_size", 128)),
            top_k=int(baseline_cfg.get("top_k", 5)),
            threshold=float(args.threshold),
            pollution_fraction=float(fraction),
            seed=int(args.seed),
        )
        payload["rows"].append(metrics)
        write_jsonl(rows, out_dir / f"scores_pollution_{fraction:g}.jsonl")
    write_json(payload, out_dir / "pollution_summary.json")
    write_json(
        {
            "rows": [
                {
                    "pollution_fraction": row["pollution_fraction"],
                    "macro_f1": row["macro_f1"],
                    "auroc": row.get("auroc"),
                    "auprc": row.get("auprc"),
                    "attack_recall": row["attack_recall"],
                    "false_positive_rate": row["false_positive_rate"],
                    "poisoned_updates": row["poisoned_updates"],
                }
                for row in payload["rows"]
            ]
        },
        out_dir / "pollution_table.json",
    )
    print(payload["rows"])


if __name__ == "__main__":
    main()
