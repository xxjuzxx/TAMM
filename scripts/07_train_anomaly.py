#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.evaluation.metrics import classification_metrics
from src.utils.io import read_yaml, write_json, write_jsonl


def _indices(token_data: dict[str, Any], split: str) -> np.ndarray:
    return np.array([idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == split], dtype=np.int64)


def _histograms(token_data: dict[str, Any], indices: np.ndarray, normalize: str) -> np.ndarray:
    input_ids = token_data["input_ids"].cpu().numpy()
    attention_mask = token_data["attention_mask"].cpu().numpy()
    vocab = token_data["vocab"]
    skip = {vocab[token] for token in ("[PAD]", "[CLS]", "[SEP]", "[MASK]") if token in vocab}
    rows = np.zeros((len(indices), len(vocab)), dtype=np.float32)
    for row_idx, data_idx in enumerate(indices.tolist()):
        active = input_ids[data_idx][attention_mask[data_idx] > 0]
        active = np.array([token_id for token_id in active if int(token_id) not in skip], dtype=np.int64)
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


def _cosine_distance(features: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    proto_norm = float(np.linalg.norm(prototype))
    if proto_norm == 0.0:
        return np.ones(features.shape[0], dtype=np.float32)
    feature_norms = np.linalg.norm(features, axis=1)
    sim = np.divide(features @ prototype, feature_norms * proto_norm, out=np.zeros(features.shape[0], dtype=np.float32), where=(feature_norms * proto_norm) > 0)
    return 1.0 - sim


def _score_rows(token_data: dict[str, Any], indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    meta_rows = token_data.get("meta", [])
    for data_idx, label, pred, score in zip(indices.tolist(), y_true.tolist(), y_pred.tolist(), scores.tolist()):
        meta = meta_rows[data_idx] if data_idx < len(meta_rows) else {}
        rows.append(
            {
                "index": int(data_idx),
                "flow_id": meta.get("flow_id"),
                "split": meta.get("split"),
                "label": meta.get("label"),
                "binary_label": int(label),
                "prediction": int(pred),
                "anomaly_score": float(score),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--config", default="configs/anomaly.yaml")
    parser.add_argument("--normalize", choices=["none", "l1", "l2"], default="l2")
    parser.add_argument("--threshold_percentile", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config) if Path(args.config).exists() else {}
    anomaly_cfg = cfg.get("anomaly", {})
    percentile = float(args.threshold_percentile if args.threshold_percentile is not None else anomaly_cfg.get("threshold_percentile", 99))
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    labels = token_data["binary_labels"].cpu().numpy()
    train_idx = _indices(token_data, "train")
    val_idx = _indices(token_data, "val")
    test_idx = _indices(token_data, "test")
    benign_train_idx = train_idx[labels[train_idx] == 0]
    benign_val_idx = val_idx[labels[val_idx] == 0]
    if benign_train_idx.size == 0:
        raise ValueError("No benign train rows available for benign-only anomaly prototype")
    if benign_val_idx.size == 0:
        raise ValueError("No benign val rows available for val-benign threshold calibration")

    train_features = _histograms(token_data, benign_train_idx, normalize=args.normalize)
    val_features = _histograms(token_data, benign_val_idx, normalize=args.normalize)
    test_features = _histograms(token_data, test_idx, normalize=args.normalize)
    prototype = train_features.mean(axis=0)
    val_scores = _cosine_distance(val_features, prototype)
    threshold = float(np.percentile(val_scores, percentile))
    test_scores = _cosine_distance(test_features, prototype)
    y_true = labels[test_idx]
    y_pred = (test_scores >= threshold).astype(int)
    metrics = classification_metrics(y_true.tolist(), y_pred.tolist(), np.column_stack([1.0 - test_scores, test_scores]))
    metrics.update(
        {
            "threshold": threshold,
            "threshold_source": "val_benign",
            "threshold_percentile": percentile,
            "prototype_source": "train_benign",
            "num_train": int(len(train_idx)),
            "num_train_benign_used": int(len(benign_train_idx)),
            "num_val": int(len(val_idx)),
            "num_val_benign_used": int(len(benign_val_idx)),
            "num_test": int(len(test_idx)),
            "normalize": args.normalize,
            "train_only": True,
        }
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics, out_dir / "metrics.json")
    write_json(
        classification_report(y_true, y_pred, labels=[0, 1], target_names=["BENIGN", "ATTACK"], output_dict=True, zero_division=0),
        out_dir / "classification_report.json",
    )
    write_json(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(), out_dir / "confusion_matrix.json")
    write_jsonl(_score_rows(token_data, test_idx, y_true, y_pred, test_scores), out_dir / "scores.jsonl")
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "tokens": args.tokens,
            "config_path": args.config,
            "prototype_source": "train_benign",
            "threshold_source": "val_benign",
            "train_only": True,
        },
        out_dir / "run_meta.json",
    )
    print(metrics)


if __name__ == "__main__":
    main()
