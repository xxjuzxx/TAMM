#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import classification_metrics, confusion
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.training.classifier_trainer import _split_indices
from src.utils.io import read_yaml, write_json


def _name_from_path(path: str) -> str:
    name = Path(path).stem
    prefix = "cicids2017_tokens_2k_behavior_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _split(token_data: dict, cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    labels = token_data["binary_labels"].numpy()
    order_values = np.array([float(meta.get("start_ts") or idx) for idx, meta in enumerate(token_data.get("meta", []))])
    _, val_idx, test_idx = _split_indices(
        labels,
        val_ratio=float(cfg.get("training", {}).get("val_ratio", 0.1)),
        test_ratio=float(cfg.get("training", {}).get("test_ratio", 0.2)),
        seed=int(cfg.get("split_seed", cfg.get("seed", 42))),
        split=split,
        order_values=order_values,
    )
    return val_idx, test_idx


def _loader(token_data: dict, indices: np.ndarray, batch_size: int) -> DataLoader:
    idx = torch.tensor(indices, dtype=torch.long)
    return DataLoader(
        TensorDataset(
            token_data["input_ids"][idx],
            token_data["attention_mask"][idx],
            token_data["token_type_ids"][idx],
            token_data["binary_labels"][idx],
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def _model(token_data: dict, cfg: dict, checkpoint: str, device: torch.device) -> BehaviorComposer:
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


def _predict(model: BehaviorComposer, loader: DataLoader, device: torch.device) -> tuple[list[int], np.ndarray]:
    y_true: list[int] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in loader:
            logits = model(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
            y_true.extend(labels.tolist())
            scores.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return y_true, np.concatenate(scores, axis=0)


def _best_threshold(y_true: list[int], scores: np.ndarray, metric: str) -> tuple[float, float]:
    best_threshold = 0.5
    best_value = -1.0
    for threshold in np.linspace(0.001, 0.999, 999):
        pred = (scores[:, 1] >= threshold).astype(int).tolist()
        metrics = classification_metrics(y_true, pred, scores)
        value = float(metrics[metric])
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold, best_value


def _metrics_at(y_true: list[int], scores: np.ndarray, threshold: float) -> dict:
    pred = (scores[:, 1] >= threshold).astype(int).tolist()
    out = classification_metrics(y_true, pred, scores)
    out["threshold"] = float(threshold)
    out["confusion_matrix"] = confusion(y_true, pred)
    return out


def _best_maximin_threshold(items: list[dict], metric: str) -> tuple[float, float]:
    best_threshold = 0.5
    best_value = -1.0
    best_mean = -1.0
    for threshold in np.linspace(0.001, 0.999, 999):
        values = []
        for item in items:
            pred = (item["val_scores"][:, 1] >= threshold).astype(int).tolist()
            values.append(float(classification_metrics(item["val_true"], pred, item["val_scores"])[metric]))
        min_value = float(min(values))
        mean_value = float(np.mean(values))
        if min_value > best_value or (min_value == best_value and mean_value > best_mean):
            best_value = min_value
            best_mean = mean_value
            best_threshold = float(threshold)
    return best_threshold, best_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", action="append", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--calibration_names", nargs="*", default=None)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--split", choices=["stratified", "chronological", "temporal_stratified"], default="temporal_stratified")
    parser.add_argument("--fixed_threshold", type=float, default=None)
    parser.add_argument("--metric", choices=["macro_f1", "weighted_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--pooled_policy", choices=["pooled_samples", "maximin"], default="pooled_samples")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    train_cfg = cfg.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 64))
    names = args.names or [_name_from_path(path) for path in args.tokens]
    if len(names) != len(args.tokens):
        raise ValueError("--names length must match --tokens length")
    if args.calibration_names == []:
        raise ValueError("--calibration_names must include at least one name when provided")

    datasets = []
    for name, path in zip(names, args.tokens):
        token_data = torch.load(path, map_location="cpu", weights_only=False)
        val_idx, test_idx = _split(token_data, cfg, args.split)
        datasets.append({"name": name, "path": path, "data": token_data, "val_idx": val_idx, "test_idx": test_idx})

    base_vocab = datasets[0]["data"]["vocab"]
    for item in datasets[1:]:
        if item["data"]["vocab"] != base_vocab:
            raise ValueError(f"vocab mismatch for {item['path']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(datasets[0]["data"], cfg, args.checkpoint, device)

    pooled_val_true: list[int] = []
    pooled_val_scores: list[np.ndarray] = []
    calibration_name_set = set(args.calibration_names or names)
    unknown_calibration_names = sorted(calibration_name_set - set(names))
    if unknown_calibration_names:
        raise ValueError(f"unknown calibration names: {unknown_calibration_names}")
    for item in datasets:
        val_true, val_scores = _predict(model, _loader(item["data"], item["val_idx"], batch_size), device)
        test_true, test_scores = _predict(model, _loader(item["data"], item["test_idx"], batch_size), device)
        item["val_true"] = val_true
        item["val_scores"] = val_scores
        item["test_true"] = test_true
        item["test_scores"] = test_scores
        if item["name"] in calibration_name_set:
            pooled_val_true.extend(val_true)
            pooled_val_scores.append(val_scores)

    calibration_items = [item for item in datasets if item["name"] in calibration_name_set]
    if args.pooled_policy == "maximin":
        pooled_threshold, pooled_value = _best_maximin_threshold(calibration_items, args.metric)
    else:
        pooled_scores = np.concatenate(pooled_val_scores, axis=0)
        pooled_threshold, pooled_value = _best_threshold(pooled_val_true, pooled_scores, args.metric)

    rows = []
    for item in datasets:
        own_threshold, own_value = _best_threshold(item["val_true"], item["val_scores"], args.metric)
        row = {
            "name": item["name"],
            "path": item["path"],
            "num_val": int(len(item["val_idx"])),
            "num_test": int(len(item["test_idx"])),
            "own_val_threshold": own_threshold,
            f"own_val_{args.metric}": own_value,
            "own_val_calibrated": _metrics_at(item["test_true"], item["test_scores"], own_threshold),
            "pooled_val_threshold": pooled_threshold,
            f"pooled_val_{args.metric}": pooled_value,
            "pooled_val_calibrated": _metrics_at(item["test_true"], item["test_scores"], pooled_threshold),
        }
        if args.fixed_threshold is not None:
            row["fixed_threshold"] = args.fixed_threshold
            row["fixed_threshold_metrics"] = _metrics_at(item["test_true"], item["test_scores"], args.fixed_threshold)
        rows.append(row)

    out = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "metric": args.metric,
        "pooled_policy": args.pooled_policy,
        "calibration_names": [name for name in names if name in calibration_name_set],
        "pooled_val_threshold": pooled_threshold,
        f"pooled_val_{args.metric}": pooled_value,
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, out_path)
    summary = {
        row["name"]: {
            "own_threshold": row["own_val_threshold"],
            "own_macro_f1": row["own_val_calibrated"]["macro_f1"],
            "pooled_macro_f1": row["pooled_val_calibrated"]["macro_f1"],
            "fixed_macro_f1": row.get("fixed_threshold_metrics", {}).get("macro_f1"),
        }
        for row in rows
    }
    print(summary)


if __name__ == "__main__":
    main()
