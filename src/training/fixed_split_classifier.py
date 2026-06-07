from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.training.classifier_trainer import (
    FocalLoss,
    LDAMLoss,
    LogitAdjustedCrossEntropyLoss,
    _class_priors_from_counts,
    _effective_number_weights_from_counts,
)
from src.utils.seed import set_seed


@dataclass
class FixedSplitTrainResult:
    metrics: dict[str, Any]
    report: dict[str, Any]
    confusion_matrix: list[list[int]]
    history: list[dict[str, float]]
    state_dict: dict[str, Any]
    predictions: list[dict[str, Any]]
    target_names: list[str]


def _indices_for_split(token_data: dict[str, Any], split_name: str) -> np.ndarray:
    indices = [idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == split_name]
    return np.array(indices, dtype=np.int64)


def _labels(token_data: dict[str, Any], task: str) -> tuple[torch.Tensor, list[str]]:
    if task == "binary":
        mapping = token_data.get("binary_label_to_id", {"BENIGN": 0, "ATTACK": 1})
        inv = {int(value): str(key) for key, value in mapping.items()}
        return token_data["binary_labels"].long(), [inv[idx] for idx in range(len(inv))]
    if task == "multiclass":
        mapping = token_data.get("label_to_id", {})
        inv = {int(value): str(key) for key, value in mapping.items()}
        return token_data["labels"].long(), [inv[idx] for idx in range(len(inv))]
    raise ValueError(f"Unsupported fixed-split task: {task}")


def _balanced_sampler_for_indices(labels: torch.Tensor, indices: np.ndarray, num_classes: int, seed: int) -> WeightedRandomSampler:
    selected = labels[torch.tensor(indices, dtype=torch.long)]
    counts = torch.bincount(selected, minlength=num_classes).float().clamp(min=1.0)
    sample_weights = (1.0 / counts)[selected]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(sample_weights.double(), num_samples=int(len(selected)), replacement=True, generator=generator)


def _class_counts(labels: torch.Tensor, indices: np.ndarray, num_classes: int) -> torch.Tensor:
    selected = labels[torch.tensor(indices, dtype=torch.long)]
    return torch.bincount(selected, minlength=num_classes).float()


def _class_weights(labels: torch.Tensor, indices: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = _class_counts(labels, indices, num_classes).clamp(min=1.0)
    return counts.sum() / (num_classes * counts)


def _criterion_for_training(train_cfg: dict[str, Any], labels: torch.Tensor, train_idx: np.ndarray, num_classes: int, device: torch.device) -> tuple[nn.Module, str]:
    loss_name = str(train_cfg.get("loss_type", train_cfg.get("loss", "ce")))
    if loss_name == "class_balanced":
        loss_name = "class_balanced_ce"
    class_counts = _class_counts(labels, train_idx, num_classes)
    class_weights = _class_weights(labels, train_idx, num_classes).to(device)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(), loss_name
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights), loss_name
    if loss_name == "focal":
        return FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0))), loss_name
    if loss_name == "weighted_focal":
        return FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)), weight=class_weights), loss_name
    if loss_name == "class_balanced_ce":
        cb_beta = float(train_cfg.get("cb_beta", 0.9999))
        return nn.CrossEntropyLoss(weight=_effective_number_weights_from_counts(class_counts, cb_beta).to(device)), loss_name
    if loss_name == "class_balanced_focal":
        cb_beta = float(train_cfg.get("cb_beta", 0.9999))
        return (
            FocalLoss(
                gamma=float(train_cfg.get("focal_gamma", 2.0)),
                weight=_effective_number_weights_from_counts(class_counts, cb_beta).to(device),
            ),
            loss_name,
        )
    if loss_name == "logit_adjusted_ce":
        tau = float(train_cfg.get("logit_adjust_tau", 1.0))
        return LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(class_counts).to(device), tau=tau), loss_name
    if loss_name == "weighted_logit_adjusted_ce":
        tau = float(train_cfg.get("logit_adjust_tau", 1.0))
        return LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(class_counts).to(device), tau=tau, weight=class_weights), loss_name
    if loss_name == "ldam":
        return (
            LDAMLoss(
                class_counts,
                max_m=float(train_cfg.get("ldam_max_m", 0.5)),
                scale=float(train_cfg.get("ldam_scale", 30.0)),
            ),
            loss_name,
        )
    if loss_name == "weighted_ldam":
        return (
            LDAMLoss(
                class_counts,
                max_m=float(train_cfg.get("ldam_max_m", 0.5)),
                scale=float(train_cfg.get("ldam_scale", 30.0)),
                weight=class_weights,
            ),
            loss_name,
        )
    raise ValueError(f"Unsupported fixed-split training loss: {loss_name}")


def _loader(
    token_data: dict[str, Any],
    labels: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    idx = torch.tensor(indices, dtype=torch.long)
    dataset = TensorDataset(
        token_data["input_ids"][idx],
        token_data["attention_mask"][idx],
        token_data["token_type_ids"][idx],
        labels[idx],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler)


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, list[int], list[int], np.ndarray]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            losses.append(float(loss.item()))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            scores.append(probs.cpu().numpy())
    if not scores:
        return 0.0, y_true, y_pred, np.zeros((0, 0), dtype=np.float32)
    return float(np.mean(losses)), y_true, y_pred, np.concatenate(scores, axis=0)


def _best_binary_threshold(y_true: list[int], scores: np.ndarray) -> float:
    if scores.shape[1] < 2:
        return 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (scores[:, 1] >= threshold).astype(int).tolist()
        macro_f1 = classification_metrics(y_true, pred, scores)["macro_f1"]
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = float(threshold)
    return best_threshold


def _prediction_rows(
    token_data: dict[str, Any],
    indices: np.ndarray,
    y_true: list[int],
    y_pred: list[int],
    y_score: np.ndarray,
    target_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta_rows = token_data.get("meta", [])
    for offset, idx in enumerate(indices.tolist()):
        meta = meta_rows[int(idx)] if int(idx) < len(meta_rows) else {}
        true_id = int(y_true[offset])
        pred_id = int(y_pred[offset])
        score_row = y_score[offset]
        rows.append(
            {
                "index": int(idx),
                "flow_id": meta.get("flow_id"),
                "split": meta.get("split"),
                "label": meta.get("label"),
                "binary_label": meta.get("binary_label"),
                "true_label": target_names[true_id] if true_id < len(target_names) else str(true_id),
                "pred_label": target_names[pred_id] if pred_id < len(target_names) else str(pred_id),
                "pred_confidence": float(score_row[pred_id]),
                "scores": {target_names[class_idx]: float(score) for class_idx, score in enumerate(score_row[: len(target_names)])},
            }
        )
    return rows


def train_fixed_split_classifier(
    token_data: dict[str, Any],
    config: dict[str, Any],
    *,
    task: str = "binary",
    train_seed: int | None = None,
    checkpoint: str | None = None,
    freeze_encoder: bool = False,
) -> FixedSplitTrainResult:
    started = time.perf_counter()
    seed = int(config.get("seed", 42))
    resolved_train_seed = int(train_seed if train_seed is not None else config.get("train_seed", seed))
    set_seed(resolved_train_seed)
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    labels, target_names = _labels(token_data, task)
    train_idx = _indices_for_split(token_data, "train")
    val_idx = _indices_for_split(token_data, "val")
    test_idx = _indices_for_split(token_data, "test")
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"fixed split requires non-empty train/val/test, got {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    batch_size = int(train_cfg.get("batch_size", 64))

    num_classes = int(labels.max().item()) + 1
    train_sampling = str(train_cfg.get("sampling", "shuffle"))
    sampler = None
    if train_sampling == "class_balanced":
        sampler = _balanced_sampler_for_indices(labels, train_idx, num_classes, resolved_train_seed)
    elif train_sampling != "shuffle":
        raise ValueError(f"Unsupported fixed-split training sampling: {train_sampling}")
    train_loader = _loader(token_data, labels, train_idx, batch_size=batch_size, shuffle=True, sampler=sampler)
    val_loader = _loader(token_data, labels, val_idx, batch_size=batch_size, shuffle=False)
    test_loader = _loader(token_data, labels, test_idx, batch_size=batch_size, shuffle=False)

    pooling_params = resolve_pooling_config(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = max(int(model_cfg.get("max_seq_len", 0) or 0), int(token_data.get("max_len", 256)))
    model = BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        intermediate_size=int(model_cfg.get("intermediate_size", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        **pooling_params,
    ).to(device)
    checkpoint_info = None
    if checkpoint:
        from src.training.classifier_trainer import _load_encoder_checkpoint

        checkpoint_info = _load_encoder_checkpoint(model, checkpoint, device, freeze_encoder=freeze_encoder)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    criterion, loss_name = _criterion_for_training(train_cfg, labels, train_idx, num_classes, device)
    epochs = int(train_cfg.get("epochs", 3))
    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for input_ids, attention_mask, token_type_ids, batch_labels in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, val_true, val_pred, val_score = _evaluate(model, val_loader, criterion, device)
        val_metrics = classification_metrics(val_true, val_pred, val_score)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": val_loss,
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    _, val_true, _, val_score = _evaluate(model, val_loader, criterion, device)
    test_loss, y_true, y_pred, y_score = _evaluate(model, test_loader, criterion, device)
    threshold = None
    if task == "binary":
        threshold = _best_binary_threshold(val_true, val_score)
        y_pred = (y_score[:, 1] >= threshold).astype(int).tolist()
    metrics = classification_metrics(y_true, y_pred, y_score)
    metrics.update(
        {
            "test_loss": test_loss,
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "split": "predefined",
            "task": task,
            "seed": seed,
            "train_seed": resolved_train_seed,
            "train_seconds": float(time.perf_counter() - started),
            "device": str(device),
            "vocab_size": int(len(token_data["vocab"])),
            "max_len": int(token_data.get("max_len", 0)),
            "model_max_seq_len": int(max_seq_len),
            "loss": loss_name,
            "sampling": train_sampling,
        }
    )
    if loss_name in {"focal", "weighted_focal", "class_balanced_focal"}:
        metrics["focal_gamma"] = float(train_cfg.get("focal_gamma", 2.0))
    if loss_name in {"class_balanced_ce", "class_balanced_focal"}:
        metrics["cb_beta"] = float(train_cfg.get("cb_beta", 0.9999))
    if loss_name in {"logit_adjusted_ce", "weighted_logit_adjusted_ce"}:
        metrics["logit_adjust_tau"] = float(train_cfg.get("logit_adjust_tau", 1.0))
    if loss_name in {"ldam", "weighted_ldam"}:
        metrics["ldam_max_m"] = float(train_cfg.get("ldam_max_m", 0.5))
        metrics["ldam_scale"] = float(train_cfg.get("ldam_scale", 30.0))
    if checkpoint_info is not None:
        metrics["checkpoint"] = checkpoint_info
    if threshold is not None:
        metrics["threshold"] = threshold
    return FixedSplitTrainResult(
        metrics=metrics,
        report=report_dict(y_true, y_pred, target_names=target_names),
        confusion_matrix=confusion(y_true, y_pred),
        history=history,
        state_dict=best_state or {key: value.detach().cpu() for key, value in model.state_dict().items()},
        predictions=_prediction_rows(token_data, test_idx, y_true, y_pred, y_score, target_names),
        target_names=target_names,
    )
