#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shlex
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.training.classifier_trainer import FocalLoss
from src.training.classifier_trainer import LogitAdjustedCrossEntropyLoss
from src.training.classifier_trainer import ContextAwareClassifier
from src.training.classifier_trainer import DEFAULT_STAT_FEATURE_NAMES
from src.training.classifier_trainer import _anomaly_feature_names
from src.training.classifier_trainer import _binary_class_weights_for_parts
from src.training.classifier_trainer import _binary_targets
from src.training.classifier_trainer import _class_counts_from_labels
from src.training.classifier_trainer import _class_priors_from_counts
from src.training.classifier_trainer import _class_weights
from src.training.classifier_trainer import _dataset_from_parts_with_features
from src.training.classifier_trainer import _effective_number_weights_from_counts
from src.training.classifier_trainer import _evaluate
from src.training.classifier_trainer import _fit_prototype_bank
from src.training.classifier_trainer import _fit_stat_normalizer
from src.training.classifier_trainer import _loader
from src.training.classifier_trainer import _loader_from_parts_with_features
from src.training.classifier_trainer import _maybe_benign_label_id
from src.training.classifier_trainer import _split_indices
from src.training.classifier_trainer import supervised_contrastive_loss
from src.utils.io import read_yaml, write_json
from src.utils.seed import set_seed


def _encoder(token_data: dict[str, Any], config: dict[str, Any], num_classes: int) -> BehaviorComposer:
    model_cfg = config.get("model", {})
    return BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=num_classes,
        max_seq_len=int(model_cfg.get("max_seq_len", token_data.get("max_len", 256))),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        intermediate_size=int(model_cfg.get("intermediate_size", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        **resolve_pooling_config(model_cfg),
    )


def _target_names(token_data: dict[str, Any]) -> list[str]:
    inv = {idx: label for label, idx in token_data["label_to_id"].items()}
    return [inv[idx] for idx in range(len(inv))]


def _external_loader(token_data: dict[str, Any], labels: torch.Tensor, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(
            token_data["input_ids"],
            token_data["attention_mask"],
            token_data["token_type_ids"],
            labels,
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def _order_values(token_data: dict[str, Any], labels: torch.Tensor) -> np.ndarray:
    meta_rows = token_data.get("meta", [])
    if len(meta_rows) == len(labels):
        return np.array([float(meta.get("start_ts") or idx) for idx, meta in enumerate(meta_rows)])
    return np.arange(len(labels), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tokens", required=True)
    parser.add_argument("--test_tokens", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--split", choices=["stratified", "chronological", "temporal_stratified"], default="stratified")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    cfg = copy.deepcopy(cfg)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    train_data = torch.load(args.train_tokens, map_location="cpu", weights_only=False)
    test_data = torch.load(args.test_tokens, map_location="cpu", weights_only=False)
    if train_data["label_to_id"] != test_data["label_to_id"]:
        raise ValueError("train/test label_to_id mismatch; build test tokens with the train label vocabulary")
    if train_data["vocab"] != test_data["vocab"]:
        raise ValueError("train/test vocab mismatch; build test tokens with --base_vocab from the train token file")

    labels = train_data["labels"]
    labels_np = labels.numpy()
    num_classes = int(labels.max().item()) + 1
    train_idx, val_idx, _ = _split_indices(
        labels_np,
        val_ratio=float(train_cfg.get("val_ratio", 0.1)),
        test_ratio=float(train_cfg.get("test_ratio", 0.2)),
        seed=seed,
        split=args.split,
        order_values=_order_values(train_data, labels),
    )
    batch_size = int(train_cfg.get("batch_size", 64))

    use_anomaly_features = bool(model_cfg.get("use_anomaly_features", False))
    use_stat_fusion = bool(model_cfg.get("use_stat_fusion", False))
    use_app_projection = bool(model_cfg.get("use_app_projection", train_cfg.get("use_app_projection", False)))
    app_projection_dim = int(model_cfg.get("app_projection_dim", train_cfg.get("app_projection_dim", model_cfg.get("hidden_size", 128))))
    use_hierarchical_classifier = bool(train_cfg.get("use_hierarchical_classifier", False))
    hierarchical_gated_inference = bool(train_cfg.get("hierarchical_gated_inference", False))
    lambda_binary = float(train_cfg.get("lambda_binary", 1.0))
    lambda_coarse = float(train_cfg.get("lambda_coarse", 1.0))
    use_supcon = bool(train_cfg.get("use_supcon", False))
    lambda_supcon = float(train_cfg.get("supcon_weight", train_cfg.get("lambda_supcon", 0.0)))
    supcon_temperature = float(train_cfg.get("supcon_temperature", 0.1))
    detach_shared_for_aux_loss = bool(train_cfg.get("detach_shared_for_aux_loss", False))
    use_hierarchical_binary_loss = bool(use_hierarchical_classifier and lambda_binary > 0.0)
    benign_label_id = _maybe_benign_label_id(train_data, "multiclass", train_data.get("label_to_id"))
    if use_hierarchical_classifier and (use_hierarchical_binary_loss or hierarchical_gated_inference) and benign_label_id is None:
        raise ValueError("Could not resolve benign label id for BENIGN-gated hierarchical transfer")
    binary_benign_label_id = int(benign_label_id) if benign_label_id is not None else 0

    anomaly_cfg = {
        **train_cfg.get("anomaly_features", {}),
        "use_service_prototype_distance": model_cfg.get("use_service_prototype_distance", train_cfg.get("anomaly_features", {}).get("use_service_prototype_distance", False)),
        "use_class_prototype_distance": model_cfg.get("use_class_prototype_distance", train_cfg.get("anomaly_features", {}).get("use_class_prototype_distance", False)),
        "anomaly_feature_dim": model_cfg.get("anomaly_feature_dim", train_cfg.get("anomaly_features", {}).get("anomaly_feature_dim", 0)),
    }
    use_service_prototype_distance = bool(anomaly_cfg.get("use_service_prototype_distance", False)) and use_anomaly_features
    use_class_prototype_distance = bool(anomaly_cfg.get("use_class_prototype_distance", False)) and use_anomaly_features
    anomaly_score_method = str(anomaly_cfg.get("score_method", "cosine"))
    anomaly_normalize = str(anomaly_cfg.get("normalize", "l2"))
    anomaly_include_special = bool(anomaly_cfg.get("include_special", False))
    anomaly_feature_dim = int(anomaly_cfg.get("anomaly_feature_dim", 0))
    anomaly_bank = None
    if use_anomaly_features:
        anomaly_bank = _fit_prototype_bank(
            train_data,
            labels,
            train_idx,
            "multiclass",
            train_data.get("label_to_id"),
            include_special=anomaly_include_special,
            normalize=anomaly_normalize,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
            score_method=anomaly_score_method,
        )

    stat_feature_names = list(model_cfg.get("stat_feature_names") or DEFAULT_STAT_FEATURE_NAMES)
    stat_mlp_dim = int(model_cfg.get("stat_mlp_dim", 16))
    stat_normalizer = _fit_stat_normalizer(train_data, train_idx, stat_feature_names) if use_stat_fusion else None
    train_parts = [(train_data, labels, train_idx)]
    val_parts = [(train_data, labels, val_idx)]
    external_parts = [(test_data, test_data["labels"], np.arange(len(test_data["labels"]), dtype=np.int64))]
    if use_anomaly_features or use_stat_fusion:
        train_dataset = _dataset_from_parts_with_features(
            train_parts,
            use_service_context=False,
            use_anomaly_features=use_anomaly_features,
            use_stat_fusion=use_stat_fusion,
            bank=anomaly_bank,
            stat_normalizer=stat_normalizer,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = _loader_from_parts_with_features(
            val_parts,
            use_service_context=False,
            use_anomaly_features=use_anomaly_features,
            use_stat_fusion=use_stat_fusion,
            bank=anomaly_bank,
            stat_normalizer=stat_normalizer,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
            batch_size=batch_size,
            shuffle=False,
        )
        external_loader = _loader_from_parts_with_features(
            external_parts,
            use_service_context=False,
            use_anomaly_features=use_anomaly_features,
            use_stat_fusion=use_stat_fusion,
            bank=anomaly_bank,
            stat_normalizer=stat_normalizer,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        train_loader = _loader(train_data, labels, train_idx, batch_size=batch_size, shuffle=True)
        val_loader = _loader(train_data, labels, val_idx, batch_size=batch_size, shuffle=False)
        external_loader = _external_loader(test_data, test_data["labels"], batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = _encoder(train_data, cfg, num_classes)
    anomaly_size = 0
    anomaly_feature_dim_resolved = 0
    if use_anomaly_features:
        anomaly_names = _anomaly_feature_names(
            num_classes,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
        )
        anomaly_size = len(anomaly_names)
        anomaly_feature_dim_resolved = anomaly_feature_dim if anomaly_feature_dim > 0 else max(8, min(64, anomaly_size * 2))
    stat_size = len(stat_feature_names) if use_stat_fusion else 0
    stat_mlp_dim_resolved = stat_mlp_dim if use_stat_fusion and stat_mlp_dim > 0 else 0
    if use_anomaly_features or use_stat_fusion or use_hierarchical_classifier or use_app_projection:
        model: nn.Module = ContextAwareClassifier(
            encoder,
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_classes=num_classes,
            context_size=0,
            dropout=float(model_cfg.get("dropout", 0.1)),
            anomaly_size=anomaly_size,
            anomaly_feature_dim=anomaly_feature_dim_resolved,
            use_hierarchical_classifier=use_hierarchical_classifier,
            stat_size=stat_size,
            stat_mlp_dim=stat_mlp_dim_resolved,
            use_app_projection=use_app_projection,
            app_projection_dim=app_projection_dim,
        ).to(device)
    else:
        model = encoder.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    loss_name = str(train_cfg.get("loss_type", train_cfg.get("loss", "ce")))
    if loss_name == "class_balanced":
        loss_name = "class_balanced_ce"
    loss_weights = _class_weights(labels, train_idx, num_classes).to(device)
    binary_loss_weights = _binary_class_weights_for_parts(train_parts, binary_benign_label_id).to(device)
    class_counts = _class_counts_from_labels(labels[torch.tensor(train_idx, dtype=torch.long)], num_classes)
    binary_class_counts = _class_counts_from_labels(_binary_targets(labels[torch.tensor(train_idx, dtype=torch.long)], binary_benign_label_id), 2)
    if loss_name == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        binary_criterion = nn.CrossEntropyLoss(weight=binary_loss_weights)
    elif loss_name == "ce":
        criterion = nn.CrossEntropyLoss()
        binary_criterion = nn.CrossEntropyLoss()
    elif loss_name == "focal":
        criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)))
        binary_criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)))
    elif loss_name == "weighted_focal":
        criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)), weight=loss_weights)
        binary_criterion = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)), weight=binary_loss_weights)
    elif loss_name == "class_balanced_ce":
        cb_beta = float(train_cfg.get("cb_beta", 0.9999))
        criterion = nn.CrossEntropyLoss(weight=_effective_number_weights_from_counts(class_counts, cb_beta).to(device))
        binary_criterion = nn.CrossEntropyLoss(weight=_effective_number_weights_from_counts(binary_class_counts, cb_beta).to(device))
    elif loss_name == "logit_adjusted_ce":
        tau = float(train_cfg.get("logit_adjust_tau", 1.0))
        criterion = LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(class_counts).to(device), tau=tau)
        binary_criterion = LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(binary_class_counts).to(device), tau=tau)
    elif loss_name == "weighted_logit_adjusted_ce":
        tau = float(train_cfg.get("logit_adjust_tau", 1.0))
        criterion = LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(class_counts).to(device), tau=tau, weight=loss_weights)
        binary_criterion = LogitAdjustedCrossEntropyLoss(_class_priors_from_counts(binary_class_counts).to(device), tau=tau, weight=binary_loss_weights)
    else:
        raise ValueError(f"Unsupported training loss: {loss_name}")

    best_state = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, int(train_cfg.get("epochs", 3)) + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            input_ids, attention_mask, token_type_ids = batch[:3]
            batch_labels = batch[-1]
            cursor = 3
            anomaly_features = None
            stat_features = None
            if use_anomaly_features:
                anomaly_features = batch[cursor]
                cursor += 1
            if use_stat_fusion:
                stat_features = batch[cursor]
            optimizer.zero_grad(set_to_none=True)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            batch_labels = batch_labels.to(device)
            anomaly_arg = anomaly_features.to(device) if anomaly_features is not None else None
            stat_arg = stat_features.to(device) if stat_features is not None else None
            if use_hierarchical_classifier:
                binary_logits, logits = model.forward_heads(input_ids, attention_mask, token_type_ids, None, anomaly_arg, stat_arg)
                loss = lambda_coarse * criterion(logits, batch_labels)
                if use_hierarchical_binary_loss:
                    loss = loss + lambda_binary * binary_criterion(binary_logits, _binary_targets(batch_labels, binary_benign_label_id))
            elif use_anomaly_features or use_stat_fusion or use_app_projection:
                logits = model(input_ids, attention_mask, token_type_ids, None, anomaly_arg, stat_arg)
                loss = criterion(logits, batch_labels)
            else:
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, batch_labels)
            if use_supcon and lambda_supcon > 0.0:
                if use_app_projection and hasattr(model, "app_features"):
                    reg_features = model.app_features(
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        detach_shared=detach_shared_for_aux_loss,
                    )
                elif hasattr(model, "fused_features"):
                    reg_features = model.fused_features(input_ids, attention_mask, token_type_ids, None, anomaly_arg, stat_arg)
                else:
                    reg_features = model.encode(input_ids, attention_mask, token_type_ids)
                loss = loss + lambda_supcon * supervised_contrastive_loss(reg_features, batch_labels, temperature=supcon_temperature)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, val_true, val_pred, val_score = _evaluate(
            model,
            val_loader,
            device,
            use_hierarchical_classifier=use_hierarchical_classifier,
            use_hierarchical_binary_loss=use_hierarchical_binary_loss,
            lambda_binary=lambda_binary,
            lambda_coarse=lambda_coarse,
            benign_label_id=binary_benign_label_id,
            gated_inference=hierarchical_gated_inference,
        )
        val_metrics = classification_metrics(val_true, val_pred, val_score)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses) if losses else 0.0),
                "val_loss": val_loss,
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, y_true, y_pred, y_score = _evaluate(
        model,
        external_loader,
        device,
        use_hierarchical_classifier=use_hierarchical_classifier,
        use_hierarchical_binary_loss=use_hierarchical_binary_loss,
        lambda_binary=lambda_binary,
        lambda_coarse=lambda_coarse,
        benign_label_id=binary_benign_label_id,
        gated_inference=hierarchical_gated_inference,
    )
    metrics = classification_metrics(y_true, y_pred, y_score)
    metrics.update(
        {
            "test_loss": test_loss,
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_external_test": int(len(test_data["labels"])),
            "split": args.split,
            "task": "multiclass_cross_dataset",
            "train_tokens": args.train_tokens,
            "test_tokens": args.test_tokens,
            "device": str(device),
            "use_anomaly_features": use_anomaly_features,
            "use_stat_fusion": use_stat_fusion,
            "use_hierarchical_classifier": use_hierarchical_classifier,
            "use_app_projection": use_app_projection,
            "pooling_strategy": str(getattr(encoder, "pooling_strategy", model_cfg.get("pooling_strategy", "cls"))),
            "class_aware_pooling": bool(getattr(encoder, "class_aware_pooling", False)),
            "loss": loss_name,
            "use_supcon": use_supcon,
        }
    )
    if use_app_projection:
        metrics["app_projection"] = {
            "input": "residual_class_aware_pool" if getattr(encoder, "residual_class_aware", False) else "shared_encode",
            "dim": int(app_projection_dim),
            "detach_shared_for_aux_loss": bool(detach_shared_for_aux_loss),
        }
    if use_supcon:
        metrics["supcon"] = {
            "lambda_supcon": lambda_supcon,
            "supcon_weight": lambda_supcon,
            "temperature": supcon_temperature,
            "feature_space": "app_projection" if use_app_projection else "shared_or_fused",
        }
    if loss_name in {"focal", "weighted_focal"}:
        metrics["focal_gamma"] = float(train_cfg.get("focal_gamma", 2.0))
    if loss_name == "class_balanced_ce":
        metrics["cb_beta"] = float(train_cfg.get("cb_beta", 0.9999))
    if loss_name in {"logit_adjusted_ce", "weighted_logit_adjusted_ce"}:
        metrics["logit_adjust_tau"] = float(train_cfg.get("logit_adjust_tau", 1.0))
    if use_anomaly_features:
        metrics["anomaly_features"] = {
            "score_method": anomaly_score_method,
            "normalize": anomaly_normalize,
            "include_special": anomaly_include_special,
            "prototype_scope": anomaly_bank.prototype_scope if anomaly_bank else None,
            "benign_label_id": anomaly_bank.benign_label_id if anomaly_bank else None,
            "warnings": anomaly_bank.warnings if anomaly_bank else [],
            "use_service_prototype_distance": use_service_prototype_distance,
            "use_class_prototype_distance": use_class_prototype_distance,
            "feature_dim": anomaly_size,
            "projection_dim": anomaly_feature_dim_resolved,
        }
    if use_stat_fusion:
        metrics["stat_fusion"] = {
            "feature_names": stat_feature_names,
            "feature_dim": stat_size,
            "projection_dim": stat_mlp_dim_resolved,
            "missing_or_unavailable_features": stat_normalizer.missing_names if stat_normalizer else [],
        }
    if use_hierarchical_classifier:
        metrics["hierarchical_classifier"] = {
            "lambda_binary": lambda_binary,
            "lambda_coarse": lambda_coarse,
            "gated_inference": hierarchical_gated_inference,
            "binary_loss_enabled": use_hierarchical_binary_loss,
            "benign_label_id": benign_label_id,
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = _target_names(train_data)
    write_json(cfg, out_dir / "resolved_config.json")
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "train_tokens": args.train_tokens,
            "test_tokens": args.test_tokens,
            "config_path": args.config,
            "split": args.split,
            "seed": seed,
        },
        out_dir / "run_manifest.json",
    )
    write_json(metrics, out_dir / "metrics.json")
    write_json(report_dict(y_true, y_pred, target_names=names), out_dir / "classification_report.json")
    write_json(confusion(y_true, y_pred), out_dir / "confusion_matrix.json")
    write_json(history, out_dir / "history.json")
    torch.save(best_state or model.state_dict(), out_dir / "best_model.pt")
    print(metrics)


if __name__ == "__main__":
    main()
