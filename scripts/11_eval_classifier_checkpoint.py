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
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.training.classifier_trainer import (
    ContextAwareClassifier,
    DEFAULT_STAT_FEATURE_NAMES,
    _anomaly_feature_rows,
    _fit_prototype_bank,
    _fit_stat_normalizer,
    _raw_label_group_ids,
    _split_indices,
    _stat_feature_rows,
    _task_labels,
    _temporal_stratified_by_group_indices,
)
from src.utils.io import read_yaml, write_json, write_jsonl


def _state_dict_from_checkpoint(checkpoint: str, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)!r}")
    return state_dict


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


def _context_matrix(meta_rows: list[dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for idx in indices.tolist():
        meta = meta_rows[int(idx)] if int(idx) < len(meta_rows) else {}
        context = meta.get("service_context") or {}
        rows.append(
            [
                float(context.get("recent_count", 0.0)),
                float(context.get("recent_short", 0.0)),
                float(context.get("recent_packets", 0.0)),
                float(context.get("short_ratio", 0.0)),
                float(context.get("last_gap") or 0.0),
                float(context.get("recent_count", 0.0) > 0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _prepare_vocab(token_data: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    vocab = dict(token_data["vocab"])
    embedding_weight = state_dict.get("token_embedding.weight")
    if embedding_weight is None:
        embedding_weight = state_dict.get("encoder.token_embedding.weight")
    if embedding_weight is None:
        return vocab
    target_size = int(embedding_weight.shape[0])
    if len(vocab) > target_size:
        raise ValueError(
            f"checkpoint vocab is smaller than token vocab: checkpoint={target_size}, token_data={len(vocab)}"
        )
    if len(vocab) < target_size:
        pad_start = len(vocab)
        for idx in range(pad_start, target_size):
            vocab[f"[CTX_{idx - pad_start}]"] = idx
    return vocab


def _remap_input_ids(input_ids: torch.Tensor, source_vocab: dict[str, int], target_vocab: dict[str, int]) -> torch.Tensor:
    if source_vocab == target_vocab:
        return input_ids
    source_tokens = [None] * len(source_vocab)
    for token, idx in source_vocab.items():
        if 0 <= int(idx) < len(source_tokens):
            source_tokens[int(idx)] = str(token)
    unk_id = int(target_vocab.get("[UNK]", 0))
    remap = torch.full((len(source_tokens),), unk_id, dtype=torch.long)
    for idx, token in enumerate(source_tokens):
        if token is None:
            continue
        remap[idx] = int(target_vocab.get(token, unk_id))
    return remap[input_ids]


def _checkpoint_feature_specs(state_dict: dict[str, torch.Tensor]) -> dict[str, int | bool]:
    use_service_context = any(str(key).startswith("context_proj.") for key in state_dict)
    use_anomaly_features = any(str(key).startswith("anomaly_proj.") for key in state_dict)
    use_stat_fusion = any(str(key).startswith("stat_proj.") for key in state_dict)
    use_hierarchical_classifier = any(str(key).startswith("binary_classifier.") for key in state_dict)
    use_app_projection = any(str(key).startswith("app_projection.") for key in state_dict)
    context_size = 0
    anomaly_size = 0
    anomaly_feature_dim = 0
    stat_size = 0
    stat_mlp_dim = 0
    app_projection_dim = 0
    if use_service_context and "context_proj.0.weight" in state_dict:
        context_size = int(state_dict["context_proj.0.weight"].shape[0])
    if use_anomaly_features and "anomaly_proj.0.weight" in state_dict:
        anomaly_size = int(state_dict["anomaly_proj.0.weight"].shape[0])
    if use_anomaly_features and "anomaly_proj.1.weight" in state_dict:
        anomaly_feature_dim = int(state_dict["anomaly_proj.1.weight"].shape[0])
    if use_stat_fusion and "stat_proj.0.weight" in state_dict:
        stat_size = int(state_dict["stat_proj.0.weight"].shape[0])
    if use_stat_fusion and "stat_proj.1.weight" in state_dict:
        stat_mlp_dim = int(state_dict["stat_proj.1.weight"].shape[0])
    if use_app_projection and "app_projection.3.weight" in state_dict:
        app_projection_dim = int(state_dict["app_projection.3.weight"].shape[0])
    elif use_app_projection and "app_projection.0.weight" in state_dict:
        app_projection_dim = int(state_dict["app_projection.0.weight"].shape[0])
    return {
        "use_service_context": use_service_context,
        "use_anomaly_features": use_anomaly_features,
        "use_stat_fusion": use_stat_fusion,
        "use_hierarchical_classifier": use_hierarchical_classifier,
        "use_app_projection": use_app_projection,
        "context_size": context_size,
        "anomaly_size": anomaly_size,
        "anomaly_feature_dim": anomaly_feature_dim,
        "stat_size": stat_size,
        "stat_mlp_dim": stat_mlp_dim,
        "app_projection_dim": app_projection_dim,
    }


def _split_for_task(
    token_data: dict[str, Any],
    labels_np: np.ndarray,
    split: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    meta_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order_values = np.array([float(meta.get("start_ts") or idx) for idx, meta in enumerate(meta_rows)])
    if split == "temporal_stratified_raw_label":
        return _temporal_stratified_by_group_indices(
            labels_np,
            _raw_label_group_ids(token_data),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            order_values=order_values,
        )
    return _split_indices(
        labels_np,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        split=split,
        order_values=order_values,
    )


def _target_names(task: str, label_to_id: dict[str, int] | None) -> list[str]:
    if task == "binary":
        return ["BENIGN", "ATTACK"]
    if label_to_id is None:
        raise ValueError(f"{task} requires a label mapping")
    inv = {idx: label for label, idx in label_to_id.items()}
    return [inv[idx] for idx in range(len(inv))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--split_tokens", default=None, help="Clean token dataset used only to compute split indices.")
    parser.add_argument("--metadata_tokens", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--task", choices=["binary", "multiclass", "multiclass_merged"], default="binary")
    parser.add_argument(
        "--split",
        choices=["stratified", "chronological", "temporal_stratified", "temporal_stratified_raw_label"],
        default="temporal_stratified",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval_split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    split_token_data = torch.load(args.split_tokens, map_location="cpu", weights_only=False) if args.split_tokens else token_data
    if len(split_token_data["input_ids"]) != len(token_data["input_ids"]):
        raise ValueError(
            f"--split_tokens row count must match --tokens for checkpoint evaluation: "
            f"split={len(split_token_data['input_ids'])}, tokens={len(token_data['input_ids'])}"
        )
    metadata_data = torch.load(args.metadata_tokens, map_location="cpu", weights_only=False) if args.metadata_tokens else None
    meta_rows = _merged_metadata_rows(token_data, metadata_data)
    split_meta_rows = _merged_metadata_rows(split_token_data, None)
    labels, label_to_id = _task_labels(split_token_data, args.task)
    labels_np = labels.numpy()
    train_idx, val_idx, test_idx = _split_for_task(
        split_token_data,
        labels_np,
        val_ratio=float(train_cfg.get("val_ratio", 0.1)),
        test_ratio=float(train_cfg.get("test_ratio", 0.2)),
        seed=int(cfg.get("seed", 42)),
        split=args.split,
        meta_rows=split_meta_rows,
    )
    if args.eval_split == "train":
        eval_idx = train_idx
    elif args.eval_split == "val":
        eval_idx = val_idx
    else:
        eval_idx = test_idx
    idx = torch.tensor(eval_idx, dtype=torch.long)
    eval_labels, eval_label_to_id = _task_labels(token_data, args.task)
    if label_to_id != eval_label_to_id:
        raise ValueError(f"Label mapping mismatch between --split_tokens and --tokens: {label_to_id} != {eval_label_to_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = _state_dict_from_checkpoint(args.checkpoint, device)
    source_vocab = dict(token_data["vocab"])
    feature_specs = _checkpoint_feature_specs(state_dict)
    use_service_context = bool(feature_specs["use_service_context"])
    use_anomaly_features = bool(feature_specs["use_anomaly_features"])
    use_stat_fusion = bool(feature_specs["use_stat_fusion"])
    use_hierarchical_classifier = bool(feature_specs["use_hierarchical_classifier"])
    use_app_projection = bool(feature_specs["use_app_projection"])
    target_vocab = dict(metadata_data["vocab"]) if (use_service_context and metadata_data is not None) else source_vocab
    vocab = _prepare_vocab({"vocab": target_vocab}, state_dict)
    anomaly_cfg = train_cfg.get("anomaly_features", {})
    anomaly_score_method = str(anomaly_cfg.get("score_method", model_cfg.get("anomaly_score_method", "cosine")))
    anomaly_normalize = str(anomaly_cfg.get("normalize", model_cfg.get("anomaly_normalize", "l2")))
    anomaly_include_special = bool(anomaly_cfg.get("include_special", model_cfg.get("anomaly_include_special", False)))
    use_service_prototype_distance = bool(
        model_cfg.get(
            "use_service_prototype_distance",
            anomaly_cfg.get("use_service_prototype_distance", False),
        )
    )
    use_class_prototype_distance = bool(
        model_cfg.get(
            "use_class_prototype_distance",
            anomaly_cfg.get("use_class_prototype_distance", False),
        )
    )
    anomaly_feature_dim = int(feature_specs["anomaly_feature_dim"] or model_cfg.get("anomaly_feature_dim", 0))
    pooling_params = resolve_pooling_config(model_cfg)
    pooling_strategy = str(pooling_params["pooling_strategy"])
    class_aware_pooling = bool(pooling_params["class_aware_pooling"])
    stat_feature_names = list(model_cfg.get("stat_feature_names") or DEFAULT_STAT_FEATURE_NAMES)
    if use_stat_fusion and len(stat_feature_names) != int(feature_specs["stat_size"]):
        stat_feature_names = stat_feature_names[: int(feature_specs["stat_size"])]
        while len(stat_feature_names) < int(feature_specs["stat_size"]):
            stat_feature_names.append(f"stat_{len(stat_feature_names)}")
    encoder = BehaviorComposer(
        vocab_size=len(vocab),
        num_classes=int(labels.max().item()) + 1,
        max_seq_len=int(model_cfg.get("max_seq_len", token_data.get("max_len", 256))),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        intermediate_size=int(model_cfg.get("intermediate_size", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        **pooling_params,
    )
    anomaly_bank = None
    if use_anomaly_features:
        anomaly_bank = _fit_prototype_bank(
            split_token_data,
            labels,
            train_idx,
            args.task,
            label_to_id,
            include_special=anomaly_include_special,
            normalize=anomaly_normalize,
            use_service_prototype_distance=use_service_prototype_distance,
            use_class_prototype_distance=use_class_prototype_distance,
            score_method=anomaly_score_method,
        )
    stat_normalizer = None
    if use_stat_fusion:
        stat_normalizer = _fit_stat_normalizer(split_token_data, train_idx, stat_feature_names)
    if use_service_context or use_anomaly_features or use_stat_fusion or use_hierarchical_classifier or use_app_projection:
        model = ContextAwareClassifier(
            encoder,
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_classes=int(labels.max().item()) + 1,
            context_size=int(feature_specs["context_size"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            anomaly_size=int(feature_specs["anomaly_size"]),
            anomaly_feature_dim=int(anomaly_feature_dim or max(8, min(64, int(feature_specs["anomaly_size"]) * 2 if feature_specs["anomaly_size"] else 8))),
            use_hierarchical_classifier=use_hierarchical_classifier,
            stat_size=int(feature_specs["stat_size"]),
            stat_mlp_dim=int(feature_specs["stat_mlp_dim"]),
            use_app_projection=use_app_projection,
            app_projection_dim=int(feature_specs["app_projection_dim"] or model_cfg.get("app_projection_dim", model_cfg.get("hidden_size", 128))),
        ).to(device)
    else:
        model = encoder.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    if use_service_context or use_anomaly_features or use_stat_fusion:
        if use_service_context and not any((row.get("service_context") is not None) for row in meta_rows):
            raise ValueError(
                "checkpoint expects service_context features, but neither --tokens nor --metadata_tokens contains them"
            )
        input_ids = _remap_input_ids(token_data["input_ids"][idx], source_vocab, target_vocab)
        tensors: list[torch.Tensor] = [
            input_ids,
            token_data["attention_mask"][idx],
            token_data["token_type_ids"][idx],
        ]
        if use_service_context:
            tensors.append(torch.tensor(_context_matrix(meta_rows, eval_idx), dtype=torch.float32))
        if use_anomaly_features:
            assert anomaly_bank is not None
            tensors.append(
                torch.tensor(
                    _anomaly_feature_rows(
                        token_data,
                        eval_idx,
                        anomaly_bank,
                        use_service_prototype_distance=use_service_prototype_distance,
                        use_class_prototype_distance=use_class_prototype_distance,
                    ),
                    dtype=torch.float32,
                )
            )
        if use_stat_fusion:
            assert stat_normalizer is not None
            tensors.append(torch.tensor(_stat_feature_rows(token_data, eval_idx, stat_normalizer), dtype=torch.float32))
        tensors.append(eval_labels[idx])
        loader = DataLoader(TensorDataset(*tensors), batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False)
    else:
        loader = DataLoader(
            TensorDataset(
                token_data["input_ids"][idx],
                token_data["attention_mask"][idx],
                token_data["token_type_ids"][idx],
                eval_labels[idx],
            ),
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=False,
        )
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[np.ndarray] = []
    row_indices: list[int] = []
    with torch.no_grad():
        row_cursor = 0
        for batch in loader:
            input_ids, attention_mask, token_type_ids = batch[:3]
            batch_labels = batch[-1]
            feature_cursor = 3
            batch_context = None
            batch_anomaly = None
            batch_stats = None
            if use_service_context:
                batch_context = batch[feature_cursor]
                feature_cursor += 1
            if use_anomaly_features:
                batch_anomaly = batch[feature_cursor]
                feature_cursor += 1
            if use_stat_fusion:
                batch_stats = batch[feature_cursor]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            context_arg = batch_context.to(device) if batch_context is not None else None
            anomaly_arg = batch_anomaly.to(device) if batch_anomaly is not None else None
            stat_arg = batch_stats.to(device) if batch_stats is not None else None
            if use_hierarchical_classifier:
                binary_logits, logits = model.forward_heads(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    context_arg,
                    anomaly_arg,
                    stat_arg,
                )
                if train_cfg.get("hierarchical_gated_inference", False):
                    benign_label_id = 0
                    binary_pred = torch.argmax(binary_logits, dim=-1).cpu().numpy()
                else:
                    binary_pred = None
            elif use_service_context or use_anomaly_features or use_stat_fusion:
                logits = model(input_ids, attention_mask, token_type_ids, context_arg, anomaly_arg, stat_arg)
                binary_pred = None
            else:
                logits = model(input_ids, attention_mask, token_type_ids)
                binary_pred = None
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            y_true.extend(batch_labels.tolist())
            if args.task == "binary":
                y_pred.extend((probs[:, 1] >= args.threshold).astype(int).tolist())
            elif use_hierarchical_classifier and binary_pred is not None:
                pred = np.argmax(probs, axis=1).astype(int)
                attack_logits = probs.copy()
                attack_logits[:, benign_label_id] = -np.inf
                attack_pred = np.argmax(attack_logits, axis=1).astype(int)
                pred = np.where(binary_pred == 0, benign_label_id, attack_pred)
                y_pred.extend(pred.tolist())
            else:
                y_pred.extend(np.argmax(probs, axis=1).astype(int).tolist())
            scores.append(probs)
            batch_size = int(batch_labels.shape[0])
            row_indices.extend(eval_idx[row_cursor : row_cursor + batch_size].tolist())
            row_cursor += batch_size
    y_score = np.concatenate(scores, axis=0)
    metrics = classification_metrics(y_true, y_pred, y_score)
    target_names = _target_names(args.task, label_to_id)
    metrics.update(
        {
            "threshold": args.threshold if args.task == "binary" else None,
            "split": args.split,
            "eval_split": args.eval_split,
            "task": args.task,
            "num_test": int(len(eval_idx)),
            "num_eval": int(len(eval_idx)),
            "checkpoint": args.checkpoint,
            "split_tokens": args.split_tokens,
        }
    )
    metrics["use_service_context"] = bool(use_service_context)
    metrics["use_anomaly_features"] = bool(use_anomaly_features)
    metrics["use_stat_fusion"] = bool(use_stat_fusion)
    metrics["use_hierarchical_classifier"] = bool(use_hierarchical_classifier)
    metrics["use_app_projection"] = bool(use_app_projection)
    if use_app_projection:
        metrics["app_projection"] = {
            "dim": int(feature_specs["app_projection_dim"] or model_cfg.get("app_projection_dim", model_cfg.get("hidden_size", 128))),
        }
    metrics["pooling_strategy"] = str(getattr(encoder, "pooling_strategy", pooling_strategy))
    metrics["pooling"] = str(getattr(encoder, "pooling_strategy", pooling_strategy))
    metrics["class_aware_pooling"] = bool(getattr(encoder, "class_aware_pooling", False))
    if getattr(encoder, "pooling_strategy", pooling_strategy) == "residual_class_aware":
        metrics["class_aware_alpha"] = float(getattr(encoder, "class_aware_alpha", model_cfg.get("class_aware_alpha", 0.5)))
        metrics["cls_beta"] = float(getattr(encoder, "cls_beta", model_cfg.get("cls_beta", 0.0)))
    if use_anomaly_features:
        metrics["anomaly_features"] = {
            "score_method": anomaly_score_method,
            "normalize": anomaly_normalize,
            "include_special": anomaly_include_special,
            "use_service_prototype_distance": use_service_prototype_distance,
            "use_class_prototype_distance": use_class_prototype_distance,
            "feature_dim": int(feature_specs["anomaly_size"]),
            "projection_dim": int(feature_specs["anomaly_feature_dim"]),
        }
    if use_stat_fusion:
        metrics["stat_fusion"] = {
            "feature_names": stat_feature_names,
            "feature_dim": int(feature_specs["stat_size"]),
            "projection_dim": int(feature_specs["stat_mlp_dim"]),
            "missing_or_unavailable_features": stat_normalizer.missing_names if stat_normalizer else [],
        }
    metrics["seed"] = int(cfg.get("seed", 42))
    if args.metadata_tokens is not None:
        metrics["metadata_tokens"] = args.metadata_tokens
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, out_dir / "resolved_config.json")
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "tokens": args.tokens,
            "split_tokens": args.split_tokens,
            "metadata_tokens": args.metadata_tokens,
            "checkpoint": args.checkpoint,
            "config_path": args.config,
            "task": args.task,
            "split": args.split,
            "eval_split": args.eval_split,
            "threshold": args.threshold,
            "seed": int(cfg.get("seed", 42)),
        },
        out_dir / "run_manifest.json",
    )
    write_json(metrics, out_dir / "metrics.json")
    write_json(report_dict(y_true, y_pred, target_names=target_names), out_dir / "classification_report.json")
    write_json(confusion(y_true, y_pred), out_dir / "confusion_matrix.json")
    score_rows = []
    for data_idx, true_label, pred_label, probs in zip(row_indices, y_true, y_pred, y_score.tolist()):
        meta = meta_rows[int(data_idx)] if int(data_idx) < len(meta_rows) else {}
        score_rows.append(
            {
                "index": int(data_idx),
                "flow_id": meta.get("flow_id"),
                "label": meta.get("label"),
                "true_label": target_names[int(true_label)] if int(true_label) < len(target_names) else str(true_label),
                "pred_label": target_names[int(pred_label)] if int(pred_label) < len(target_names) else str(pred_label),
                "binary_label": meta.get("binary_label"),
                "prediction": int(pred_label),
                "attack_probability": float(probs[1]) if len(probs) > 1 else None,
                "scores": {target_names[class_idx]: float(score) for class_idx, score in enumerate(probs[: len(target_names)])},
                "packet_count": meta.get("packet_count"),
                "token_count": meta.get("token_count"),
                "start_ts": meta.get("start_ts"),
                "end_ts": meta.get("end_ts"),
                "duration": meta.get("duration"),
                "dataset_file": meta.get("dataset_file"),
                "src_ip": meta.get("src_ip"),
                "dst_ip": meta.get("dst_ip"),
                "src_port": meta.get("src_port"),
                "dst_port": meta.get("dst_port"),
                "protocol": meta.get("protocol"),
                "service_key": meta.get("service_key"),
                "service_context": meta.get("service_context"),
            }
        )
    write_jsonl(score_rows, out_dir / "scores.jsonl")
    write_json(score_rows, out_dir / f"{args.eval_split}_predictions.json")
    print(metrics)


if __name__ == "__main__":
    main()
