#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.models.behavior_composer import BehaviorComposer, resolve_pooling_config
from src.training.classifier_trainer import _task_labels
from src.training.session_aggregation_trainer import (
    SessionAggregationClassifier,
    build_session_windows,
    merge_flow_metadata,
)
from src.utils.io import read_jsonl, read_yaml, write_json
from src.utils.seed import set_seed


def _state_dict_from_checkpoint(checkpoint: str, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)!r}")
    return state_dict


def _target_names(task: str, label_to_id: dict[str, int] | None) -> list[str]:
    if task == "binary":
        return ["BENIGN", "ATTACK"]
    if label_to_id is None:
        raise ValueError(f"{task} requires a label mapping")
    inv = {idx: label for label, idx in label_to_id.items()}
    return [inv[idx] for idx in range(len(inv))]


def _infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int | None:
    for key in ("classifier.2.weight", "classifier.1.weight"):
        weight = state_dict.get(key)
        if weight is not None and weight.ndim == 2:
            return int(weight.shape[0])
    return None


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int], np.ndarray]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, token_type_ids, flow_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            flow_mask = flow_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask, token_type_ids, flow_mask)
            probs = torch.softmax(logits, dim=-1)
            losses.append(float(criterion(logits, labels).item()))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            scores.append(probs.cpu().numpy())
    return float(np.mean(losses) if losses else 0.0), y_true, y_pred, np.concatenate(scores, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True, help="Target token file to evaluate.")
    parser.add_argument("--metadata_flows", default=None, help="Optional flow JSONL used to fill missing src/dst metadata by flow_id.")
    parser.add_argument("--config", required=True, help="Session aggregation config used to build the checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Session aggregation checkpoint.")
    parser.add_argument("--task", choices=["binary", "multiclass", "multiclass_merged"], default="multiclass")
    parser.add_argument("--group_by", default=None)
    parser.add_argument("--window_n", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--min_flows", type=int, default=None)
    parser.add_argument("--eval_min_purity", type=float, default=None)
    parser.add_argument("--session_pooling_strategy", choices=["mean", "attentive", "transformer_mean", "transformer_attentive"], default=None)
    parser.add_argument(
        "--session_flow_representation",
        choices=["shared_encode", "classification_embedding", "class_aware_summary", "residual_class_aware_pool"],
        default=None,
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cfg = copy.deepcopy(read_yaml(args.config))
    model_cfg = cfg.setdefault("model", {})
    train_cfg = cfg.setdefault("training", {})
    if args.group_by is not None:
        model_cfg["session_group_by"] = args.group_by
    if args.window_n is not None:
        model_cfg["session_window_n"] = args.window_n
    if args.stride is not None:
        model_cfg["session_stride"] = args.stride
    if args.min_flows is not None:
        model_cfg["session_min_flows"] = args.min_flows
    if args.eval_min_purity is not None:
        model_cfg["session_eval_min_purity"] = args.eval_min_purity
    if args.session_pooling_strategy is not None:
        model_cfg["session_pooling_strategy"] = args.session_pooling_strategy
    if args.session_flow_representation is not None:
        model_cfg["session_flow_representation"] = args.session_flow_representation

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    metadata_summary = None
    if args.metadata_flows:
        token_data, metadata_summary = merge_flow_metadata(token_data, read_jsonl(args.metadata_flows))
    labels, label_to_id = _task_labels(token_data, args.task)
    num_classes = int(labels.max().item()) + 1
    target_names = _target_names(args.task, label_to_id)

    session_group_by = str(model_cfg.get("session_group_by", "service_host_proto"))
    session_window_n = int(model_cfg.get("session_window_n", 8))
    session_stride = int(model_cfg.get("session_stride", 1))
    session_min_flows = int(model_cfg.get("session_min_flows", 2))
    session_eval_min_purity = float(model_cfg.get("session_eval_min_purity", 0.0))
    session_pooling_strategy = str(model_cfg.get("session_pooling_strategy", "mean"))
    session_flow_representation = str(model_cfg.get("session_flow_representation", "shared_encode"))
    session_transformer_layers = int(model_cfg.get("session_transformer_layers", 1))
    session_transformer_heads = int(model_cfg.get("session_transformer_heads", model_cfg.get("num_heads", 4)))
    session_transformer_intermediate_size = int(model_cfg.get("session_transformer_intermediate_size", int(model_cfg.get("hidden_size", 128)) * 2))

    indices = np.arange(len(labels), dtype=np.int64)
    eval_dataset, eval_stats, eval_sessions = build_session_windows(
        token_data,
        labels,
        indices,
        group_by=session_group_by,
        window_n=session_window_n,
        stride=session_stride,
        min_flows=session_min_flows,
        min_purity=session_eval_min_purity,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = _state_dict_from_checkpoint(args.checkpoint, device)
    checkpoint_classes = _infer_num_classes(state_dict)
    if checkpoint_classes is not None and checkpoint_classes != num_classes:
        raise ValueError(
            f"checkpoint class count ({checkpoint_classes}) does not match token labels ({num_classes}); "
            "use a target token file with the same label_to_id mapping"
        )
    encoder = BehaviorComposer(
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
    model = SessionAggregationClassifier(
        encoder,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_classes=num_classes,
        dropout=float(model_cfg.get("dropout", 0.1)),
        flow_representation=session_flow_representation,
        session_pooling_strategy=session_pooling_strategy,
        session_window_n=session_window_n,
        session_transformer_layers=session_transformer_layers,
        session_transformer_heads=session_transformer_heads,
        session_transformer_intermediate_size=session_transformer_intermediate_size,
        encoder_frozen=True,
    ).to(device)
    model.load_state_dict(state_dict)

    loader = DataLoader(eval_dataset, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False)
    test_loss, y_true, y_pred, y_score = _evaluate(model, loader, device)
    metrics = classification_metrics(y_true, y_pred, y_score)
    metrics.update(
        {
            "test_loss": float(test_loss),
            "num_flows": int(len(labels)),
            "num_session_eval": int(len(eval_sessions)),
            "task": args.task,
            "seed": seed,
            "device": str(device),
            "checkpoint": args.checkpoint,
            "tokens": args.tokens,
            "session_group_by": session_group_by,
            "session_window_n": int(session_window_n),
            "session_stride": int(session_stride),
            "session_min_flows": int(session_min_flows),
            "session_eval_min_purity": float(session_eval_min_purity),
            "session_pooling_strategy": session_pooling_strategy,
            "session_transformer_layers": int(session_transformer_layers),
            "session_transformer_heads": int(session_transformer_heads),
            "session_transformer_intermediate_size": int(session_transformer_intermediate_size),
            "session_flow_representation": session_flow_representation,
            "session_stats": {"eval": eval_stats},
            "metadata_summary": metadata_summary,
            "eval_seconds": float(time.perf_counter() - started),
        }
    )

    meta_rows = list(token_data.get("meta", []))
    predictions: list[dict[str, Any]] = []
    for offset, session_row in enumerate(eval_sessions):
        source_indices = session_row.get("source_indices", [])
        first_meta = meta_rows[int(source_indices[0])] if source_indices else {}
        true_id = int(y_true[offset])
        pred_id = int(y_pred[offset])
        score_row = y_score[offset]
        predictions.append(
            {
                "index": int(source_indices[0]) if source_indices else -1,
                "flow_id": first_meta.get("flow_id"),
                "label": first_meta.get("label"),
                "true_label": target_names[true_id] if true_id < len(target_names) else str(true_id),
                "pred_label": target_names[pred_id] if pred_id < len(target_names) else str(pred_id),
                "pred_confidence": float(score_row[pred_id]),
                "start_ts": first_meta.get("start_ts"),
                "end_ts": first_meta.get("end_ts"),
                "duration": first_meta.get("duration"),
                "dataset_file": first_meta.get("dataset_file"),
                "service_key": first_meta.get("service_key"),
                "src_ip": first_meta.get("src_ip"),
                "dst_ip": first_meta.get("dst_ip"),
                "src_port": first_meta.get("src_port"),
                "dst_port": first_meta.get("dst_port"),
                "protocol": first_meta.get("protocol"),
                "session_id": session_row.get("session_id"),
                "session_group_key": session_row.get("group_key"),
                "session_flow_ids": session_row.get("source_flow_ids"),
                "session_flow_labels": session_row.get("source_labels"),
                "session_label_purity": session_row.get("label_purity"),
                "scores": {target_names[class_idx]: float(score) for class_idx, score in enumerate(score_row[: len(target_names)])},
            }
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, out_dir / "resolved_config.json")
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "tokens": args.tokens,
            "metadata_flows": args.metadata_flows,
            "metadata_summary": metadata_summary,
            "config_path": args.config,
            "checkpoint": args.checkpoint,
            "task": args.task,
            "seed": seed,
        },
        out_dir / "run_manifest.json",
    )
    write_json(metrics, out_dir / "metrics.json")
    write_json(report_dict(y_true, y_pred, target_names=target_names), out_dir / "classification_report.json")
    write_json(confusion(y_true, y_pred), out_dir / "confusion_matrix.json")
    write_json(predictions, out_dir / "session_predictions.json")
    print(metrics)


if __name__ == "__main__":
    main()
