#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shlex
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.training.session_aggregation_trainer import train_session_classifier
from src.training.session_aggregation_trainer import merge_flow_metadata
from src.utils.io import read_jsonl, read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--metadata_flows", default=None)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--task", choices=["binary", "multiclass", "multiclass_merged"], default="multiclass")
    parser.add_argument("--split", choices=["stratified", "chronological", "temporal_stratified", "temporal_stratified_raw_label"], default="temporal_stratified")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--group_by", default=None)
    parser.add_argument("--window_n", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--min_flows", type=int, default=None)
    parser.add_argument("--min_purity", type=float, default=None)
    parser.add_argument("--train_min_purity", type=float, default=None)
    parser.add_argument("--eval_min_purity", type=float, default=None)
    parser.add_argument("--session_pooling_strategy", choices=["mean", "attentive", "transformer_mean", "transformer_attentive"], default=None)
    parser.add_argument(
        "--session_flow_representation",
        choices=["shared_encode", "classification_embedding", "class_aware_summary", "residual_class_aware_pool"],
        default=None,
    )
    parser.add_argument("--freeze_encoder_only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    cfg = copy.deepcopy(cfg)
    if args.group_by is not None:
        cfg.setdefault("model", {})["session_group_by"] = args.group_by
    if args.window_n is not None:
        cfg.setdefault("model", {})["session_window_n"] = args.window_n
    if args.stride is not None:
        cfg.setdefault("model", {})["session_stride"] = args.stride
    if args.min_flows is not None:
        cfg.setdefault("model", {})["session_min_flows"] = args.min_flows
    if args.min_purity is not None:
        cfg.setdefault("model", {})["session_min_purity"] = args.min_purity
    if args.train_min_purity is not None:
        cfg.setdefault("model", {})["session_train_min_purity"] = args.train_min_purity
    if args.eval_min_purity is not None:
        cfg.setdefault("model", {})["session_eval_min_purity"] = args.eval_min_purity
    if args.session_pooling_strategy is not None:
        cfg.setdefault("model", {})["session_pooling_strategy"] = args.session_pooling_strategy
    if args.session_flow_representation is not None:
        cfg.setdefault("model", {})["session_flow_representation"] = args.session_flow_representation

    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    metadata_summary = None
    if args.metadata_flows:
        token_data, metadata_summary = merge_flow_metadata(token_data, read_jsonl(args.metadata_flows))
    result = train_session_classifier(
        token_data,
        cfg,
        task=args.task,
        split=args.split,
        checkpoint=args.checkpoint,
        freeze_encoder=args.freeze_encoder or args.freeze_encoder_only or bool(args.checkpoint),
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
            "task": args.task,
            "split": args.split,
            "checkpoint": args.checkpoint,
            "freeze_encoder": bool(args.freeze_encoder),
            "freeze_encoder_only": bool(args.freeze_encoder_only),
            "group_by": args.group_by,
            "window_n": args.window_n,
            "stride": args.stride,
            "min_flows": args.min_flows,
            "min_purity": args.min_purity,
            "train_min_purity": args.train_min_purity,
            "eval_min_purity": args.eval_min_purity,
            "session_pooling_strategy": args.session_pooling_strategy,
            "session_flow_representation": args.session_flow_representation,
            "seed": result.metrics.get("seed"),
        },
        out_dir / "run_manifest.json",
    )
    write_json(result.metrics, out_dir / "metrics.json")
    write_json(result.report, out_dir / "classification_report.json")
    write_json(result.confusion_matrix, out_dir / "confusion_matrix.json")
    write_json(result.history, out_dir / "history.json")
    write_json(result.predictions, out_dir / "test_predictions.json")
    torch.save(result.state_dict, out_dir / "best_model.pt")
    print(result.metrics)


if __name__ == "__main__":
    main()
