#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shlex
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.training.classifier_trainer import train_classifier
from src.utils.io import read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--task", choices=["binary", "multiclass", "multiclass_merged"], default="binary")
    parser.add_argument(
        "--split",
        choices=["stratified", "chronological", "temporal_stratified", "temporal_stratified_raw_label"],
        default="stratified",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--augment_tokens", action="append", default=[])
    parser.add_argument("--augment_all_labels", action="store_true")
    parser.add_argument("--augment_fraction", type=float, default=1.0)
    parser.add_argument("--augment_fractions", nargs="*", type=float, default=None)
    parser.add_argument("--extra_train_tokens", action="append", default=[])
    parser.add_argument("--extra_train_fraction", type=float, default=1.0)
    parser.add_argument("--use_service_context", action="store_true")
    parser.add_argument("--seed", "--train_seed", dest="train_seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.augment_fractions is not None and len(args.augment_fractions) != len(args.augment_tokens):
        parser.error("--augment_fractions length must match --augment_tokens length")

    cfg = read_yaml(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    cfg = copy.deepcopy(cfg)
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    augment_token_data = [torch.load(path, map_location="cpu", weights_only=False) for path in args.augment_tokens]
    extra_train_token_data = [torch.load(path, map_location="cpu", weights_only=False) for path in args.extra_train_tokens]
    result = train_classifier(
        token_data,
        cfg,
        task=args.task,
        split=args.split,
        checkpoint=args.checkpoint,
        freeze_encoder=args.freeze_encoder,
        label_fraction=args.label_fraction,
        augment_token_data=augment_token_data,
        augment_attack_only=not args.augment_all_labels,
        augment_fraction=args.augment_fraction,
        augment_fractions=args.augment_fractions,
        extra_train_token_data=extra_train_token_data,
        extra_train_fraction=args.extra_train_fraction,
        train_seed=args.train_seed,
        use_service_context=args.use_service_context,
    )
    if args.augment_tokens and "augmentation" in result.metrics:
        result.metrics["augmentation"]["sources"] = args.augment_tokens
    if args.extra_train_tokens and "extra_train" in result.metrics:
        result.metrics["extra_train"]["sources"] = args.extra_train_tokens
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, out_dir / "resolved_config.json")
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "tokens": args.tokens,
            "config_path": args.config,
            "task": args.task,
            "split": args.split,
            "checkpoint": args.checkpoint,
            "freeze_encoder": bool(args.freeze_encoder),
            "label_fraction": float(args.label_fraction),
            "augment_tokens": args.augment_tokens,
            "augment_all_labels": bool(args.augment_all_labels),
            "augment_fraction": float(args.augment_fraction),
            "augment_fractions": args.augment_fractions,
            "extra_train_tokens": args.extra_train_tokens,
            "extra_train_fraction": float(args.extra_train_fraction),
            "use_service_context": bool(args.use_service_context),
            "seed": result.metrics.get("seed"),
            "split_seed": result.metrics.get("split_seed"),
            "train_seed": result.metrics.get("train_seed"),
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
