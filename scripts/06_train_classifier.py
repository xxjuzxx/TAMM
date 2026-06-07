#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.data.leakage_check import LeakageError
from src.training.fixed_split_classifier import train_fixed_split_classifier
from src.utils.io import read_yaml, write_json


def _write_confusion_csv(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def _write_predictions_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "flow_id", "split", "label", "binary_label", "true_label", "pred_label", "pred_confidence", "scores"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            packed = dict(row)
            packed["scores"] = json.dumps(packed.get("scores", {}), ensure_ascii=True, sort_keys=True)
            writer.writerow({key: packed.get(key) for key in fieldnames})


def _per_class_metrics(report: dict) -> dict:
    return {
        key: value
        for key, value in report.items()
        if isinstance(value, dict) and {"precision", "recall", "f1-score", "support"}.issubset(value.keys())
    }


def _assert_leakage_passed(path: str | None) -> None:
    if not path:
        return
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not report.get("passed", False):
        failed = [name for name, item in report.get("checks", {}).items() if not item.get("ok")]
        raise LeakageError(f"Refusing to train because leakage report failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--leakage_report", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--seed", "--train_seed", dest="train_seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    _assert_leakage_passed(args.leakage_report)
    cfg = read_yaml(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    result = train_fixed_split_classifier(
        token_data,
        cfg,
        task=args.task,
        train_seed=args.train_seed,
        checkpoint=args.checkpoint,
        freeze_encoder=args.freeze_encoder,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(result.metrics, out_dir / "metrics.json")
    write_json(result.report, out_dir / "classification_report.json")
    write_json(_per_class_metrics(result.report), out_dir / "per_class_metrics.json")
    write_json(cfg, out_dir / "config.yaml")
    write_json(result.history, out_dir / "history.json")
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "tokens": args.tokens,
            "config_path": args.config,
            "task": args.task,
            "leakage_report": args.leakage_report,
            "checkpoint": args.checkpoint,
            "freeze_encoder": bool(args.freeze_encoder),
            "seed": result.metrics.get("seed"),
            "train_seed": result.metrics.get("train_seed"),
            "target_names": result.target_names,
        },
        out_dir / "run_meta.json",
    )
    _write_confusion_csv(result.confusion_matrix, result.target_names, out_dir / "confusion_matrix.csv")
    _write_predictions_csv(result.predictions, out_dir / "predictions.csv")
    torch.save(result.state_dict, out_dir / "best_model.pt")
    print(result.metrics)


if __name__ == "__main__":
    main()
