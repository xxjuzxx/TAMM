#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.training.anomaly_trainer import train_anomaly_detector
from src.utils.io import write_json, write_jsonl, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument(
        "--split",
        choices=["stratified", "chronological", "temporal_stratified", "predefined"],
        default="temporal_stratified",
    )
    parser.add_argument("--score_method", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--normalize", choices=["none", "l1", "l2"], default="l2")
    parser.add_argument("--feature", choices=["token_histogram", "encoder_embedding"], default="token_histogram")
    parser.add_argument("--baseline", choices=["global_prototype", "service_memory"], default="global_prototype")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--include_special", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    result = train_anomaly_detector(
        token_data,
        cfg,
        split=args.split,
        score_method=args.score_method,
        normalize=args.normalize,
        include_special=args.include_special,
        feature=args.feature,
        checkpoint=args.checkpoint,
        baseline=args.baseline,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, out_dir / "resolved_config.json")
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "tokens": args.tokens,
            "config_path": args.config,
            "split": args.split,
            "score_method": args.score_method,
            "normalize": args.normalize,
            "feature": args.feature,
            "baseline": args.baseline,
            "checkpoint": args.checkpoint,
            "include_special": bool(args.include_special),
            "seed": cfg.get("seed", 42),
        },
        out_dir / "run_manifest.json",
    )
    write_json(result.metrics, out_dir / "metrics.json")
    write_json(result.report, out_dir / "classification_report.json")
    write_json(result.confusion_matrix, out_dir / "confusion_matrix.json")
    write_json(result.split_summary, out_dir / "split_summary.json")
    write_jsonl(result.score_rows, out_dir / "scores.jsonl")
    print(result.metrics)


if __name__ == "__main__":
    main()
