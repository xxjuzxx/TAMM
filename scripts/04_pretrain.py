#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.training.pretrain_trainer import pretrain
from src.utils.io import read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    if args.epochs is not None:
        cfg.setdefault("pretraining", {})["epochs"] = args.epochs
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    result = pretrain(token_data, cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, out_dir / "resolved_config.json")
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "tokens": args.tokens,
            "config_path": args.config,
            "epochs": cfg.get("pretraining", {}).get("epochs"),
            "out": args.out,
        },
        out_dir / "run_manifest.json",
    )
    write_json(result.metrics, out_dir / "metrics.json")
    write_json(result.history, out_dir / "history.json")
    torch.save({"state_dict": result.state_dict, "metrics": result.metrics}, out_dir / "best.pt")
    print(result.metrics)


if __name__ == "__main__":
    main()
