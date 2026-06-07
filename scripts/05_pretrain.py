#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import torch

from src.training.pretrain_trainer import pretrain
from src.utils.io import read_yaml, write_json


def _subset_token_data(token_data: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    idx = torch.tensor(indices, dtype=torch.long)
    out = dict(token_data)
    for key in ("input_ids", "attention_mask", "token_type_ids", "labels", "binary_labels"):
        if key in token_data:
            out[key] = token_data[key][idx]
    out["meta"] = [token_data.get("meta", [])[item] for item in indices]
    if "attack_family" in token_data:
        out["attack_family"] = [token_data["attack_family"][item] for item in indices]
    out["source_num_rows"] = int(len(token_data["input_ids"]))
    out["pretrain_source_split"] = "train"
    out["train_only"] = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    if args.epochs is not None:
        cfg.setdefault("pretraining", {})["epochs"] = int(args.epochs)
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    train_indices = [idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == "train"]
    if not train_indices:
        raise ValueError("Guide-compatible pretraining requires token meta split=train rows")
    train_only_data = _subset_token_data(token_data, train_indices)
    result = pretrain(train_only_data, cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, out_dir / "config.yaml")
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "tokens": args.tokens,
            "config_path": args.config,
            "epochs": cfg.get("pretraining", {}).get("epochs"),
            "out": args.out,
            "pretrain_source_split": "train",
            "num_source_rows": int(len(token_data["input_ids"])),
            "num_pretrain_rows": int(len(train_indices)),
            "train_only": True,
        },
        out_dir / "run_meta.json",
    )
    metrics = dict(result.metrics)
    metrics["num_source_rows"] = int(len(token_data["input_ids"]))
    metrics["num_pretrain_rows"] = int(len(train_indices))
    metrics["pretrain_source_split"] = "train"
    metrics["train_only"] = True
    write_json(metrics, out_dir / "metrics.json")
    write_json(result.history, out_dir / "history.json")
    torch.save({"state_dict": result.state_dict, "metrics": metrics}, out_dir / "best.pt")
    print(metrics)


if __name__ == "__main__":
    main()
