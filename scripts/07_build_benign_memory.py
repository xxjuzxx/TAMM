#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.pipeline.common import ROOT, command_record, ensure_dirs, write_csv, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight benign-memory manifests from category token corpora.")
    parser.add_argument("--token-dir", default="paper_icdm_applied_2026/experiments/unknown/tokens_category")
    parser.add_argument("--output-dir", default="data/processed/benign_memory")
    parser.add_argument("--max-corpora", type=int, default=1)
    args = parser.parse_args()

    ensure_dirs()
    rows: list[dict[str, Any]] = []
    for idx, token_path in enumerate(sorted((ROOT / args.token_dir).glob("*.pt"))):
        token_data = torch.load(token_path, map_location="cpu", weights_only=False)
        train_idx = [i for i, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == "train"]
        val_idx = [i for i, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == "val"]
        labels = token_data["binary_labels"].cpu().numpy().astype(int)
        if any(labels[i] != 0 for i in train_idx):
            raise ValueError(f"train split contains attack labels: {token_path}")
        if any(labels[i] != 0 for i in val_idx):
            raise ValueError(f"validation split contains attack labels: {token_path}")
        out_manifest = ROOT / args.output_dir / f"{token_path.stem}_memory_manifest.json"
        status = "inventoried"
        if idx < args.max_corpora:
            write_json(
                out_manifest,
                {
                    "source_token_corpus": str(token_path),
                    "base_memory_size": len(train_idx),
                    "validation_size": len(val_idx),
                    "memory_representation": "behavior-token histogram built by experiment scorer",
                    "strategy": "train_split_benign_only",
                    "attack_labels_used_for_threshold": False,
                    "raw_ip_used_as_token": False,
                    "absolute_time_used_as_token": False,
                    "five_tuple_used_as_token": False,
                    "command_used": command_record(sys.argv)["command"],
                },
            )
            status = "manifest_built"
        rows.append(
            {
                "source_token_corpus": str(token_path),
                "output_manifest": str(out_manifest),
                "memory_size": len(train_idx),
                "validation_size": len(val_idx),
                "status": status,
                "attack_labels_used_for_threshold": False,
            }
        )
    write_csv(ROOT / "data/manifests/benign_memory_manifest.csv", rows)
    write_json(ROOT / "data/manifests/benign_memory_summary.json", {"command": command_record(sys.argv), "rows": rows})
    write_md(ROOT / "reports/benign_memory_summary.md", ["# Benign Memory Summary", "", f"- Token corpora inventoried: {len(rows)}", "- Memory construction uses train split benign rows only."])
    print(ROOT / "data/manifests/benign_memory_manifest.csv")


if __name__ == "__main__":
    main()
