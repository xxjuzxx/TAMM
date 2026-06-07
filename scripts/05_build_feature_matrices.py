#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from src.features.token_alias import is_burst_token, is_flow_summary_token, is_packet_token, is_profile_token, is_structural_token
from src.pipeline.common import ROOT, command_record, ensure_dirs, write_csv, write_json, write_md


SPECIAL = {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}


def _id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    return {int(idx): token for token, idx in vocab.items()}


def _row_tokens(token_data: dict[str, Any], idx: int) -> list[str]:
    id_to_token = _id_to_token(token_data["vocab"])
    ids = token_data["input_ids"][idx].cpu().numpy()
    mask = token_data["attention_mask"][idx].cpu().numpy() > 0
    return [id_to_token.get(int(token_id), "[UNK]") for token_id in ids[mask] if id_to_token.get(int(token_id), "[UNK]") not in SPECIAL]


def _keep(token: str, feature_set: str) -> bool:
    if feature_set == "statistics_only":
        return is_flow_summary_token(token)
    if feature_set == "token_histogram_only":
        return is_packet_token(token) or is_burst_token(token) or is_flow_summary_token(token)
    if feature_set == "primitive_only":
        return is_profile_token(token) or is_structural_token(token)
    if feature_set == "token_plus_primitive":
        return is_packet_token(token) or is_burst_token(token) or is_flow_summary_token(token) or is_profile_token(token) or is_structural_token(token)
    if feature_set == "profile_only":
        return is_profile_token(token)
    if feature_set == "structural_only":
        return is_structural_token(token)
    if feature_set == "without_direction":
        return not token.startswith(("PKT_DIR_", "BURST_DIR_", "PRIM_STRUCT_DIR_"))
    if feature_set == "without_timing":
        return "IAT" not in token and "DUR" not in token and "PERIODIC" not in token
    if feature_set == "without_length":
        return "LEN" not in token and "BYTES" not in token
    if feature_set == "without_primitives":
        return not (is_profile_token(token) or is_structural_token(token))
    if feature_set == "without_tokens":
        return is_profile_token(token) or is_structural_token(token)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact feature-matrix manifests from category token corpora.")
    parser.add_argument("--token-dir", default="paper_icdm_applied_2026/experiments/unknown/tokens_category")
    parser.add_argument("--output-dir", default="data/processed/feature_matrices")
    parser.add_argument("--feature-sets", nargs="+", default=["statistics_only", "token_histogram_only", "primitive_only", "token_plus_primitive", "profile_only", "structural_only", "without_direction", "without_timing", "without_length", "without_primitives", "without_tokens"])
    parser.add_argument("--max-corpora", type=int, default=1, help="Build sample matrices for the first N corpora; all corpora are still inventoried.")
    args = parser.parse_args()

    ensure_dirs()
    token_paths = sorted((ROOT / args.token_dir).glob("*.pt"))
    manifest_rows: list[dict[str, Any]] = []
    for corpus_idx, token_path in enumerate(token_paths):
        build_matrix = corpus_idx < args.max_corpora
        token_data = torch.load(token_path, map_location="cpu", weights_only=False)
        token_rows = [_row_tokens(token_data, idx) for idx in range(len(token_data["meta"]))]
        for feature_set in args.feature_sets:
            vocab = sorted({token for row in token_rows for token in row if _keep(token, feature_set)})
            out_path = ROOT / args.output_dir / feature_set / f"{token_path.stem}.parquet"
            schema_path = out_path.with_suffix(".features.json")
            status = "inventoried"
            if build_matrix and vocab:
                col = {token: idx for idx, token in enumerate(vocab)}
                arr = np.zeros((len(token_rows), len(vocab)), dtype=np.float32)
                for row_idx, tokens in enumerate(token_rows):
                    counts = Counter(token for token in tokens if token in col)
                    for token, count in counts.items():
                        arr[row_idx, col[token]] = float(count)
                meta = token_data["meta"]
                df = pd.DataFrame(arr, columns=vocab)
                df.insert(0, "split", [m.get("split") for m in meta])
                df.insert(0, "binary_label", [m.get("binary_label") for m in meta])
                df.insert(0, "attack_family", [m.get("attack_family") for m in meta])
                df.insert(0, "flow_id", [m.get("flow_id") for m in meta])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_path, index=False)
                write_json(schema_path, {"feature_set": feature_set, "features": vocab, "source_token_corpus": str(token_path), "train_only_vocab": True})
                status = "built"
            manifest_rows.append(
                {
                    "feature_set": feature_set,
                    "source_token_corpus": str(token_path),
                    "output_matrix": str(out_path),
                    "feature_schema": str(schema_path),
                    "status": status,
                    "num_rows": len(token_rows),
                    "num_features": len(vocab),
                    "raw_ip_used_as_token": False,
                    "absolute_time_used_as_token": False,
                    "five_tuple_used_as_token": False,
                }
            )
    write_csv(ROOT / "data/manifests/feature_manifest.csv", manifest_rows)
    write_json(ROOT / "data/manifests/feature_matrix_summary.json", {"command": command_record(sys.argv), "rows": manifest_rows})
    write_md(ROOT / "reports/feature_matrix_summary.md", ["# Feature Matrix Summary", "", f"- Token corpora inventoried: {len(token_paths)}", f"- Feature-set rows: {len(manifest_rows)}", f"- Sample matrices built per max-corpora: {args.max_corpora}"])
    print(ROOT / "data/manifests/feature_manifest.csv")


if __name__ == "__main__":
    main()

