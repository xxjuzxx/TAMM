#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import torch

from src.utils.io import write_json


def _label_id(mapping: dict[str, int], label: str) -> int:
    if label in mapping:
        return int(mapping[label])
    lowered = {str(key).lower(): int(value) for key, value in mapping.items()}
    key = label.lower()
    if key not in lowered:
        raise ValueError(f"label {label!r} not found in mapping {mapping}")
    return lowered[key]


def _subset_value(value: Any, indices: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor) and int(value.shape[0]) >= int(indices.max().item()) + 1:
        return value[indices]
    if isinstance(value, list) and len(value) >= int(indices.max().item()) + 1:
        return [value[int(idx)] for idx in indices.tolist()]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--binary_label", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    if bool(args.binary_label) == bool(args.label):
        parser.error("provide exactly one of --binary_label or --label")

    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    if args.binary_label is not None:
        mapping = token_data.get("binary_label_to_id") or {}
        source = token_data["binary_labels"]
        target = _label_id(mapping, args.binary_label)
        filter_desc = {"binary_label": args.binary_label, "binary_label_id": target}
    else:
        mapping = token_data.get("label_to_id") or {}
        source = token_data["labels"]
        target = _label_id(mapping, args.label)
        filter_desc = {"label": args.label, "label_id": target}

    indices = torch.nonzero(source == int(target), as_tuple=False).flatten()
    if args.max_rows is not None:
        indices = indices[: int(args.max_rows)]
    if indices.numel() == 0:
        raise ValueError(f"filter selected no rows: {filter_desc}")

    subset = {key: _subset_value(value, indices) for key, value in token_data.items()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subset, out_path)
    stats = {
        "command": shlex.join(sys.argv),
        "script": Path(__file__).name,
        "tokens": args.tokens,
        "out": str(out_path),
        "filter": filter_desc,
        "num_rows_in": int(token_data["input_ids"].shape[0]),
        "num_rows_out": int(indices.numel()),
        "max_rows": args.max_rows,
        "vocab_size": int(len(token_data["vocab"])),
        "max_len": int(token_data.get("max_len", token_data["input_ids"].shape[1])),
    }
    write_json(stats, out_path.with_suffix("").with_name(out_path.stem + "_stats.json"))
    print(stats)


if __name__ == "__main__":
    main()
