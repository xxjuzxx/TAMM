#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.utils.io import write_json


def _safe_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if value is None:
        return ("NONE",)
    return (str(value),)


def _group_key(meta: dict[str, Any], group_by: str) -> tuple[str, ...]:
    service_key = _safe_tuple(meta.get("service_key"))
    if group_by == "service_key":
        return ("service_key", *service_key)
    if group_by == "service_host_proto":
        host = service_key[0] if len(service_key) >= 1 else "NONE"
        proto = service_key[2] if len(service_key) >= 3 else str(meta.get("protocol") or "NONE")
        return ("service_host_proto", host, proto)
    if group_by == "global_time":
        return ("global_time",)
    raise ValueError(f"Unsupported group_by: {group_by}")


def _sort_key(index: int, meta_rows: list[dict[str, Any]]) -> tuple[float, int, str]:
    meta = meta_rows[index] if index < len(meta_rows) else {}
    try:
        ts = float(meta.get("start_ts"))
    except (TypeError, ValueError):
        ts = math.inf
    return ts, index, str(meta.get("flow_id") or "")


def _active_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, *, cls_id: int, sep_id: int, pad_id: int) -> list[int]:
    out: list[int] = []
    for token, mask in zip(input_ids.tolist(), attention_mask.tolist()):
        token_id = int(token)
        if int(mask) <= 0:
            continue
        if token_id in {cls_id, sep_id, pad_id}:
            continue
        out.append(token_id)
    return out


def _majority_label(indices: list[int], meta_rows: list[dict[str, Any]]) -> tuple[str | None, float, dict[str, int]]:
    labels = [str(meta_rows[idx].get("label")) for idx in indices if idx < len(meta_rows) and meta_rows[idx].get("label") is not None]
    if not labels:
        return None, 0.0, {}
    counts = Counter(labels)
    label, count = counts.most_common(1)[0]
    return label, float(count / len(labels)), dict(counts)


def _windows_for_group(indices: list[int], meta_rows: list[dict[str, Any]], *, window_n: int, stride: int) -> list[list[int]]:
    ordered = sorted(indices, key=lambda idx: _sort_key(idx, meta_rows))
    size = max(1, int(window_n))
    step = max(1, int(stride))
    if len(ordered) <= size:
        return [ordered]
    return [ordered[start : start + size] for start in range(0, len(ordered) - size + 1, step)]


def build_paragraph_token_data(
    token_data: dict[str, Any],
    *,
    group_by: str,
    window_n: int,
    stride: int,
    max_len: int,
    min_flows: int,
    paragraph_sep_token: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vocab = dict(token_data["vocab"])
    if paragraph_sep_token not in vocab:
        vocab[paragraph_sep_token] = len(vocab)
    paragraph_sep_id = int(vocab[paragraph_sep_token])
    pad_id = int(vocab.get("[PAD]", 0))
    cls_id = int(vocab.get("[CLS]", 1))
    sep_id = int(vocab.get("[SEP]", 2))
    meta_rows = list(token_data.get("meta", []))
    if len(meta_rows) != len(token_data["input_ids"]):
        raise ValueError("token_data meta row count must match input_ids row count")

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, meta in enumerate(meta_rows):
        groups[_group_key(meta, group_by)].append(idx)

    paragraph_ids: list[torch.Tensor] = []
    paragraph_masks: list[torch.Tensor] = []
    paragraph_types: list[torch.Tensor] = []
    paragraph_meta: list[dict[str, Any]] = []
    source_labels: list[int] = []
    label_to_id = token_data.get("label_to_id")
    if isinstance(label_to_id, dict):
        label_to_id = {str(key): int(value) for key, value in label_to_id.items()}

    truncated_count = 0
    skipped_short = 0
    group_sizes = []
    source_flow_counts = []
    for group_key, group_indices in groups.items():
        group_sizes.append(len(group_indices))
        for window in _windows_for_group(group_indices, meta_rows, window_n=window_n, stride=stride):
            if len(window) < int(min_flows):
                skipped_short += 1
                continue
            tokens: list[int] = [cls_id]
            for pos, flow_idx in enumerate(window):
                if pos > 0:
                    tokens.append(paragraph_sep_id)
                tokens.extend(
                    _active_tokens(
                        token_data["input_ids"][flow_idx],
                        token_data["attention_mask"][flow_idx],
                        cls_id=cls_id,
                        sep_id=sep_id,
                        pad_id=pad_id,
                    )
                )
            tokens.append(sep_id)
            if len(tokens) > max_len:
                truncated_count += 1
                tokens = tokens[: max_len - 1] + [sep_id]
            attention = [1] * len(tokens)
            if len(tokens) < max_len:
                pad_count = max_len - len(tokens)
                tokens.extend([pad_id] * pad_count)
                attention.extend([0] * pad_count)
            token_type_ids = [0] * max_len
            majority, purity, label_counts = _majority_label(window, meta_rows)
            if label_to_id is not None and majority is not None and majority in label_to_id:
                source_labels.append(int(label_to_id[majority]))
            else:
                source_labels.append(-1)
            paragraph_ids.append(torch.tensor(tokens, dtype=torch.long))
            paragraph_masks.append(torch.tensor(attention, dtype=torch.long))
            paragraph_types.append(torch.tensor(token_type_ids, dtype=torch.long))
            source_flow_counts.append(len(window))
            paragraph_meta.append(
                {
                    "paragraph_id": len(paragraph_meta),
                    "group_key": list(group_key),
                    "group_by": group_by,
                    "source_indices": [int(item) for item in window],
                    "source_flow_ids": [meta_rows[item].get("flow_id") for item in window],
                    "source_start_ts": [meta_rows[item].get("start_ts") for item in window],
                    "source_labels": [meta_rows[item].get("label") for item in window],
                    "label": majority,
                    "label_purity": purity,
                    "label_counts": label_counts,
                    "flow_count": len(window),
                    "token_count": int(sum(attention)),
                }
            )

    if not paragraph_ids:
        raise ValueError("No paragraph samples were built. Lower --min_flows or check grouping metadata.")
    out = {
        "input_ids": torch.stack(paragraph_ids, dim=0),
        "attention_mask": torch.stack(paragraph_masks, dim=0),
        "token_type_ids": torch.stack(paragraph_types, dim=0),
        "labels": torch.tensor(source_labels, dtype=torch.long),
        "meta": paragraph_meta,
        "label_to_id": token_data.get("label_to_id"),
        "binary_label_to_id": token_data.get("binary_label_to_id"),
        "vocab": vocab,
        "max_len": int(max_len),
        "profile_mode": token_data.get("profile_mode"),
        "paragraph_source": {
            "group_by": group_by,
            "window_n": int(window_n),
            "stride": int(stride),
            "min_flows": int(min_flows),
            "paragraph_sep_token": paragraph_sep_token,
        },
    }
    stats = {
        "num_source_flows": int(len(meta_rows)),
        "num_groups": int(len(groups)),
        "avg_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
        "num_paragraphs": int(len(paragraph_meta)),
        "avg_flows_per_paragraph": float(np.mean(source_flow_counts)) if source_flow_counts else 0.0,
        "max_flows_per_paragraph": int(max(source_flow_counts)) if source_flow_counts else 0,
        "truncated_paragraphs": int(truncated_count),
        "truncation_ratio": float(truncated_count / len(paragraph_meta)) if paragraph_meta else 0.0,
        "skipped_short_windows": int(skipped_short),
        "vocab_size": int(len(vocab)),
        "max_len": int(max_len),
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--group_by", choices=["service_key", "service_host_proto", "global_time"], default="service_host_proto")
    parser.add_argument("--window_n", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--min_flows", type=int, default=2)
    parser.add_argument("--paragraph_sep_token", default="[FLOW_SEP]")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False)
    paragraph_data, stats = build_paragraph_token_data(
        token_data,
        group_by=args.group_by,
        window_n=args.window_n,
        stride=args.stride,
        max_len=args.max_len,
        min_flows=args.min_flows,
        paragraph_sep_token=args.paragraph_sep_token,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(paragraph_data, out_path)
    stats["command"] = shlex.join(sys.argv)
    stats["tokens"] = args.tokens
    stats["out"] = args.out
    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    write_json(stats, stats_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
