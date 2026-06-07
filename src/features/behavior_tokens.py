from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from src.data.splits import split_lookup
from src.features.tokenizer import PrimitiveTrafficTokenizer, Vocabulary, _service_context_rows
from src.features.token_alias import is_burst_token, is_flow_summary_token, is_packet_token, is_profile_token, is_rhythm_token


EMPTY_PROFILE_PRIMITIVES = {
    "short": None,
    "same": None,
    "packet": [],
    "local": [],
    "repeat": [],
    "duplicate": [],
}


def _normal_binary_label(flow: dict[str, Any]) -> str:
    value = str(flow.get("binary_label") or "").upper()
    if value in {"BENIGN", "ATTACK"}:
        return value
    return "BENIGN" if str(flow.get("label", "")).lower() == "benign" else "ATTACK"


def _class_label(flow: dict[str, Any], label_field: str) -> str:
    if label_field == "attack_family":
        return str(flow.get("attack_family") or flow.get("label") or "")
    return str(flow.get("label") or flow.get("attack_family") or "")


def _label_mapping(flows: list[dict[str, Any]], label_field: str) -> dict[str, int]:
    labels = sorted({_class_label(flow, label_field) for flow in flows})
    if "BENIGN" in labels:
        labels = ["BENIGN"] + [label for label in labels if label != "BENIGN"]
    elif "Benign" in labels:
        labels = ["Benign"] + [label for label in labels if label != "Benign"]
    return {label: idx for idx, label in enumerate(labels)}


def _context_id(flow: dict[str, Any]) -> str:
    dataset = str(flow.get("dataset") or "dataset")
    day = str(flow.get("day") or "day")
    proto = str(flow.get("proto") or flow.get("protocol") or "proto").lower()
    return f"{dataset}:{day}:{proto}"


def _pad(ids: list[int], max_len: int, pad_id: int) -> tuple[list[int], list[int], list[int]]:
    ids = ids[:max_len]
    attention = [1] * len(ids)
    token_types = [0] * len(ids)
    pad_len = max_len - len(ids)
    if pad_len > 0:
        ids = ids + [pad_id] * pad_len
        attention = attention + [0] * pad_len
        token_types = token_types + [0] * pad_len
    return ids, attention, token_types


def _profile_by_id(profile_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("flow_id")): row for row in profile_rows}


def _default_profile_row(flow_id: str) -> dict[str, Any]:
    return {"flow_id": flow_id, "profile": dict(EMPTY_PROFILE_PRIMITIVES)}


def build_behavior_token_dataset(
    flows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    split_payload: dict[str, Any],
    tokenizer_config: dict[str, Any],
    *,
    max_len: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = dict(tokenizer_config)
    if max_len is not None:
        cfg["max_len"] = int(max_len)
    cfg["use_service_tokens"] = bool(cfg.get("use_service_tokens", False))
    has_port_tokens = bool(cfg["use_service_tokens"])
    include_flow_summary = bool(cfg.pop("include_flow_summary", True))
    include_packet_tokens = bool(cfg.pop("include_packet_tokens", True))
    include_burst_tokens = bool(cfg.pop("include_burst_tokens", True))
    include_rhythm_tokens = bool(cfg.pop("include_rhythm_tokens", True))
    label_field = str(cfg.pop("label_field", "label"))
    tokenizer = PrimitiveTrafficTokenizer(**cfg, vocab=Vocabulary())
    tokenizer.vocab = Vocabulary()
    max_len_resolved = int(tokenizer.max_len)
    lookup = split_lookup(split_payload)
    profile_map = _profile_by_id(profile_rows)

    flows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for flow in flows:
        split_name = lookup.get(str(flow.get("flow_id")))
        if split_name in flows_by_split:
            flows_by_split[split_name].append(flow)

    service_context_by_id: dict[str, dict[str, Any]] = {}
    if tokenizer.record_service_context:
        for split_flows in flows_by_split.values():
            service_context_by_id.update(
                _service_context_rows(
                    split_flows,
                    window_seconds=tokenizer.service_context_window_seconds,
                    short_packet_threshold=tokenizer.service_context_short_packet_threshold,
                    count_bin_max=tokenizer.count_bin_max,
                )
            )

    train_ids = {str(flow.get("flow_id")) for flow in flows_by_split["train"]}
    raw_token_lengths: list[int] = []
    train_selected_token_lengths: list[int] = []
    for flow in flows:
        flow_id = str(flow.get("flow_id"))
        if flow_id not in train_ids:
            continue
        profile = profile_map.get(flow_id, _default_profile_row(flow_id))
        raw_tokens = _select_ablation_tokens(
            tokenizer.raw_flow_tokens(flow, profile, service_context=service_context_by_id.get(flow_id)),
            include_flow_summary=include_flow_summary,
            include_packet_tokens=include_packet_tokens,
            include_burst_tokens=include_burst_tokens,
            include_rhythm_tokens=include_rhythm_tokens,
        )
        raw_token_lengths.append(len(raw_tokens))
        train_selected_token_lengths.append(min(len(raw_tokens), max_len_resolved))
        for token in raw_tokens[:max_len_resolved]:
            tokenizer.vocab.add(token)

    label_to_id = _label_mapping(flows, label_field)
    binary_label_to_id = {"BENIGN": 0, "ATTACK": 1}
    encoded: list[dict[str, Any]] = []
    unknown_counts: Counter[str] = Counter()
    total_token_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for flow in flows:
        flow_id = str(flow.get("flow_id"))
        split_name = lookup.get(flow_id)
        if split_name is None:
            continue
        profile = profile_map.get(flow_id, _default_profile_row(flow_id))
        raw_tokens = _select_ablation_tokens(
            tokenizer.raw_flow_tokens(flow, profile, service_context=service_context_by_id.get(flow_id)),
            include_flow_summary=include_flow_summary,
            include_packet_tokens=include_packet_tokens,
            include_burst_tokens=include_burst_tokens,
            include_rhythm_tokens=include_rhythm_tokens,
        )
        tokens = raw_tokens[:max_len_resolved]
        unknown_counts[split_name] += sum(1 for token in tokens if token not in tokenizer.vocab.token_to_id)
        total_token_counts[split_name] += len(tokens)
        input_ids, attention_mask, token_type_ids = _pad(tokenizer.vocab.encode(tokens), max_len_resolved, tokenizer.vocab.pad_id)
        label = _class_label(flow, label_field)
        binary_label = _normal_binary_label(flow)
        row = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": int(label_to_id[label]),
            "binary_labels": int(binary_label_to_id[binary_label]),
            "attack_family": str(flow.get("attack_family") or label),
            "meta": {
                "flow_id": flow_id,
                "split": split_name,
                "dataset": str(flow.get("dataset") or "CICIDS2017"),
                "context_id": _context_id(flow),
                "token_len": min(len(tokens), max_len_resolved),
                "token_count": min(len(tokens), max_len_resolved),
                "raw_token_count": len(raw_tokens),
                "truncated": len(raw_tokens) > max_len_resolved,
                "has_profile_token": any(is_profile_token(token) for token in tokens),
                "has_rhythm_token": any(is_rhythm_token(token) for token in tokens),
                "has_port_token": has_port_tokens,
                "has_ip_token": False,
                "has_abs_time_token": False,
                "label": label,
                "binary_label": binary_label,
                "attack_family": str(flow.get("attack_family") or label),
                "label_field": label_field,
            },
        }
        encoded.append(row)
        split_counts[split_name] += 1

    token_data = {
        "input_ids": torch.tensor([row["input_ids"] for row in encoded], dtype=torch.long),
        "attention_mask": torch.tensor([row["attention_mask"] for row in encoded], dtype=torch.long),
        "token_type_ids": torch.tensor([row["token_type_ids"] for row in encoded], dtype=torch.long),
        "labels": torch.tensor([row["labels"] for row in encoded], dtype=torch.long),
        "binary_labels": torch.tensor([row["binary_labels"] for row in encoded], dtype=torch.long),
        "attack_family": [row["attack_family"] for row in encoded],
        "meta": [row["meta"] for row in encoded],
        "label_to_id": label_to_id,
        "binary_label_to_id": binary_label_to_id,
        "vocab": tokenizer.vocab.to_dict(),
        "max_len": max_len_resolved,
        "tokenizer_config": cfg,
        "ablation_features": {
            "include_flow_summary": include_flow_summary,
            "include_packet_tokens": include_packet_tokens,
            "include_burst_tokens": include_burst_tokens,
            "include_rhythm_tokens": include_rhythm_tokens,
            "profile_mode": cfg.get("profile_mode", "full"),
        },
        "vocab_provenance": "train_only",
        "train_only": True,
        "threshold_tuning_split": "val",
        "label_field": label_field,
    }
    lengths = [int(row["meta"]["token_len"]) for row in encoded]
    raw_lengths = [int(row["meta"]["raw_token_count"]) for row in encoded]
    token_stats = _token_statistics(encoded, unknown_counts, total_token_counts)
    stats = {
        "num_rows": len(encoded),
        "num_flows": len(encoded),
        "split_counts": dict(sorted(split_counts.items())),
        "labels": label_to_id,
        "binary_labels": binary_label_to_id,
        "vocab_size": len(tokenizer.vocab.token_to_id),
        "vocab_provenance": "train_only",
        "train_only": True,
        "train_raw_token_count_avg": float(sum(raw_token_lengths) / len(raw_token_lengths)) if raw_token_lengths else 0.0,
        "avg_token_length": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "avg_raw_token_length": float(sum(raw_lengths) / len(raw_lengths)) if raw_lengths else 0.0,
        "truncated_count": int(sum(1 for row in encoded if row["meta"]["truncated"])),
        "unknown_token_counts": dict(sorted(unknown_counts.items())),
        "has_port_token": has_port_tokens,
        "has_ip_token": False,
        "has_abs_time_token": False,
        **token_stats,
        "ablation_features": {
            "include_flow_summary": include_flow_summary,
            "include_packet_tokens": include_packet_tokens,
            "include_burst_tokens": include_burst_tokens,
            "include_rhythm_tokens": include_rhythm_tokens,
            "profile_mode": cfg.get("profile_mode", "full"),
        },
    }
    return token_data, stats


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round(float(q) * (len(ordered) - 1)))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def _token_statistics(
    encoded: list[dict[str, Any]],
    unknown_counts: Counter[str],
    total_token_counts: Counter[str],
) -> dict[str, Any]:
    raw_lengths = [int(row["meta"]["raw_token_count"]) for row in encoded]
    kept_lengths = [int(row["meta"]["token_count"]) for row in encoded]
    truncated = [bool(row["meta"]["truncated"]) for row in encoded]
    total_tokens = sum(total_token_counts.values())
    total_unknown = sum(unknown_counts.values())
    per_split_stats: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        rows = [row for row in encoded if row["meta"]["split"] == split_name]
        split_raw = [int(row["meta"]["raw_token_count"]) for row in rows]
        split_kept = [int(row["meta"]["token_count"]) for row in rows]
        split_total = int(total_token_counts.get(split_name, 0))
        per_split_stats[split_name] = {
            "count": len(rows),
            "avg_len": float(sum(split_kept) / len(split_kept)) if split_kept else 0.0,
            "p50_len": _percentile(split_kept, 0.5),
            "p95_len": _percentile(split_kept, 0.95),
            "avg_raw_len": float(sum(split_raw) / len(split_raw)) if split_raw else 0.0,
            "truncation_rate": (sum(1 for row in rows if row["meta"]["truncated"]) / len(rows)) if rows else 0.0,
            "unk_token_rate": (float(unknown_counts.get(split_name, 0)) / float(split_total)) if split_total else 0.0,
        }
    profile_rows = 0
    rhythm_rows = 0
    for row in encoded:
        if bool(row["meta"].get("has_profile_token")):
            profile_rows += 1
        if bool(row["meta"].get("has_rhythm_token")):
            rhythm_rows += 1
    return {
        "avg_len": float(sum(kept_lengths) / len(kept_lengths)) if kept_lengths else 0.0,
        "p50_len": _percentile(kept_lengths, 0.5),
        "p95_len": _percentile(kept_lengths, 0.95),
        "avg_raw_len": float(sum(raw_lengths) / len(raw_lengths)) if raw_lengths else 0.0,
        "truncation_rate": (float(sum(1 for item in truncated if item)) / float(len(truncated))) if truncated else 0.0,
        "unk_token_rate": (float(total_unknown) / float(total_tokens)) if total_tokens else 0.0,
        "unk_profile_rate": 0.0,
        "rhythm_token_coverage": (float(rhythm_rows) / float(len(encoded))) if encoded else 0.0,
        "profile_token_coverage": (float(profile_rows) / float(len(encoded))) if encoded else 0.0,
        "per_split_stats": per_split_stats,
    }


def _select_ablation_tokens(
    tokens: list[str],
    *,
    include_flow_summary: bool,
    include_packet_tokens: bool,
    include_burst_tokens: bool,
    include_rhythm_tokens: bool,
) -> list[str]:
    out: list[str] = []
    for token in tokens:
        if token in {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}:
            out.append(token)
        elif is_flow_summary_token(token):
            if include_flow_summary:
                out.append(token)
        elif is_rhythm_token(token):
            if include_rhythm_tokens:
                out.append(token)
        elif is_burst_token(token):
            if include_burst_tokens:
                out.append(token)
        elif is_packet_token(token) or is_profile_token(token, include_none=True):
            if include_packet_tokens:
                out.append(token)
        else:
            out.append(token)
    return out
