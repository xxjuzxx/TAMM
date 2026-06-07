from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from src.features.tokenizer import SPECIAL_TOKENS


@dataclass(frozen=True)
class TokenCorpusSource:
    path: str
    name: str
    corpus: dict[str, Any]


def infer_source_name(path: str | Path) -> str:
    stem = Path(path).stem
    for marker in ("_tokens", "_vocab", "_stats", "_manifest"):
        idx = stem.find(marker)
        if idx > 0:
            stem = stem[:idx]
            break
    return stem


def _inverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    inverse: dict[int, str] = {}
    for token, idx in vocab.items():
        inverse[int(idx)] = str(token)
    return inverse


def _copy_vocab_by_index(vocab: dict[str, int]) -> dict[str, int]:
    items = sorted(((str(token), int(idx)) for token, idx in vocab.items()), key=lambda item: item[1])
    expected = list(range(len(items)))
    observed = [idx for _token, idx in items]
    if observed != expected:
        raise ValueError("base vocabulary ids must be contiguous from 0 to vocab_size - 1")
    return {token: idx for token, idx in items}


def _normalize_service_key(meta: dict[str, Any], source_name: str) -> list[str]:
    key = meta.get("service_key")
    if isinstance(key, (list, tuple)) and key:
        return [str(item) for item in key]
    if key not in (None, ""):
        return [str(key)]
    return ["SOURCE", source_name, "GLOBAL"]


def _normalize_meta(
    meta: dict[str, Any],
    *,
    source_name: str,
    source_path: str,
    source_row_index: int,
    target_max_len: int,
    source_sequence_length: int,
) -> dict[str, Any]:
    out = dict(meta)
    out["source_dataset"] = source_name
    out["source_file"] = source_path
    out["source_row_index"] = int(source_row_index)
    out["merged_max_len"] = int(target_max_len)
    out["source_sequence_length"] = int(source_sequence_length)
    out["merged_truncated"] = bool(source_sequence_length > target_max_len)
    out["service_key"] = _normalize_service_key(out, source_name)
    if out.get("dataset_file") is None:
        out["dataset_file"] = source_path
    return out


def _pad_or_truncate_tensor(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    if tensor.shape[0] >= target_len:
        return tensor[:target_len]
    pad = torch.full((target_len - tensor.shape[0],), pad_value, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def merge_token_corpora(
    sources: Sequence[TokenCorpusSource],
    *,
    target_max_len: int | None = None,
    max_rows_per_source: int | None = None,
    source_names: Sequence[str] | None = None,
    preserve_first_vocab: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not sources:
        raise ValueError("merge_token_corpora requires at least one source")
    if source_names is not None and len(source_names) != len(sources):
        raise ValueError("source_names length must match sources length")

    resolved_names = [str(source_names[idx]) if source_names is not None else sources[idx].name for idx in range(len(sources))]
    source_lengths = [int(corpus["input_ids"].shape[1]) for _src, corpus in ((src, src.corpus) for src in sources)]
    resolved_max_len = int(target_max_len) if target_max_len is not None else max(source_lengths)
    if preserve_first_vocab:
        merged_vocab = _copy_vocab_by_index(sources[0].corpus["vocab"])
    else:
        merged_vocab: dict[str, int] = {}
        for token in SPECIAL_TOKENS:
            merged_vocab.setdefault(token, len(merged_vocab))

    source_stats: list[dict[str, Any]] = []
    per_source_rows: list[list[dict[str, Any]]] = []
    source_inverse_vocabs = [_inverse_vocab(src.corpus["vocab"]) for src in sources]
    for source_idx, source in enumerate(sources):
        corpus = source.corpus
        name = resolved_names[source_idx]
        input_ids: torch.Tensor = corpus["input_ids"]
        attention_mask: torch.Tensor = corpus["attention_mask"]
        token_type_ids: torch.Tensor = corpus["token_type_ids"]
        meta_rows: list[dict[str, Any]] = list(corpus.get("meta", []))
        if len(meta_rows) != int(input_ids.shape[0]):
            raise ValueError(f"meta row count does not match input_ids for source {source.path}")
        row_limit = int(max_rows_per_source) if max_rows_per_source is not None else int(input_ids.shape[0])
        row_limit = min(row_limit, int(input_ids.shape[0]))
        selected_rows: list[dict[str, Any]] = []
        truncated_rows = 0
        inverse_vocab = source_inverse_vocabs[source_idx]
        for row_idx in range(row_limit):
            ids = input_ids[row_idx]
            mask = attention_mask[row_idx]
            types = token_type_ids[row_idx]
            source_sequence_length = int(torch.count_nonzero(mask).item())
            trimmed_len = min(int(ids.shape[0]), resolved_max_len)
            trimmed_ids = ids[:trimmed_len].tolist()
            trimmed_mask = mask[:trimmed_len]
            trimmed_types = types[:trimmed_len]
            if source_sequence_length > resolved_max_len:
                truncated_rows += 1
            tokens = [inverse_vocab.get(int(idx), "[UNK]") for idx in trimmed_ids]
            mapped_ids = []
            for token in tokens:
                if token not in merged_vocab:
                    merged_vocab[token] = len(merged_vocab)
                mapped_ids.append(merged_vocab[token])
            pad_id = merged_vocab["[PAD]"]
            remapped_ids = _pad_or_truncate_tensor(torch.tensor(mapped_ids, dtype=torch.long), resolved_max_len, pad_id)
            remapped_mask = _pad_or_truncate_tensor(trimmed_mask.to(dtype=torch.long), resolved_max_len, 0)
            remapped_types = _pad_or_truncate_tensor(trimmed_types.to(dtype=torch.long), resolved_max_len, 0)
            selected_rows.append(
                {
                    "input_ids": remapped_ids.tolist(),
                    "attention_mask": remapped_mask.tolist(),
                    "token_type_ids": remapped_types.tolist(),
                    "meta": _normalize_meta(
                        meta_rows[row_idx],
                        source_name=name,
                        source_path=source.path,
                        source_row_index=row_idx,
                        target_max_len=resolved_max_len,
                        source_sequence_length=source_sequence_length,
                    ),
                }
            )
        per_source_rows.append(selected_rows)
        source_stats.append(
            {
                "source_name": name,
                "path": source.path,
                "rows": len(selected_rows),
                "source_max_len": int(input_ids.shape[1]),
                "target_max_len": resolved_max_len,
                "truncated_rows": int(truncated_rows),
                "truncation_ratio": float(truncated_rows / len(selected_rows)) if selected_rows else 0.0,
                "source_vocab_size": int(len(corpus["vocab"])),
            }
        )

    merged_rows = [row for rows in per_source_rows for row in rows]
    merged = {
        "input_ids": torch.tensor([row["input_ids"] for row in merged_rows], dtype=torch.long),
        "attention_mask": torch.tensor([row["attention_mask"] for row in merged_rows], dtype=torch.long),
        "token_type_ids": torch.tensor([row["token_type_ids"] for row in merged_rows], dtype=torch.long),
        "meta": [row["meta"] for row in merged_rows],
        "vocab": dict(merged_vocab),
        "max_len": resolved_max_len,
        "profile_mode": "mixed",
        "use_service_context": any(bool(src.corpus.get("use_service_context", False)) for src in sources),
        "record_service_context": any(bool(src.corpus.get("record_service_context", False)) for src in sources),
        "use_service_tokens": any(bool(src.corpus.get("use_service_tokens", False)) for src in sources),
        "use_burst_shape_tokens": any(bool(src.corpus.get("use_burst_shape_tokens", False)) for src in sources),
        "use_first_k_signature": any(bool(src.corpus.get("use_first_k_signature", False)) for src in sources),
        "first_k": max(int(src.corpus.get("first_k", 0) or 0) for src in sources) if any("first_k" in src.corpus for src in sources) else 0,
        "use_context_profile_tokens": any(bool(src.corpus.get("use_context_profile_tokens", False)) for src in sources),
        "use_transition_profile_tokens": any(bool(src.corpus.get("use_transition_profile_tokens", False)) for src in sources),
        "sources": source_stats,
    }
    stats = {
        "num_sources": len(sources),
        "num_rows": len(merged_rows),
        "vocab_size": len(merged_vocab),
        "max_len": resolved_max_len,
        "source_stats": source_stats,
        "avg_rows_per_source": float(len(merged_rows) / len(sources)) if sources else 0.0,
        "truncated_rows_total": int(sum(item["truncated_rows"] for item in source_stats)),
        "truncation_ratio_total": float(sum(item["truncated_rows"] for item in source_stats) / len(merged_rows)) if merged_rows else 0.0,
        "preserve_first_vocab": bool(preserve_first_vocab),
    }
    return merged, stats
