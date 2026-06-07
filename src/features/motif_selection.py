from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from src.features.token_alias import SPECIAL_TOKENS, canonical_tokens


DEFAULT_MOTIF_PREFIXES = ("PRIM_PROFILE_", "PRIM_STRUCT_", "DM_SEQ_", "DM_BURST_", "DM_IAT_", "DM_TRANS_")


@dataclass(frozen=True)
class MotifSelectionConfig:
    min_support: float = 0.005
    max_support: float = 0.95
    bootstrap_samples: int = 20
    random_seed: int = 43
    core_quantile: float = 0.50
    tail_quantile: float = 0.95
    alpha: float = 0.25
    beta: float = 0.20
    gamma: float = 0.15
    delta: float = 0.30
    lambda_red: float = 0.20
    dictionary_size: int = 200
    redundancy_threshold: float = 0.90
    profile_cap_fraction: float | None = None
    structural_cap_fraction: float | None = None
    data_mined_cap_fraction: float | None = None


def load_token_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load a token corpus into normalized flow records.

    The loader supports the current PyTorch token corpus artifacts and simple
    JSONL records. Each output record has at least flow_id, tokens, split, and
    label fields. Endpoint, raw time, protocol, service, and five-tuple fields
    are intentionally ignored by this motif-selection layer.
    """

    path = Path(path)
    if path.suffix == ".pt":
        corpus = torch.load(path, map_location="cpu", weights_only=False)
        inv_vocab = {int(idx): str(token) for token, idx in corpus["vocab"].items()}
        input_ids = corpus["input_ids"].cpu().numpy()
        attention_mask = corpus["attention_mask"].cpu().numpy()
        records: list[dict[str, Any]] = []
        labels = corpus.get("binary_labels")
        label_values = labels.cpu().numpy().tolist() if labels is not None else [None] * len(corpus.get("meta", []))
        for row_idx, meta in enumerate(corpus.get("meta", [])):
            active = input_ids[row_idx][attention_mask[row_idx] > 0]
            tokens = canonical_tokens(inv_vocab.get(int(token_id), "[UNK]") for token_id in active)
            records.append(
                {
                    "flow_id": meta.get("flow_id") or meta.get("id") or str(row_idx),
                    "tokens": tokens,
                    "split": meta.get("split", ""),
                    "label": meta.get("binary_label") or meta.get("label") or label_values[row_idx],
                    "row_index": row_idx,
                }
            )
        return records
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for row_idx, line in enumerate(handle):
                if not line.strip():
                    continue
                obj = json.loads(line)
                tokens = obj.get("tokens") or obj.get("behavior_tokens") or obj.get("token_sequence") or []
                records.append(
                    {
                        "flow_id": obj.get("flow_id") or obj.get("id") or str(row_idx),
                        "tokens": canonical_tokens(tokens),
                        "split": obj.get("split", ""),
                        "label": obj.get("binary_label") or obj.get("label"),
                        "row_index": row_idx,
                        "score": obj.get("score") or obj.get("anomaly_score"),
                    }
                )
        return records
    raise ValueError(f"Unsupported token corpus format: {path}")


def is_candidate_motif(token: str, motif_prefixes: Sequence[str] = DEFAULT_MOTIF_PREFIXES) -> bool:
    tok = str(token)
    if tok in SPECIAL_TOKENS:
        return False
    return any(tok.startswith(prefix) for prefix in motif_prefixes)


def motif_family(token: str) -> str:
    if token.startswith("PRIM_PROFILE_"):
        return "profile"
    if token.startswith("PRIM_STRUCT_"):
        rest = token.removeprefix("PRIM_STRUCT_")
        return "structural_" + (rest.split("_", 1)[0].lower() if rest else "unknown")
    if token.startswith("DM_"):
        return "data_mined"
    return "motif"


def extract_candidate_motifs(records: Sequence[dict[str, Any]], motif_prefixes: Sequence[str] = DEFAULT_MOTIF_PREFIXES) -> list[str]:
    """Return stable-sorted candidate motif tokens observed in the supplied records."""

    motifs = {token for record in records for token in set(record.get("tokens", [])) if is_candidate_motif(token, motif_prefixes)}
    return sorted(motifs)


def build_occurrence_matrix(records: Sequence[dict[str, Any]], candidate_motifs: Sequence[str]) -> np.ndarray:
    """Build a binary flow-by-motif occurrence matrix."""

    col = {motif: idx for idx, motif in enumerate(candidate_motifs)}
    mat = np.zeros((len(records), len(candidate_motifs)), dtype=bool)
    for row_idx, record in enumerate(records):
        for token in set(record.get("tokens", [])):
            idx = col.get(token)
            if idx is not None:
                mat[row_idx, idx] = True
    return mat


def compute_support(occurrence: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Compute train-only binary flow-level support."""

    if len(train_idx) == 0:
        return np.zeros(occurrence.shape[1], dtype=np.float64)
    return occurrence[train_idx].mean(axis=0).astype(np.float64)


def compute_stability(
    occurrence: np.ndarray,
    train_idx: np.ndarray,
    *,
    bootstrap_samples: int = 20,
    random_seed: int = 43,
    eps: float = 1e-9,
) -> np.ndarray:
    """Compute bootstrap support stability on training flows."""

    if len(train_idx) == 0:
        return np.zeros(occurrence.shape[1], dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    supports = []
    for _ in range(max(1, int(bootstrap_samples))):
        sample = rng.choice(train_idx, size=len(train_idx), replace=True)
        supports.append(occurrence[sample].mean(axis=0))
    boot = np.vstack(supports).astype(np.float64)
    mu = boot.mean(axis=0)
    sigma = boot.std(axis=0)
    return np.clip(1.0 - np.minimum(1.0, sigma / (mu + eps)), 0.0, 1.0)


def compute_coverage_or_compression(
    records: Sequence[dict[str, Any]],
    candidate_motifs: Sequence[str],
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a log-count coverage proxy and raw train occurrence counts."""

    col = {motif: idx for idx, motif in enumerate(candidate_motifs)}
    counts = np.zeros(len(candidate_motifs), dtype=np.float64)
    for idx in train_idx.tolist():
        token_counts = Counter(records[int(idx)].get("tokens", []))
        for token, count in token_counts.items():
            pos = col.get(token)
            if pos is not None:
                counts[pos] += float(count)
    values = np.log1p(counts)
    max_value = float(values.max()) if values.size else 0.0
    coverage = values / max_value if max_value > 0 else values
    return coverage.astype(np.float64), counts.astype(np.float64)


def compute_tail_sensitivity(
    occurrence: np.ndarray,
    val_idx: np.ndarray,
    val_scores: np.ndarray | None,
    *,
    core_quantile: float = 0.50,
    tail_quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate benign-tail sensitivity from validation scores without attack labels."""

    if len(val_idx) == 0 or val_scores is None or len(val_scores) != len(val_idx):
        zeros = np.zeros(occurrence.shape[1], dtype=np.float64)
        return zeros, zeros, zeros
    val_scores = np.asarray(val_scores, dtype=np.float64)
    core_cut = np.quantile(val_scores, float(core_quantile))
    tail_cut = np.quantile(val_scores, float(tail_quantile))
    core_local = np.flatnonzero(val_scores <= core_cut)
    tail_local = np.flatnonzero(val_scores >= tail_cut)
    if core_local.size == 0 or tail_local.size == 0:
        zeros = np.zeros(occurrence.shape[1], dtype=np.float64)
        return zeros, zeros, zeros
    core_idx = val_idx[core_local]
    tail_idx = val_idx[tail_local]
    core_rate = occurrence[core_idx].mean(axis=0).astype(np.float64)
    tail_rate = occurrence[tail_idx].mean(axis=0).astype(np.float64)
    return np.abs(tail_rate - core_rate), core_rate, tail_rate


def _normalize_term(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if math.isclose(lo, hi):
        return np.ones_like(values) if hi > 0 else np.zeros_like(values)
    return (values - lo) / (hi - lo)


def compute_redundancy(candidate_vector: np.ndarray, selected_vectors: list[np.ndarray]) -> float:
    """Compute maximum Jaccard redundancy to selected motifs."""

    if not selected_vectors:
        return 0.0
    cand = candidate_vector.astype(bool)
    max_sim = 0.0
    for vec in selected_vectors:
        other = vec.astype(bool)
        union = np.logical_or(cand, other).sum()
        sim = float(np.logical_and(cand, other).sum() / union) if union else 0.0
        max_sim = max(max_sim, sim)
    return max_sim


def select_motif_dictionary(
    records: Sequence[dict[str, Any]],
    candidate_motifs: Sequence[str],
    occurrence: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    val_scores: np.ndarray | None,
    *,
    config: MotifSelectionConfig | None = None,
    strategy: str = "full_utility",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a motif dictionary and return selected entries plus a full report."""

    cfg = config or MotifSelectionConfig()
    support = compute_support(occurrence, train_idx)
    stability = compute_stability(occurrence, train_idx, bootstrap_samples=cfg.bootstrap_samples, random_seed=cfg.random_seed)
    coverage, train_counts = compute_coverage_or_compression(records, candidate_motifs, train_idx)
    tail, core_rate, tail_rate = compute_tail_sensitivity(
        occurrence,
        val_idx,
        val_scores,
        core_quantile=cfg.core_quantile,
        tail_quantile=cfg.tail_quantile,
    )

    valid = (support >= cfg.min_support) & (support <= cfg.max_support)
    sup_n = _normalize_term(support)
    stab_n = _normalize_term(stability)
    cov_n = _normalize_term(coverage)
    tail_n = _normalize_term(tail)

    if strategy == "all_candidates":
        base = valid.astype(np.float64)
        max_k = int(np.sum(valid))
        use_red = False
    elif strategy == "support_only":
        base = sup_n
        max_k = cfg.dictionary_size
        use_red = False
    elif strategy == "support_stability":
        base = cfg.alpha * sup_n + cfg.beta * stab_n
        max_k = cfg.dictionary_size
        use_red = False
    elif strategy == "support_stability_tail":
        base = cfg.alpha * sup_n + cfg.beta * stab_n + cfg.delta * tail_n
        max_k = cfg.dictionary_size
        use_red = False
    elif strategy == "full_utility":
        base = cfg.alpha * sup_n + cfg.beta * stab_n + cfg.gamma * cov_n + cfg.delta * tail_n
        max_k = cfg.dictionary_size
        use_red = True
    else:
        raise ValueError(f"Unknown motif selection strategy: {strategy}")

    order = sorted(range(len(candidate_motifs)), key=lambda idx: (-float(base[idx]), candidate_motifs[idx]))
    selected: list[dict[str, Any]] = []
    selected_vectors: list[np.ndarray] = []
    selected_family_counts: Counter[str] = Counter()
    caps = {
        "profile": cfg.profile_cap_fraction,
        "data_mined": cfg.data_mined_cap_fraction,
    }
    for idx in order:
        if not valid[idx]:
            continue
        token = candidate_motifs[idx]
        family = motif_family(token)
        family_key = "profile" if family == "profile" else ("data_mined" if family == "data_mined" else "structural")
        cap_fraction = cfg.structural_cap_fraction if family_key == "structural" else caps.get(family_key)
        if cap_fraction is not None and selected_family_counts[family_key] >= int(max_k * float(cap_fraction)):
            continue
        red = compute_redundancy(occurrence[train_idx, idx], selected_vectors) if use_red else 0.0
        if use_red and red > cfg.redundancy_threshold:
            continue
        final_utility = float(base[idx] - (cfg.lambda_red * red if use_red else 0.0))
        if strategy != "all_candidates" and final_utility <= 0:
            continue
        entry = {
            "motif": token,
            "motif_family": family,
            "rank": len(selected) + 1,
            "support": float(support[idx]),
            "stability": float(stability[idx]),
            "coverage": float(coverage[idx]),
            "tail_sensitivity": float(tail[idx]),
            "redundancy_at_selection": float(red),
            "utility_score": final_utility,
            "train_occurrence_count": int(train_counts[idx]),
            "validation_core_occurrence_rate": float(core_rate[idx]),
            "validation_tail_occurrence_rate": float(tail_rate[idx]),
            "selection_strategy": strategy,
            "selected": True,
        }
        selected.append(entry)
        selected_vectors.append(occurrence[train_idx, idx])
        selected_family_counts[family_key] += 1
        if len(selected) >= max_k:
            break

    selected_set = {entry["motif"] for entry in selected}
    report: list[dict[str, Any]] = []
    for idx, token in enumerate(candidate_motifs):
        report.append(
            {
                "motif": token,
                "motif_family": motif_family(token),
                "selected": token in selected_set,
                "rank": next((entry["rank"] for entry in selected if entry["motif"] == token), ""),
                "support": float(support[idx]),
                "stability": float(stability[idx]),
                "coverage": float(coverage[idx]),
                "tail_sensitivity": float(tail[idx]),
                "utility_score": float(base[idx]),
                "train_occurrence_count": int(train_counts[idx]),
                "validation_core_occurrence_rate": float(core_rate[idx]),
                "validation_tail_occurrence_rate": float(tail_rate[idx]),
                "filtered_by_support": bool(not valid[idx]),
                "selection_strategy": strategy,
            }
        )
    return selected, report


def save_motif_dictionary(
    selected: Sequence[dict[str, Any]],
    report: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    config: MotifSelectionConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save selected dictionary JSON, full CSV report, and config JSON."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "config": asdict(config),
        "selected_motifs": list(selected),
        "selected_count": len(selected),
    }
    (out / "motif_dictionary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "motif_selection_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames: list[str] = []
    for row in report:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (out / "motif_selection_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def filter_records_by_dictionary(records: Sequence[dict[str, Any]], motif_dictionary: Iterable[str]) -> list[dict[str, Any]]:
    """Return records retaining only selected motif tokens."""

    keep = set(motif_dictionary)
    out = []
    for record in records:
        copied = dict(record)
        copied["tokens"] = [token for token in record.get("tokens", []) if token in keep]
        out.append(copied)
    return out
