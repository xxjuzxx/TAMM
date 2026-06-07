#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.features.motif_selection import (
    DEFAULT_MOTIF_PREFIXES,
    MotifSelectionConfig,
    build_occurrence_matrix,
    extract_candidate_motifs,
    load_token_corpus,
    motif_family,
    save_motif_dictionary,
    select_motif_dictionary,
)
from src.features.structural_primitives import (
    StructuralPrimitiveConfig,
    build_train_only_structural_primitive_vocabulary,
    extract_structural_primitive_candidates,
    filter_triggers,
)
from src.features.token_alias import SPECIAL_TOKENS, is_packet_burst_token


ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "scripts" / "52_sweep_anomaly_low_fpr.py"
DEFAULT_TOKEN_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown" / "tokens_category"
DEFAULT_OUT = ROOT / "results" / "motif_selection"
EXPERT_MOTIF_PREFIXES = ("PRIM_PROFILE_", "PRIM_STRUCT_")
DATA_MINED_MOTIF_PREFIXES = ("DM_SEQ_", "DM_BURST_", "DM_IAT_", "DM_TRANS_")
ATTACK_SLUG = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
}


def _load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("tamm_low_fpr_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sweep module from {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tamm_low_fpr_sweep"] = module
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _token_path(token_dir: Path, attack: str, seed: int) -> Path:
    return token_dir / f"cicids2017_leave_one_{ATTACK_SLUG[attack]}_anomaly_seed{seed}_a3_full_rhythm.pt"


def _read_token_data(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _split_indices(token_data: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray([idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == split], dtype=np.int64)


def _id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    return {int(idx): str(token) for token, idx in vocab.items()}


def _token_rows(token_data: dict[str, Any]) -> list[list[str]]:
    inv = _id_to_token(token_data["vocab"])
    input_ids = token_data["input_ids"].cpu().numpy()
    attention_mask = token_data["attention_mask"].cpu().numpy()
    rows = []
    for row_idx in range(input_ids.shape[0]):
        active = input_ids[row_idx][attention_mask[row_idx] > 0]
        rows.append([inv.get(int(token_id), "[UNK]") for token_id in active])
    return rows


def _augment_structural_rows(rows: list[list[str]], train_idx: np.ndarray, *, min_support: int = 5) -> tuple[list[list[str]], dict[str, int]]:
    cfg = StructuralPrimitiveConfig(
        enabled=True,
        enable_packet_shape_primitives=True,
        enable_burst_shape_primitives=True,
        enable_timing_rhythm_primitives=True,
        enable_direction_transition_primitives=True,
        enable_composite_primitives=True,
        min_support=min_support,
    )
    raw = [extract_structural_primitive_candidates(tokens, cfg) for tokens in rows]
    vocab = build_train_only_structural_primitive_vocabulary(raw, train_idx, min_support=min_support)
    out: list[list[str]] = []
    for tokens, triggers in zip(rows, raw):
        added: list[str] = []
        for trigger in filter_triggers(triggers, vocab):
            added.extend([trigger.name] * max(1, int(trigger.count)))
        if "[SEP]" in tokens:
            sep_idx = len(tokens) - 1 - tokens[::-1].index("[SEP]")
            out.append(tokens[:sep_idx] + added + tokens[sep_idx:])
        else:
            out.append(list(tokens) + added)
    return out, vocab


def _packet_dir_len_events(tokens: list[str]) -> list[str]:
    """Extract packet direction-length symbols without using endpoint identity."""

    events: list[str] = []
    pending_dir: str | None = None
    for token in tokens:
        if token.startswith("PKT_DIR_"):
            pending_dir = token.removeprefix("PKT_DIR_")
        elif pending_dir is not None and token.startswith("PKT_LEN_") and token.removeprefix("PKT_LEN_").isdigit():
            events.append(f"{pending_dir}_L{token.removeprefix('PKT_LEN_')}")
            pending_dir = None
    return events


def _packet_iat_events(tokens: list[str]) -> list[str]:
    return [f"I{token.removeprefix('PKT_IAT_')}" for token in tokens if token.startswith("PKT_IAT_")]


def _packet_dir_events(tokens: list[str]) -> list[str]:
    return [token.removeprefix("PKT_DIR_") for token in tokens if token.startswith("PKT_DIR_")]


def _burst_dir_events(tokens: list[str]) -> list[str]:
    return [token.removeprefix("BURST_DIR_") for token in tokens if token.startswith("BURST_DIR_")]


def _ngrams(symbols: list[str], n: int) -> list[str]:
    if n <= 0 or len(symbols) < n:
        return []
    return ["-".join(symbols[idx : idx + n]) for idx in range(0, len(symbols) - n + 1)]


def _data_mined_candidates_for_row(tokens: list[str], ngram_lengths: list[int]) -> list[str]:
    """Generate interpretable sequential motif candidates from symbolic behavior tokens."""

    out: list[str] = []
    dir_len = _packet_dir_len_events(tokens)
    iat = _packet_iat_events(tokens)
    burst_dir = _burst_dir_events(tokens)
    dirs = _packet_dir_events(tokens)
    for n in ngram_lengths:
        out.extend(f"DM_SEQ_DIRLEN_N{n}_{pattern}" for pattern in _ngrams(dir_len, n))
        out.extend(f"DM_IAT_NGRAM_N{n}_{pattern}" for pattern in _ngrams(iat, n))
        out.extend(f"DM_BURST_NGRAM_N{n}_{pattern}" for pattern in _ngrams(burst_dir, n))
    out.extend(f"DM_TRANS_{pattern}" for pattern in _ngrams(dirs, 2))
    return out


def _augment_data_mined_rows(
    rows: list[list[str]],
    train_idx: np.ndarray,
    *,
    ngram_lengths: list[int],
    min_support: float,
) -> tuple[list[list[str]], dict[str, int]]:
    """Append train-fitted data-mined motif tokens to rows.

    Candidate names are generated for every row from symbolic packet/burst
    sequences, but the deployable vocabulary is fixed using train-split
    flow-level support only. Test-only motifs are therefore not appended.
    """

    raw = [_data_mined_candidates_for_row(tokens, ngram_lengths) for tokens in rows]
    train_size = max(1, len(train_idx))
    min_count = int(np.ceil(float(min_support) * train_size)) if float(min_support) < 1.0 else int(min_support)
    min_count = max(1, min_count)
    support: Counter[str] = Counter()
    for idx in train_idx.tolist():
        support.update(set(raw[int(idx)]))
    vocab = {token: count for token, count in sorted(support.items()) if count >= min_count}
    out: list[list[str]] = []
    for tokens, candidates in zip(rows, raw):
        added = [token for token in candidates if token in vocab]
        if "[SEP]" in tokens:
            sep_idx = len(tokens) - 1 - tokens[::-1].index("[SEP]")
            out.append(tokens[:sep_idx] + added + tokens[sep_idx:])
        else:
            out.append(list(tokens) + added)
    return out, vocab


def _records_from_rows(token_data: dict[str, Any], rows: list[list[str]]) -> list[dict[str, Any]]:
    labels = token_data.get("binary_labels")
    label_values = labels.cpu().numpy().tolist() if labels is not None else [None] * len(token_data.get("meta", []))
    records = []
    for idx, meta in enumerate(token_data.get("meta", [])):
        records.append(
            {
                "flow_id": meta.get("flow_id") or str(idx),
                "tokens": rows[idx],
                "split": meta.get("split", ""),
                "label": meta.get("binary_label") or meta.get("label") or label_values[idx],
                "row_index": idx,
            }
        )
    return records


def _features_from_rows(rows: list[list[str]], keep_tokens: set[str], *, transform: str, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    kept = sorted(token for token in keep_tokens if token not in SPECIAL_TOKENS)
    if not kept:
        raise ValueError("No tokens selected for feature view")
    col = {token: idx for idx, token in enumerate(kept)}
    mat = np.zeros((len(rows), len(kept)), dtype=np.float32)
    for row_idx, tokens in enumerate(rows):
        counts = Counter(token for token in tokens if token in col)
        for token, count in counts.items():
            mat[row_idx, col[token]] = float(count)
    if transform.startswith("binary"):
        mat = (mat > 0).astype(np.float32)
        norm = transform.removeprefix("binary_") or "none"
    elif transform.startswith("count"):
        norm = transform.removeprefix("count_") or "none"
    elif transform.startswith("tfidf"):
        df = np.sum(mat[train_idx] > 0, axis=0)
        idf = np.log((1.0 + len(train_idx)) / (1.0 + df)) + 1.0
        mat = mat * idf.reshape(1, -1).astype(np.float32)
        norm = transform.removeprefix("tfidf_") or "none"
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    if norm == "l2":
        denom = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = np.divide(mat, denom, out=np.zeros_like(mat, dtype=np.float32), where=denom > 0)
    elif norm == "l1":
        denom = np.sum(np.abs(mat), axis=1, keepdims=True)
        mat = np.divide(mat, denom, out=np.zeros_like(mat, dtype=np.float32), where=denom > 0)
    elif norm != "none":
        raise ValueError(f"Unsupported norm: {norm}")
    return mat.astype(np.float32, copy=False), {
        "num_features": int(mat.shape[1]),
        "mean_nonzero": float(np.mean(np.sum(mat != 0, axis=1))),
        "kept_tokens": kept,
        "transform": transform,
    }


def _evaluate_features(
    token_data: dict[str, Any],
    features: np.ndarray,
    stats: dict[str, Any],
    *,
    feature_view: str,
    transform: str,
    scorer: str,
    k: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = _split_indices(token_data, "train")
    val_idx = _split_indices(token_data, "val")
    test_idx = _split_indices(token_data, "test")
    groups = ["GLOBAL"] * len(token_data.get("meta", []))
    start = time.perf_counter()
    val_scores = S._scores(features, train_idx, val_idx, groups, scorer=scorer, k=k)
    test_scores = S._scores(features, train_idx, test_idx, groups, scorer=scorer, k=k)
    score_seconds = time.perf_counter() - start
    metrics = S._evaluate(
        features,
        stats,
        groups,
        labels,
        train_idx,
        val_idx,
        test_idx,
        feature_filter=feature_view,
        transform=transform,
        scorer=scorer,
        k=k,
        group_mode="global",
    )
    return metrics, val_scores, test_scores, float(score_seconds)


def _metric_row(
    metrics: dict[str, Any],
    *,
    seed: int,
    attack: str,
    strategy: str,
    feature_view: str,
    candidate_source_mode: str,
    dictionary_size: int,
    candidate_count: int,
    selected_count: int,
    elapsed: float,
    score_seconds: float,
    transform: str,
    scorer: str,
    k: int,
) -> dict[str, Any]:
    p99_fpr = metrics.get("val_p99_0_false_positive_rate")
    return {
        "seed": seed,
        "heldout_attack": attack,
        "motif_selection_strategy": strategy,
        "feature_view": feature_view,
        "candidate_source_mode": candidate_source_mode,
        "dictionary_size_k": dictionary_size,
        "candidate_motif_count": candidate_count,
        "selected_motif_count": selected_count,
        "vocab_size": metrics.get("num_features"),
        "memory_size": metrics.get("train_size", ""),
        "transform": transform,
        "scorer": scorer,
        "k": k,
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "fpr95": metrics.get("fpr95"),
        "recall_at_0_1pct_fpr": metrics.get("recall_at_0_1pct_fpr"),
        "recall_at_1pct_fpr": metrics.get("recall_at_1pct_fpr"),
        "recall_at_5pct_fpr": metrics.get("recall_at_5pct_fpr"),
        "val_p99_threshold": metrics.get("val_p99_0_threshold"),
        "val_p99_realized_fpr": p99_fpr,
        "false_alerts_per_10k_benign": float(p99_fpr) * 10000.0 if p99_fpr not in (None, "") else "",
        "best_macro_f1": metrics.get("best_macro_f1"),
        "mean_nonzero": metrics.get("mean_nonzero"),
        "selection_seconds": elapsed,
        "score_seconds": score_seconds,
        "query_ms_per_flow": metrics.get("query_ms_per_flow", ""),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = [
        "auroc",
        "auprc",
        "fpr95",
        "recall_at_0_1pct_fpr",
        "recall_at_1pct_fpr",
        "recall_at_5pct_fpr",
        "val_p99_realized_fpr",
        "false_alerts_per_10k_benign",
        "selected_motif_count",
        "vocab_size",
        "selection_seconds",
        "score_seconds",
    ]
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["motif_selection_strategy"]),
            str(row.get("candidate_source_mode", "expert_plus_data_mined")),
            str(row["feature_view"]),
            int(row["dictionary_size_k"]),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for (strategy, mode, view, k), vals in sorted(groups.items()):
        agg: dict[str, Any] = {
            "motif_selection_strategy": strategy,
            "candidate_source_mode": mode,
            "feature_view": view,
            "dictionary_size_k": k,
            "runs": len(vals),
        }
        for col in numeric:
            nums = []
            for row in vals:
                try:
                    nums.append(float(row[col]))
                except (KeyError, TypeError, ValueError):
                    pass
            if nums:
                agg[col] = float(sum(nums) / len(nums))
                agg[f"{col}_std"] = float(statistics.pstdev(nums)) if len(nums) > 1 else 0.0
        out.append(agg)
    return out


def _conformal_rows(
    token_data: dict[str, Any],
    test_scores: np.ndarray,
    val_scores: np.ndarray,
    *,
    seed: int,
    attack: str,
    strategy: str,
    feature_view: str,
    candidate_source_mode: str,
) -> list[dict[str, Any]]:
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    test_idx = _split_indices(token_data, "test")
    y_true = labels[test_idx]
    out = []
    for alpha in [0.01, 0.005, 0.001]:
        pvals = np.asarray([(1.0 + float(np.sum(val_scores >= score))) / (len(val_scores) + 1.0) for score in test_scores])
        y_pred = (pvals <= alpha).astype(np.int64)
        benign = y_true == 0
        attack_mask = y_true == 1
        fp = int(np.sum((y_pred == 1) & benign))
        tn = int(np.sum((y_pred == 0) & benign))
        tp = int(np.sum((y_pred == 1) & attack_mask))
        fn = int(np.sum((y_pred == 0) & attack_mask))
        fpr = float(fp / max(fp + tn, 1))
        recall = float(tp / max(tp + fn, 1))
        out.append(
            {
                "seed": seed,
                "heldout_attack": attack,
                "motif_selection_strategy": strategy,
                "feature_view": feature_view,
                "candidate_source_mode": candidate_source_mode,
                "calibration": "conformal",
                "target_alpha": alpha,
                "realized_fpr": fpr,
                "attack_recall": recall,
                "false_alerts_per_10k_benign": fpr * 10000.0,
            }
        )
    return out


def run_one(
    token_path: Path,
    *,
    attack: str,
    seed: int,
    out_dir: Path,
    strategies: list[str],
    dictionary_sizes: list[int],
    transform: str,
    scorer: str,
    k: int,
    bootstrap_samples: int,
    include_data_mined_candidates: bool,
    dm_ngram_lengths: list[int],
    dm_min_support: float,
    candidate_source_modes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token_data = _read_token_data(token_path)
    train_idx = _split_indices(token_data, "train")
    token_rows, structural_vocab = _augment_structural_rows(_token_rows(token_data), train_idx)
    data_mined_vocab: dict[str, int] = {}
    if include_data_mined_candidates:
        token_rows, data_mined_vocab = _augment_data_mined_rows(
            token_rows,
            train_idx,
            ngram_lengths=dm_ngram_lengths,
            min_support=dm_min_support,
        )
    records = _records_from_rows(token_data, token_rows)
    val_idx = _split_indices(token_data, "val")
    if not np.all(token_data["binary_labels"].cpu().numpy().astype(np.int64)[train_idx] == 0):
        raise ValueError(f"Train split is not benign-only: {token_path}")
    packet_burst_tokens = {token for tokens in token_rows for token in tokens if is_packet_burst_token(token)}
    pb_features, pb_stats = _features_from_rows(token_rows, packet_burst_tokens, transform=transform, train_idx=train_idx)
    pb_metrics, pb_val_scores, _pb_test_scores, _ = _evaluate_features(
        token_data,
        pb_features,
        pb_stats,
        feature_view="packet_burst_only",
        transform=transform,
        scorer=scorer,
        k=k,
    )
    # Tail sensitivity is based on packet/burst benign-memory scores, not on the selected motif dictionary.
    del pb_metrics

    rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    for dictionary_size in dictionary_sizes:
        for candidate_source_mode in candidate_source_modes:
            if candidate_source_mode == "expert_only":
                motif_prefixes = EXPERT_MOTIF_PREFIXES
            elif candidate_source_mode == "data_mined_only":
                motif_prefixes = DATA_MINED_MOTIF_PREFIXES
            elif candidate_source_mode in {"expert_plus_data_mined", "all"}:
                motif_prefixes = DEFAULT_MOTIF_PREFIXES
            else:
                raise ValueError(f"Unsupported candidate source mode: {candidate_source_mode}")
            candidate_motifs = extract_candidate_motifs(records, motif_prefixes)
            if not candidate_motifs:
                continue
            occurrence = build_occurrence_matrix(records, candidate_motifs)
            for strategy in strategies:
                cfg = MotifSelectionConfig(
                    dictionary_size=dictionary_size,
                    bootstrap_samples=bootstrap_samples,
                    random_seed=seed,
                )
                start = time.perf_counter()
                selected, report = select_motif_dictionary(
                    records,
                    candidate_motifs,
                    occurrence,
                    train_idx,
                    val_idx,
                    pb_val_scores,
                    config=cfg,
                    strategy=strategy,
                )
                elapsed = time.perf_counter() - start
                artifact_dir = out_dir / "dictionaries" / f"{attack}_seed{seed}_{candidate_source_mode}_{strategy}_k{dictionary_size}"
                save_motif_dictionary(
                    selected,
                    report,
                    artifact_dir,
                    config=cfg,
                    metadata={
                        "token_path": str(token_path),
                        "heldout_attack": attack,
                        "seed": seed,
                        "candidate_source_mode": candidate_source_mode,
                        "motif_prefixes": list(motif_prefixes),
                        "structural_vocab_size": len(structural_vocab),
                        "data_mined_vocab_size": len(data_mined_vocab),
                        "data_mined_ngram_lengths": dm_ngram_lengths,
                        "data_mined_min_support": dm_min_support,
                        "tail_scores": "packet_burst_only_benign_validation",
                        "leakage_control": "train support/stability/coverage; benign-validation tail; no held-out attack labels",
                    },
                )
                selected_tokens = {entry["motif"] for entry in selected}
                if not selected_tokens:
                    continue
                for view, keep_tokens in [
                    ("selected_motifs_only", selected_tokens),
                    ("packet_burst_selected_motifs", packet_burst_tokens | selected_tokens),
                ]:
                    features, stats = _features_from_rows(token_rows, keep_tokens, transform=transform, train_idx=train_idx)
                    metrics, val_scores, test_scores, score_seconds = _evaluate_features(
                        token_data,
                        features,
                        stats,
                        feature_view=view,
                        transform=transform,
                        scorer=scorer,
                        k=k,
                    )
                    metrics["query_ms_per_flow"] = float(score_seconds / max(len(val_idx) + len(_split_indices(token_data, "test")), 1) * 1000.0)
                    rows.append(
                        _metric_row(
                            metrics,
                            seed=seed,
                            attack=attack,
                            strategy=strategy,
                            feature_view=view,
                            candidate_source_mode=candidate_source_mode,
                            dictionary_size=dictionary_size,
                            candidate_count=len(candidate_motifs),
                            selected_count=len(selected_tokens),
                            elapsed=elapsed,
                            score_seconds=score_seconds,
                            transform=transform,
                            scorer=scorer,
                            k=k,
                        )
                    )
                    cal_rows.extend(
                        _conformal_rows(
                            token_data,
                            test_scores,
                            val_scores,
                            seed=seed,
                            attack=attack,
                            strategy=strategy,
                            feature_view=view,
                            candidate_source_mode=candidate_source_mode,
                        )
                    )
    return rows, cal_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train-only motif dictionary selection ablations on existing token corpora.")
    parser.add_argument("--token_dir", default=str(DEFAULT_TOKEN_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--strategies", nargs="+", default=["all_candidates", "support_only", "support_stability", "support_stability_tail", "full_utility"])
    parser.add_argument("--dictionary_sizes", nargs="+", type=int, default=[200])
    parser.add_argument("--transform", default="binary_l2")
    parser.add_argument("--scorer", default="knn_cosine")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--bootstrap_samples", type=int, default=20)
    parser.add_argument("--include_data_mined_candidates", action="store_true")
    parser.add_argument("--dm_ngram_lengths", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--dm_min_support", type=float, default=0.005)
    parser.add_argument("--candidate_source_modes", nargs="+", default=["expert_plus_data_mined"])
    parser.add_argument("--quick", action="store_true", help="Run DDoS seed 43 only with K=200.")
    args = parser.parse_args()

    attacks = ["DDoS"] if args.quick else args.attacks
    seeds = [43] if args.quick else args.seeds
    dictionary_sizes = [200] if args.quick else args.dictionary_sizes
    out_dir = Path(args.out_dir)
    metric_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for attack in attacks:
        for seed in seeds:
            path = _token_path(Path(args.token_dir), attack, seed)
            if not path.exists():
                skipped.append({"heldout_attack": attack, "seed": seed, "reason": f"missing token corpus: {path}"})
                continue
            rows, crows = run_one(
                path,
                attack=attack,
                seed=seed,
                out_dir=out_dir,
                strategies=args.strategies,
                dictionary_sizes=dictionary_sizes,
                transform=args.transform,
                scorer=args.scorer,
                k=args.k,
                bootstrap_samples=args.bootstrap_samples,
                include_data_mined_candidates=args.include_data_mined_candidates,
                dm_ngram_lengths=args.dm_ngram_lengths,
                dm_min_support=args.dm_min_support,
                candidate_source_modes=args.candidate_source_modes,
            )
            metric_rows.extend(rows)
            cal_rows.extend(crows)

    _write_csv(metric_rows, out_dir / "motif_selection_metrics.csv")
    _write_csv(_aggregate(metric_rows), out_dir / "motif_selection_summary.csv")
    _write_csv(cal_rows, out_dir / "motif_selection_conformal_calibration.csv")
    _write_csv(skipped, out_dir / "motif_selection_skipped.csv")
    manifest = {
        "metric_rows": len(metric_rows),
        "calibration_rows": len(cal_rows),
        "skipped": skipped,
        "strategies": args.strategies,
        "dictionary_sizes": dictionary_sizes,
        "include_data_mined_candidates": bool(args.include_data_mined_candidates),
        "dm_ngram_lengths": args.dm_ngram_lengths,
        "dm_min_support": args.dm_min_support,
        "candidate_source_modes": args.candidate_source_modes,
        "leakage_control": [
            "support/stability/coverage computed on train split only",
            "tail sensitivity computed on benign validation scores from packet_burst_only memory",
            "held-out attack labels not used for motif selection or threshold calibration",
            "raw IP/time/five-tuple/protocol/service not used as motif tokens or memory keys",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "motif_selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "metric_rows": len(metric_rows), "skipped": len(skipped)}, sort_keys=True))


if __name__ == "__main__":
    main()
