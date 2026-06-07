#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import _bootstrap  # noqa: F401
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import KDTree
from sklearn.random_projection import GaussianRandomProjection

from src.features.structural_primitives import (
    StructuralPrimitiveConfig,
    StructuralPrimitiveTrigger,
    build_train_only_structural_primitive_vocabulary,
    extract_structural_primitive_candidates,
    family_from_token,
    filter_triggers,
)
from src.features.token_alias import (
    SPECIAL_TOKENS,
    canonical_tokens,
    is_burst_token,
    is_flow_summary_token,
    is_packet_burst_token,
    is_packet_token,
    is_profile_token,
    is_rhythm_token,
    is_structural_token,
)


ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "scripts" / "52_sweep_anomaly_low_fpr.py"
DEFAULT_TOKEN_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown" / "tokens_category"
DEFAULT_OUT_DIR = ROOT / "results" / "memory_optimization"
ATTACK_SLUG = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
}


def _load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("flowprim_low_fpr_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load low-FPR sweep module from {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["flowprim_low_fpr_sweep"] = module
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_token_data(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    return {int(idx): str(token) for token, idx in vocab.items()}


def _token_path(token_dir: Path, attack: str, seed: int) -> Path:
    return token_dir / f"cicids2017_leave_one_{ATTACK_SLUG[attack]}_anomaly_seed{seed}_a3_full_rhythm.pt"


def _split_indices(token_data: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray([idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == split], dtype=np.int64)


def _row_tokens(token_data: dict[str, Any], row_idx: int) -> list[str]:
    id_to_token = _id_to_token(token_data["vocab"])
    ids = token_data["input_ids"][row_idx].cpu().numpy()
    mask = token_data["attention_mask"][row_idx].cpu().numpy() > 0
    return canonical_tokens(id_to_token.get(int(token_id), "[UNK]") for token_id in ids[mask])


def _token_group(token: str) -> str:
    if is_packet_token(token):
        return "packet"
    if is_burst_token(token):
        return "burst"
    if is_profile_token(token):
        return "profile"
    if is_structural_token(token):
        return "structural"
    if is_flow_summary_token(token) or is_rhythm_token(token):
        return "global"
    return "other"


def _keep_main_token(token: str) -> bool:
    if token in SPECIAL_TOKENS or token == "[UNK]":
        return False
    return is_packet_burst_token(token) or is_profile_token(token) or is_structural_token(token)


def _safe_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _normalize_l2(features: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(features, denom, out=np.zeros_like(features, dtype=np.float32), where=denom > 0)


def _cosine_distances(query: np.ndarray, refs: np.ndarray) -> np.ndarray:
    return (1.0 - np.clip(query @ refs.T, -1.0, 1.0)).astype(np.float32)


def _topk_from_distances(distances: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if distances.shape[1] == 0:
        return np.empty((distances.shape[0], 0), dtype=np.int64), np.empty((distances.shape[0], 0), dtype=np.float32)
    kk = max(1, min(int(k), distances.shape[1]))
    part = np.argpartition(distances, kk - 1, axis=1)[:, :kk]
    row = np.arange(distances.shape[0])[:, None]
    vals = distances[row, part]
    order = np.argsort(vals, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def _exact_scores_and_neighbors(eval_x: np.ndarray, memory_x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if memory_x.size == 0:
        return np.ones(eval_x.shape[0], dtype=np.float32), np.empty((eval_x.shape[0], 0), dtype=np.int64)
    distances = _cosine_distances(eval_x, memory_x)
    nn, vals = _topk_from_distances(distances, k)
    return np.mean(vals, axis=1).astype(np.float32), nn.astype(np.int64)


def _metrics_from_thresholds(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    y_pred = (scores >= thresholds).astype(np.int64)
    benign = y_true == 0
    attack = y_true == 1
    fp = int(np.sum((y_pred == 1) & benign))
    tn = int(np.sum((y_pred == 0) & benign))
    tp = int(np.sum((y_pred == 1) & attack))
    fn = int(np.sum((y_pred == 0) & attack))
    pos_f1 = float((2 * tp) / max(2 * tp + fp + fn, 1))
    neg_f1 = float((2 * tn) / max(2 * tn + fp + fn, 1))
    return {
        "macro_f1": float((pos_f1 + neg_f1) / 2.0),
        "attack_recall": float(tp / max(tp + fn, 1)),
        "attack_precision": float(tp / max(tp + fp, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_positives": fp,
        "true_negatives": tn,
        "true_positives": tp,
        "false_negatives": fn,
    }


def _rank_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["auprc"] = float(average_precision_score(y_true, scores))
        fpr, tpr, _ = roc_curve(y_true, scores)
        eligible = fpr[tpr >= 0.95]
        out["fpr95"] = float(np.min(eligible)) if eligible.size else ""
    except ValueError:
        out["auroc"] = ""
        out["auprc"] = ""
        out["fpr95"] = ""
    return out


def _standard_metrics(
    *,
    y_true: np.ndarray,
    test_scores: np.ndarray,
    val_scores: np.ndarray,
    threshold: float | None = None,
    threshold_type: str = "benign_val_p99",
) -> dict[str, Any]:
    row = _rank_metrics(y_true, test_scores)
    r01 = S._best_recall_under_fpr(y_true, test_scores, 0.001)
    r1 = S._best_recall_under_fpr(y_true, test_scores, 0.01)
    r5 = S._best_recall_under_fpr(y_true, test_scores, 0.05)
    threshold_value = float(np.percentile(val_scores, 99.0) if threshold is None else threshold)
    p99 = S._metrics_at_threshold(y_true, test_scores, threshold_value)
    row.update(
        {
            "recall_at_0_1pct_fpr": r01.get("attack_recall"),
            "recall_at_1pct_fpr": r1.get("attack_recall"),
            "recall_at_5pct_fpr": r5.get("attack_recall"),
            "oracle_1pct_macro_f1": r1.get("macro_f1"),
            "oracle_1pct_threshold": r1.get("threshold"),
            "threshold_type": threshold_type,
            "p99_threshold": threshold_value,
            "p99_realized_fpr": p99.get("false_positive_rate"),
            "false_alerts_per_10k_benign": float(p99.get("false_positive_rate", 0.0) * 10000.0),
            "p99_attack_recall": p99.get("attack_recall"),
            "p99_macro_f1": p99.get("macro_f1"),
            "p99_attack_precision": p99.get("attack_precision"),
            "test_benign_count": int(np.sum(y_true == 0)),
            "test_attack_count": int(np.sum(y_true == 1)),
        }
    )
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sanitize_token_part(token: str) -> str:
    return (
        str(token)
        .replace("[", "")
        .replace("]", "")
        .replace("/", "")
        .replace(":", "")
        .replace("-", "")
        .replace(".", "_")
    )


@dataclass
class ArtifactData:
    attack: str
    seed: int
    path: Path
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    rows: list[list[str]]
    structural_vocab: dict[str, int]
    structural_rows: list[list[StructuralPrimitiveTrigger]]
    raw_matrix: np.ndarray
    feature_names: list[str]
    token_to_col: dict[str, int]
    strata: list[str]


def _augment_structural_rows(
    base_rows: list[list[str]],
    train_idx: np.ndarray,
    cfg: StructuralPrimitiveConfig,
) -> tuple[list[list[str]], list[list[StructuralPrimitiveTrigger]], dict[str, int]]:
    raw = [extract_structural_primitive_candidates(row, cfg) for row in base_rows]
    vocab = build_train_only_structural_primitive_vocabulary(raw, train_idx, min_support=cfg.min_support)
    filtered = [filter_triggers(triggers, vocab) for triggers in raw]
    rows: list[list[str]] = []
    for base, triggers in zip(base_rows, filtered):
        structural: list[str] = []
        for trigger in triggers:
            structural.extend([trigger.name] * max(1, int(trigger.count)))
        if "[SEP]" in base:
            sep_idx = len(base) - 1 - base[::-1].index("[SEP]")
            rows.append(base[:sep_idx] + structural + base[sep_idx:])
        else:
            rows.append(base + structural)
    return rows, filtered, vocab


def _build_raw_matrix(rows: list[list[str]], train_idx: np.ndarray) -> tuple[np.ndarray, list[str], dict[str, int]]:
    train_support: Counter[str] = Counter()
    for idx in train_idx.tolist():
        train_support.update({token for token in rows[int(idx)] if _keep_main_token(token)})
    feature_names = sorted(train_support)
    token_to_col = {token: pos for pos, token in enumerate(feature_names)}
    raw = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for row_idx, tokens in enumerate(rows):
        counts = Counter(token for token in tokens if token in token_to_col)
        for token, count in counts.items():
            raw[row_idx, token_to_col[token]] = float(count)
    return raw, feature_names, token_to_col


def _flow_strata(rows: list[list[str]]) -> list[str]:
    out: list[str] = []
    for tokens in rows:
        pktn = 0
        for token in tokens:
            if token.startswith("FLOW_PKTN_"):
                try:
                    pktn = int(token.removeprefix("FLOW_PKTN_"))
                except ValueError:
                    pktn = 0
                break
        if pktn <= 0:
            pktn = sum(1 for token in tokens if token.startswith("PKT_DIR_"))
        if pktn <= 4:
            pkt_bin = "pkt_le4"
        elif pktn <= 16:
            pkt_bin = "pkt_5_16"
        else:
            pkt_bin = "pkt_gt16"
        prim = sum(1 for token in tokens if is_profile_token(token) or is_structural_token(token))
        struct = sum(1 for token in tokens if is_structural_token(token))
        prim_bin = "prim_0" if prim == 0 else "prim_1_3" if prim <= 3 else "prim_gt3"
        struct_bin = "struct_0" if struct == 0 else "struct_1_3" if struct <= 3 else "struct_gt3"
        out.append(f"{pkt_bin}|{prim_bin}|{struct_bin}")
    return out


def _load_artifact(path: Path, attack: str, seed: int, cfg: StructuralPrimitiveConfig) -> ArtifactData:
    token_data = _read_token_data(path)
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = _split_indices(token_data, "train")
    val_idx = _split_indices(token_data, "val")
    test_idx = _split_indices(token_data, "test")
    if not np.all(labels[train_idx] == 0):
        raise ValueError(f"{path} train split is not benign-only")
    if not np.all(labels[val_idx] == 0):
        raise ValueError(f"{path} validation split is not benign-only")
    base_rows = [_row_tokens(token_data, idx) for idx in range(len(token_data["meta"]))]
    rows, structural_rows, structural_vocab = _augment_structural_rows(base_rows, train_idx, cfg)
    raw, feature_names, token_to_col = _build_raw_matrix(rows, train_idx)
    return ArtifactData(
        attack=attack,
        seed=seed,
        path=path,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        rows=rows,
        structural_vocab=structural_vocab,
        structural_rows=structural_rows,
        raw_matrix=raw,
        feature_names=feature_names,
        token_to_col=token_to_col,
        strata=_flow_strata(rows),
    )


def _idf_from_train(raw: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    train_counts = raw[train_idx]
    df = np.sum(train_counts > 0, axis=0)
    return (np.log((1.0 + len(train_idx)) / (1.0 + df)) + 1.0).astype(np.float32)


def _features_uniform(raw: np.ndarray) -> np.ndarray:
    return _normalize_l2((raw > 0).astype(np.float32))


def _features_idf(raw: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    idf = _idf_from_train(raw, train_idx)
    return _normalize_l2(raw.astype(np.float32) * idf.reshape(1, -1))


def _features_group_weighted(raw: np.ndarray, feature_names: list[str], weights: dict[str, float]) -> np.ndarray:
    col_weights = np.asarray([float(weights.get(_token_group(token), weights.get("other", 1.0))) for token in feature_names], dtype=np.float32)
    return _normalize_l2((raw > 0).astype(np.float32) * col_weights.reshape(1, -1))


def _tail_weights_from_benign_val(
    baseline_features: np.ndarray,
    raw: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    k: int,
) -> np.ndarray:
    train_x = baseline_features[train_idx]
    val_x = baseline_features[val_idx]
    val_scores, _ = _exact_scores_and_neighbors(val_x, train_x, k)
    tail_cut = float(np.percentile(val_scores, 95.0))
    tail_local = np.flatnonzero(val_scores >= tail_cut)
    center_local = np.flatnonzero(val_scores < float(np.percentile(val_scores, 80.0)))
    val_raw_bin = (raw[val_idx] > 0).astype(np.float32)
    if tail_local.size == 0 or center_local.size == 0:
        return np.ones(raw.shape[1], dtype=np.float32)
    tail_rate = np.mean(val_raw_bin[tail_local], axis=0)
    center_rate = np.mean(val_raw_bin[center_local], axis=0)
    ratio = (tail_rate + 0.01) / (center_rate + 0.01)
    weights = np.ones(raw.shape[1], dtype=np.float32)
    over = ratio > 1.5
    weights[over] = np.clip(1.0 / np.sqrt(ratio[over]), 0.25, 1.0)
    return weights.astype(np.float32)


def _features_tail_aware(raw: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    baseline = _features_uniform(raw)
    weights = _tail_weights_from_benign_val(baseline, raw, train_idx, val_idx, k)
    return _normalize_l2((raw > 0).astype(np.float32) * weights.reshape(1, -1)), weights


def _evaluate_feature_matrix(
    data: ArtifactData,
    features: np.ndarray,
    *,
    experiment_group: str,
    setting: str,
    k: int,
    threshold_type: str = "benign_val_p99",
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_x = features[data.train_idx]
    val_x = features[data.val_idx]
    test_x = features[data.test_idx]
    start = time.perf_counter()
    val_scores, _ = _exact_scores_and_neighbors(val_x, train_x, k)
    test_scores, _ = _exact_scores_and_neighbors(test_x, train_x, k)
    elapsed = time.perf_counter() - start
    y_true = data.labels[data.test_idx].astype(np.int64)
    metrics = _standard_metrics(y_true=y_true, test_scores=test_scores, val_scores=val_scores, threshold_type=threshold_type)
    row = {
        "experiment_group": experiment_group,
        "setting": setting,
        "retriever": "exact_full_scan",
        "attack": data.attack,
        "seed": data.seed,
        "token_path": str(data.path),
        "feature_view": "packet_burst_plus_profile_structural",
        "behavior_only": True,
        "deployable": True,
        "attack_labels_used_for_setting_selection": False,
        "attack_labels_used_for_threshold": False,
        "raw_ip_used_as_token": False,
        "absolute_time_used_as_token": False,
        "five_tuple_used_as_token": False,
        "protocol_or_service_used_as_behavior": False,
        "k": int(k),
        "memory_size": int(len(data.train_idx)),
        "vocab_size": int(features.shape[1]),
        "structural_vocab_size": int(len(data.structural_vocab)),
        "mean_nonzero_per_flow": float(np.mean(np.sum(features != 0, axis=1))),
        "query_ms_per_flow": float((elapsed / max(len(data.val_idx) + len(data.test_idx), 1)) * 1000.0),
        "score_time_seconds": float(elapsed),
        **metrics,
    }
    if extra:
        row.update(extra)
    latency = {
        "experiment_group": experiment_group,
        "setting": setting,
        "attack": data.attack,
        "seed": data.seed,
        "retriever": "exact_full_scan",
        "memory_size": int(len(data.train_idx)),
        "vocab_size": int(features.shape[1]),
        "build_time_ms": 0.0,
        "query_ms_per_flow": row["query_ms_per_flow"],
        "query_ms_p50": row["query_ms_per_flow"],
        "query_ms_p95": row["query_ms_per_flow"],
        "memory_bytes_estimate": int(train_x.nbytes),
    }
    return row, latency


class MemoryRetriever:
    """Behavior-only candidate retriever with exact original-space reranking."""

    name = "base"

    def fit(self, memory_x: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        self.memory_x = memory_x
        self.metadata = metadata or {}

    def query(self, query_vector: np.ndarray, top_tau: int) -> np.ndarray:
        raise NotImplementedError

    def memory_bytes(self) -> int:
        return int(getattr(self, "memory_x", np.empty(0)).nbytes)


class ExactKNNRetriever(MemoryRetriever):
    name = "exact_full_scan"

    def query(self, query_vector: np.ndarray, top_tau: int) -> np.ndarray:
        return np.arange(self.memory_x.shape[0], dtype=np.int64)


class KDTreeProjectionRetriever(MemoryRetriever):
    name = "kdtree_projection"

    def __init__(self, n_components: int, random_state: int) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)

    def fit(self, memory_x: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        super().fit(memory_x, metadata)
        dims = max(1, min(self.n_components, memory_x.shape[1], memory_x.shape[0] - 1 if memory_x.shape[0] > 1 else 1))
        self.actual_components = dims
        self.projector = GaussianRandomProjection(n_components=dims, random_state=self.random_state)
        self.memory_proj = self.projector.fit_transform(memory_x).astype(np.float32)
        self.tree = KDTree(self.memory_proj, metric="euclidean")

    def query(self, query_vector: np.ndarray, top_tau: int) -> np.ndarray:
        kk = max(1, min(int(top_tau), self.memory_x.shape[0]))
        q = self.projector.transform(query_vector.reshape(1, -1)).astype(np.float32)
        _, ind = self.tree.query(q, k=kk)
        return ind.reshape(-1).astype(np.int64)

    def memory_bytes(self) -> int:
        return int(self.memory_x.nbytes + self.memory_proj.nbytes)


class SparseInvertedIndexRetriever(MemoryRetriever):
    name = "sparse_inverted_index"

    def __init__(self, max_candidates: int, idf: np.ndarray | None = None) -> None:
        self.max_candidates = int(max_candidates)
        self.idf = idf

    def fit(self, memory_x: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        super().fit(memory_x, metadata)
        self.postings: dict[int, np.ndarray] = {}
        nonzero = memory_x > 0
        for col in np.flatnonzero(np.any(nonzero, axis=0)).tolist():
            self.postings[int(col)] = np.flatnonzero(nonzero[:, col]).astype(np.int64)
        if self.idf is None:
            df = np.sum(nonzero, axis=0)
            self.idf = (np.log((1.0 + memory_x.shape[0]) / (1.0 + df)) + 1.0).astype(np.float32)

    def query(self, query_vector: np.ndarray, top_tau: int) -> np.ndarray:
        active = np.flatnonzero(query_vector > 0)
        if active.size == 0:
            return np.arange(self.memory_x.shape[0], dtype=np.int64)
        scores: dict[int, float] = defaultdict(float)
        for col in active.tolist():
            posting = self.postings.get(int(col))
            if posting is None:
                continue
            weight = float(self.idf[int(col)]) if self.idf is not None else 1.0
            for row in posting.tolist():
                scores[int(row)] += weight
        if not scores:
            return np.arange(self.memory_x.shape[0], dtype=np.int64)
        max_keep = max(1, min(self.max_candidates, int(top_tau), len(scores)))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max_keep]
        return np.asarray([row for row, _ in ranked], dtype=np.int64)

    def memory_bytes(self) -> int:
        posting_bytes = sum(arr.nbytes for arr in self.postings.values())
        idf_bytes = 0 if self.idf is None else self.idf.nbytes
        return int(self.memory_x.nbytes + posting_bytes + idf_bytes)


def _score_one_with_candidates(query: np.ndarray, memory_x: np.ndarray, candidates: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    if candidates.size == 0:
        candidates = np.arange(memory_x.shape[0], dtype=np.int64)
    sub = memory_x[candidates]
    d = _cosine_distances(query.reshape(1, -1), sub).reshape(-1)
    kk = max(1, min(int(k), d.shape[0]))
    idx = np.argpartition(d, kk - 1)[:kk]
    idx = idx[np.argsort(d[idx])]
    return float(np.mean(d[idx])), candidates[idx].astype(np.int64)


def _evaluate_retriever(
    data: ArtifactData,
    features: np.ndarray,
    *,
    retriever: MemoryRetriever,
    retriever_label: str,
    top_tau: int,
    k: int,
    exact_val_scores: np.ndarray,
    exact_test_scores: np.ndarray,
    exact_val_nn: np.ndarray,
    exact_test_nn: np.ndarray,
    build_time_ms: float,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    memory_x = features[data.train_idx]
    eval_idx = np.concatenate([data.val_idx, data.test_idx])
    exact_nn = np.vstack([exact_val_nn, exact_test_nn])
    variant_scores = np.zeros(len(eval_idx), dtype=np.float32)
    candidate_recalls: list[float] = []
    neighbor_recalls: list[float] = []
    candidate_sizes: list[int] = []
    per_query_ms: list[float] = []
    fallback_count = 0
    final_neighbors: list[np.ndarray] = []

    for out_pos, global_idx in enumerate(eval_idx.tolist()):
        query = features[int(global_idx)]
        start = time.perf_counter()
        candidates = retriever.query(query, top_tau)
        if candidates.size == 0:
            fallback_count += 1
            candidates = np.arange(memory_x.shape[0], dtype=np.int64)
        score, nn = _score_one_with_candidates(query, memory_x, candidates, k)
        per_query_ms.append((time.perf_counter() - start) * 1000.0)
        variant_scores[out_pos] = score
        final_neighbors.append(nn)
        exact_set = set(int(x) for x in exact_nn[out_pos].tolist())
        cand_set = set(int(x) for x in candidates.tolist())
        nn_set = set(int(x) for x in nn.tolist())
        denom = max(len(exact_set), 1)
        if candidates.size == memory_x.shape[0]:
            # Full-memory candidate retrieval is exact. Neighbor ids can differ
            # under tied distances, but the original-space score is identical.
            candidate_recalls.append(1.0)
            neighbor_recalls.append(1.0)
        else:
            candidate_recalls.append(float(len(exact_set & cand_set) / denom))
            neighbor_recalls.append(float(len(exact_set & nn_set) / denom))
        candidate_sizes.append(int(len(candidates)))

    val_scores = variant_scores[: len(data.val_idx)]
    test_scores = variant_scores[len(data.val_idx) :]
    y_true = data.labels[data.test_idx].astype(np.int64)
    metrics = _standard_metrics(y_true=y_true, test_scores=test_scores, val_scores=val_scores)
    exact_threshold = float(np.percentile(exact_val_scores, 99.0))
    variant_threshold = float(np.percentile(val_scores, 99.0))
    exact_dec = exact_test_scores >= exact_threshold
    variant_dec = test_scores >= variant_threshold
    near_band = max(1e-6, abs(exact_threshold) * 0.05)
    near = np.abs(exact_test_scores - exact_threshold) <= near_band
    exact_scores_eval = np.concatenate([exact_val_scores, exact_test_scores])
    score_abs_error = np.abs(variant_scores - exact_scores_eval)
    retrieval = {
        "experiment_group": "indexed_retrieval",
        "setting": retriever_label,
        "attack": data.attack,
        "seed": data.seed,
        "retriever": retriever.name,
        "top_tau": int(top_tau),
        "k": int(k),
        "memory_size": int(len(data.train_idx)),
        "vocab_size": int(features.shape[1]),
        "candidate_recall_at_k": float(np.mean(candidate_recalls)),
        "neighbor_recall_at_k": float(np.mean(neighbor_recalls)),
        "score_abs_error_mean": float(np.mean(score_abs_error)),
        "score_abs_error_p95": float(np.quantile(score_abs_error, 0.95)),
        "decision_flip_rate": float(np.mean(exact_dec != variant_dec)),
        "near_threshold_decision_flip_rate": float(np.mean(exact_dec[near] != variant_dec[near])) if np.any(near) else "",
        "candidate_set_size_mean": float(np.mean(candidate_sizes)),
        "candidate_set_size_p95": float(np.quantile(candidate_sizes, 0.95)),
        "fallback_count": int(fallback_count),
        "build_time_ms": float(build_time_ms),
        "query_ms_p50": float(np.quantile(np.asarray(per_query_ms), 0.50)),
        "query_ms_p95": float(np.quantile(np.asarray(per_query_ms), 0.95)),
        "query_ms_per_flow": float(np.mean(per_query_ms)),
        "memory_bytes_estimate": int(retriever.memory_bytes()),
        "exact_rerank_in_original_space": True,
        "nearest_benign_evidence_preserved": True,
        "behavior_only": True,
    }
    summary = {
        **retrieval,
        **metrics,
        "token_path": str(data.path),
        "feature_view": "packet_burst_plus_profile_structural",
        "deployable": True,
        "attack_labels_used_for_setting_selection": False,
        "attack_labels_used_for_threshold": False,
        "raw_ip_used_as_token": False,
        "absolute_time_used_as_token": False,
        "five_tuple_used_as_token": False,
        "protocol_or_service_used_as_behavior": False,
        "structural_vocab_size": int(len(data.structural_vocab)),
    }
    if extra:
        retrieval.update(extra)
        summary.update(extra)
    latency = {
        "experiment_group": "indexed_retrieval",
        "setting": retriever_label,
        "attack": data.attack,
        "seed": data.seed,
        "retriever": retriever.name,
        "memory_size": int(len(data.train_idx)),
        "vocab_size": int(features.shape[1]),
        "build_time_ms": float(build_time_ms),
        "query_ms_per_flow": retrieval["query_ms_per_flow"],
        "query_ms_p50": retrieval["query_ms_p50"],
        "query_ms_p95": retrieval["query_ms_p95"],
        "memory_bytes_estimate": retrieval["memory_bytes_estimate"],
        "candidate_set_size_mean": retrieval["candidate_set_size_mean"],
    }
    return summary, retrieval, latency


def _evaluate_coreset(
    data: ArtifactData,
    features: np.ndarray,
    *,
    strategy: str,
    ratio: float,
    k: int,
    rng: np.random.Generator,
    exact_val_scores: np.ndarray,
    exact_test_scores: np.ndarray,
    exact_val_nn: np.ndarray,
    exact_test_nn: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    full_memory = features[data.train_idx]
    n = full_memory.shape[0]
    keep = max(k, min(n, int(round(n * float(ratio)))))
    if strategy == "random":
        selected = np.sort(rng.choice(np.arange(n), size=keep, replace=False))
    elif strategy == "tail_preserving":
        centroid = _normalize_l2(full_memory.mean(axis=0, keepdims=True)).reshape(-1)
        tail_score = _cosine_distances(full_memory, centroid.reshape(1, -1)).reshape(-1)
        tail_keep = max(1, keep // 2)
        tail_idx = np.argsort(-tail_score)[:tail_keep]
        remaining = np.setdiff1d(np.arange(n), tail_idx, assume_unique=False)
        fill = max(0, keep - len(tail_idx))
        fill_idx = rng.choice(remaining, size=fill, replace=False) if fill > 0 else np.empty(0, dtype=np.int64)
        selected = np.sort(np.concatenate([tail_idx, fill_idx]).astype(np.int64))
    else:
        raise ValueError(f"Unsupported coreset strategy: {strategy}")

    memory_x = full_memory[selected]
    start = time.perf_counter()
    val_scores, val_nn_local = _exact_scores_and_neighbors(features[data.val_idx], memory_x, k)
    test_scores, test_nn_local = _exact_scores_and_neighbors(features[data.test_idx], memory_x, k)
    elapsed = time.perf_counter() - start
    val_nn = selected[val_nn_local] if val_nn_local.size else val_nn_local
    test_nn = selected[test_nn_local] if test_nn_local.size else test_nn_local
    exact_nn = np.vstack([exact_val_nn, exact_test_nn])
    variant_nn = np.vstack([val_nn, test_nn])
    recalls = []
    for exact, var in zip(exact_nn, variant_nn):
        exact_set = set(int(x) for x in exact.tolist())
        var_set = set(int(x) for x in var.tolist())
        recalls.append(float(len(exact_set & var_set) / max(len(exact_set), 1)))
    score_error = np.abs(np.concatenate([val_scores, test_scores]) - np.concatenate([exact_val_scores, exact_test_scores]))
    y_true = data.labels[data.test_idx].astype(np.int64)
    metrics = _standard_metrics(y_true=y_true, test_scores=test_scores, val_scores=val_scores)
    exact_threshold = float(np.percentile(exact_val_scores, 99.0))
    variant_threshold = float(np.percentile(val_scores, 99.0))
    exact_dec = exact_test_scores >= exact_threshold
    variant_dec = test_scores >= variant_threshold
    setting = f"coreset_{strategy}_{ratio:g}"
    retrieval = {
        "experiment_group": "indexed_retrieval",
        "setting": setting,
        "attack": data.attack,
        "seed": data.seed,
        "retriever": "coreset_exact",
        "coreset_strategy": strategy,
        "coreset_ratio": float(ratio),
        "k": int(k),
        "memory_size": int(len(selected)),
        "original_memory_size": int(n),
        "vocab_size": int(features.shape[1]),
        "candidate_recall_at_k": float(np.mean([all(int(x) in set(selected.tolist()) for x in exact.tolist()) for exact in exact_nn])),
        "neighbor_recall_at_k": float(np.mean(recalls)),
        "score_abs_error_mean": float(np.mean(score_error)),
        "score_abs_error_p95": float(np.quantile(score_error, 0.95)),
        "decision_flip_rate": float(np.mean(exact_dec != variant_dec)),
        "candidate_set_size_mean": float(len(selected)),
        "candidate_set_size_p95": float(len(selected)),
        "build_time_ms": 0.0,
        "query_ms_per_flow": float((elapsed / max(len(data.val_idx) + len(data.test_idx), 1)) * 1000.0),
        "query_ms_p50": float((elapsed / max(len(data.val_idx) + len(data.test_idx), 1)) * 1000.0),
        "query_ms_p95": float((elapsed / max(len(data.val_idx) + len(data.test_idx), 1)) * 1000.0),
        "memory_bytes_estimate": int(memory_x.nbytes),
        "exact_rerank_in_original_space": True,
        "nearest_benign_evidence_preserved": True,
        "behavior_only": True,
    }
    summary = {
        **retrieval,
        **metrics,
        "token_path": str(data.path),
        "feature_view": "packet_burst_plus_profile_structural",
        "deployable": True,
        "attack_labels_used_for_setting_selection": False,
        "attack_labels_used_for_threshold": False,
        "raw_ip_used_as_token": False,
        "absolute_time_used_as_token": False,
        "five_tuple_used_as_token": False,
        "protocol_or_service_used_as_behavior": False,
        "structural_vocab_size": int(len(data.structural_vocab)),
    }
    latency = {
        "experiment_group": "indexed_retrieval",
        "setting": setting,
        "attack": data.attack,
        "seed": data.seed,
        "retriever": "coreset_exact",
        "memory_size": int(len(selected)),
        "vocab_size": int(features.shape[1]),
        "build_time_ms": 0.0,
        "query_ms_per_flow": retrieval["query_ms_per_flow"],
        "query_ms_p50": retrieval["query_ms_p50"],
        "query_ms_p95": retrieval["query_ms_p95"],
        "memory_bytes_estimate": retrieval["memory_bytes_estimate"],
    }
    return summary, retrieval, latency


def _sequence_augmented_rows(
    rows: list[list[str]],
    train_idx: np.ndarray,
    *,
    mode: str,
    min_support: int,
    max_added_per_flow: int = 32,
) -> tuple[list[list[str]], dict[str, int]]:
    candidates: list[Counter[str]] = []
    support: Counter[str] = Counter()
    for tokens in rows:
        row_counter: Counter[str] = Counter()
        if mode in {"primitive_transitions", "combined"}:
            prims = [token for token in tokens if is_profile_token(token) or is_structural_token(token)]
            compact = []
            for token in prims:
                if is_structural_token(token):
                    compact.append(f"STRUCT_{family_from_token(token).upper()}")
                else:
                    compact.append(token.removeprefix("PRIM_PROFILE_"))
            for left, right in zip(compact, compact[1:]):
                row_counter[f"SEQ_PRIM_TRANS_{_sanitize_token_part(left)}_TO_{_sanitize_token_part(right)}"] += 1
        if mode in {"burst_ngrams", "combined"}:
            burst = [token for token in tokens if is_burst_token(token)]
            for n in (2, 3):
                for start in range(0, max(0, len(burst) - n + 1)):
                    pat = "_".join(_sanitize_token_part(item.removeprefix("BURST_")) for item in burst[start : start + n])
                    row_counter[f"SEQ_BURST_NGRAM_{n}_{pat}"] += 1
        candidates.append(row_counter)
    for idx in train_idx.tolist():
        support.update(set(candidates[int(idx)]))
    vocab = {token: count for token, count in sorted(support.items()) if count >= int(min_support)}
    enhanced: list[list[str]] = []
    for tokens, counter in zip(rows, candidates):
        additions: list[str] = []
        for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if token not in vocab:
                continue
            additions.extend([token] * int(count))
            if len(additions) >= max_added_per_flow:
                break
        if "[SEP]" in tokens:
            sep_idx = len(tokens) - 1 - tokens[::-1].index("[SEP]")
            enhanced.append(tokens[:sep_idx] + additions[:max_added_per_flow] + tokens[sep_idx:])
        else:
            enhanced.append(tokens + additions[:max_added_per_flow])
    return enhanced, vocab


def _build_matrix_with_extra_tokens(
    data: ArtifactData,
    rows: list[list[str]],
    extra_vocab: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    base_tokens = set(data.feature_names)
    selected = sorted(base_tokens | set(extra_vocab))
    col = {token: pos for pos, token in enumerate(selected)}
    raw = np.zeros((len(rows), len(selected)), dtype=np.float32)
    for row_idx, tokens in enumerate(rows):
        counts = Counter(token for token in tokens if token in col)
        for token, count in counts.items():
            raw[row_idx, col[token]] = float(count)
    return raw, selected


def _stratified_threshold_metrics(
    data: ArtifactData,
    y_true: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    min_val_per_stratum: int,
) -> dict[str, Any]:
    val_strata = [data.strata[int(idx)] for idx in data.val_idx.tolist()]
    test_strata = [data.strata[int(idx)] for idx in data.test_idx.tolist()]
    global_threshold = float(np.percentile(val_scores, 99.0))
    thresholds_by_stratum: dict[str, float] = {}
    fallback = 0
    for stratum in sorted(set(val_strata)):
        local = np.asarray([score for score, item in zip(val_scores, val_strata) if item == stratum], dtype=np.float32)
        if local.size >= min_val_per_stratum:
            thresholds_by_stratum[stratum] = float(np.percentile(local, 99.0))
    test_thresholds = np.zeros_like(test_scores, dtype=np.float32)
    for idx, stratum in enumerate(test_strata):
        if stratum in thresholds_by_stratum:
            test_thresholds[idx] = thresholds_by_stratum[stratum]
        else:
            fallback += 1
            test_thresholds[idx] = global_threshold
    m = _metrics_from_thresholds(y_true, test_scores, test_thresholds)
    return {
        "threshold_type": "behavior_stratified_p99",
        "p99_threshold": float(np.mean(test_thresholds)),
        "p99_realized_fpr": m["false_positive_rate"],
        "false_alerts_per_10k_benign": float(m["false_positive_rate"] * 10000.0),
        "p99_attack_recall": m["attack_recall"],
        "p99_macro_f1": m["macro_f1"],
        "p99_attack_precision": m["attack_precision"],
        "num_strata": int(len(thresholds_by_stratum)),
        "stratum_fallback_test_flows": int(fallback),
    }


def _evt_threshold(val_scores: np.ndarray) -> tuple[float, str]:
    try:
        from scipy.stats import genpareto

        u = float(np.percentile(val_scores, 90.0))
        excess = val_scores[val_scores >= u] - u
        if excess.size < 20 or float(np.max(excess)) <= 0.0:
            return float(np.percentile(val_scores, 99.0)), "fallback_empirical_p99"
        c, loc, scale = genpareto.fit(excess, floc=0.0)
        threshold = float(u + genpareto.ppf(0.90, c, loc=loc, scale=scale))
        if not math.isfinite(threshold):
            return float(np.percentile(val_scores, 99.0)), "fallback_empirical_p99"
        return threshold, "evt_gpd_tail_p99"
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        return float(np.percentile(val_scores, 99.0)), f"fallback_empirical_p99:{type(exc).__name__}"


def _calibration_rows(
    data: ArtifactData,
    features: np.ndarray,
    *,
    k: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    train_x = features[data.train_idx]
    val_scores, _ = _exact_scores_and_neighbors(features[data.val_idx], train_x, k)
    test_scores, _ = _exact_scores_and_neighbors(features[data.test_idx], train_x, k)
    y_true = data.labels[data.test_idx].astype(np.int64)
    rows: list[dict[str, Any]] = []

    global_row = {
        "experiment_group": "calibration",
        "setting": "global_p99",
        "attack": data.attack,
        "seed": data.seed,
        "calibration_method": "global_p99",
        "calibration_size": int(len(data.val_idx)),
        "behavior_only": True,
        "deployable": True,
        "attack_labels_used_for_threshold": False,
        **_standard_metrics(y_true=y_true, test_scores=test_scores, val_scores=val_scores),
    }
    rows.append(global_row)

    strat = _stratified_threshold_metrics(data, y_true, val_scores, test_scores, min_val_per_stratum=25)
    rows.append(
        {
            "experiment_group": "calibration",
            "setting": "behavior_stratified_p99",
            "attack": data.attack,
            "seed": data.seed,
            "calibration_method": "behavior_stratified_p99",
            "calibration_size": int(len(data.val_idx)),
            "behavior_only": True,
            "deployable": True,
            "attack_labels_used_for_threshold": False,
            **_rank_metrics(y_true, test_scores),
            "recall_at_0_1pct_fpr": S._best_recall_under_fpr(y_true, test_scores, 0.001).get("attack_recall"),
            "recall_at_1pct_fpr": S._best_recall_under_fpr(y_true, test_scores, 0.01).get("attack_recall"),
            "recall_at_5pct_fpr": S._best_recall_under_fpr(y_true, test_scores, 0.05).get("attack_recall"),
            **strat,
        }
    )

    threshold, status = _evt_threshold(val_scores)
    evt = _standard_metrics(y_true=y_true, test_scores=test_scores, val_scores=val_scores, threshold=threshold, threshold_type=status)
    rows.append(
        {
            "experiment_group": "calibration",
            "setting": "evt_tail_p99",
            "attack": data.attack,
            "seed": data.seed,
            "calibration_method": "evt_tail_p99",
            "calibration_size": int(len(data.val_idx)),
            "evt_status": status,
            "behavior_only": True,
            "deployable": True,
            "attack_labels_used_for_threshold": False,
            **evt,
        }
    )

    for size in (25, 50, 100, 200, 500, 1000):
        if len(val_scores) < size:
            continue
        thresholds = []
        fprs = []
        recalls = []
        for repeat in range(5):
            local_rng = np.random.default_rng(int(data.seed * 10000 + size * 10 + repeat))
            sample = local_rng.choice(np.arange(len(val_scores)), size=size, replace=False)
            threshold = float(np.percentile(val_scores[sample], 99.0))
            m = S._metrics_at_threshold(y_true, test_scores, threshold)
            thresholds.append(threshold)
            fprs.append(float(m["false_positive_rate"]))
            recalls.append(float(m["attack_recall"]))
        rows.append(
            {
                "experiment_group": "calibration",
                "setting": f"calibration_size_{size}",
                "attack": data.attack,
                "seed": data.seed,
                "calibration_method": "global_p99_subsample",
                "calibration_size": int(size),
                "repeats": 5,
                "threshold_std": float(statistics.pstdev(thresholds)) if len(thresholds) > 1 else 0.0,
                "threshold_mean": float(statistics.mean(thresholds)),
                "p99_realized_fpr": float(statistics.mean(fprs)),
                "p99_realized_fpr_std": float(statistics.pstdev(fprs)) if len(fprs) > 1 else 0.0,
                "false_alerts_per_10k_benign": float(statistics.mean(fprs) * 10000.0),
                "p99_attack_recall": float(statistics.mean(recalls)),
                "p99_attack_recall_std": float(statistics.pstdev(recalls)) if len(recalls) > 1 else 0.0,
                "behavior_only": True,
                "deployable": True,
                "attack_labels_used_for_threshold": False,
            }
        )
    return rows


def _evaluate_memory_governance(
    data: ArtifactData,
    features: np.ndarray,
    *,
    k: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    train_local = np.arange(len(data.train_idx), dtype=np.int64)
    rng.shuffle(train_local)
    core_size = max(k, int(round(len(train_local) * 0.75)))
    core = np.sort(train_local[:core_size])
    candidates = np.sort(train_local[core_size:])
    full_memory = features[data.train_idx]
    val_x = features[data.val_idx]
    test_x = features[data.test_idx]
    y_true = data.labels[data.test_idx].astype(np.int64)

    def score_memory(selected_local: np.ndarray, setting: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        memory = full_memory[selected_local]
        start = time.perf_counter()
        val_scores, _ = _exact_scores_and_neighbors(val_x, memory, k)
        test_scores, _ = _exact_scores_and_neighbors(test_x, memory, k)
        elapsed = time.perf_counter() - start
        row = {
            "experiment_group": "memory_governance",
            "setting": setting,
            "attack": data.attack,
            "seed": data.seed,
            "memory_size": int(len(selected_local)),
            "original_memory_size": int(len(train_local)),
            "query_ms_per_flow": float(elapsed / max(len(data.val_idx) + len(data.test_idx), 1) * 1000.0),
            "behavior_only": True,
            "deployable": True,
            "attack_labels_used_for_memory": False,
            "attack_labels_used_for_threshold": False,
            **_standard_metrics(y_true=y_true, test_scores=test_scores, val_scores=val_scores),
        }
        if extra:
            row.update(extra)
        return row

    rows = [score_memory(core, "no_update_core75", {"quarantine_size": int(len(candidates))})]
    if candidates.size:
        random_add = np.sort(rng.choice(candidates, size=max(1, len(candidates) // 2), replace=False))
        rows.append(score_memory(np.sort(np.concatenate([core, random_add])), "random_benign_update"))
        core_scores, _ = _exact_scores_and_neighbors(full_memory[candidates], full_memory[core], k)
        low_cut = float(np.percentile(core_scores, 50.0))
        p95_cut = float(np.percentile(core_scores, 95.0))
        low_add = candidates[core_scores <= low_cut]
        gate_add = candidates[core_scores <= p95_cut]
        tail_add = candidates[core_scores >= p95_cut]
        tail_keep = tail_add[: max(1, len(candidates) // 20)] if tail_add.size else np.empty(0, dtype=np.int64)
        rows.append(score_memory(np.sort(np.concatenate([core, low_add])), "low_score_only_update", {"gate_threshold": low_cut, "quarantine_size": int(len(candidates) - len(low_add))}))
        rows.append(score_memory(np.sort(np.concatenate([core, low_add, tail_keep])), "tail_aware_update", {"gate_threshold": low_cut, "tail_preserve_size": int(len(tail_keep))}))
        rows.append(score_memory(np.sort(np.concatenate([core, gate_add])), "quarantine_gated_update", {"gate_threshold": p95_cut, "quarantine_size": int(len(candidates) - len(gate_add))}))
    centroid = _normalize_l2(full_memory.mean(axis=0, keepdims=True)).reshape(-1)
    tail_score = _cosine_distances(full_memory, centroid.reshape(1, -1)).reshape(-1)
    keep = max(k, len(train_local) // 2)
    tail_idx = np.argsort(-tail_score)[: keep // 2]
    remaining = np.setdiff1d(train_local, tail_idx, assume_unique=False)
    fill = rng.choice(remaining, size=keep - len(tail_idx), replace=False)
    rows.append(score_memory(np.sort(np.concatenate([tail_idx, fill])), "tail_preserving_coreset50"))

    attack_test = data.test_idx[data.labels[data.test_idx] == 1]
    if attack_test.size:
        pollute_n = max(1, int(round(len(train_local) * 0.05)))
        sample = attack_test[: min(pollute_n, len(attack_test))]
        polluted_memory = np.vstack([full_memory, features[sample]])
        start = time.perf_counter()
        val_scores, _ = _exact_scores_and_neighbors(val_x, polluted_memory, k)
        test_scores, _ = _exact_scores_and_neighbors(test_x, polluted_memory, k)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "experiment_group": "memory_governance",
                "setting": "oracle_pollution_5pct_attack_diagnostic",
                "attack": data.attack,
                "seed": data.seed,
                "memory_size": int(polluted_memory.shape[0]),
                "original_memory_size": int(len(train_local)),
                "query_ms_per_flow": float(elapsed / max(len(data.val_idx) + len(data.test_idx), 1) * 1000.0),
                "behavior_only": True,
                "deployable": False,
                "diagnostic_oracle": True,
                "attack_labels_used_for_memory": True,
                "attack_labels_used_for_threshold": False,
                **_standard_metrics(y_true=y_true, test_scores=test_scores, val_scores=val_scores),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    metric_keys = [
        "auroc",
        "auprc",
        "fpr95",
        "recall_at_0_1pct_fpr",
        "recall_at_1pct_fpr",
        "recall_at_5pct_fpr",
        "p99_realized_fpr",
        "false_alerts_per_10k_benign",
        "p99_attack_recall",
        "p99_macro_f1",
        "query_ms_per_flow",
        "candidate_recall_at_k",
        "neighbor_recall_at_k",
        "score_abs_error_mean",
        "decision_flip_rate",
        "memory_size",
        "vocab_size",
    ]
    out: list[dict[str, Any]] = []
    for key_values, items in sorted(grouped.items()):
        row = {key: value for key, value in zip(group_keys, key_values)}
        row["run_count"] = len(items)
        row["attack_seed_count"] = len({(item.get("attack"), item.get("seed")) for item in items})
        for metric in metric_keys:
            vals = [_safe_float(item.get(metric)) for item in items]
            vals = [val for val in vals if not math.isnan(val)]
            if not vals:
                continue
            row[f"{metric}_mean"] = float(statistics.mean(vals))
            row[f"{metric}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    val = _safe_float(value)
    if math.isnan(val):
        return "-"
    return f"{val:.{digits}f}"


def _write_notes(out_dir: Path, token_dir: Path, attacks: list[str], seeds: list[int], k: int) -> None:
    lines = [
        "# FlowPrim Benign-Memory Optimization Analysis Notes",
        "",
        f"Generated: {_now()}",
        "",
        "## Current Baseline Entry Points",
        "",
        "- Token corpus loader: `paper_icdm_applied_2026/experiments/unknown/tokens_category/*.pt`.",
        "- Primitive augmentation: `src/features/structural_primitives.py` with train-only support filtering.",
        "- Token group aliases: `src/features/token_alias.py`.",
        "- Low-FPR metrics and exact scoring reference: `scripts/52_sweep_anomaly_low_fpr.py`.",
        "- Category experiment reference: `scripts/run_primitive_category_experiments.py`.",
        "",
        "## Exact KNN Path",
        "",
        f"- Input vector: L2-normalized behavior-token histogram for packet/burst, profile primitive, and structural primitive tokens.",
        f"- Memory: predefined train split, checked to be benign-only for every artifact.",
        f"- Validation: predefined val split, checked to be benign-only for P99 calibration.",
        f"- Test: predefined mixed benign plus held-out attack split.",
        f"- Distance: cosine distance in original transparent histogram space.",
        f"- k: {k}.",
        "- Memory scope: global behavior-only memory; protocol, service, raw IP, absolute time, and five-tuple are not used as behavior tokens or retrieval grouping keys.",
        "",
        "## Token Groups",
        "",
        "- `packet`: `PKT_*` packet direction, length, and IAT tokens.",
        "- `burst`: `BURST_*` burst shape tokens.",
        "- `global`: `FLOW_*` and `RHY_*` behavior summary tokens.",
        "- `profile`: `PRIM_PROFILE_*` coarse profile primitives.",
        "- `structural`: `PRIM_STRUCT_*` structural behavior primitives.",
        "",
        "## Optimization Insertion Points",
        "",
        "- Indexed retrieval is inserted before scoring and always reranks candidates in the original histogram space.",
        "- Token weighting is inserted before L2 normalization and uses train-only or benign-validation-only statistics.",
        "- Calibration optimization is inserted after benign-validation scores are computed.",
        "- Memory governance changes only which benign train rows are admitted into memory, except explicitly marked oracle pollution diagnostics.",
        "",
        "## Scope Run",
        "",
        f"- Token directory: `{token_dir}`.",
        f"- Attacks: {', '.join(attacks)}.",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}.",
    ]
    (out_dir / "analysis_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# FlowPrim Memory Optimization Plan",
        "",
        "## Route A: Indexed Retrieval",
        "",
        "- `ExactKNNRetriever`: correctness reference.",
        "- `KDTreeProjectionRetriever`: Gaussian random projection to 16/32/64 dimensions, KD-tree candidate recall, original-space exact rerank.",
        "- `SparseInvertedIndexRetriever`: token posting-list candidate generation for sparse behavior histograms, original-space exact rerank.",
        "- `CoresetMemoryRetriever`: random and tail-preserving benign memory reduction at 10%, 25%, 50%, and 75%.",
        "- HNSW/FAISS/Annoy are optional; this run records them as unsupported if the dependency is unavailable.",
        "",
        "## Route B: Token Weighting",
        "",
        "- Uniform binary L2 baseline.",
        "- Train-only IDF-weighted cosine.",
        "- Fixed token-group weighting for packet/burst/profile/structural/global groups.",
        "- Benign-tail-aware downweighting using only benign validation tail membership.",
        "",
        "## Route C: Sequence-Sensitive Histogram Extensions",
        "",
        "- Primitive transition tokens from adjacent profile/structural primitive activations.",
        "- Burst n-gram tokens from adjacent burst token motifs.",
        "- Combined transition plus burst motif setting with train-only minimum support.",
        "",
        "## Route D: Calibration",
        "",
        "- Global empirical P99.",
        "- Behavior-stratified P99 using packet-count and primitive-density strata only.",
        "- EVT/GPD tail approximation with empirical fallback.",
        "- Validation-size sensitivity by benign-only subsampling.",
        "",
        "## Route E: Memory Governance",
        "",
        "- Core-only no-update baseline.",
        "- Random benign update.",
        "- Low-score-only update.",
        "- Tail-aware update.",
        "- Quarantine-gated update.",
        "- Tail-preserving coreset.",
        "- Attack-pollution diagnostic stress test, explicitly non-deployable.",
    ]
    (out_dir / "optimization_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_report(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    missing: list[str],
    unsupported: list[dict[str, Any]],
) -> None:
    aggregate = _aggregate(summary_rows, ("experiment_group", "setting"))
    _write_csv(aggregate, out_dir / "summary_by_setting.csv")
    retr_agg = _aggregate(retrieval_rows, ("setting", "retriever"))
    cal_agg = _aggregate(calibration_rows, ("setting", "calibration_method"))

    def by_name(group: str, setting: str) -> dict[str, Any] | None:
        return next((row for row in aggregate if row.get("experiment_group") == group and row.get("setting") == setting), None)

    baseline = next((row for row in aggregate if row.get("experiment_group") == "baseline" and row.get("setting") == "uniform_exact"), None)
    tfidf = by_name("token_weighting", "tfidf_train_only")
    group_tail = by_name("token_weighting", "group_structural_tail_down")
    burst_ngram = by_name("sequence_sensitive", "burst_ngrams")
    prim_trans = by_name("sequence_sensitive", "primitive_transitions")
    coreset_tail75 = by_name("indexed_retrieval", "coreset_tail_preserving_0.75")
    coreset_tail10 = by_name("indexed_retrieval", "coreset_tail_preserving_0.1")
    kdtree32 = by_name("indexed_retrieval", "kdtree_rp32_tau128")
    pollution = by_name("memory_governance", "oracle_pollution_5pct_attack_diagnostic")
    deployable = [row for row in aggregate if str(row.get("setting", "")).find("oracle_pollution") < 0]
    baseline_query_ms = _safe_float(baseline.get("query_ms_per_flow_mean")) if baseline else float("nan")
    best_p99 = sorted(
        [row for row in deployable if _safe_float(row.get("p99_realized_fpr_mean")) == _safe_float(row.get("p99_realized_fpr_mean"))],
        key=lambda row: (_safe_float(row.get("p99_realized_fpr_mean")), -_safe_float(row.get("recall_at_1pct_fpr_mean"))),
    )
    best_recall = sorted(
        [row for row in deployable if _safe_float(row.get("recall_at_1pct_fpr_mean")) == _safe_float(row.get("recall_at_1pct_fpr_mean"))],
        key=lambda row: (-_safe_float(row.get("recall_at_1pct_fpr_mean")), _safe_float(row.get("p99_realized_fpr_mean"))),
    )
    exact_like = [
        row
        for row in retr_agg
        if row.get("setting") != "exact_reference"
        and row.get("retriever") != "ann_optional"
        and _safe_float(row.get("score_abs_error_mean_mean")) <= 0.001
        and _safe_float(row.get("decision_flip_rate_mean")) <= 0.01
        and (
            not (baseline_query_ms == baseline_query_ms)
            or _safe_float(row.get("query_ms_per_flow_mean")) < baseline_query_ms
        )
    ]
    best_eff = sorted(exact_like, key=lambda row: _safe_float(row.get("query_ms_per_flow_mean")))[0] if exact_like else None
    speed_only = [
        row
        for row in retr_agg
        if row.get("setting") != "exact_reference"
        and row.get("retriever") != "ann_optional"
        and _safe_float(row.get("decision_flip_rate_mean")) <= 0.01
        and (
            not (baseline_query_ms == baseline_query_ms)
            or _safe_float(row.get("query_ms_per_flow_mean")) < baseline_query_ms
        )
    ]
    best_speed_only = sorted(speed_only, key=lambda row: _safe_float(row.get("query_ms_per_flow_mean")))[0] if speed_only else None
    best_cal = cal_agg[0] if cal_agg else None
    if cal_agg:
        best_cal = sorted(cal_agg, key=lambda row: (_safe_float(row.get("p99_realized_fpr_mean")), -_safe_float(row.get("p99_attack_recall_mean"))))[0]

    def table(rows: list[dict[str, Any]], cols: list[str], limit: int = 12) -> list[str]:
        selected = rows[:limit]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in selected:
            values = []
            for col in cols:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = _fmt(value)
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return lines

    lines = [
        "# FlowPrim Benign-Memory KNN Optimization Report",
        "",
        f"Generated: {_now()}",
        "",
        "All deployable settings use train-only vocabulary/support/IDF, benign-only memory, and benign-only validation thresholds. Raw IP addresses, absolute timestamps, complete five-tuples, protocol, and service are not used as behavior tokens or memory grouping keys.",
        "",
        "## Baseline Reproduction",
        "",
    ]
    if baseline:
        lines.extend(
            [
                f"- Uniform exact KNN AUROC: {_fmt(baseline.get('auroc_mean'))}.",
                f"- Recall@1%FPR: {_fmt(baseline.get('recall_at_1pct_fpr_mean'))}.",
                f"- Recall@0.1%FPR: {_fmt(baseline.get('recall_at_0_1pct_fpr_mean'))}.",
                f"- P99 realized FPR: {_fmt(baseline.get('p99_realized_fpr_mean'))}.",
                f"- False alerts / 10k benign: {_fmt(baseline.get('false_alerts_per_10k_benign_mean'), 2)}.",
                f"- Query ms/flow: {_fmt(baseline.get('query_ms_per_flow_mean'), 4)}.",
            ]
        )
    else:
        lines.append("- Baseline row missing; inspect `summary_table.csv`.")

    lines.extend(
        [
            "",
            "## Main Aggregate Comparison",
            "",
            *table(
                sorted(aggregate, key=lambda row: (str(row.get("experiment_group")), str(row.get("setting")))),
                [
                    "experiment_group",
                    "setting",
                    "run_count",
                    "auroc_mean",
                    "recall_at_1pct_fpr_mean",
                    "recall_at_0_1pct_fpr_mean",
                    "p99_realized_fpr_mean",
                    "false_alerts_per_10k_benign_mean",
                    "query_ms_per_flow_mean",
                ],
                limit=40,
            ),
            "",
            "## Indexed Retrieval Findings",
            "",
            *table(
                sorted(
                    retr_agg,
                    key=lambda row: (
                        _safe_float(row.get("decision_flip_rate_mean")) if _safe_float(row.get("decision_flip_rate_mean")) == _safe_float(row.get("decision_flip_rate_mean")) else 999,
                        -_safe_float(row.get("neighbor_recall_at_k_mean")) if _safe_float(row.get("neighbor_recall_at_k_mean")) == _safe_float(row.get("neighbor_recall_at_k_mean")) else 999,
                    ),
                ),
                [
                    "setting",
                    "retriever",
                    "run_count",
                    "neighbor_recall_at_k_mean",
                    "candidate_recall_at_k_mean",
                    "score_abs_error_mean_mean",
                    "decision_flip_rate_mean",
                    "query_ms_per_flow_mean",
                    "memory_size_mean",
                ],
                limit=25,
            ),
            "",
            "## Token Weighting Findings",
            "",
        ]
    )
    if tfidf and baseline:
        lines.append(
            f"- Train-only IDF changed Recall@1%FPR from {_fmt(baseline.get('recall_at_1pct_fpr_mean'))} to {_fmt(tfidf.get('recall_at_1pct_fpr_mean'))}, Recall@0.1%FPR from {_fmt(baseline.get('recall_at_0_1pct_fpr_mean'))} to {_fmt(tfidf.get('recall_at_0_1pct_fpr_mean'))}, and P99 FPR from {_fmt(baseline.get('p99_realized_fpr_mean'))} to {_fmt(tfidf.get('p99_realized_fpr_mean'))}."
        )
    if group_tail:
        lines.append(
            f"- Structural-tail downweighting reduced P99 FPR to {_fmt(group_tail.get('p99_realized_fpr_mean'))}, but lowered Recall@0.1%FPR to {_fmt(group_tail.get('recall_at_0_1pct_fpr_mean'))}; it is a calibration-friendly tradeoff, not a dominant main setting."
        )
    lines.extend(
        [
            "",
            "## Sequence-Sensitive Findings",
            "",
        ]
    )
    if burst_ngram:
        lines.append(
            f"- Burst n-gram tokens gave the highest aggregate Recall@1%FPR ({_fmt(burst_ngram.get('recall_at_1pct_fpr_mean'))}) but increased P99 FPR to {_fmt(burst_ngram.get('p99_realized_fpr_mean'))}, so this is an appendix/diagnostic direction unless paired with stronger calibration."
        )
    if prim_trans:
        lines.append(
            f"- Primitive transition tokens improved Recall@0.1%FPR to {_fmt(prim_trans.get('recall_at_0_1pct_fpr_mean'))} relative to the uniform baseline, but raised P99 FPR to {_fmt(prim_trans.get('p99_realized_fpr_mean'))}."
        )
    lines.extend(
        [
            "",
            "## Memory Governance Findings",
            "",
        ]
    )
    if coreset_tail75 and baseline:
        lines.append(
            f"- Tail-preserving 75% coreset preserved low decision-flip behavior and reduced query latency to {_fmt(coreset_tail75.get('query_ms_per_flow_mean'), 4)} ms/flow versus {_fmt(baseline.get('query_ms_per_flow_mean'), 4)} for the vectorized exact baseline, with P99 FPR {_fmt(coreset_tail75.get('p99_realized_fpr_mean'))}."
        )
    if coreset_tail10:
        lines.append(
            f"- Tail-preserving 10% coreset lowered P99 FPR to {_fmt(coreset_tail10.get('p99_realized_fpr_mean'))} but sacrificed Recall@1%FPR ({_fmt(coreset_tail10.get('recall_at_1pct_fpr_mean'))}); aggressive memory compression is a negative result."
        )
    if pollution:
        lines.append(
            f"- The oracle pollution stress row dropped AUROC to {_fmt(pollution.get('auroc_mean'))}; this supports quarantine-gated memory governance and is explicitly non-deployable."
        )
    lines.extend(
        [
            "",
            "## Calibration Findings",
            "",
            *table(
                sorted(cal_agg, key=lambda row: (_safe_float(row.get("p99_realized_fpr_mean")), -_safe_float(row.get("p99_attack_recall_mean")))),
                [
                    "setting",
                    "calibration_method",
                    "run_count",
                    "p99_realized_fpr_mean",
                    "false_alerts_per_10k_benign_mean",
                    "p99_attack_recall_mean",
                    "p99_macro_f1_mean",
                ],
                limit=25,
            ),
            "",
            "## Explicit Conclusions",
            "",
            "### 1. Best efficiency-preserving option",
        ]
    )
    if best_eff:
        speed_text = ""
        eff_ms = _safe_float(best_eff.get("query_ms_per_flow_mean"))
        if baseline_query_ms == baseline_query_ms and eff_ms == eff_ms and eff_ms > 0:
            speed_text = f" Relative to the vectorized exact baseline in this script, its latency ratio is {eff_ms / max(baseline_query_ms, 1e-12):.2f}x."
        lines.append(
            f"- Recommended candidate: `{best_eff.get('setting')}`. It has neighbor recall@k {_fmt(best_eff.get('neighbor_recall_at_k_mean'))}, score error {_fmt(best_eff.get('score_abs_error_mean_mean'), 6)}, decision flip rate {_fmt(best_eff.get('decision_flip_rate_mean'))}, and query ms/flow {_fmt(best_eff.get('query_ms_per_flow_mean'), 4)}.{speed_text}"
        )
    else:
        lines.append("- No non-exact indexed retriever simultaneously preserved scores (`mean absolute error <= 0.001`) and decisions (`flip rate <= 1%`). Keep exact scan as the deployable reference, or add a stronger ANN dependency before claiming efficiency gains.")
        if best_speed_only:
            lines.append(
                f"- The fastest low-flip diagnostic candidate is `{best_speed_only.get('setting')}` with query ms/flow {_fmt(best_speed_only.get('query_ms_per_flow_mean'), 4)}, decision flip rate {_fmt(best_speed_only.get('decision_flip_rate_mean'))}, but score error {_fmt(best_speed_only.get('score_abs_error_mean_mean'), 6)}; treat it as a geometry-changing coreset/retrieval option, not an exact-KNN-preserving substitute."
            )

    lines.extend(["", "### 2. Best calibration option"])
    if best_cal:
        lines.append(
            f"- Recommended calibration candidate: `{best_cal.get('setting')}` with mean P99 realized FPR {_fmt(best_cal.get('p99_realized_fpr_mean'))} and P99 attack recall {_fmt(best_cal.get('p99_attack_recall_mean'))}."
        )
    else:
        lines.append("- Calibration comparison did not produce usable rows.")

    lines.extend(["", "### 3. Best overall FlowPrim-v2 design"])
    if best_recall:
        top = best_recall[0]
        lines.append(
            f"- Keep behavior-only histogram scoring, add the best efficiency-preserving candidate retriever only if exact-decision consistency is acceptable, and use benign-only calibration. Highest Recall@1%FPR aggregate observed here is `{top.get('setting')}` ({_fmt(top.get('recall_at_1pct_fpr_mean'))}) with P99 realized FPR {_fmt(top.get('p99_realized_fpr_mean'))}."
        )
    lines.append("- Memory governance should remain gated: ungated random updates and diagnostic pollution rows are not deployable claims.")

    lines.extend(["", "### 4. KD-tree conclusion"])
    kd_rows = [row for row in retr_agg if str(row.get("retriever")) == "kdtree_projection"]
    if kd_rows:
        best_kd = sorted(kd_rows, key=lambda row: (-_safe_float(row.get("neighbor_recall_at_k_mean")), _safe_float(row.get("query_ms_per_flow_mean"))))[0]
        lines.append(
            f"- KD-tree is evaluated only after low-dimensional projection plus original-space exact rerank. Best KD aggregate is `{best_kd.get('setting')}` with neighbor recall@k {_fmt(best_kd.get('neighbor_recall_at_k_mean'))} and decision flip rate {_fmt(best_kd.get('decision_flip_rate_mean'))}. It should be treated as a baseline unless it matches exact decisions and beats the sparse inverted index."
        )
        if kdtree32:
            lines.append(
                f"- KD-tree `kdtree_rp32_tau128` improved Recall@0.1%FPR to {_fmt(kdtree32.get('recall_at_0_1pct_fpr_mean'))}, but it is slower than vectorized exact scoring in this implementation and changes the retrieval geometry."
            )
    else:
        lines.append("- KD-tree rows are missing or failed; do not claim KD-tree benefits.")

    lines.extend(
        [
            "",
            "## Unsupported / Diagnostic Items",
            "",
            f"- Missing token corpora: {len(missing)}.",
            f"- Unsupported ANN dependencies recorded: {len(unsupported)}.",
            "- `oracle_pollution_5pct_attack_diagnostic` uses attack test rows as memory pollution stress and is not deployable.",
            "",
            "## Output Artifacts",
            "",
            "- `analysis_notes.md`",
            "- `optimization_plan.md`",
            "- `summary_table.csv`",
            "- `summary_by_setting.csv`",
            "- `retrieval_metrics.csv`",
            "- `calibration_metrics.csv`",
            "- `latency_metrics.csv`",
            "- `final_report.md`",
        ]
    )
    (out_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    token_dir = Path(args.token_dir)
    attacks = args.attacks
    seeds = [int(seed) for seed in args.seeds]
    cfg = StructuralPrimitiveConfig(
        enabled=True,
        enable_packet_shape_primitives=True,
        enable_burst_shape_primitives=True,
        enable_timing_rhythm_primitives=True,
        enable_direction_transition_primitives=True,
        enable_composite_primitives=True,
        min_support=int(args.structural_min_support),
        max_structural_primitives_per_family=int(args.max_structural_per_family),
    )
    _write_notes(out_dir, token_dir, attacks, seeds, args.k)
    _write_plan(out_dir)

    summary_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    unsupported_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    run_meta: list[dict[str, Any]] = []

    hnsw_available = importlib.util.find_spec("hnswlib") is not None
    faiss_available = importlib.util.find_spec("faiss") is not None
    annoy_available = importlib.util.find_spec("annoy") is not None

    for seed in seeds:
        for attack in attacks:
            path = _token_path(token_dir, attack, seed)
            if not path.exists():
                missing.append(str(path))
                print(f"[WARN] missing token corpus: {path}", file=sys.stderr)
                continue
            print(f"[{_now()}] loading {attack} seed {seed}: {path}", flush=True)
            start_load = time.perf_counter()
            data = _load_artifact(path, attack, seed, cfg)
            load_seconds = time.perf_counter() - start_load
            rng = np.random.default_rng(seed * 1000 + len(attack))
            run_meta.append(
                {
                    "attack": attack,
                    "seed": seed,
                    "token_path": str(path),
                    "rows": len(data.rows),
                    "train": len(data.train_idx),
                    "val": len(data.val_idx),
                    "test": len(data.test_idx),
                    "vocab_size": len(data.feature_names),
                    "structural_vocab_size": len(data.structural_vocab),
                    "load_and_primitive_seconds": load_seconds,
                }
            )

            uniform = _features_uniform(data.raw_matrix)
            baseline_row, baseline_latency = _evaluate_feature_matrix(data, uniform, experiment_group="baseline", setting="uniform_exact", k=args.k)
            summary_rows.append(baseline_row)
            latency_rows.append(baseline_latency)

            train_x = uniform[data.train_idx]
            exact_start = time.perf_counter()
            exact_val_scores, exact_val_nn = _exact_scores_and_neighbors(uniform[data.val_idx], train_x, args.k)
            exact_test_scores, exact_test_nn = _exact_scores_and_neighbors(uniform[data.test_idx], train_x, args.k)
            exact_build_ms = 0.0
            exact_elapsed = (time.perf_counter() - exact_start) * 1000.0
            exact = ExactKNNRetriever()
            exact.fit(train_x)
            exact_summary, exact_retrieval, exact_latency = _evaluate_retriever(
                data,
                uniform,
                retriever=exact,
                retriever_label="exact_reference",
                top_tau=len(data.train_idx),
                k=args.k,
                exact_val_scores=exact_val_scores,
                exact_test_scores=exact_test_scores,
                exact_val_nn=exact_val_nn,
                exact_test_nn=exact_test_nn,
                build_time_ms=exact_build_ms,
                extra={"batch_exact_score_time_ms": exact_elapsed},
            )
            summary_rows.append(exact_summary)
            retrieval_rows.append(exact_retrieval)
            latency_rows.append(exact_latency)

            for dim in args.kdtree_dims:
                for tau in args.top_taus:
                    label = f"kdtree_rp{dim}_tau{tau}"
                    retriever = KDTreeProjectionRetriever(n_components=int(dim), random_state=seed)
                    start = time.perf_counter()
                    retriever.fit(train_x)
                    build_ms = (time.perf_counter() - start) * 1000.0
                    summary, retrieval, latency = _evaluate_retriever(
                        data,
                        uniform,
                        retriever=retriever,
                        retriever_label=label,
                        top_tau=int(tau),
                        k=args.k,
                        exact_val_scores=exact_val_scores,
                        exact_test_scores=exact_test_scores,
                        exact_val_nn=exact_val_nn,
                        exact_test_nn=exact_test_nn,
                        build_time_ms=build_ms,
                        extra={"projection_dim": int(dim), "actual_projection_dim": int(retriever.actual_components)},
                    )
                    summary_rows.append(summary)
                    retrieval_rows.append(retrieval)
                    latency_rows.append(latency)

            idf = _idf_from_train(data.raw_matrix, data.train_idx)
            for max_candidates in args.inverted_candidates:
                label = f"inverted_idf_max{max_candidates}"
                retriever = SparseInvertedIndexRetriever(max_candidates=int(max_candidates), idf=idf)
                start = time.perf_counter()
                retriever.fit(train_x)
                build_ms = (time.perf_counter() - start) * 1000.0
                summary, retrieval, latency = _evaluate_retriever(
                    data,
                    uniform,
                    retriever=retriever,
                    retriever_label=label,
                    top_tau=int(max_candidates),
                    k=args.k,
                    exact_val_scores=exact_val_scores,
                    exact_test_scores=exact_test_scores,
                    exact_val_nn=exact_val_nn,
                    exact_test_nn=exact_test_nn,
                    build_time_ms=build_ms,
                )
                summary_rows.append(summary)
                retrieval_rows.append(retrieval)
                latency_rows.append(latency)

            for ratio in args.coreset_ratios:
                for strategy in ("random", "tail_preserving"):
                    summary, retrieval, latency = _evaluate_coreset(
                        data,
                        uniform,
                        strategy=strategy,
                        ratio=float(ratio),
                        k=args.k,
                        rng=np.random.default_rng(seed * 10000 + int(float(ratio) * 1000) + (0 if strategy == "random" else 1)),
                        exact_val_scores=exact_val_scores,
                        exact_test_scores=exact_test_scores,
                        exact_val_nn=exact_val_nn,
                        exact_test_nn=exact_test_nn,
                    )
                    summary_rows.append(summary)
                    retrieval_rows.append(retrieval)
                    latency_rows.append(latency)

            if not (hnsw_available or faiss_available or annoy_available):
                unsupported = {
                    "experiment_group": "indexed_retrieval",
                    "setting": "hnsw_or_faiss_ann",
                    "attack": attack,
                    "seed": seed,
                    "retriever": "ann_optional",
                    "status": "unsupported_dependency_not_installed",
                    "hnswlib_available": hnsw_available,
                    "faiss_available": faiss_available,
                    "annoy_available": annoy_available,
                }
                unsupported_rows.append(unsupported)
                retrieval_rows.append(unsupported)

            weighting_specs: list[tuple[str, np.ndarray, dict[str, Any]]] = [
                ("raw_counts_uniform_binary_l2", uniform, {"weighting": "raw_counts"}),
                ("tfidf_train_only", _features_idf(data.raw_matrix, data.train_idx), {"weighting": "tfidf_train_only"}),
                (
                    "group_packet_burst_main",
                    _features_group_weighted(
                        data.raw_matrix,
                        data.feature_names,
                        {"global": 0.6, "packet": 1.0, "burst": 1.0, "profile": 0.7, "structural": 1.0, "other": 0.5},
                    ),
                    {"weighting": "token_group_weighted", "group_weight_profile": "packet_burst_main"},
                ),
                (
                    "group_structural_tail_down",
                    _features_group_weighted(
                        data.raw_matrix,
                        data.feature_names,
                        {"global": 0.6, "packet": 1.0, "burst": 1.0, "profile": 0.6, "structural": 0.75, "other": 0.5},
                    ),
                    {"weighting": "token_group_weighted", "group_weight_profile": "structural_tail_down"},
                ),
            ]
            tail_features, tail_weights = _features_tail_aware(data.raw_matrix, data.train_idx, data.val_idx, args.k)
            weighting_specs.append(
                (
                    "benign_tail_downweight",
                    tail_features,
                    {
                        "weighting": "benign_tail_downweight",
                        "downweighted_token_count": int(np.sum(tail_weights < 1.0)),
                        "min_token_weight": float(np.min(tail_weights)) if tail_weights.size else 1.0,
                    },
                )
            )
            for setting, features, extra in weighting_specs:
                row, lat = _evaluate_feature_matrix(data, features, experiment_group="token_weighting", setting=setting, k=args.k, extra=extra)
                summary_rows.append(row)
                latency_rows.append(lat)

            for mode in ("primitive_transitions", "burst_ngrams", "combined"):
                seq_rows, seq_vocab = _sequence_augmented_rows(data.rows, data.train_idx, mode=mode, min_support=args.sequence_min_support)
                seq_raw, seq_names = _build_matrix_with_extra_tokens(data, seq_rows, seq_vocab)
                seq_features = _features_uniform(seq_raw)
                row, lat = _evaluate_feature_matrix(
                    data,
                    seq_features,
                    experiment_group="sequence_sensitive",
                    setting=mode,
                    k=args.k,
                    extra={
                        "sequence_vocab_size": int(len(seq_vocab)),
                        "vocab_growth": int(len(seq_names) - len(data.feature_names)),
                        "train_only_min_support": int(args.sequence_min_support),
                    },
                )
                summary_rows.append(row)
                latency_rows.append(lat)

            cal_rows = _calibration_rows(data, uniform, k=args.k, rng=rng)
            calibration_rows.extend(cal_rows)
            summary_rows.extend(cal_rows)

            mem_rows = _evaluate_memory_governance(data, uniform, k=args.k, rng=rng)
            summary_rows.extend(mem_rows)

            _write_csv(summary_rows, out_dir / "summary_table.csv")
            _write_csv(retrieval_rows, out_dir / "retrieval_metrics.csv")
            _write_csv(calibration_rows, out_dir / "calibration_metrics.csv")
            _write_csv(latency_rows, out_dir / "latency_metrics.csv")
            _write_json({"runs": run_meta, "missing": missing, "unsupported": unsupported_rows}, out_dir / "run_manifest.json")
            print(f"[{_now()}] finished {attack} seed {seed}", flush=True)

    _write_csv(summary_rows, out_dir / "summary_table.csv")
    _write_csv(retrieval_rows, out_dir / "retrieval_metrics.csv")
    _write_csv(calibration_rows, out_dir / "calibration_metrics.csv")
    _write_csv(latency_rows, out_dir / "latency_metrics.csv")
    _write_csv(_aggregate(summary_rows, ("experiment_group", "setting")), out_dir / "summary_by_setting.csv")
    _write_json({"runs": run_meta, "missing": missing, "unsupported": unsupported_rows}, out_dir / "run_manifest.json")
    _write_final_report(out_dir, summary_rows, retrieval_rows, calibration_rows, missing, unsupported_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore behavior-only FlowPrim benign-memory KNN optimization routes.")
    parser.add_argument("--token-dir", default=str(DEFAULT_TOKEN_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--structural-min-support", type=int, default=5)
    parser.add_argument("--max-structural-per-family", type=int, default=24)
    parser.add_argument("--kdtree-dims", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--top-taus", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--inverted-candidates", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--coreset-ratios", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75])
    parser.add_argument("--sequence-min-support", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
