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
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.features.structural_primitives import (
    StructuralPrimitiveConfig,
    StructuralPrimitiveTrigger,
    build_train_only_structural_primitive_vocabulary,
    extract_structural_primitive_candidates,
    family_from_token,
    filter_triggers,
)
from src.features.token_alias import canonical_tokens, is_packet_burst_token, is_profile_token, is_structural_token


ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "scripts" / "52_sweep_anomaly_low_fpr.py"
UNKNOWN_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown"
DEFAULT_TOKEN_DIR = UNKNOWN_DIR / "tokens_category"
DEFAULT_OUT = ROOT / "results" / "primitive_categories"
ATTACK_SLUG = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "DoS": "dos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
    "Infiltration": "infiltration",
}
SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}


def _load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("flowprim_low_fpr_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sweep module from {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["flowprim_low_fpr_sweep"] = module
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _read_token_data(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _token_path(attack: str, seed: int, token_dir: Path = DEFAULT_TOKEN_DIR, artifact_prefix: str = "cicids2017") -> Path:
    slug = ATTACK_SLUG[attack]
    return token_dir / f"{artifact_prefix}_leave_one_{slug}_anomaly_seed{seed}_a3_full_rhythm.pt"


def _split_indices(token_data: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray([idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == split], dtype=np.int64)


def _row_tokens(token_data: dict[str, Any], row_idx: int) -> list[str]:
    id_to_token = S._id_to_token(token_data["vocab"])
    ids = token_data["input_ids"][row_idx].cpu().numpy()
    mask = token_data["attention_mask"][row_idx].cpu().numpy() > 0
    return canonical_tokens(id_to_token.get(int(token_id), "[UNK]") for token_id in ids[mask])


def _profile_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if is_profile_token(token)]


def _packet_burst_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if is_packet_burst_token(token)]


def _structural_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if is_structural_token(token)]


def _keep_token_for_view(token: str, feature_view: str) -> bool:
    if token in SPECIAL_TOKENS:
        return False
    is_v1 = is_profile_token(token)
    is_structural = is_structural_token(token)
    is_packet_burst = bool(_packet_burst_tokens([token]))
    if feature_view == "profile_only":
        return is_v1
    if feature_view == "structural_only":
        return is_structural
    if feature_view == "profile_plus_structural":
        return is_v1 or is_structural
    if feature_view == "packet_burst_only":
        return is_packet_burst
    if feature_view == "packet_burst_plus_profile":
        return is_packet_burst or is_v1
    if feature_view == "packet_burst_plus_structural":
        return is_packet_burst or is_structural
    if feature_view == "packet_burst_plus_profile_structural":
        return is_packet_burst or is_v1 or is_structural
    raise ValueError(f"Unknown feature view: {feature_view}")


def _raw_counts_for_view(token_rows: list[list[str]], vocab: dict[str, int], feature_view: str) -> tuple[np.ndarray, list[str]]:
    kept = sorted(token for token in vocab if _keep_token_for_view(token, feature_view))
    if not kept:
        raise ValueError(f"Feature view kept no tokens: {feature_view}")
    col = {token: idx for idx, token in enumerate(kept)}
    features = np.zeros((len(token_rows), len(kept)), dtype=np.float32)
    for row_idx, tokens in enumerate(token_rows):
        counts = Counter(token for token in tokens if token in col)
        for token, count in counts.items():
            features[row_idx, col[token]] = float(count)
    return features, kept


def _normalize(features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return features.astype(np.float32, copy=False)
    if mode == "l1":
        denom = np.sum(np.abs(features), axis=1, keepdims=True)
    elif mode == "l2":
        denom = np.linalg.norm(features, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported normalization: {mode}")
    return np.divide(features, denom, out=np.zeros_like(features, dtype=np.float32), where=denom > 0)


def _features_for_view(
    token_rows: list[list[str]],
    vocab: dict[str, int],
    train_idx: np.ndarray,
    *,
    feature_view: str,
    transform: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw, kept = _raw_counts_for_view(token_rows, vocab, feature_view)
    features = raw
    if transform.startswith("binary"):
        features = (features > 0).astype(np.float32)
        features = _normalize(features, transform.removeprefix("binary_") or "none")
    elif transform.startswith("tfidf"):
        train_counts = features[train_idx]
        df = np.sum(train_counts > 0, axis=0)
        idf = np.log((1.0 + len(train_idx)) / (1.0 + df)) + 1.0
        features = features * idf.reshape(1, -1).astype(np.float32)
        features = _normalize(features, transform.removeprefix("tfidf_") or "none")
    elif transform.startswith("count"):
        features = _normalize(features, transform.removeprefix("count_") or "none")
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    stats = {
        "feature_filter": feature_view,
        "transform": transform,
        "num_features": int(features.shape[1]),
        "kept_tokens": kept,
        "mean_nonzero": float(np.mean(np.sum(features != 0, axis=1))),
    }
    return features.astype(np.float32, copy=False), stats


def _groups(token_data: dict[str, Any], memory_scope: str) -> list[str]:
    if memory_scope != "global":
        raise ValueError("FlowPrim category experiments use global benign memory only.")
    return ["GLOBAL"] * len(token_data.get("meta", []))


def _safe_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _evaluate_view(
    token_data: dict[str, Any],
    token_rows: list[list[str]],
    vocab: dict[str, int],
    *,
    feature_view: str,
    transform: str,
    scorer: str,
    k: int,
    memory_scope: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = _split_indices(token_data, "train")
    val_idx = _split_indices(token_data, "val")
    test_idx = _split_indices(token_data, "test")
    features, feature_stats = _features_for_view(token_rows, vocab, train_idx, feature_view=feature_view, transform=transform)
    groups = _groups(token_data, memory_scope)
    start_score = time.perf_counter()
    val_scores = S._scores(features, train_idx, val_idx, groups, scorer=scorer, k=k)
    test_scores = S._scores(features, train_idx, test_idx, groups, scorer=scorer, k=k)
    score_seconds = time.perf_counter() - start_score
    metrics = S._evaluate(
        features,
        feature_stats,
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
    metrics["score_time_seconds"] = score_seconds
    metrics["query_ms_per_flow"] = float((score_seconds / max(len(val_idx) + len(test_idx), 1)) * 1000.0)
    return metrics, features, val_scores, test_scores, feature_stats


def _encode_enhanced_rows(base_rows: list[list[str]], structural_rows: list[list[StructuralPrimitiveTrigger]], structural_vocab_support: dict[str, int]) -> tuple[list[list[str]], dict[str, int]]:
    structural_vocab = set(structural_vocab_support)
    rows: list[list[str]] = []
    vocab_counter: Counter[str] = Counter()
    for base, triggers in zip(base_rows, structural_rows):
        structural = []
        for trigger in triggers:
            if trigger.name not in structural_vocab:
                continue
            repeats = max(1, int(trigger.count))
            structural.extend([trigger.name] * repeats)
        merged = list(base)
        if "[SEP]" in merged:
            sep_idx = len(merged) - 1 - merged[::-1].index("[SEP]")
            merged = merged[:sep_idx] + structural + merged[sep_idx:]
        else:
            merged.extend(structural)
        rows.append(merged)
        vocab_counter.update(merged)
    return rows, {token: idx for idx, token in enumerate(sorted(vocab_counter))}


def _primitive_category_for_view(view: str) -> str:
    if "profile_structural" in view or "profile_plus_structural" in view:
        return "profile_plus_structural"
    if "structural" in view:
        return "structural"
    if "profile" in view or view == "packet_burst_plus_profile":
        return "profile"
    if view == "packet_burst_plus_structural":
        return "structural"
    return "none"


def _view_uses_structural(view: str) -> bool:
    """Return whether structural primitive tokens are part of the active feature view."""

    return view in {
        "structural_only",
        "profile_plus_structural",
        "packet_burst_plus_structural",
        "packet_burst_plus_profile_structural",
    }


def _aggregate_numeric(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        vals = [_safe_float(row.get(key)) for row in rows]
        vals = [val for val in vals if not math.isnan(val)]
        if not vals:
            out[key] = ""
            continue
        out[key] = float(sum(vals) / len(vals))
        out[f"{key}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return out


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


def _top_lift_rows(
    token_rows: list[list[str]],
    labels: np.ndarray,
    *,
    seed: int,
    attack: str,
    max_rows: int = 50,
) -> list[dict[str, Any]]:
    benign = labels == 0
    attack_mask = labels == 1
    support: dict[str, tuple[int, int]] = {}
    structural_vocab = sorted({token for tokens in token_rows for token in set(tokens) if is_structural_token(token)})
    for token in structural_vocab:
        has = np.asarray([token in set(tokens) for tokens in token_rows], dtype=bool)
        b_rate = float(np.mean(has[benign])) if np.any(benign) else 0.0
        a_rate = float(np.mean(has[attack_mask])) if np.any(attack_mask) else 0.0
        lift = (a_rate + 1e-6) / (b_rate + 1e-6)
        support[token] = (int(np.sum(has[benign])), int(np.sum(has[attack_mask])))
        yield_row = {
            "seed": seed,
            "heldout_attack": attack,
            "primitive": token,
            "family": family_from_token(token),
            "benign_rate": b_rate,
            "attack_rate": a_rate,
            "attack_to_benign_lift": lift,
            "benign_support": support[token][0],
            "attack_support": support[token][1],
        }
        yield yield_row


def _scale_summary(
    token_rows: list[list[str]],
    structural_rows: list[list[StructuralPrimitiveTrigger]],
    structural_vocab: dict[str, int],
    labels: np.ndarray,
    *,
    seed: int,
    attack: str,
) -> list[dict[str, Any]]:
    counts = [sum(1 for token in tokens if is_structural_token(token)) for tokens in token_rows]
    med = float(statistics.median(counts)) if counts else 0.0
    p95 = float(np.quantile(np.asarray(counts, dtype=np.float32), 0.95)) if counts else 0.0
    rows = [
        {
            "seed": seed,
            "heldout_attack": attack,
            "primitive_category": "structural",
            "primitive_vocab_size": len(structural_vocab),
            "avg_primitive_count_per_flow": float(sum(counts) / max(len(counts), 1)),
            "median_primitive_count_per_flow": med,
            "p95_primitive_count_per_flow": p95,
            "flows_with_at_least_one_structural": float(sum(1 for count in counts if count > 0) / max(len(counts), 1)),
            "train_only_min_support": min(structural_vocab.values()) if structural_vocab else 0,
        }
    ]
    family_counts: Counter[str] = Counter()
    flow_family_hits: dict[str, int] = Counter()
    for tokens in token_rows:
        fams = {family_from_token(token) for token in tokens if is_structural_token(token)}
        for fam in fams:
            flow_family_hits[fam] += 1
        for token in tokens:
            if is_structural_token(token):
                family_counts[family_from_token(token)] += 1
    for fam in sorted(set(family_counts) | set(flow_family_hits)):
        rows.append(
            {
                    "seed": seed,
                    "heldout_attack": attack,
                    "primitive_category": "structural",
                    "family": fam,
                "family_token_count": family_counts[fam],
                "family_flow_trigger_rate": float(flow_family_hits[fam] / max(len(token_rows), 1)),
            }
        )
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_total: Counter[str] = Counter()
    for row_tokens, label in zip(token_rows, labels.tolist()):
        label_name = "BENIGN" if int(label) == 0 else attack
        class_total[label_name] += 1
        flow_families = {family_from_token(token) for token in row_tokens if is_structural_token(token)}
        for fam in flow_families:
            class_counts[label_name][fam] += 1
    for label_name, counts_by_family in sorted(class_counts.items()):
        for fam, count in sorted(counts_by_family.items()):
            rows.append(
                {
                    "seed": seed,
                    "heldout_attack": attack,
                    "class": label_name,
                    "family": fam,
                    "class_family_trigger_rate": float(count / max(class_total[label_name], 1)),
                }
            )
    return rows


def _explanation_coverage(
    token_data: dict[str, Any],
    token_rows: list[list[str]],
    metrics_by_view: dict[str, dict[str, Any]],
    test_scores_by_view: dict[str, np.ndarray],
    *,
    seed: int,
    attack: str,
) -> list[dict[str, Any]]:
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    test_idx = _split_indices(token_data, "test")
    rows: list[dict[str, Any]] = []
    for view, scores in test_scores_by_view.items():
        view_uses_structural = _view_uses_structural(view)
        threshold = float(metrics_by_view[view].get("threshold_at_1pct_fpr") or metrics_by_view[view].get("best_threshold") or 0.0)
        alerts = scores >= threshold
        global_alert_indices = test_idx[alerts]
        if len(global_alert_indices) == 0:
            rows.append(
                {
                    "seed": seed,
                    "heldout_attack": attack,
                    "feature_view": view,
                    "view_uses_structural": view_uses_structural,
                    "coverage_scope": "view_active" if view_uses_structural else "audit_availability_only",
                    "alerts": 0,
                    "alerts_with_structural_available": 0,
                    "structural_available_in_alert_rate": 0.0,
                    "alerts_with_structural_as_used": 0,
                    "explanation_coverage": "" if not view_uses_structural else 0.0,
                }
            )
            continue
        structural_hit = 0
        family_counter: Counter[str] = Counter()
        for idx in global_alert_indices.tolist():
            fams = {family_from_token(token) for token in token_rows[int(idx)] if is_structural_token(token)}
            if fams:
                structural_hit += 1
                family_counter.update(fams)
        base = {
            "seed": seed,
            "heldout_attack": attack,
            "feature_view": view,
            "view_uses_structural": view_uses_structural,
            "coverage_scope": "view_active" if view_uses_structural else "audit_availability_only",
            "alerts": int(len(global_alert_indices)),
            "alerts_with_structural_available": int(structural_hit),
            "structural_available_in_alert_rate": float(structural_hit / max(len(global_alert_indices), 1)),
            "alerts_with_structural_as_used": int(structural_hit if view_uses_structural else 0),
            "explanation_coverage": float(structural_hit / max(len(global_alert_indices), 1)) if view_uses_structural else "",
            "attack_alerts": int(np.sum(alerts & (labels[test_idx] == 1))),
            "benign_alerts": int(np.sum(alerts & (labels[test_idx] == 0))),
            "attack_alerts_with_structural_available": int(
                sum(1 for idx in global_alert_indices.tolist() if labels[idx] == 1 and any(is_structural_token(token) for token in token_rows[int(idx)]))
            ),
            "benign_alerts_with_structural_available": int(
                sum(1 for idx in global_alert_indices.tolist() if labels[idx] == 0 and any(is_structural_token(token) for token in token_rows[int(idx)]))
            ),
        }
        rows.append(base)
        if view_uses_structural:
            for fam, count in sorted(family_counter.items()):
                rows.append({**base, "structural_family": fam, "structural_family_alert_hits": int(count)})
    return rows


def _augment_one(
    token_data: dict[str, Any],
    cfg: StructuralPrimitiveConfig,
) -> tuple[list[list[str]], list[list[StructuralPrimitiveTrigger]], dict[str, int], dict[str, int], dict[str, Any]]:
    base_rows = [_row_tokens(token_data, idx) for idx in range(len(token_data["meta"]))]
    train_idx = _split_indices(token_data, "train")
    start_extract = time.perf_counter()
    structural_rows_raw = [extract_structural_primitive_candidates(tokens, cfg) for tokens in base_rows]
    extract_seconds = time.perf_counter() - start_extract
    structural_vocab = build_train_only_structural_primitive_vocabulary(structural_rows_raw, train_idx, min_support=cfg.min_support)
    structural_rows = [filter_triggers(triggers, structural_vocab) for triggers in structural_rows_raw]
    start_encode = time.perf_counter()
    enhanced_rows, vocab = _encode_enhanced_rows(base_rows, structural_rows, structural_vocab)
    encode_seconds = time.perf_counter() - start_encode
    timing = {
        "extraction_time_seconds": extract_seconds,
        "extraction_ms_per_flow": float(extract_seconds / max(len(base_rows), 1) * 1000.0),
        "tokenization_time_seconds": encode_seconds,
        "tokenization_ms_per_flow": float(encode_seconds / max(len(base_rows), 1) * 1000.0),
    }
    return enhanced_rows, structural_rows, structural_vocab, vocab, timing


def _metrics_row(
    row: dict[str, Any],
    *,
    seed: int,
    attack: str,
    feature_view: str,
    primitive_category: str,
    memory_scope: str,
    structural_vocab_size: int,
    memory_size: int,
    test_size: int,
) -> dict[str, Any]:
    p99_fpr = row.get("val_p99_0_false_positive_rate")
    return {
        "seed": seed,
        "heldout_attack": attack,
        "feature_view": feature_view,
        "primitive_category": primitive_category,
        "memory_scope": memory_scope,
        "vocab_size": row.get("num_features"),
        "structural_vocab_size": structural_vocab_size,
        "memory_size": memory_size,
        "threshold_type": "oracle_1pct_and_benign_val_p99",
        "transform": row.get("transform"),
        "scorer": row.get("scorer"),
        "k": row.get("k"),
        "macro_f1_oracle_1pct": "",
        "best_macro_f1": row.get("best_macro_f1"),
        "auroc": row.get("auroc"),
        "auprc": row.get("auprc"),
        "fpr95": row.get("fpr95"),
        "recall_at_0_1pct_fpr": row.get("recall_at_0_1pct_fpr"),
        "recall_at_1pct_fpr": row.get("recall_at_1pct_fpr"),
        "recall_at_5pct_fpr": row.get("recall_at_5pct_fpr"),
        "actual_fpr_at_1pct_fpr": row.get("actual_fpr_at_1pct_fpr"),
        "val_p99_threshold": row.get("val_p99_0_threshold"),
        "val_p99_realized_fpr": p99_fpr,
        "val_p99_attack_recall": row.get("val_p99_0_attack_recall"),
        "val_p99_macro_f1": row.get("val_p99_0_macro_f1"),
        "false_alerts_per_10k_benign": float(p99_fpr * 10000.0) if p99_fpr not in {None, ""} else "",
        "query_ms_per_flow": row.get("query_ms_per_flow"),
        "test_size": test_size,
        "attack_labels_used_for_threshold": False,
        "raw_ip_used_as_token": False,
        "absolute_time_used_as_token": False,
        "five_tuple_used_as_token": False,
    }


def _fix_oracle_macro_f1(
    out_row: dict[str, Any],
    labels: np.ndarray,
    test_idx: np.ndarray,
    test_scores: np.ndarray,
) -> None:
    y_true = labels[test_idx].astype(np.int64)
    r1 = S._best_recall_under_fpr(y_true, test_scores, 0.01)
    out_row["macro_f1_oracle_1pct"] = r1.get("macro_f1")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = StructuralPrimitiveConfig(
        enabled=True,
        enable_packet_shape_primitives=bool(args.enable_packet_shape_primitives),
        enable_burst_shape_primitives=bool(args.enable_burst_shape_primitives),
        enable_timing_rhythm_primitives=bool(args.enable_timing_rhythm_primitives),
        enable_direction_transition_primitives=bool(args.enable_direction_transition_primitives),
        enable_composite_primitives=bool(args.enable_composite_primitives),
        min_support=int(args.min_support),
        max_structural_primitives_per_family=int(args.max_structural_primitives_per_family),
    )
    memory_scope = "global"
    feature_views = args.feature_views
    metric_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    top_lift_rows: list[dict[str, Any]] = []
    missing_token_corpora: list[str] = []
    token_out_dir = out_dir / "tokens"
    token_out_dir.mkdir(exist_ok=True)

    for seed in args.seeds:
        for attack in args.attacks:
            token_path = _token_path(attack, seed, Path(args.token_dir), args.artifact_prefix)
            if not token_path.exists():
                missing_token_corpora.append(str(token_path))
                print(f"[WARN] missing token corpus: {token_path}", file=sys.stderr)
                continue
            token_data = _read_token_data(token_path)
            labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
            train_idx = _split_indices(token_data, "train")
            val_idx = _split_indices(token_data, "val")
            test_idx = _split_indices(token_data, "test")
            if not np.all(labels[train_idx] == 0):
                raise ValueError(f"{token_path} train split is not benign-only")
            if not np.all(labels[val_idx] == 0):
                raise ValueError(f"{token_path} val split is not benign-only")
            enhanced_rows, structural_rows, structural_vocab, vocab, timing = _augment_one(token_data, cfg)
            token_payload = {
                "source_token_path": str(token_path),
                "dataset_name": args.dataset_name,
                "artifact_prefix": args.artifact_prefix,
                "seed": seed,
                "heldout_attack": attack,
                "structural_config": cfg.__dict__,
                "structural_vocab": structural_vocab,
                "enhanced_vocab": vocab,
                "canonical_primitive_prefixes": ["PRIM_PROFILE_", "PRIM_STRUCT_"],
                "feature_token_source": "base behavior tokens plus train-filtered structural primitives inserted before [SEP]",
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
                "memory_scope": memory_scope,
                "rows": [
                    {
                        "flow_id": token_data["meta"][idx].get("flow_id"),
                        "split": token_data["meta"][idx].get("split"),
                        "label": token_data["meta"][idx].get("binary_label"),
                        "attack_family": token_data["meta"][idx].get("attack_family"),
                        "structural_primitives": [trigger.to_dict() for trigger in structural_rows[idx]],
                    }
                    for idx in range(len(structural_rows))
                ],
            }
            token_json = token_out_dir / f"structural_{ATTACK_SLUG[attack]}_seed{seed}.json"
            token_json.write_text(json.dumps(token_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (token_out_dir / f"structural_{ATTACK_SLUG[attack]}_seed{seed}_vocab.json").write_text(
                json.dumps(vocab, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            scale_rows.extend(_scale_summary(enhanced_rows, structural_rows, structural_vocab, labels, seed=seed, attack=attack))
            lift = sorted(
                _top_lift_rows(enhanced_rows, labels, seed=seed, attack=attack),
                key=lambda row: (float(row["attack_to_benign_lift"]), float(row["attack_rate"])),
                reverse=True,
            )[: args.top_lift_k]
            top_lift_rows.extend(lift)

            metrics_by_view: dict[str, dict[str, Any]] = {}
            test_scores_by_view: dict[str, np.ndarray] = {}
            for feature_view in feature_views:
                scorer = args.scorer
                transform = args.transform
                k = int(args.k)
                start_total = time.perf_counter()
                row, _features, _val_scores, test_scores, feature_stats = _evaluate_view(
                    token_data,
                    enhanced_rows,
                    {token: idx for idx, token in enumerate(sorted({tok for row_tokens in enhanced_rows for tok in row_tokens}))},
                    feature_view=feature_view,
                    transform=transform,
                    scorer=scorer,
                    k=k,
                    memory_scope=memory_scope,
                )
                total_seconds = time.perf_counter() - start_total + timing["extraction_time_seconds"] + timing["tokenization_time_seconds"]
                out_row = _metrics_row(
                    row,
                    seed=seed,
                    attack=attack,
                    feature_view=feature_view,
                    primitive_category=_primitive_category_for_view(feature_view),
                    memory_scope=memory_scope,
                    structural_vocab_size=len(structural_vocab),
                    memory_size=len(train_idx),
                    test_size=len(test_idx),
                )
                _fix_oracle_macro_f1(out_row, labels, test_idx, test_scores)
                metric_rows.append(out_row)
                metrics_by_view[feature_view] = row
                test_scores_by_view[feature_view] = test_scores
                runtime_rows.append(
                    {
                        "seed": seed,
                        "heldout_attack": attack,
                        "feature_view": feature_view,
                        "primitive_category": out_row["primitive_category"],
                        "memory_scope": memory_scope,
                        **timing,
                        "knn_scoring_ms_per_flow": row.get("query_ms_per_flow"),
                        "total_flow_ready_ms_per_flow": float(total_seconds / max(len(token_data["meta"]), 1) * 1000.0),
                        "memory_size": len(train_idx),
                        "feature_vocab_size": feature_stats["num_features"],
                        "structural_vocab_size": len(structural_vocab),
                    }
                )
            coverage_rows.extend(_explanation_coverage(token_data, enhanced_rows, metrics_by_view, test_scores_by_view, seed=seed, attack=attack))

    numeric_keys = [
        "best_macro_f1",
        "macro_f1_oracle_1pct",
        "auroc",
        "auprc",
        "fpr95",
        "recall_at_0_1pct_fpr",
        "recall_at_1pct_fpr",
        "recall_at_5pct_fpr",
        "val_p99_realized_fpr",
        "val_p99_attack_recall",
        "val_p99_macro_f1",
        "false_alerts_per_10k_benign",
        "query_ms_per_flow",
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["feature_view"]), str(row["primitive_category"]), str(row["memory_scope"]))].append(row)
    for (view, category, memory_scope_value), rows in sorted(grouped.items()):
        attribution_rows.append(
            {
                "feature_view": view,
                "primitive_category": category,
                "memory_scope": memory_scope_value,
                "runs": len(rows),
                **_aggregate_numeric(rows, numeric_keys),
            }
        )

    _write_csv(metric_rows, out_dir / "primitive_category_unknown_metrics.csv")
    _write_csv(attribution_rows, out_dir / "primitive_category_feature_attribution.csv")
    _write_csv(scale_rows, out_dir / "primitive_category_scale_summary.csv")
    _write_csv(coverage_rows, out_dir / "primitive_category_explanation_coverage.csv")
    _write_csv(runtime_rows, out_dir / "primitive_category_runtime.csv")
    _write_csv(top_lift_rows, out_dir / "top_structural_primitives_by_lift.csv")
    summary = {
        "config": cfg.__dict__,
        "dataset_name": args.dataset_name,
        "artifact_prefix": args.artifact_prefix,
        "attacks": args.attacks,
        "seeds": args.seeds,
        "feature_views": feature_views,
        "memory_scope": memory_scope,
        "metric_rows": len(metric_rows),
        "missing_token_corpora": missing_token_corpora,
        "output": str(out_dir),
        "notes": [
            "Structural primitive vocabulary is fit from train split only.",
            "Held-out attack labels are not used for vocab, memory, or benign-validation P99 thresholds.",
            "Existing raw IP, absolute timestamp, full five-tuple, protocol, and service fields are not used as behavior primitive tokens or memory-grouping keys.",
        ],
    }
    (out_dir / "primitive_category_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_readme(out_dir, attribution_rows, scale_rows, coverage_rows, runtime_rows, top_lift_rows, args, cfg, missing_token_corpora)
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        if value == "":
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_readme(
    out_dir: Path,
    attribution_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    top_lift_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    cfg: StructuralPrimitiveConfig,
    missing_token_corpora: list[str],
) -> None:
    by_view = {str(row["feature_view"]): row for row in attribution_rows}
    profile_row = by_view.get("profile_only")
    structural_row = by_view.get("structural_only")
    pb = by_view.get("packet_burst_only")
    verdict_lines: list[str] = []
    if profile_row and structural_row:
        verdict_lines.append(
            f"- structural primitive only vs profile primitive only: AUROC {_fmt(structural_row.get('auroc'))} vs {_fmt(profile_row.get('auroc'))}; "
            f"R@1%FPR {_fmt(structural_row.get('recall_at_1pct_fpr'))} vs {_fmt(profile_row.get('recall_at_1pct_fpr'))}; "
            f"P99 FPR {_fmt(structural_row.get('val_p99_realized_fpr'))} vs {_fmt(profile_row.get('val_p99_realized_fpr'))}."
        )
    if structural_row and pb:
        verdict_lines.append(
            f"- structural primitive gap to packet+burst: AUROC gap {_fmt(float(pb.get('auroc') or 0) - float(structural_row.get('auroc') or 0))}; "
            f"R@1%FPR gap {_fmt(float(pb.get('recall_at_1pct_fpr') or 0) - float(structural_row.get('recall_at_1pct_fpr') or 0))}."
        )
    cov = [
        row
        for row in coverage_rows
        if row.get("feature_view") in {"structural_only", "packet_burst_plus_structural", "packet_burst_plus_profile_structural"}
        and row.get("view_uses_structural")
        and not row.get("structural_family")
    ]
    if cov:
        avg_cov = sum(float(row.get("explanation_coverage") or 0.0) for row in cov) / len(cov)
        verdict_lines.append(f"- Average structural primitive alert explanation coverage across reported structural primitive views: {_fmt(avg_cov)}.")
    if not verdict_lines:
        verdict_lines.append("- No completed metric rows were available; inspect the CSV files for partial runs.")

    lines = [
        "# Primitive Category Experiment Artifacts",
        "",
        "This directory evaluates profile and structural behavior primitives as two categories under the same FlowPrim primitive miner.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/run_primitive_category_experiments.sh --mode quick",
        "# or",
        "bash scripts/run_primitive_category_experiments.sh --mode full",
        "```",
        "",
        "## Split and Leakage Controls",
        "",
        f"- Source token corpora are canonicalized {args.dataset_name} leave-one unknown artifacts under `{args.token_dir}`.",
        "- Structural primitive vocabulary and min-support filtering are fit from the `train` split only.",
        "- Existing leave-one anomaly splits keep train/validation benign-only; held-out attack labels are used only for test metrics.",
        "- Raw IP addresses, absolute timestamps, and complete five-tuples are not used as behavior primitive tokens.",
        "- Protocol and service fields are not used as behavior tokens or memory-grouping keys in the reported category experiment.",
    ]
    if missing_token_corpora:
        lines.extend(
            [
                "- Missing token corpora were skipped rather than filled with synthetic or fabricated data:",
                *[f"  - `{path}`" for path in missing_token_corpora],
            ]
        )
    lines.extend(
        [
            "",
        "## Structural Primitive Configuration",
        "",
        "```json",
        json.dumps(cfg.__dict__, indent=2, sort_keys=True),
        "```",
        "",
        "## Feature Views",
        "",
        "- `profile_only`: profile primitive evidence using canonical `PRIM_PROFILE_*` tokens only.",
        "- `structural_only`: structural primitive evidence using canonical `PRIM_STRUCT_*` tokens only.",
        "- `profile_plus_structural`: profile plus structural primitives.",
        "- `packet_burst_only`: packet/burst behavior-token baseline.",
        "- `packet_burst_plus_profile`: packet/burst plus profile primitives.",
        "- `packet_burst_plus_structural`: packet/burst plus `PRIM_STRUCT_*`.",
        "- `packet_burst_plus_profile_structural`: all behavior evidence.",
        "",
        "## Current Verdict",
        "",
        *verdict_lines,
        "",
        "## Key Files",
        "",
        "- `primitive_category_feature_attribution.csv`: aggregate view-level metrics.",
        "- `primitive_category_unknown_metrics.csv`: per seed/heldout/view metrics.",
        "- `primitive_category_scale_summary.csv`: vocabulary size, primitive counts, trigger rates.",
        "- `primitive_category_explanation_coverage.csv`: alert coverage by structural primitives.",
        "- `primitive_category_runtime.csv`: extraction/tokenization/scoring time.",
        "- `top_structural_primitives_by_lift.csv`: structural primitives ranked by attack-vs-benign lift.",
        "- `tokens/structural_*_seed*.json`: enhanced token-vocabulary metadata and per-flow structural primitive trigger provenance.",
        "- `tokens/structural_*_seed*_vocab.json`: enhanced behavior-token vocabulary used by the category scorer.",
        "",
        "## Notes",
        "",
        "- `PRIM_STRUCT_PKT_*` encodes packet-shape structures such as direction+length n-grams, small-packet runs, same-length runs, spikes, and ramps.",
        "- `PRIM_STRUCT_BURST_*` encodes burst templates, request-response pairs, fast alternation, asymmetry, short-burst multiplicity, and duplicate templates.",
        "- `PRIM_STRUCT_IAT_*` / `PRIM_STRUCT_FLOW_*` encodes timing rhythm such as low variance, beacon-like timing, bursty-then-idle, heavy tail, slow drip, and periodic burst intervals.",
        "- `PRIM_STRUCT_DIR_*` encodes direction-transition structure.",
        "- `PRIM_STRUCT_COMP_*` encodes lightweight composites across packet, burst, direction, and timing evidence.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structural primitive feature attribution over FlowPrim leave-one unknown token corpora.")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--token-dir", default=str(DEFAULT_TOKEN_DIR))
    parser.add_argument("--dataset-name", default="CICIDS2017")
    parser.add_argument("--artifact-prefix", default="cicids2017")
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[43])
    parser.add_argument("--feature_views", nargs="+", default=[
        "profile_only",
        "structural_only",
        "profile_plus_structural",
        "packet_burst_only",
        "packet_burst_plus_profile",
        "packet_burst_plus_structural",
        "packet_burst_plus_profile_structural",
    ])
    parser.add_argument("--transform", default="binary_l2")
    parser.add_argument("--scorer", default="knn_cosine")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min_support", type=int, default=5)
    parser.add_argument("--max_structural_primitives_per_family", type=int, default=24)
    parser.add_argument("--top_lift_k", type=int, default=50)
    parser.add_argument("--enable_packet_shape_primitives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_burst_shape_primitives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_timing_rhythm_primitives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_direction_transition_primitives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_composite_primitives", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
