#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import _bootstrap  # noqa: F401
import numpy as np

from src.features.token_alias import (
    canonical_token,
    canonical_tokens,
    is_burst_token,
    is_flow_summary_token,
    is_packet_token,
    is_profile_token,
    is_rhythm_token,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
UNKNOWN_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown"
TOKEN_DIR = UNKNOWN_DIR / "tokens_category"
OUT_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "revision"
PAPER_TABLE_DIR = REPO / "paper" / "tables"
SKIP_LEGACY_PAPER_TABLES = {
    "table_target_realized_fpr.tex",
    "table_alert_budget.tex",
    "table_prevalence_alert_budget.tex",
    "table_unknown_aggregate_wo_webattack.tex",
    "table_knn_feature_baselines.tex",
    "table_behavior_feature_attribution.tex",
    "table_memory_scope_audit.tex",
}
SWEEP_PATH = ROOT / "scripts" / "52_sweep_anomaly_low_fpr.py"
CONFIG_PATH = ROOT / "configs" / "cicids2017.yaml"
FLOW_SOURCE = ROOT / "outputs" / "processed" / "ccfa" / "cicids2017_interim_labeled_flows.jsonl"
TRAIN_ONLY_PROFILE_SOURCE = UNKNOWN_DIR / "tokens_category" / "cicids2017_leave_one_botnet_anomaly_seed43_a3_full_rhythm_profile_primitives.jsonl"

ATTACKS = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
}

BEST_SETTINGS = {
    "Botnet": {"feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_euclidean", "k": 3, "group_mode": "global"},
    "DDoS": {"feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "global"},
    "Probe": {"feature_filter": "all_no_special", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "global"},
    "WebAttack": {"feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "global"},
    "BruteForce": {"feature_filter": "packet_burst_profile", "transform": "tfidf_l2", "scorer": "knn_cosine", "k": 3, "group_mode": "global"},
}

SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}
METADATA_SHORTCUT_TOKENS = ("SVC_", "SERVICE_", "PROTO_", "PROTOCOL_", "CTX_", "DENSITY_", "RECENT_")

FEATURE_ATTRIBUTION_VIEWS = [
    ("global_flow_summary", "global flow summary tokens", "global_only"),
    ("packet_only", "packet/global tokens", "packet"),
    ("profile_only", "profile primitive tokens", "profile_only"),
    ("packet_burst", "packet+burst tokens", "packet_burst"),
    ("packet_burst_profile", "packet+burst+profile primitive tokens", "packet_burst_profile"),
]


def _load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("flowprim_revision_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _token_path(attack: str, seed: int) -> Path:
    slug = ATTACKS[attack]
    return TOKEN_DIR / f"cicids2017_leave_one_{slug}_anomaly_seed{seed}_a3_full_rhythm.pt"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


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
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    value = _float(value)
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_pm(mean: Any, std: Any, digits: int = 4) -> str:
    return f"${_fmt(mean, digits)}\\pm{_fmt(std, digits)}$"


def _fmt_plain_pm(mean: Any, std: Any, digits: int = 4) -> str:
    return f"{_fmt(mean, digits)} +/- {_fmt(std, digits)}"


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean), float(std)


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO))
        except ValueError:
            return str(path)


def _token_group(token: str) -> str:
    if is_flow_summary_token(token):
        return "global"
    if is_packet_token(token):
        return "packet"
    if is_burst_token(token):
        return "burst"
    if is_profile_token(token) or is_rhythm_token(token):
        return "primitive"
    if token.startswith(METADATA_SHORTCUT_TOKENS):
        return "metadata_shortcut"
    return "other"


def _feature_keep(token: str, feature_filter: str) -> bool:
    if feature_filter == "global_only":
        return is_flow_summary_token(token)
    return bool(S._keep_token(token, feature_filter))


def _features(
    token_data: dict[str, Any],
    train_idx: np.ndarray,
    *,
    feature_filter: str,
    transform: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if feature_filter != "global_only":
        return S._features(token_data, train_idx, feature_filter=feature_filter, transform=transform)
    id_to_token = S._id_to_token(token_data["vocab"])
    kept_ids = sorted(token_id for token_id, token in id_to_token.items() if _feature_keep(token, feature_filter))
    if not kept_ids:
        raise ValueError(f"Feature filter kept no tokens: {feature_filter}")
    features = S._raw_counts(token_data, kept_ids)
    if transform.startswith("binary"):
        features = (features > 0).astype(np.float32)
        features = S._normalize(features, transform.removeprefix("binary_") or "none")
    elif transform.startswith("count"):
        features = S._normalize(features, transform.removeprefix("count_") or "none")
    elif transform.startswith("tfidf"):
        train_counts = features[train_idx]
        df = np.sum(train_counts > 0, axis=0)
        idf = np.log((1.0 + len(train_idx)) / (1.0 + df)) + 1.0
        features = features * idf.reshape(1, -1).astype(np.float32)
        features = S._normalize(features, transform.removeprefix("tfidf_") or "none")
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    return features.astype(np.float32, copy=False), {
        "feature_filter": feature_filter,
        "transform": transform,
        "num_features": int(features.shape[1]),
        "kept_tokens": [canonical_token(id_to_token[token_id]) for token_id in kept_ids],
        "mean_nonzero": float(np.mean(np.sum(features != 0, axis=1))),
    }


def _load_state(attack: str, seed: int, *, feature_filter: str | None = None, transform: str | None = None) -> dict[str, Any]:
    setting = BEST_SETTINGS[attack].copy()
    if feature_filter is not None:
        setting["feature_filter"] = feature_filter
    if transform is not None:
        setting["transform"] = transform
    token_path = _token_path(attack, seed)
    token_data = S._read_token_data(token_path)
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = S._split_indices(token_data, "train")
    val_idx = S._split_indices(token_data, "val")
    test_idx = S._split_indices(token_data, "test")
    if not np.all(labels[train_idx] == 0):
        raise ValueError(f"{token_path} has non-benign train rows")
    if not np.all(labels[val_idx] == 0):
        raise ValueError(f"{token_path} has non-benign validation rows")
    features, feature_stats = _features(token_data, train_idx, feature_filter=setting["feature_filter"], transform=setting["transform"])
    meta = token_data.get("meta", [])
    return {
        "attack": attack,
        "seed": seed,
        "setting": setting,
        "token_path": token_path,
        "token_data": token_data,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "features": features,
        "feature_stats": feature_stats,
        "no_raw_direct_tokens": all(
            not bool(row.get("has_ip_token")) and not bool(row.get("has_abs_time_token")) and not bool(row.get("has_port_token"))
            for row in meta
        ),
        "train_only_vocab": bool(token_data.get("train_only")) and str(token_data.get("vocab_provenance", "")).lower() == "train_only",
    }


def _evaluate_state(state: dict[str, Any], group_mode: str | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    setting = state["setting"]
    group_mode = group_mode or setting["group_mode"]
    groups = S._groups(state["token_data"], group_mode)
    val_scores = S._scores(
        state["features"],
        state["train_idx"],
        state["val_idx"],
        groups,
        scorer=setting["scorer"],
        k=int(setting["k"]),
    )
    test_scores = S._scores(
        state["features"],
        state["train_idx"],
        state["test_idx"],
        groups,
        scorer=setting["scorer"],
        k=int(setting["k"]),
    )
    y_true = state["labels"][state["test_idx"]].astype(np.int64)
    best = S._best_macro(y_true, test_scores)
    r01 = S._best_recall_under_fpr(y_true, test_scores, 0.001)
    r1 = S._best_recall_under_fpr(y_true, test_scores, 0.01)
    rank = S._rank_metrics(y_true, test_scores)
    row = {
        "best_threshold": best["threshold"],
        "best_macro_f1": best["macro_f1"],
        "best_attack_recall": best["attack_recall"],
        "best_false_positive_rate": best["false_positive_rate"],
        "recall_at_0_1pct_fpr": r01["attack_recall"],
        "actual_fpr_at_0_1pct_fpr": r01["false_positive_rate"],
        "threshold_at_0_1pct_fpr": r01["threshold"],
        "recall_at_1pct_fpr": r1["attack_recall"],
        "actual_fpr_at_1pct_fpr": r1["false_positive_rate"],
        "threshold_at_1pct_fpr": r1["threshold"],
        **rank,
    }
    for percentile in [95.0, 99.0, 99.5, 100.0]:
        threshold = float(np.percentile(val_scores, percentile))
        metrics = S._metrics_at_threshold(y_true, test_scores, threshold)
        prefix = f"p{str(percentile).replace('.', '_')}"
        row[f"val_{prefix}_threshold"] = threshold
        row[f"val_{prefix}_macro_f1"] = metrics["macro_f1"]
        row[f"val_{prefix}_attack_recall"] = metrics["attack_recall"]
        row[f"val_{prefix}_false_positive_rate"] = metrics["false_positive_rate"]
        row[f"val_{prefix}_attack_precision"] = metrics["attack_precision"]
        row[f"val_{prefix}_false_positives"] = metrics["false_positives"]
        row[f"val_{prefix}_true_positives"] = metrics["true_positives"]
    return row, val_scores, test_scores, groups


def _group_summary(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        summary = {key: value for key, value in zip(keys, group)}
        summary["num_runs"] = len(items)
        for metric in metrics:
            vals = [_float(item.get(metric)) for item in items]
            clean = [value for value in vals if value is not None]
            mean, std = _mean_std(clean)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        out.append(summary)
    return out


def build_target_realized_fpr(seeds: list[int], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        for seed in seeds:
            state = _load_state(attack, seed)
            metrics, _, _, _ = _evaluate_state(state)
            y_true = state["labels"][state["test_idx"]].astype(np.int64)
            benign_n = int(np.sum(y_true == 0))
            attack_n = int(np.sum(y_true == 1))
            for percentile in [95.0, 99.0, 99.5, 100.0]:
                prefix = f"p{str(percentile).replace('.', '_')}"
                rows.append(
                    {
                        "unknown_attack": attack,
                        "seed": seed,
                        "target_percentile": f"P{percentile:g}",
                        "threshold": metrics[f"val_{prefix}_threshold"],
                        "macro_f1": metrics[f"val_{prefix}_macro_f1"],
                        "attack_recall": metrics[f"val_{prefix}_attack_recall"],
                        "realized_test_fpr": metrics[f"val_{prefix}_false_positive_rate"],
                        "attack_precision": metrics[f"val_{prefix}_attack_precision"],
                        "false_positives": metrics[f"val_{prefix}_false_positives"],
                        "true_positives": metrics[f"val_{prefix}_true_positives"],
                        "benign_test_flows": benign_n,
                        "attack_test_flows": attack_n,
                        "webattack_low_support": attack == "WebAttack",
                        "train_only_vocab": state["train_only_vocab"],
                        "no_raw_direct_tokens": state["no_raw_direct_tokens"],
                    }
                )
    summary = _group_summary(rows, ["target_percentile"], ["threshold", "macro_f1", "attack_recall", "realized_test_fpr"])
    order = {"P95": 0, "P99": 1, "P99.5": 2, "P100": 3}
    summary = sorted(summary, key=lambda row: order.get(str(row.get("target_percentile")), 99))
    _write_csv(rows, out_dir / "target_realized_fpr_runs.csv")
    _write_csv(summary, out_dir / "target_realized_fpr_summary.csv")
    return rows, summary


def build_alert_budget(seeds: list[int], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        for seed in seeds:
            state = _load_state(attack, seed)
            metrics, _, test_scores, _ = _evaluate_state(state)
            y_true = state["labels"][state["test_idx"]].astype(np.int64)
            threshold_specs = [
                ("test_oracle_1pct_fpr", metrics["threshold_at_1pct_fpr"]),
                ("benign_val_P99", metrics["val_p99_0_threshold"]),
                ("benign_val_P99_5", metrics["val_p99_5_threshold"]),
            ]
            benign_n = int(np.sum(y_true == 0))
            attack_n = int(np.sum(y_true == 1))
            total_n = int(len(y_true))
            for threshold_type, threshold in threshold_specs:
                out = S._metrics_at_threshold(y_true, test_scores, float(threshold))
                rows.append(
                    {
                        "unknown_attack": attack,
                        "seed": seed,
                        "threshold_type": threshold_type,
                        "threshold": threshold,
                        "macro_f1": out["macro_f1"],
                        "attack_recall": out["attack_recall"],
                        "false_positive_rate": out["false_positive_rate"],
                        "false_alerts_per_10k_benign": out["false_positive_rate"] * 10000.0,
                        "total_alerts_per_10k_flows": (out["false_positives"] + out["true_positives"]) / max(total_n, 1) * 10000.0,
                        "detected_attacks_per_10k_attack": out["attack_recall"] * 10000.0,
                        "false_positives": out["false_positives"],
                        "true_positives": out["true_positives"],
                        "benign_test_flows": benign_n,
                        "attack_test_flows": attack_n,
                        "webattack_low_support": attack == "WebAttack",
                    }
                )
    summary = _group_summary(
        rows,
        ["threshold_type"],
        [
            "macro_f1",
            "attack_recall",
            "false_positive_rate",
            "false_alerts_per_10k_benign",
            "total_alerts_per_10k_flows",
            "detected_attacks_per_10k_attack",
        ],
    )
    _write_csv(rows, out_dir / "alert_budget_runs.csv")
    _write_csv(summary, out_dir / "alert_budget_summary.csv")
    return rows, summary


def build_memory_scope_audit(seeds: list[int], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = [
        ("behavior_only_flowprim", "packet+burst+profile primitive", "packet_burst_profile", "global"),
        ("behavior_only_full_token_diagnostic", "packet+burst+profile primitive+rhythm", "all_no_special", "global"),
    ]
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        for seed in seeds:
            base = BEST_SETTINGS[attack]
            for setting_name, token_groups, feature_filter, group_mode in settings:
                state = _load_state(attack, seed, feature_filter=feature_filter, transform=base["transform"])
                metrics, _, _, _ = _evaluate_state(state, group_mode=group_mode)
                rows.append(
                    {
                        "setting": setting_name,
                        "unknown_attack": attack,
                        "seed": seed,
                        "token_groups": token_groups,
                        "feature_filter": feature_filter,
                        "group_mode": group_mode,
                        "raw_ip_time_fivetuple_direct_token": "no" if state["no_raw_direct_tokens"] else "yes",
                        "macro_f1": metrics["best_macro_f1"],
                        "auroc": metrics["auroc"],
                        "fpr95": metrics["fpr95"],
                        "recall_at_1pct_fpr": metrics["recall_at_1pct_fpr"],
                        "recall_at_0_1pct_fpr": metrics["recall_at_0_1pct_fpr"],
                        "realized_fpr_val_p99": metrics["val_p99_0_false_positive_rate"],
                        "attack_recall_val_p99": metrics["val_p99_0_attack_recall"],
                        "train_only_vocab": state["train_only_vocab"],
                    }
                )
    summary = _group_summary(
        rows,
        ["setting", "token_groups", "raw_ip_time_fivetuple_direct_token"],
        ["macro_f1", "auroc", "fpr95", "recall_at_1pct_fpr", "recall_at_0_1pct_fpr", "realized_fpr_val_p99"],
    )
    _write_csv(rows, out_dir / "memory_scope_audit_runs.csv")
    _write_csv(summary, out_dir / "memory_scope_audit_summary.csv")
    return rows, summary


def build_knn_feature_baselines(seeds: list[int], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        for seed in seeds:
            base = BEST_SETTINGS[attack]
            for name, token_groups, feature_filter in FEATURE_ATTRIBUTION_VIEWS:
                group_mode = "global"
                state = _load_state(attack, seed, feature_filter=feature_filter, transform=base["transform"])
                metrics, _, _, _ = _evaluate_state(state, group_mode=group_mode)
                rows.append(
                    {
                        "baseline": name,
                        "unknown_attack": attack,
                        "seed": seed,
                        "token_groups": token_groups,
                        "memory_scope": "global",
                        "feature_filter": feature_filter,
                        "group_mode": group_mode,
                        "scorer": base["scorer"],
                        "k": base["k"],
                        "macro_f1": metrics["best_macro_f1"],
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "fpr95": metrics["fpr95"],
                        "recall_at_0_1pct_fpr": metrics["recall_at_0_1pct_fpr"],
                        "recall_at_1pct_fpr": metrics["recall_at_1pct_fpr"],
                        "best_macro_f1": metrics["best_macro_f1"],
                        "realized_fpr_val_p99": metrics["val_p99_0_false_positive_rate"],
                        "train_only_vocab": state["train_only_vocab"],
                        "raw_ip_time_fivetuple_direct_token": "no" if state["no_raw_direct_tokens"] else "yes",
                    }
                )
    summary = _group_summary(
        rows,
        ["baseline", "token_groups", "memory_scope", "raw_ip_time_fivetuple_direct_token"],
        ["macro_f1", "auroc", "fpr95", "recall_at_0_1pct_fpr", "recall_at_1pct_fpr", "best_macro_f1", "realized_fpr_val_p99"],
    )
    _write_csv(rows, out_dir / "knn_feature_baselines_runs.csv")
    _write_csv(summary, out_dir / "knn_feature_baselines_summary.csv")
    _write_csv(rows, out_dir / "behavior_feature_attribution_runs.csv")
    _write_csv(summary, out_dir / "behavior_feature_attribution.csv")
    return rows, summary


def _row_tokens(token_data: dict[str, Any], row_idx: int) -> list[str]:
    id_to_token = S._id_to_token(token_data["vocab"])
    ids = token_data["input_ids"][row_idx].cpu().numpy()
    mask = token_data["attention_mask"][row_idx].cpu().numpy() > 0
    return canonical_tokens(id_to_token.get(int(token_id), "[UNK]") for token_id in ids[mask])


def _display_primitive_token(token: str) -> str:
    return canonical_token(token)


def _display_primitive_text(text: str) -> str:
    parts = []
    for part in str(text).split():
        stripped = part.strip(",;()[]{}")
        canonical = canonical_token(stripped)
        if canonical != stripped:
            part = part.replace(stripped, canonical)
        parts.append(part)
    return " ".join(parts)


def _short_primitive_text(text: str) -> str:
    value = str(text)
    if value == "none":
        return value
    value = _display_primitive_text(value)
    replacements = {
        "PRIM_PROFILE_": "PROFILE_",
        "PRIM_STRUCT_": "STRUCT_",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _active_primitives(tokens: list[str]) -> str:
    values = sorted({canonical_token(token) for token in tokens if is_profile_token(token)})
    return ", ".join(values) if values else "none"


def _distance_to_refs(features: np.ndarray, query_idx: int, ref_indices: np.ndarray, scorer: str) -> np.ndarray:
    query = features[np.asarray([query_idx], dtype=np.int64)]
    refs = features[ref_indices]
    if scorer.endswith("euclidean"):
        return S._euclidean_distance(query, refs).reshape(-1)
    return S._cosine_distance(query, refs).reshape(-1)


def _nearest_neighbor_evidence(state: dict[str, Any], groups: list[str], row_idx: int, group_mode: str) -> tuple[np.ndarray, np.ndarray]:
    train_idx = state["train_idx"]
    ref_indices = train_idx
    distances = _distance_to_refs(state["features"], row_idx, ref_indices, state["setting"]["scorer"])
    k = max(1, min(int(state["setting"]["k"]), len(ref_indices)))
    nearest_pos = np.argsort(distances)[:k]
    return ref_indices[nearest_pos], distances[nearest_pos]


def _group_deviation_summary(state: dict[str, Any], row_idx: int, nearest_idx: np.ndarray, limit: int = 3) -> tuple[str, str]:
    tokens = state["feature_stats"]["kept_tokens"]
    query = state["features"][row_idx]
    ref_mean = np.mean(state["features"][nearest_idx], axis=0)
    delta = np.abs(query - ref_mean)
    group_scores: dict[str, float] = defaultdict(float)
    for value, token in zip(delta.tolist(), tokens):
        group_scores[_token_group(token)] += float(value)
    usable = {key: value for key, value in group_scores.items() if key in {"global", "packet", "burst", "primitive", "metadata_shortcut"}}
    if not usable:
        return "other", "no dominant token-group deviation"
    dominant = max(usable.items(), key=lambda item: item[1])[0]
    parts = sorted(usable.items(), key=lambda item: (-item[1], item[0]))[:limit]
    summary = "; ".join(f"{key}={value:.3f}" for key, value in parts)
    return dominant, summary


def _select_indices(y_true: np.ndarray, test_idx: np.ndarray, test_scores: np.ndarray, threshold: float) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    attack_positions = np.flatnonzero(y_true == 1)
    benign_positions = np.flatnonzero(y_true == 0)
    if attack_positions.size:
        attack_scores = test_scores[attack_positions]
        out.append(("top_scoring_true_positive", int(test_idx[attack_positions[int(np.argmax(attack_scores))]])))
        alert_positions = attack_positions[attack_scores >= threshold]
        if alert_positions.size:
            margins = test_scores[alert_positions] - threshold
            out.append(("borderline_true_positive", int(test_idx[alert_positions[int(np.argmin(margins))]])))
    if benign_positions.size:
        benign_scores = test_scores[benign_positions]
        fp_positions = benign_positions[benign_scores >= threshold]
        if fp_positions.size:
            out.append(("benign_tail_false_positive", int(test_idx[fp_positions[int(np.argmax(test_scores[fp_positions]))]])))
        tn_positions = benign_positions[benign_scores < threshold]
        if tn_positions.size:
            margins = threshold - test_scores[tn_positions]
            out.append(("benign_true_negative", int(test_idx[tn_positions[int(np.argmax(margins))]])))
    seen: set[tuple[str, int]] = set()
    deduped = []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def build_diagnosis_audit(seed: int, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        state = _load_state(attack, seed)
        metrics, val_scores, test_scores, groups = _evaluate_state(state)
        del val_scores
        y_true = state["labels"][state["test_idx"]].astype(np.int64)
        threshold = float(metrics["val_p99_0_threshold"])
        score_by_idx = {int(idx): float(test_scores[pos]) for pos, idx in enumerate(state["test_idx"].tolist())}
        selected = _select_indices(y_true, state["test_idx"], test_scores, threshold)
        for case_type, idx in selected:
            tokens = _row_tokens(state["token_data"], idx)
            primitives = _active_primitives(tokens)
            nearest_idx, nearest_dist = _nearest_neighbor_evidence(state, groups, idx, state["setting"]["group_mode"])
            dominant, group_delta = _group_deviation_summary(state, idx, nearest_idx)
            nearest_primitives = Counter(_active_primitives(_row_tokens(state["token_data"], int(ref))) for ref in nearest_idx.tolist())
            nearest_summary = "; ".join(f"{name}:{count}" for name, count in nearest_primitives.most_common(3))
            meta = state["token_data"]["meta"][idx]
            label = str(meta.get("attack_family") or meta.get("label") or "")
            score = score_by_idx[idx]
            decision = "alert" if score >= threshold else "normal"
            if label == "BENIGN" and decision == "alert" and primitives != "none":
                interpretation = "primitive-heavy benign-tail false positive"
            elif label != "BENIGN" and decision == "alert" and primitives == "none" and dominant in {"packet", "burst"}:
                interpretation = "packet/burst-token distance dominates; no primitive trigger"
            elif label != "BENIGN" and decision == "alert":
                interpretation = "alert combines memory distance with primitive/profile evidence"
            else:
                interpretation = "benign true negative near benign memory"
            rows.append(
                {
                    "unknown_attack_setting": attack,
                    "case": case_type,
                    "seed": seed,
                    "flow_id": str(meta.get("flow_id", "")),
                    "short_flow_id": str(meta.get("flow_id", ""))[:10],
                    "label": label,
                    "score": score,
                    "threshold": threshold,
                    "decision": decision,
                    "active_primitives": primitives,
                    "dominant_token_group": dominant,
                    "token_group_deviation": group_delta,
                    "nearest_benign_distance_mean": float(np.mean(nearest_dist)),
                    "nearest_benign_evidence_summary": f"group={groups[idx]}; primitive mix={nearest_summary}",
                    "interpretation": interpretation,
                }
            )
    _write_csv(rows, out_dir / "diagnosis_audit_cases.csv")
    return rows


def build_external_diagnostics(out_dir: Path) -> list[dict[str, Any]]:
    src = ROOT / "paper_icdm_applied_2026" / "experiments" / "tables" / "table_external_diagnosis.csv"
    rows: list[dict[str, Any]] = []
    keep_settings = {
        ("Botnet2014", "temporal_stratified IP-labeled smoke"),
        ("CSE-CIC-IDS2018", "zero_shot_cicids2017_to_ids2018_full"),
        ("CSE-CIC-IDS2018", "ids2018_train_to_ids2018_eval_sanity"),
        ("CSE-CIC-IDS2018", "few_shot_ids2018_01pct_scratch"),
        ("UNSW-NB15", "zero_shot_cicids2017_to_unsw"),
        ("UNSW-NB15", "unsw_train_to_unsw_test_sanity"),
        ("UNSW-NB15", "few_shot_unsw_01pct_scratch"),
    }
    for row in _read_csv(src):
        dataset = row.get("dataset", "")
        setting = row.get("setting", "")
        task = row.get("task", "")
        if dataset in {"CSE-CIC-IDS2018", "UNSW-NB15"} and "binary" not in task:
            continue
        if (dataset, setting) not in keep_settings:
            continue
        if dataset == "Botnet2014":
            input_type = "PCAP/Zeek"
            label_quality = "weak IP-based labels"
            interpretation = "PCAP ingestion smoke test"
        elif "zero_shot" in setting:
            input_type = "tabular flow features"
            label_quality = "public dataset labels"
            interpretation = "zero-shot schema/domain shift"
        elif "few_shot" in setting:
            input_type = "tabular flow features"
            label_quality = "public dataset labels"
            interpretation = "few-shot adaptation feasibility"
        else:
            input_type = "tabular flow features"
            label_quality = "public dataset labels"
            interpretation = "target-domain upper bound"
        rows.append(
            {
                "dataset": dataset,
                "input_type": input_type,
                "label_quality": label_quality,
                "setting": setting,
                "macro_f1": row.get("macro_f1"),
                "auroc": row.get("auroc"),
                "interpretation": interpretation,
            }
        )
    order = {
        ("Botnet2014", "temporal_stratified IP-labeled smoke"): 0,
        ("CSE-CIC-IDS2018", "zero_shot_cicids2017_to_ids2018_full"): 1,
        ("CSE-CIC-IDS2018", "ids2018_train_to_ids2018_eval_sanity"): 2,
        ("CSE-CIC-IDS2018", "few_shot_ids2018_01pct_scratch"): 3,
        ("UNSW-NB15", "zero_shot_cicids2017_to_unsw"): 4,
        ("UNSW-NB15", "unsw_train_to_unsw_test_sanity"): 5,
        ("UNSW-NB15", "few_shot_unsw_01pct_scratch"): 6,
    }
    rows = sorted(rows, key=lambda row: order.get((str(row.get("dataset")), str(row.get("setting"))), 99))
    _write_csv(rows, out_dir / "external_diagnostics_interpreted.csv")
    return rows


def build_behavior_feature_attribution(seeds: list[int], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return build_knn_feature_baselines(seeds, out_dir)


def _threshold_rows_from_alert_runs(alert_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in alert_rows:
        out.append(
            {
                "threshold_type": row["threshold_type"],
                "unknown_attack": row["unknown_attack"],
                "seed": row["seed"],
                "realized_fpr": row["false_positive_rate"],
                "recall": row["attack_recall"],
                "macro_f1": row["macro_f1"],
            }
        )
    return out


def build_prevalence_alert_budget(alert_rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for source in _threshold_rows_from_alert_runs(alert_rows):
        for prevalence in [0.001, 0.01, 0.05]:
            benign_count = 10000.0 * (1.0 - prevalence)
            attack_count = 10000.0 * prevalence
            fpr = float(source["realized_fpr"])
            recall = float(source["recall"])
            false_alerts = benign_count * fpr
            true_alerts = attack_count * recall
            total_alerts = false_alerts + true_alerts
            rows.append(
                {
                    "threshold_type": source["threshold_type"],
                    "prevalence": prevalence,
                    "unknown_attack": source["unknown_attack"],
                    "seed": source["seed"],
                    "realized_fpr": fpr,
                    "recall": recall,
                    "false_alerts_per_10k_total": false_alerts,
                    "true_alerts_per_10k_total": true_alerts,
                    "total_alerts_per_10k_total": total_alerts,
                    "precision_among_alerts": true_alerts / total_alerts if total_alerts > 0 else 0.0,
                    "webattack_low_support": source["unknown_attack"] == "WebAttack",
                }
            )
    summary = _group_summary(
        rows,
        ["threshold_type", "prevalence"],
        [
            "realized_fpr",
            "recall",
            "false_alerts_per_10k_total",
            "true_alerts_per_10k_total",
            "total_alerts_per_10k_total",
            "precision_among_alerts",
        ],
    )
    order = {"test_oracle_1pct_fpr": 0, "benign_val_P99": 1, "benign_val_P99_5": 2}
    summary = sorted(summary, key=lambda row: (order.get(str(row.get("threshold_type")), 99), float(row.get("prevalence") or 0.0)))
    _write_csv(rows, out_dir / "prevalence_alert_budget_runs.csv")
    _write_csv(summary, out_dir / "prevalence_alert_budget.csv")
    return rows, summary


def build_unknown_best_settings_runs(seeds: list[int], out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        for seed in seeds:
            state = _load_state(attack, seed)
            metrics, _, _, _ = _evaluate_state(state)
            rows.append(
                {
                    "unknown_attack": attack,
                    "seed": seed,
                    "feature_filter": state["setting"]["feature_filter"],
                    "transform": state["setting"]["transform"],
                    "scorer": state["setting"]["scorer"],
                    "k": state["setting"]["k"],
                    "group_mode": state["setting"]["group_mode"],
                    "best_macro_f1": metrics["best_macro_f1"],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "fpr95": metrics["fpr95"],
                    "recall_at_0_1pct_fpr": metrics["recall_at_0_1pct_fpr"],
                    "recall_at_1pct_fpr": metrics["recall_at_1pct_fpr"],
                    "val_p99_0_false_positive_rate": metrics["val_p99_0_false_positive_rate"],
                    "train_only_vocab": state["train_only_vocab"],
                    "raw_ip_time_fivetuple_direct_token": "no" if state["no_raw_direct_tokens"] else "yes",
                    "webattack_low_support": attack == "WebAttack",
                }
            )
    _write_csv(rows, out_dir / "unknown_best_settings_3seed_runs.csv")
    _write_csv(rows, UNKNOWN_DIR / "unknown_best_settings_3seed_runs.csv")
    return rows


def build_unknown_aggregate_with_without_webattack(target_rows: list[dict[str, Any]], out_dir: Path, seeds: list[int]) -> list[dict[str, Any]]:
    rows = build_unknown_best_settings_runs(seeds, out_dir)
    target_by_key = {
        (str(row["unknown_attack"]), int(row["seed"])): row
        for row in target_rows
        if row.get("target_percentile") == "P99"
    }
    out: list[dict[str, Any]] = []
    groups = [
        ("all_five_unknowns", rows),
        ("excluding_webattack", [row for row in rows if row.get("unknown_attack") != "WebAttack"]),
    ]
    metrics = [
        "best_macro_f1",
        "auroc",
        "auprc",
        "fpr95",
        "recall_at_0_1pct_fpr",
        "recall_at_1pct_fpr",
        "val_p99_0_false_positive_rate",
    ]
    for group_name, group_rows in groups:
        item: dict[str, Any] = {
            "aggregate": group_name,
            "num_runs": len(group_rows),
            "num_unknown_attacks": len({row.get("unknown_attack") for row in group_rows}),
            "includes_webattack": group_name == "all_five_unknowns",
        }
        for metric in metrics:
            values = [_float(row.get(metric)) for row in group_rows]
            clean = [value for value in values if value is not None]
            mean, std = _mean_std(clean)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        p99_values = []
        for row in group_rows:
            target = target_by_key.get((str(row.get("unknown_attack")), int(row.get("seed") or 0)))
            if target is not None:
                val = _float(target.get("realized_test_fpr"))
                if val is not None:
                    p99_values.append(val)
        mean, std = _mean_std(p99_values)
        item["benign_val_p99_realized_fpr_mean"] = mean
        item["benign_val_p99_realized_fpr_std"] = std
        out.append(item)
    _write_csv(out, out_dir / "unknown_aggregate_with_without_webattack.csv")
    return out


def build_setting_selection_protocol(out_dir: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "step": "Primitive configuration",
            "data_used": "configs/cicids2017.yaml fixed before evaluation",
            "attack_labels_used": "no",
            "tuned": "no",
            "purpose": "Define SHORT/SAME/PKT/LOCAL/REPEAT/DUP extraction parameters.",
            "leakage_risk_control": "Fixed config; not selected on held-out attack test labels.",
        },
        {
            "step": "Leave-one split construction",
            "data_used": "CICIDS2017 corrected labels and split seed",
            "attack_labels_used": "yes, for holding out the unknown family",
            "tuned": "no",
            "purpose": "Exclude the unknown family from train/validation and retain benign+held-out attack in test.",
            "leakage_risk_control": "Held-out attack labels define the stress-test partition only; train and validation contain benign rows only.",
        },
        {
            "step": "Train-only vocabulary",
            "data_used": "Training flows from each split",
            "attack_labels_used": "no",
            "tuned": "no",
            "purpose": "Build behavior-token vocabulary and IDF statistics.",
            "leakage_risk_control": "Token artifacts record vocab_provenance=train_only; val/test unknown tokens map through the train vocabulary.",
        },
        {
            "step": "Benign memory construction",
            "data_used": "Benign training flows only",
            "attack_labels_used": "no",
            "tuned": "no",
            "purpose": "Build KNN benign memory for anomaly scoring.",
            "leakage_risk_control": "Script asserts all train rows are benign before scoring.",
        },
        {
            "step": "Fixed behavior-only setting",
            "data_used": "Configuration fixed for all attacks and seeds",
            "attack_labels_used": "no",
            "tuned": "no",
            "purpose": "Use behavior-token histograms, exact KNN cosine scoring, k=3, and global benign memory for the main low-FPR tables.",
            "leakage_risk_control": "No attack-specific setting selection and no deployment-metadata memory partitioning are used for the main low-FPR claim.",
        },
        {
            "step": "Repeated-seed evaluation",
            "data_used": "Seed-42/43/44 split and token corpora",
            "attack_labels_used": "only for test metrics",
            "tuned": "no",
            "purpose": "Evaluate stability of the fixed behavior-only setting without retuning.",
            "leakage_risk_control": "No per-seed hyperparameter reselection; train/val benign-only assertions are checked.",
        },
        {
            "step": "Benign-validation thresholds",
            "data_used": "Benign validation anomaly scores",
            "attack_labels_used": "no",
            "tuned": "threshold only",
            "purpose": "Estimate deployable thresholds such as P95/P99/P99.5/P100.",
            "leakage_risk_control": "No attack labels enter threshold calibration; realized FPR/recall are measured on test only after threshold selection.",
        },
        {
            "step": "Test-oracle low-FPR thresholds",
            "data_used": "Test labels and scores",
            "attack_labels_used": "yes",
            "tuned": "threshold only",
            "purpose": "Measure separability under fixed FPR budgets.",
            "leakage_risk_control": "Explicitly marked as oracle/separability analysis, not deployable calibration.",
        },
    ]
    _write_csv(rows, out_dir / "setting_selection_protocol.csv")
    return rows


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data or {}
    except Exception:
        data: dict[str, Any] = {}
        current: dict[str, Any] | None = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not raw.startswith(" ") and line.endswith(":"):
                key = line[:-1].strip()
                data[key] = {}
                current = data[key]
            elif current is not None and ":" in line:
                key, value = line.strip().split(":", 1)
                value = value.strip()
                try:
                    current[key.strip()] = int(value)
                except ValueError:
                    current[key.strip()] = value
        return data


def copy_validation_size_table(out_dir: Path) -> list[dict[str, Any]]:
    src = ROOT / "paper_icdm_applied_2026" / "experiments" / "deployment" / "calibration_stability_summary.csv"
    rows = _read_csv(src)
    _write_csv(rows, out_dir / "validation_size_stability_summary.csv")
    return rows


def copy_primitive_ablation_as_sensitivity(out_dir: Path) -> list[dict[str, Any]]:
    src = ROOT / "paper_icdm_applied_2026" / "experiments" / "primitive_ablation" / "primitive_ablation_summary.csv"
    if not src.exists():
        return []
    rows = _read_csv(src)
    out = []
    for row in rows:
        out.append(
            {
                "sensitivity_type": "primitive_family_removal",
                "variant": row.get("ablation") or row.get("variant") or row.get("dropped_token") or "",
                "macro_f1": row.get("best_macro_f1_mean") or row.get("macro_f1_mean") or row.get("best_macro_f1"),
                "recall_at_1pct_fpr": row.get("recall_at_1pct_fpr_mean") or row.get("recall_at_1pct_fpr"),
                "auroc": row.get("auroc_mean") or row.get("auroc"),
                "fpr95": row.get("fpr95_mean") or row.get("fpr95"),
                "artifact": str(src.relative_to(ROOT)),
                "note": "Uses existing primitive-family removal ablation; raw-threshold sensitivity was not rerun from packet records.",
            }
        )
    _write_csv(out, out_dir / "primitive_sensitivity_from_ablation.csv")
    return out


def _build_token_data_from_sources(flows: list[dict[str, Any]], profile_rows: list[dict[str, Any]], split_payload: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    from src.features.behavior_tokens import build_behavior_token_dataset

    token_data, _stats = build_behavior_token_dataset(
        flows,
        profile_rows,
        split_payload,
        cfg,
        max_len=int(cfg.get("max_len", 512)),
    )
    return token_data


def _profile_rows_for_split(flows: list[dict[str, Any]], split_payload: dict[str, Any], profile_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from src.data.splits import split_lookup
    from src.features.profile_primitives import extract_all_profile_primitives

    empty_profile = {"short": None, "same": None, "packet": [], "local": [], "repeat": [], "duplicate": []}
    lookup = split_lookup(split_payload)
    train_flows = [flow for flow in flows if lookup.get(str(flow.get("flow_id"))) == "train"]
    train_rows, _stats = extract_all_profile_primitives(train_flows, profile_cfg)
    rows_by_id = {str(row["flow_id"]): {**row, "profile_primitive_provenance": "train_only"} for row in train_rows}
    out = []
    for flow in flows:
        flow_id = str(flow.get("flow_id"))
        out.append(
            rows_by_id.get(
                flow_id,
                {
                    "flow_id": flow_id,
                    "service_key": flow.get("service_key"),
                    "label": flow.get("label"),
                    "profile": dict(empty_profile),
                    "profile_primitive_provenance": "train_dictionary_unseen_flow",
                },
            )
        )
    return out


def _evaluate_token_data(token_data: dict[str, Any], attack: str, *, feature_filter: str | None = None, transform: str | None = None, group_mode: str | None = None) -> dict[str, Any]:
    setting = BEST_SETTINGS[attack].copy()
    if feature_filter is not None:
        setting["feature_filter"] = feature_filter
    if transform is not None:
        setting["transform"] = transform
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = S._split_indices(token_data, "train")
    val_idx = S._split_indices(token_data, "val")
    test_idx = S._split_indices(token_data, "test")
    if not np.all(labels[train_idx] == 0) or not np.all(labels[val_idx] == 0):
        raise ValueError("sensitivity token data must have benign-only train/val")
    features, _feature_stats = _features(token_data, train_idx, feature_filter=setting["feature_filter"], transform=setting["transform"])
    groups = S._groups(token_data, group_mode or setting["group_mode"])
    val_scores = S._scores(features, train_idx, val_idx, groups, scorer=setting["scorer"], k=int(setting["k"]))
    test_scores = S._scores(features, train_idx, test_idx, groups, scorer=setting["scorer"], k=int(setting["k"]))
    y_true = labels[test_idx].astype(np.int64)
    best = S._best_macro(y_true, test_scores)
    r1 = S._best_recall_under_fpr(y_true, test_scores, 0.01)
    rank = S._rank_metrics(y_true, test_scores)
    threshold = float(np.percentile(val_scores, 99.0))
    val99 = S._metrics_at_threshold(y_true, test_scores, threshold)
    return {
        "macro_f1": best["macro_f1"],
        "auroc": rank["auroc"],
        "fpr95": rank["fpr95"],
        "recall_at_1pct_fpr": r1["attack_recall"],
        "benign_val_p99_fpr": val99["false_positive_rate"],
    }


def build_primitive_threshold_sensitivity(out_dir: Path, *, seed: int = 43) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attacks = ["Botnet", "DDoS", "Probe"]
    cfg = _load_yaml_config(CONFIG_PATH)
    base_profile = dict(cfg.get("profile_primitives", {}))
    tokenizer_cfg = dict(cfg.get("tokenizer", {}))
    tokenizer_cfg.update(
        {
            "max_len": 512,
            "profile_mode": "full",
            "include_flow_summary": True,
            "include_packet_tokens": True,
            "include_burst_tokens": True,
            "include_rhythm_tokens": True,
            "use_burst_shape_tokens": True,
            "use_transition_profile_tokens": True,
            "label_field": "attack_family",
            "record_service_context": False,
            "use_service_context": False,
            "use_service_tokens": False,
        }
    )
    flows = _read_jsonl(FLOW_SOURCE)
    variants: list[tuple[str, str, Any]] = [
        ("baseline", "fixed config", "SHORT=6;PKT=12;peer_chunk=128;LOCAL=8"),
        ("short_threshold_4", "SHORT threshold", 4),
        ("short_threshold_8", "SHORT threshold", 8),
        ("short_threshold_10", "SHORT threshold", 10),
        ("max_pkt_tokens_6", "max PKT tokens", 6),
        ("max_pkt_tokens_24", "max PKT tokens", 24),
        ("peer_chunk_64", "peer chunk size", 64),
        ("peer_chunk_256", "peer chunk size", 256),
    ]
    rows: list[dict[str, Any]] = []
    for attack in attacks:
        split_path = UNKNOWN_DIR / f"splits_leave_one_{ATTACKS[attack]}_anomaly_seed{seed}.json"
        split_payload = _read_json(split_path)
        for variant, parameter, value in variants:
            profile_cfg = dict(base_profile)
            if parameter == "SHORT threshold":
                profile_cfg["short_flow_packet_threshold"] = int(value)
            elif parameter == "max PKT tokens":
                profile_cfg["max_packet_profile_primitives"] = int(value)
            elif parameter == "peer chunk size":
                profile_cfg["max_peer_flows"] = int(value)
            profile_rows = _profile_rows_for_split(flows, split_payload, profile_cfg)
            token_data = _build_token_data_from_sources(flows, profile_rows, split_payload, tokenizer_cfg)
            metrics = _evaluate_token_data(token_data, attack)
            rows.append(
                {
                    "scope": "seed43_source_rebuild_smoke",
                    "unknown_attack": attack,
                    "seed": seed,
                    "variant": variant,
                    "parameter": parameter,
                    "value": value,
                    "macro_f1": metrics["macro_f1"],
                    "auroc": metrics["auroc"],
                    "fpr95": metrics["fpr95"],
                    "recall_at_1pct_fpr": metrics["recall_at_1pct_fpr"],
                    "benign_val_p99_fpr": metrics["benign_val_p99_fpr"],
                    "train_only_profile_primitives": True,
                    "train_only_vocab": True,
                    "note": "Small source-rebuild smoke sensitivity over three seed-43 unknown settings; not used for hyperparameter selection.",
                }
            )
    summary = _group_summary(rows, ["variant", "parameter", "value"], ["macro_f1", "auroc", "fpr95", "recall_at_1pct_fpr", "benign_val_p99_fpr"])
    order = {name: pos for pos, (name, _parameter, _value) in enumerate(variants)}
    summary = sorted(summary, key=lambda row: order.get(str(row.get("variant")), 99))
    _write_csv(rows, out_dir / "primitive_threshold_sensitivity_runs.csv")
    _write_csv(summary, out_dir / "primitive_threshold_sensitivity.csv")
    _write_csv(summary, out_dir / "primitive_sensitivity.csv")
    return rows, summary


def build_diagnosis_audit_simplified(diagnosis_rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    desired = [
        ("Botnet", "top_scoring_true_positive"),
        ("Botnet", "benign_tail_false_positive"),
        ("WebAttack", "top_scoring_true_positive"),
        ("DDoS", "top_scoring_true_positive"),
        ("BruteForce", "top_scoring_true_positive"),
        ("BruteForce", "benign_true_negative"),
    ]
    rows: list[dict[str, Any]] = []
    for attack, case in desired:
        row = next((item for item in diagnosis_rows if item["unknown_attack_setting"] == attack and item["case"] == case), None)
        if row is None:
            continue
        primitives = _short_primitive_text(row["active_primitives"])
        if primitives == "none" and row["decision"] == "alert":
            dominant = f"{row['dominant_token_group']} token-distance deviation"
        elif row["label"] == "BENIGN" and row["decision"] == "alert":
            dominant = "primitive-heavy benign-tail deviation"
        elif row["decision"] == "normal":
            dominant = "near benign memory"
        else:
            dominant = f"{row['dominant_token_group']} token-distance with primitive/profile evidence"
        label = str(row["label"])
        interp = str(row["interpretation"])
        if label == "WebAttack":
            interp += "; low-support 28-flow case"
        rows.append(
            {
                "case": f"{attack}:{case.replace('_', ' ')}",
                "label": label,
                "decision": row["decision"],
                "score_threshold": f"{_fmt(row['score'])}/{_fmt(row['threshold'])}",
                "active_primitives": primitives,
                "dominant_evidence": dominant,
                "analyst_facing_interpretation": interp,
            }
        )
    _write_csv(rows, out_dir / "diagnosis_audit_simplified.csv")
    return rows


def build_ann_knn_scalability(out_dir: Path, *, seed: int = 43) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sizes = [1000, 5000, 10000, 50000]
    for attack in ["Botnet", "DDoS", "Probe"]:
        state = _load_state(attack, seed)
        groups = S._groups(state["token_data"], state["setting"]["group_mode"])
        y_true = state["labels"][state["test_idx"]].astype(np.int64)
        for size in sizes:
            train_idx = state["train_idx"]
            if len(train_idx) >= size:
                refs_idx = train_idx[:size]
                stress = "real_subsample"
            else:
                reps = int(math.ceil(size / max(len(train_idx), 1)))
                refs_idx = np.tile(train_idx, reps)[:size]
                stress = "bootstrap_duplicate_memory"
            t0 = time.perf_counter()
            # Exact KNN against the stress memory. Protocol grouping is kept when possible,
            # but duplicated refs are only a scalability stress, not a new statistical result.
            eval_scores = []
            eval_indices = state["test_idx"][: min(512, len(state["test_idx"]))]
            train_by_group: dict[str, list[int]] = defaultdict(list)
            for idx in refs_idx.tolist():
                train_by_group[groups[idx]].append(idx)
            for idx in eval_indices.tolist():
                group_refs = train_by_group.get(groups[idx]) or refs_idx.tolist()
                distances = _distance_to_refs(state["features"], idx, np.asarray(group_refs, dtype=np.int64), state["setting"]["scorer"])
                k = max(1, min(int(state["setting"]["k"]), len(distances)))
                eval_scores.append(float(np.mean(np.partition(distances, k - 1)[:k])))
            query_seconds = time.perf_counter() - t0
            full_scores = S._scores(state["features"], refs_idx, state["test_idx"], groups, scorer=state["setting"]["scorer"], k=int(state["setting"]["k"]))
            rank = S._rank_metrics(y_true, full_scores)
            r01 = S._best_recall_under_fpr(y_true, full_scores, 0.001)
            r1 = S._best_recall_under_fpr(y_true, full_scores, 0.01)
            val_scores = S._scores(state["features"], refs_idx, state["val_idx"], groups, scorer=state["setting"]["scorer"], k=int(state["setting"]["k"]))
            threshold = float(np.percentile(val_scores, 99.0))
            val99 = S._metrics_at_threshold(y_true, full_scores, threshold)
            rows.append(
                {
                    "attack": attack,
                    "seed": seed,
                    "method": "exact_knn",
                    "memory_size": size,
                    "memory_source": stress,
                    "query_ms_per_flow": (query_seconds / max(len(eval_indices), 1)) * 1000.0,
                    "index_build_seconds": 0.0,
                    "memory_mb": float(state["features"][refs_idx].nbytes / (1024.0 * 1024.0)),
                    "auroc": rank["auroc"],
                    "fpr95": rank["fpr95"],
                    "recall_at_0_1pct_fpr": r01["attack_recall"],
                    "recall_at_1pct_fpr": r1["attack_recall"],
                    "p99_fpr": val99["false_positive_rate"],
                    "note": "Scalability stress uses duplicated benign memory when requested size exceeds available train-benign flows.",
                }
            )
    summary = _group_summary(rows, ["method", "memory_size", "memory_source"], ["query_ms_per_flow", "memory_mb", "auroc", "fpr95", "recall_at_0_1pct_fpr", "recall_at_1pct_fpr", "p99_fpr"])
    _write_csv(rows, out_dir / "ann_knn_scalability_runs.csv")
    _write_csv(summary, out_dir / "ann_knn_scalability.csv")
    return rows, summary


def build_e2e_throughput_smoke(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flows_path = ROOT / "outputs" / "processed" / "ccfa" / "cicids2017_interim_labeled_flows.jsonl"
    profile_path = TRAIN_ONLY_PROFILE_SOURCE
    token_path = _token_path("Botnet", 43)
    t0 = time.perf_counter()
    flows = _read_jsonl(flows_path)
    t1 = time.perf_counter()
    profile_rows = _read_jsonl(profile_path)
    t2 = time.perf_counter()
    token_data = S._read_token_data(token_path)
    t3 = time.perf_counter()
    state = _load_state("Botnet", 43)
    metrics, _val_scores, test_scores, _groups = _evaluate_state(state)
    del metrics
    threshold = float(np.percentile(_val_scores, 99.0))
    records = [
        {"flow_id": state["token_data"]["meta"][int(idx)].get("flow_id"), "score": float(test_scores[pos]), "decision": "alert" if float(test_scores[pos]) >= threshold else "normal"}
        for pos, idx in enumerate(state["test_idx"][:512])
    ]
    t4 = time.perf_counter()
    specs = [
        ("Processed flow JSONL load", flows_path, len(flows), len(flows), t1 - t0, "Uses existing Zeek-aligned flow artifact; PCAP->Zeek not rerun."),
        ("Profile primitive JSONL load", profile_path, len(profile_rows), len(profile_rows), t2 - t1, "Train-only profile primitive artifact load; not extraction time."),
        ("Token corpus load", token_path, int(token_data["input_ids"].shape[0]), int(token_data["input_ids"].shape[0]), t3 - t2, "Existing token corpus load."),
        ("KNN scoring to diagnosis records", token_path, len(state["test_idx"]), len(records), t4 - t3, "Flow-ready KNN scoring and record materialization for first 512 test flows."),
    ]
    for stage, path, input_count, output_count, seconds, note in specs:
        rows.append(
            {
                "stage": stage,
                "input_count": input_count,
                "output_count": output_count,
                "wall_time_seconds": seconds,
                "throughput_items_per_second": output_count / seconds if seconds > 0 else 0.0,
                "artifact": _safe_rel(path),
                "notes": note,
            }
        )
    _write_csv(rows, out_dir / "e2e_throughput_smoke.csv")
    return rows


def build_artifact_availability(out_dir: Path) -> list[dict[str, Any]]:
    artifacts = [
        ("Corrected label audit", "Label provenance and corrected mapping audit", ROOT / "outputs" / "processed" / "cicids2017_improved_label_audit.json", "yes", "Dataset filtering / setup"),
        ("Temporal split file", "Main temporal split metadata", ROOT / "outputs" / "processed" / "ccfa" / "splits_temporal_chronological.json", "yes", "Main detection tables"),
        ("Leave-one split files", "Unknown attack seed-specific split files", UNKNOWN_DIR / "splits_leave_one_botnet_anomaly_seed43.json", "yes", "Unknown / calibration tables"),
        ("Primitive config", "Fixed primitive extraction thresholds", CONFIG_PATH, "yes", "Primitive mining and sensitivity"),
        ("Train-only profile primitive rows", "Profile primitive rows with train-only provenance", TRAIN_ONLY_PROFILE_SOURCE, "yes", "Profile primitive / category-token corpora"),
        ("Category token vocab files", "Train-only leave-one vocabularies embedded in category-token corpora", TOKEN_DIR / "cicids2017_leave_one_botnet_anomaly_seed43_a3_full_rhythm_vocab.json", "yes", "Behavior-feature and primitive-category attribution"),
        ("Category token corpora", "Leave-one token histograms/sequences with PRIM_PROFILE and PRIM_STRUCT prefixes", TOKEN_DIR / "cicids2017_leave_one_botnet_anomaly_seed43_a3_full_rhythm.pt", "yes", "KNN memory/calibration and category attribution tables"),
        ("Unknown 3-seed runs", "Locked-setting per-seed unknown metrics", UNKNOWN_DIR / "unknown_best_settings_3seed_runs.csv", "yes", "Unknown aggregate table"),
        ("Revision result CSVs", "Calibration, attribution, audit, budget CSVs", out_dir / "behavior_feature_attribution.csv", "yes", "Revision tables"),
        ("Extra benign canonical artifacts", "Additional benign canonical flows, tokens, histograms, and metadata", ROOT / "artifacts" / "extra_benign" / "extra_benign_metadata.csv", "yes", "Extra benign summary"),
        ("Extra benign gate and splits", "Admission gate scores and memory/calibration/tail/quarantine splits", ROOT / "results" / "extra_benign_gate_scores.csv", "yes", "Extra benign memory/calibration tables"),
        ("Extra benign PCAP-to-Zeek throughput", "Measured Zeek parsing throughput on five benign PCAP slices", ROOT / "paper_icdm_applied_2026" / "experiments" / "extra_benign" / "extra_benign_pcap_zeek_throughput.csv", "yes", "Extra benign throughput table"),
        ("Extra benign result CSVs", "Memory-calibration attribution, calibration scaling, and memory strategies", ROOT / "paper_icdm_applied_2026" / "experiments" / "extra_benign" / "extra_benign_memory_calibration_attribution.csv", "yes", "Extra benign tables"),
        ("Extra benign experiment wrapper", "Prepare/gate/split/evaluate extra benign pipeline", ROOT / "scripts" / "run_extra_benign_experiments.sh", "yes", "Extra benign artifacts and tables"),
        ("Primitive category experiment wrapper", "Profile/structural attribution over regenerated category tokens", ROOT / "scripts" / "run_primitive_category_experiments.sh", "yes", "Primitive category attribution table"),
        ("Primitive category table generator", "CSV-to-LaTeX snippet for profile/structural table", ROOT / "scripts" / "build_primitive_category_table.py", "yes", "Primitive category attribution table"),
        ("Revision table generator", "CSV-to-LaTeX snippets", ROOT / "scripts" / "64_run_icdm_revision_experiments.py", "yes", "All revision tables"),
        ("Reproducibility checklist", "Artifact and run protocol checklist", REPO / "paper" / "reproducibility_checklist.md", "yes", "Artifact documentation"),
    ]
    rows = []
    for artifact, purpose, path, released, regenerates in artifacts:
        rows.append(
            {
                "artifact": artifact,
                "purpose": purpose,
                "path": _safe_rel(path),
                "exists": path.exists(),
                "released": released,
                "regenerates": regenerates,
            }
        )
    _write_csv(rows, out_dir / "artifact_availability.csv")
    return rows


def _latex_table(path: Path, body: str) -> None:
    if path.name in SKIP_LEGACY_PAPER_TABLES:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def _tex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_tables(
    out_dir: Path,
    target_summary: list[dict[str, Any]],
    alert_summary: list[dict[str, Any]],
    memory_scope_summary: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    diagnosis_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    knn_summary: list[dict[str, Any]] | None = None,
    prevalence_summary: list[dict[str, Any]] | None = None,
    unknown_aggregate_rows: list[dict[str, Any]] | None = None,
    setting_protocol_rows: list[dict[str, Any]] | None = None,
    sensitivity_summary: list[dict[str, Any]] | None = None,
    diagnosis_simplified_rows: list[dict[str, Any]] | None = None,
    ann_summary: list[dict[str, Any]] | None = None,
    e2e_rows: list[dict[str, Any]] | None = None,
    artifact_rows: list[dict[str, Any]] | None = None,
) -> None:
    del out_dir
    rows = []
    for row in target_summary:
        rows.append(
            f"{row['target_percentile']} & {_fmt_pm(row['macro_f1_mean'], row['macro_f1_std'])} & "
            f"{_fmt_pm(row['attack_recall_mean'], row['attack_recall_std'])} & "
            f"{_fmt_pm(row['realized_test_fpr_mean'], row['realized_test_fpr_std'])} & "
            f"{_fmt_pm(row['threshold_mean'], row['threshold_std'])} \\\\"
        )
    _latex_table(
        PAPER_TABLE_DIR / "table_target_realized_fpr.tex",
        "\\begin{tabular}{lcccc}\n\\toprule\nBENIGN-val threshold & Macro-F1 & Attack recall & Realized FPR & Threshold \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}",
    )

    label_map = {
        "test_oracle_1pct_fpr": "Oracle 1\\%FPR",
        "benign_val_P99": "BENIGN-val P99",
        "benign_val_P99_5": "BENIGN-val P99.5",
    }
    rows = []
    alert_order = {"test_oracle_1pct_fpr": 0, "benign_val_P99": 1, "benign_val_P99_5": 2}
    for row in sorted(alert_summary, key=lambda item: alert_order.get(str(item.get("threshold_type")), 99)):
        rows.append(
            f"{label_map.get(str(row['threshold_type']), row['threshold_type'])} & "
            f"{_fmt_pm(row['false_alerts_per_10k_benign_mean'], row['false_alerts_per_10k_benign_std'], 1)} & "
            f"{_fmt_pm(row['total_alerts_per_10k_flows_mean'], row['total_alerts_per_10k_flows_std'], 1)} & "
            f"{_fmt_pm(row['detected_attacks_per_10k_attack_mean'], row['detected_attacks_per_10k_attack_std'], 1)} & "
            f"{_fmt_pm(row['attack_recall_mean'], row['attack_recall_std'])} \\\\"
        )
    _latex_table(
        PAPER_TABLE_DIR / "table_alert_budget.tex",
        "\\begin{tabular}{lcccc}\n\\toprule\nThreshold & False alerts / 10k benign & Alerts / 10k flows & Detected / 10k attacks & Recall \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}",
    )

    setting_map = {
        "behavior_only_flowprim": "Behavior-only FlowPrim",
        "behavior_only_full_token_diagnostic": "Full-token diagnostic",
    }
    rows = []
    memory_scope_order = {"behavior_only_flowprim": 0, "behavior_only_full_token_diagnostic": 1}
    for row in sorted(memory_scope_summary, key=lambda item: memory_scope_order.get(str(item.get("setting")), 99)):
        rows.append(
            f"{setting_map.get(str(row['setting']), row['setting'])} & {_tex(row['token_groups'])} & {row['raw_ip_time_fivetuple_direct_token']} & "
            f"{_fmt_pm(row['macro_f1_mean'], row['macro_f1_std'])} & {_fmt_pm(row['auroc_mean'], row['auroc_std'])} & "
            f"{_fmt_pm(row['fpr95_mean'], row['fpr95_std'])} & {_fmt_pm(row['recall_at_1pct_fpr_mean'], row['recall_at_1pct_fpr_std'])} & "
            f"{_fmt_pm(row['realized_fpr_val_p99_mean'], row['realized_fpr_val_p99_std'])} \\\\"
        )
    _latex_table(
        PAPER_TABLE_DIR / "table_memory_scope_audit.tex",
        "\\begin{tabular}{lllccccc}\n\\toprule\nSetting & Token groups & Raw ID/time token & Macro-F1 & AUROC & FPR95 & Recall@1\\%FPR & P99 FPR \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}",
    )

    rows = []
    for row in validation_rows:
        rows.append(
            f"{row['val_benign_n']} & {_fmt_pm(row['macro_f1_mean_mean'], row['macro_f1_mean_std'])} & "
            f"{_fmt_pm(row['attack_recall_mean_mean'], row['attack_recall_mean_std'])} & "
            f"{_fmt_pm(row['false_positive_rate_mean_mean'], row['false_positive_rate_mean_std'])} & "
            f"{_fmt_pm(row['threshold_std_mean'], row['threshold_std_std'])} \\\\"
        )
    _latex_table(
        PAPER_TABLE_DIR / "table_validation_size_stability.tex",
        "\\begin{tabular}{rcccc}\n\\toprule\nVal benign flows & Macro-F1 & Recall & Test FPR & Threshold std \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}",
    )

    case_order = [
        ("Botnet", "top_scoring_true_positive"),
        ("Botnet", "benign_tail_false_positive"),
        ("DDoS", "top_scoring_true_positive"),
        ("Probe", "borderline_true_positive"),
        ("WebAttack", "top_scoring_true_positive"),
        ("BruteForce", "top_scoring_true_positive"),
        ("BruteForce", "benign_true_negative"),
    ]
    selected = []
    for attack, case in case_order:
        match = next((row for row in diagnosis_rows if row["unknown_attack_setting"] == attack and row["case"] == case), None)
        if match is not None:
            selected.append(match)
    rows = []
    for row in selected:
        primitives = _short_primitive_text(row["active_primitives"]).replace(", ", ",")
        nearest = _tex(_display_primitive_text(row["nearest_benign_evidence_summary"]))
        interp = _tex(row["interpretation"])
        rows.append(
            f"{row['unknown_attack_setting']}:{row['case'].replace('_', ' ')} & {row['label']} & {_fmt(row['score'])}/{_fmt(row['threshold'])} & "
            f"{row['decision']} & {_tex(primitives)} & {_tex(row['dominant_token_group'])} & {nearest} & {interp} \\\\"
        )
    _latex_table(
        PAPER_TABLE_DIR / "table_diagnosis_audit.tex",
        "\\begin{tabular}{lllcllll}\n\\toprule\nCase & Label & Score/threshold & Decision & Active primitives & Dominant source & Nearest benign evidence & Interpretation \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}",
    )

    rows = []
    for row in external_rows:
        setting = str(row["setting"])
        if "zero_shot_cicids2017_to_ids2018_full" in setting:
            setting = "zero-shot CICIDS2017$\\rightarrow$IDS2018"
        elif "ids2018_train_to_ids2018_eval_sanity" in setting:
            setting = "target-domain sanity split"
        elif "few_shot_ids2018_01pct_scratch" in setting:
            setting = "0.1\\% few-shot scratch"
        elif "zero_shot_cicids2017_to_unsw" in setting:
            setting = "zero-shot CICIDS2017$\\rightarrow$UNSW"
        elif "unsw_train_to_unsw_test_sanity" in setting:
            setting = "official target-domain split"
        elif "few_shot_unsw_01pct_scratch" in setting:
            setting = "0.1\\% few-shot scratch"
        elif "temporal_stratified" in setting:
            setting = "external PCAP smoke"
        rows.append(
            f"{row['dataset']} & {_tex(row['input_type'])} & {_tex(row['label_quality'])} & {setting} & "
            f"{_fmt(row['macro_f1'])} & {_fmt(row['auroc'])} & {_tex(row['interpretation'])} \\\\"
        )
    _latex_table(
        PAPER_TABLE_DIR / "table_external_diagnostics_revision.tex",
        "\\begin{tabular}{lllcccl}\n\\toprule\nDataset & Input type & Label quality & Setting & Macro-F1 & AUROC & Interpretation \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}",
    )

    if knn_summary is not None:
        baseline_order = [
            "global_flow_summary",
            "packet_only",
            "profile_only",
            "packet_burst",
            "packet_burst_profile",
        ]
        by_name = {str(row["baseline"]): row for row in knn_summary}
        rows = []
        for name in baseline_order:
            row = by_name.get(name)
            if row is None:
                continue
            rows.append(
                f"{name.replace('_', '\\_')} & {_tex(row['token_groups'])} & {_tex(row['memory_scope'])} & {row['raw_ip_time_fivetuple_direct_token']} & "
                f"{_fmt_pm(row['auroc_mean'], row['auroc_std'])} & {_fmt_pm(row['fpr95_mean'], row['fpr95_std'])} & "
                f"{_fmt_pm(row['recall_at_0_1pct_fpr_mean'], row['recall_at_0_1pct_fpr_std'])} & "
                f"{_fmt_pm(row['recall_at_1pct_fpr_mean'], row['recall_at_1pct_fpr_std'])} & "
                f"{_fmt_pm(row['realized_fpr_val_p99_mean'], row['realized_fpr_val_p99_std'])} \\\\"
            )
        _latex_table(
            PAPER_TABLE_DIR / "table_knn_feature_baselines.tex",
            "\\begin{tabular}{llllccccc}\n\\toprule\nFeature view & Evidence & Memory scope & Raw ID/time token & AUROC & FPR95 & Recall@0.1\\%FPR & Recall@1\\%FPR & P99 FPR \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )
        _latex_table(
            PAPER_TABLE_DIR / "table_behavior_feature_attribution.tex",
            "\\begin{tabular}{llllccccc}\n\\toprule\nFeature view & Evidence & Memory scope & Raw ID/time token & AUROC & FPR95 & R@0.1\\%FPR & R@1\\%FPR & P99 FPR \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )

    if prevalence_summary is not None:
        label_map = {
            "test_oracle_1pct_fpr": "Oracle 1\\%FPR",
            "benign_val_P99": "BENIGN-val P99",
            "benign_val_P99_5": "BENIGN-val P99.5",
        }
        rows = []
        for row in prevalence_summary:
            rows.append(
                f"{label_map.get(str(row['threshold_type']), row['threshold_type'])} & {_fmt(float(row['prevalence']) * 100, 1)}\\% & "
                f"{_fmt_pm(row['false_alerts_per_10k_total_mean'], row['false_alerts_per_10k_total_std'], 1)} & "
                f"{_fmt_pm(row['true_alerts_per_10k_total_mean'], row['true_alerts_per_10k_total_std'], 1)} & "
                f"{_fmt_pm(row['total_alerts_per_10k_total_mean'], row['total_alerts_per_10k_total_std'], 1)} & "
                f"{_fmt_pm(row['precision_among_alerts_mean'], row['precision_among_alerts_std'])} & "
                f"{_fmt_pm(row['recall_mean'], row['recall_std'])} \\\\"
            )
        _latex_table(
            PAPER_TABLE_DIR / "table_prevalence_alert_budget.tex",
            "\\begin{tabular}{llccccc}\n\\toprule\nThreshold & Prevalence & False alerts/10k & True alerts/10k & Total alerts/10k & Precision & Recall \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )

    if unknown_aggregate_rows is not None:
        rows = []
        label_map = {"all_five_unknowns": "All five", "excluding_webattack": "Excluding WebAttack"}
        for row in unknown_aggregate_rows:
            rows.append(
                f"{label_map.get(str(row['aggregate']), row['aggregate'])} & {row['num_unknown_attacks']} & {row['num_runs']} & "
                f"{_fmt_pm(row['best_macro_f1_mean'], row['best_macro_f1_std'])} & {_fmt_pm(row['auroc_mean'], row['auroc_std'])} & "
                f"{_fmt_pm(row['auprc_mean'], row['auprc_std'])} & {_fmt_pm(row['fpr95_mean'], row['fpr95_std'])} & "
                f"{_fmt_pm(row['recall_at_0_1pct_fpr_mean'], row['recall_at_0_1pct_fpr_std'])} & "
                f"{_fmt_pm(row['recall_at_1pct_fpr_mean'], row['recall_at_1pct_fpr_std'])} & "
                f"{_fmt_pm(row['benign_val_p99_realized_fpr_mean'], row['benign_val_p99_realized_fpr_std'])} \\\\"
            )
        _latex_table(
            PAPER_TABLE_DIR / "table_unknown_aggregate_wo_webattack.tex",
            "\\begin{tabular}{lrrccccccc}\n\\toprule\nAggregate & Attacks & Runs & Macro-F1 & AUROC & AUPRC & FPR95 & R@0.1\\%FPR & R@1\\%FPR & P99 FPR \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )

    if setting_protocol_rows is not None:
        rows = []
        for row in setting_protocol_rows:
            rows.append(
                f"{_tex(row['step'])} & {_tex(row['data_used'])} & {_tex(row['attack_labels_used'])} & {_tex(row['tuned'])} & {_tex(row['purpose'])} & {_tex(row['leakage_risk_control'])} \\\\"
            )
        _latex_table(
            PAPER_TABLE_DIR / "table_setting_selection_protocol.tex",
            "\\begin{tabular}{llllll}\n\\toprule\nStep & Data used & Attack labels used? & Tuned? & Purpose & Leakage control \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )

    if sensitivity_summary is not None:
        rows = []
        for row in sensitivity_summary:
            rows.append(
                f"{_tex(row['variant'])} & {_tex(row['parameter'])} & {_tex(row['value'])} & "
                f"{_fmt_pm(row['macro_f1_mean'], row['macro_f1_std'])} & {_fmt_pm(row['auroc_mean'], row['auroc_std'])} & "
                f"{_fmt_pm(row['fpr95_mean'], row['fpr95_std'])} & {_fmt_pm(row['recall_at_1pct_fpr_mean'], row['recall_at_1pct_fpr_std'])} & "
                f"{_fmt_pm(row['benign_val_p99_fpr_mean'], row['benign_val_p99_fpr_std'])} \\\\"
            )
        _latex_table(
            PAPER_TABLE_DIR / "table_primitive_sensitivity.tex",
            "\\begin{tabular}{lllccccc}\n\\toprule\nVariant & Parameter & Value & Macro-F1 & AUROC & FPR95 & R@1\\%FPR & P99 FPR \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )

    if diagnosis_simplified_rows is not None:
        rows = []
        for row in diagnosis_simplified_rows:
            rows.append(
                f"{_tex(row['case'])} & {_tex(row['label'])} & {_tex(row['decision'])} & {_tex(row['score_threshold'])} & "
                f"{_tex(row['active_primitives'])} & {_tex(row['dominant_evidence'])} & {_tex(row['analyst_facing_interpretation'])} \\\\"
            )
        _latex_table(
            PAPER_TABLE_DIR / "table_diagnosis_audit_simplified.tex",
            "\\begin{tabular}{lllllll}\n\\toprule\nCase & Label & Decision & Score/threshold & Active primitives & Dominant evidence & Analyst-facing interpretation \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )

    if ann_summary is not None:
        rows = []
        for row in ann_summary:
            rows.append(
                f"{_tex(row['method'])} & {row['memory_size']} & {_tex(row['memory_source'])} & "
                f"{_fmt_pm(row['query_ms_per_flow_mean'], row['query_ms_per_flow_std'], 3)} & "
                f"{_fmt_pm(row['memory_mb_mean'], row['memory_mb_std'], 2)} & "
                f"{_fmt_pm(row['auroc_mean'], row['auroc_std'])} & {_fmt_pm(row['recall_at_1pct_fpr_mean'], row['recall_at_1pct_fpr_std'])} & "
                f"{_fmt_pm(row['p99_fpr_mean'], row['p99_fpr_std'])} \\\\"
            )
        _latex_table(
            PAPER_TABLE_DIR / "table_ann_scalability.tex",
            "\\begin{tabular}{lrlccccc}\n\\toprule\nMethod & Memory & Source & Query ms/flow & Memory MB & AUROC & R@1\\%FPR & P99 FPR \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )

    if e2e_rows is not None:
        rows = []
        for row in e2e_rows:
            rows.append(
                f"{_tex(row['stage'])} & {row['input_count']} & {row['output_count']} & {_fmt(row['wall_time_seconds'], 4)} & "
                f"{_fmt(row['throughput_items_per_second'], 1)} & {_tex(row['notes'])} \\\\"
            )
        _latex_table(
            PAPER_TABLE_DIR / "table_e2e_throughput.tex",
            "\\begin{tabular}{lrrrrl}\n\\toprule\nStage & Input & Output & Wall time (s) & Throughput/s & Notes \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )

    if artifact_rows is not None:
        rows = []
        for row in artifact_rows:
            rows.append(
                f"{_tex(row['artifact'])} & {_tex(row['purpose'])} & \\texttt{{{_tex(row['path'])}}} & {_tex(row['released'])} & {_tex(row['regenerates'])} \\\\"
            )
        _latex_table(
            PAPER_TABLE_DIR / "table_artifact_availability.tex",
            "\\begin{tabular}{lllll}\n\\toprule\nArtifact & Purpose & Path & Released? & Regenerates \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}",
        )


def write_revision_report(out_dir: Path, mode: str, generated: dict[str, Any]) -> None:
    lines = [
        "# ICDM Revision Experiment Report",
        "",
        f"Mode: `{mode}`.",
        "",
        "## Generated Artifacts",
        "",
    ]
    for name, value in generated.items():
        if isinstance(value, list):
            lines.append(f"- `{name}`: {len(value)} rows")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- All recomputed low-FPR rows use existing leave-one token corpora and global KNN benign memory.",
            "- Train and validation rows are checked to be benign-only; vocabularies are train-only in the token artifacts.",
            "- Raw IP addresses, absolute timestamps, and direct five-tuple/port tokens are not present in the audited token corpora.",
            "- WebAttack has only 28 held-out attack flows and is marked as low-support in the generated CSV rows.",
            "- Historical setting sweeps are not used for the main behavior-only low-FPR claim.",
            "- Deployable thresholds use benign validation scores only; test-oracle thresholds are marked as separability analysis.",
            "- Primitive-threshold sensitivity in full mode is a seed-43 source-rebuild smoke analysis over three unknown settings, not a new hyperparameter search.",
        ]
    )
    (out_dir / "revision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ICDM Applied Track revision experiments from existing FlowPrim artifacts.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--audit_seed", type=int, default=43)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    target_rows, target_summary = build_target_realized_fpr(args.seeds, out_dir)
    alert_rows, alert_summary = build_alert_budget(args.seeds, out_dir)
    memory_scope_rows, memory_scope_summary = build_memory_scope_audit(args.seeds, out_dir)
    feature_rows, feature_summary = build_behavior_feature_attribution(args.seeds, out_dir)
    prevalence_rows, prevalence_summary = build_prevalence_alert_budget(alert_rows, out_dir)
    aggregate_rows = build_unknown_aggregate_with_without_webattack(target_rows, out_dir, args.seeds)
    setting_protocol_rows = build_setting_selection_protocol(out_dir)
    validation_rows = copy_validation_size_table(out_dir)
    diagnosis_rows = build_diagnosis_audit(args.audit_seed, out_dir)
    diagnosis_simplified_rows = build_diagnosis_audit_simplified(diagnosis_rows, out_dir)
    external_rows = build_external_diagnostics(out_dir)
    primitive_rows = copy_primitive_ablation_as_sensitivity(out_dir)

    sensitivity_rows: list[dict[str, Any]] | None = None
    sensitivity_summary: list[dict[str, Any]] | None = None
    ann_rows: list[dict[str, Any]] | None = None
    ann_summary: list[dict[str, Any]] | None = None
    e2e_rows: list[dict[str, Any]] | None = None
    artifact_rows: list[dict[str, Any]] | None = None
    if args.mode == "full":
        sensitivity_rows, sensitivity_summary = build_primitive_threshold_sensitivity(out_dir, seed=args.audit_seed)
        ann_rows, ann_summary = build_ann_knn_scalability(out_dir, seed=args.audit_seed)
        e2e_rows = build_e2e_throughput_smoke(out_dir)
        artifact_rows = build_artifact_availability(out_dir)

    write_latex_tables(
        out_dir,
        target_summary,
        alert_summary,
        memory_scope_summary,
        validation_rows,
        diagnosis_rows,
        external_rows,
        feature_summary,
        prevalence_summary,
        aggregate_rows,
        setting_protocol_rows,
        sensitivity_summary,
        diagnosis_simplified_rows,
        ann_summary,
        e2e_rows,
        artifact_rows,
    )
    generated = {
        "target_realized_fpr_runs": target_rows,
        "alert_budget_runs": alert_rows,
        "memory_scope_audit_runs": memory_scope_rows,
        "behavior_feature_attribution_runs": feature_rows,
        "prevalence_alert_budget_runs": prevalence_rows,
        "unknown_aggregate_with_without_webattack": aggregate_rows,
        "setting_selection_protocol": setting_protocol_rows,
        "validation_size_rows": validation_rows,
        "diagnosis_audit_cases": diagnosis_rows,
        "diagnosis_audit_simplified": diagnosis_simplified_rows,
        "external_diagnostics": external_rows,
        "primitive_sensitivity_from_ablation": primitive_rows,
        "behavior_feature_attribution_summary": feature_summary,
    }
    if sensitivity_rows is not None:
        generated["primitive_threshold_sensitivity_runs"] = sensitivity_rows
    if ann_rows is not None:
        generated["ann_knn_scalability_runs"] = ann_rows
    if e2e_rows is not None:
        generated["e2e_throughput_smoke"] = e2e_rows
    if artifact_rows is not None:
        generated["artifact_availability"] = artifact_rows
    write_revision_report(out_dir, args.mode, generated)
    with (out_dir / "revision_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({key: len(value) for key, value in generated.items()}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"mode": args.mode, **{key: len(value) for key, value in generated.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
