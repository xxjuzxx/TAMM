#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

from src.features.token_alias import canonical_tokens, is_burst_token, is_packet_token, is_profile_token


ROOT = Path(__file__).resolve().parents[1]
TOKEN_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown" / "tokens_category"
OUT_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "deployment"
SWEEP_PATH = ROOT / "scripts" / "52_sweep_anomaly_low_fpr.py"
RUNTIME_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "runtime"

BEST_SETTINGS = {
    "Botnet": {"slug": "botnet", "feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_euclidean", "k": 3, "group_mode": "global"},
    "DDoS": {"slug": "ddos", "feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "global"},
    "Probe": {"slug": "probe", "feature_filter": "all_no_special", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "global"},
    "WebAttack": {"slug": "webattack", "feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "global"},
    "BruteForce": {"slug": "bruteforce", "feature_filter": "packet_burst_profile", "transform": "tfidf_l2", "scorer": "knn_cosine", "k": 3, "group_mode": "global"},
}

SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}


def _load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("flowprim_deployment_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _token_path(attack: str, seed: int) -> Path:
    return TOKEN_DIR / f"cicids2017_leave_one_{BEST_SETTINGS[attack]['slug']}_anomaly_seed{seed}_a3_full_rhythm.pt"


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    value = _float(value)
    return "-" if value is None else f"{value:.{digits}f}"


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize(val) for val in value]
    if isinstance(value, tuple):
        return [_sanitize(val) for val in value]
    if isinstance(value, np.ndarray):
        return [_sanitize(val) for val in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
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
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean), float(std)


def _summarize(rows: list[dict[str, Any]], group_key: str, metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)
    out: list[dict[str, Any]] = []
    for group, items in grouped.items():
        summary: dict[str, Any] = {group_key: group, "num_rows": len(items)}
        for metric in metrics:
            values = [_float(item.get(metric)) for item in items]
            clean = [value for value in values if value is not None]
            mean, std = _mean_std(clean)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        out.append(summary)
    def sort_key(row: dict[str, Any]) -> Any:
        value = row[group_key]
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    return sorted(out, key=sort_key)


def _load_attack_state(attack: str, seed: int) -> dict[str, Any]:
    setting = BEST_SETTINGS[attack]
    token_path = _token_path(attack, seed)
    token_data = S._read_token_data(token_path)
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = S._split_indices(token_data, "train")
    val_idx = S._split_indices(token_data, "val")
    test_idx = S._split_indices(token_data, "test")
    features, feature_stats = S._features(
        token_data,
        train_idx,
        feature_filter=setting["feature_filter"],
        transform=setting["transform"],
    )
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
    }


def _eval_with_train(
    state: dict[str, Any],
    train_idx: np.ndarray,
    group_mode: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    setting = state["setting"]
    groups = S._groups(state["token_data"], group_mode)
    val_scores = S._scores(
        state["features"],
        train_idx,
        state["val_idx"],
        groups,
        scorer=setting["scorer"],
        k=int(setting["k"]),
    )
    test_scores = S._scores(
        state["features"],
        train_idx,
        state["test_idx"],
        groups,
        scorer=setting["scorer"],
        k=int(setting["k"]),
    )
    labels = state["labels"]
    y_true = labels[state["test_idx"]].astype(np.int64)
    best = S._best_macro(y_true, test_scores)
    r01 = S._best_recall_under_fpr(y_true, test_scores, 0.001)
    r1 = S._best_recall_under_fpr(y_true, test_scores, 0.01)
    rank = S._rank_metrics(y_true, test_scores)
    val99 = S._metrics_at_threshold(y_true, test_scores, float(np.percentile(val_scores, 99.0)))
    row = {
        "best_macro_f1": best["macro_f1"],
        "best_attack_recall": best["attack_recall"],
        "best_false_positive_rate": best["false_positive_rate"],
        "recall_at_0_1pct_fpr": r01["attack_recall"],
        "actual_fpr_at_0_1pct_fpr": r01["false_positive_rate"],
        "recall_at_1pct_fpr": r1["attack_recall"],
        "actual_fpr_at_1pct_fpr": r1["false_positive_rate"],
        "val_p99_threshold": float(np.percentile(val_scores, 99.0)),
        "val_p99_macro_f1": val99["macro_f1"],
        "val_p99_attack_recall": val99["attack_recall"],
        "val_p99_false_positive_rate": val99["false_positive_rate"],
        "val_p99_attack_precision": val99["attack_precision"],
        **rank,
    }
    return row, val_scores, test_scores, groups


def run_memory_scope_audit(seed: int, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for attack in BEST_SETTINGS:
        state = _load_attack_state(attack, seed)
        for group_mode in ["global"]:
            metrics, _, _, _ = _eval_with_train(state, state["train_idx"], group_mode)
            rows.append(
                {
                    "attack": attack,
                    "seed": seed,
                    "feature_filter": state["setting"]["feature_filter"],
                    "transform": state["setting"]["transform"],
                    "scorer": state["setting"]["scorer"],
                    "k": state["setting"]["k"],
                    "memory_scope": group_mode,
                    "num_features": state["feature_stats"]["num_features"],
                    "train_n": len(state["train_idx"]),
                    "val_n": len(state["val_idx"]),
                    "test_n": len(state["test_idx"]),
                    **metrics,
                }
            )
    summary = _summarize(
        rows,
        "memory_scope",
        [
            "best_macro_f1",
            "auroc",
            "fpr95",
            "recall_at_0_1pct_fpr",
            "recall_at_1pct_fpr",
            "val_p99_false_positive_rate",
            "val_p99_attack_recall",
        ],
    )
    _write_csv(rows, out_dir / "memory_scope_audit.csv")
    _write_csv(summary, out_dir / "memory_scope_audit_summary.csv")
    return rows, summary


def _sample_train(train_idx: np.ndarray, size: int, seed: int) -> np.ndarray:
    if size >= len(train_idx):
        return np.asarray(train_idx, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(train_idx, size=int(size), replace=False)
    return np.asarray(sorted(chosen.tolist()), dtype=np.int64)


def _time_scores(
    state: dict[str, Any],
    train_idx: np.ndarray,
    groups: list[str],
    max_eval: int,
    repeats: int,
) -> tuple[float, float]:
    setting = state["setting"]
    eval_idx = state["test_idx"][: min(max_eval, len(state["test_idx"]))]
    if len(eval_idx) == 0:
        return 0.0, 0.0
    timings: list[float] = []
    S._scores(state["features"], train_idx, eval_idx, groups, scorer=setting["scorer"], k=int(setting["k"]))
    for _ in range(repeats):
        start = time.perf_counter()
        S._scores(state["features"], train_idx, eval_idx, groups, scorer=setting["scorer"], k=int(setting["k"]))
        elapsed = time.perf_counter() - start
        timings.append(elapsed / max(len(eval_idx), 1) * 1000.0)
    return float(statistics.fmean(timings)), float(statistics.stdev(timings) if len(timings) > 1 else 0.0)


def run_memory_scaling(seed: int, out_dir: Path, sizes: list[int], max_eval: int, repeats: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for attack_pos, attack in enumerate(BEST_SETTINGS):
        state = _load_attack_state(attack, seed)
        group_mode = state["setting"]["group_mode"]
        groups = S._groups(state["token_data"], group_mode)
        for size in sizes:
            train_sub = _sample_train(state["train_idx"], size, seed=seed * 1000 + attack_pos * 100 + size)
            metrics, _, _, _ = _eval_with_train(state, train_sub, group_mode)
            ms_mean, ms_std = _time_scores(state, train_sub, groups, max_eval=max_eval, repeats=repeats)
            rows.append(
                {
                    "attack": attack,
                    "seed": seed,
                    "memory_size": len(train_sub),
                    "feature_filter": state["setting"]["feature_filter"],
                    "transform": state["setting"]["transform"],
                    "scorer": state["setting"]["scorer"],
                    "k": state["setting"]["k"],
                    "group_mode": group_mode,
                    "num_features": state["feature_stats"]["num_features"],
                    "scoring_ms_per_flow_mean": ms_mean,
                    "scoring_ms_per_flow_std": ms_std,
                    "scoring_flows_per_second": float(1000.0 / max(ms_mean, 1e-12)),
                    **metrics,
                }
            )
    summary = _summarize(
        rows,
        "memory_size",
        [
            "best_macro_f1",
            "auroc",
            "fpr95",
            "recall_at_0_1pct_fpr",
            "recall_at_1pct_fpr",
            "val_p99_false_positive_rate",
            "val_p99_attack_recall",
            "scoring_ms_per_flow_mean",
            "scoring_flows_per_second",
        ],
    )
    _write_csv(rows, out_dir / "memory_size_scaling.csv")
    _write_csv(summary, out_dir / "memory_size_scaling_summary.csv")
    return rows, summary


def run_calibration_stability(seed: int, out_dir: Path, sizes: list[int], repeats: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for attack_pos, attack in enumerate(BEST_SETTINGS):
        state = _load_attack_state(attack, seed)
        metrics, val_scores, test_scores, _ = _eval_with_train(state, state["train_idx"], state["setting"]["group_mode"])
        del metrics
        y_true = state["labels"][state["test_idx"]].astype(np.int64)
        for size in sizes:
            size = min(int(size), len(val_scores))
            for repeat in range(repeats if size < len(val_scores) else 1):
                rng = np.random.default_rng(seed * 100000 + attack_pos * 10000 + size * 100 + repeat)
                if size >= len(val_scores):
                    sampled = val_scores
                else:
                    sampled = rng.choice(val_scores, size=size, replace=False)
                threshold = float(np.percentile(sampled, 99.0))
                out = S._metrics_at_threshold(y_true, test_scores, threshold)
                rows.append(
                    {
                        "attack": attack,
                        "seed": seed,
                        "val_benign_n": size,
                        "repeat": repeat,
                        "threshold": threshold,
                        "macro_f1": out["macro_f1"],
                        "attack_recall": out["attack_recall"],
                        "false_positive_rate": out["false_positive_rate"],
                        "attack_precision": out["attack_precision"],
                        "false_positives": out["false_positives"],
                        "true_negatives": out["true_negatives"],
                        "true_positives": out["true_positives"],
                        "false_negatives": out["false_negatives"],
                    }
                )
        for size in sizes:
            attack_size_rows = [row for row in rows if row["attack"] == attack and int(row["val_benign_n"]) == min(size, len(val_scores))]
            item = {"attack": attack, "seed": seed, "val_benign_n": min(size, len(val_scores)), "repeats": len(attack_size_rows)}
            for metric in ["threshold", "macro_f1", "attack_recall", "false_positive_rate", "attack_precision"]:
                values = [float(row[metric]) for row in attack_size_rows]
                mean, std = _mean_std(values)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = std
            summary_rows.append(item)
    aggregate = _summarize(
        summary_rows,
        "val_benign_n",
        [
            "macro_f1_mean",
            "attack_recall_mean",
            "false_positive_rate_mean",
            "threshold_std",
        ],
    )
    _write_csv(rows, out_dir / "calibration_stability_runs.csv")
    _write_csv(summary_rows, out_dir / "calibration_stability_by_attack.csv")
    _write_csv(aggregate, out_dir / "calibration_stability_summary.csv")
    return rows, summary_rows, aggregate


def _row_tokens(token_data: dict[str, Any], row_idx: int) -> list[str]:
    id_to_token = S._id_to_token(token_data["vocab"])
    ids = token_data["input_ids"][row_idx].cpu().numpy()
    mask = token_data["attention_mask"][row_idx].cpu().numpy() > 0
    return canonical_tokens(id_to_token.get(int(token_id), "[UNK]") for token_id in ids[mask])


def _token_summary(tokens: list[str], predicate=None, limit: int = 8) -> str:
    selected = [token for token in tokens if token not in SPECIAL_TOKENS]
    if predicate is not None:
        selected = [token for token in selected if predicate(token)]
    counts = Counter(selected)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return "; ".join(f"{token}:{count}" for token, count in ordered)


def _primitive_summary(tokens: list[str]) -> str:
    primitives = sorted(
        token
        for token in set(tokens)
        if is_profile_token(token)
    )
    return ", ".join(primitives) if primitives else "none"


def run_case_studies(seed: int, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack in BEST_SETTINGS:
        state = _load_attack_state(attack, seed)
        metrics, val_scores, test_scores, groups = _eval_with_train(state, state["train_idx"], state["setting"]["group_mode"])
        del metrics
        labels = state["labels"]
        test_idx = state["test_idx"]
        y_true = labels[test_idx].astype(np.int64)
        threshold = float(np.percentile(val_scores, 99.0))
        score_by_idx = {int(idx): float(test_scores[pos]) for pos, idx in enumerate(test_idx.tolist())}
        attack_indices = test_idx[y_true == 1]
        benign_indices = test_idx[y_true == 0]
        selected: list[tuple[str, int]] = []
        if len(attack_indices):
            candidates: list[tuple[float, int]] = []
            fallback: list[tuple[float, int]] = []
            for idx in attack_indices.tolist():
                tokens = _row_tokens(state["token_data"], int(idx))
                score = score_by_idx[int(idx)]
                fallback.append((score, int(idx)))
                if score >= threshold and _primitive_summary(tokens) != "none":
                    candidates.append((score, int(idx)))
            source = candidates or fallback
            selected.append(("attack_primitive_alert" if candidates else "attack_high_score", max(source)[1]))
        if len(benign_indices):
            benign_scores = np.asarray([score_by_idx[int(idx)] for idx in benign_indices], dtype=np.float64)
            selected.append(("benign_tail", int(benign_indices[int(np.argmax(benign_scores))])))
        for case_type, idx in selected:
            tokens = _row_tokens(state["token_data"], idx)
            meta = state["token_data"]["meta"][idx]
            score = score_by_idx[idx]
            prediction = int(score >= threshold)
            rows.append(
                {
                    "unknown_attack_setting": attack,
                    "case_type": case_type,
                    "seed": seed,
                    "flow_id": str(meta.get("flow_id", "")),
                    "short_flow_id": str(meta.get("flow_id", ""))[:10],
                    "label": str(meta.get("attack_family") or meta.get("label") or ""),
                    "binary_label": str(meta.get("binary_label") or int(labels[idx])),
                    "group": groups[idx],
                    "score": score,
                    "threshold_p99": threshold,
                    "margin": float(score - threshold),
                    "prediction": prediction,
                    "decision": "alert" if prediction else "normal",
                    "primitive_tokens": _primitive_summary(tokens),
                    "top_packet_tokens": _token_summary(tokens, lambda token: is_packet_token(token), limit=8),
                    "top_burst_tokens": _token_summary(tokens, lambda token: is_burst_token(token), limit=8),
                    "top_all_tokens": _token_summary(tokens, None, limit=10),
                }
            )
    _write_csv(rows, out_dir / "diagnosis_case_studies.csv")
    return rows


def run_runtime_breakdown(out_dir: Path) -> list[dict[str, Any]]:
    source = RUNTIME_DIR / "diagnosis_latency_by_attack.csv"
    if not source.exists():
        return []
    rows = _read_csv(source)
    metric_map = [
        ("feature_build_us_per_flow", "Feature construction", "us/flow"),
        ("memory_build_us_per_train_flow", "Benign memory setup", "us/train flow"),
        ("threshold_us_per_val_flow", "Validation threshold scoring", "us/val flow"),
        ("scoring_mean_ms", "Exact memory score", "ms/flow"),
        ("record_mean_ms", "Diagnosis record materialization", "ms/flow"),
        ("diagnosis_mean_ms", "Total flow-ready diagnosis", "ms/flow"),
    ]
    out: list[dict[str, Any]] = []
    for key, stage, unit in metric_map:
        values = [_float(row.get(key)) for row in rows]
        clean = [value for value in values if value is not None]
        mean, std = _mean_std(clean)
        out.append(
            {
                "stage": stage,
                "unit": unit,
                "mean": mean,
                "std": std,
                "min": min(clean) if clean else None,
                "max": max(clean) if clean else None,
            }
        )
    if out:
        _write_csv(out, out_dir / "runtime_breakdown_summary.csv")
    return out


def _write_markdown(
    out_dir: Path,
    memory_scope_summary: list[dict[str, Any]],
    memory_summary: list[dict[str, Any]],
    calibration_summary: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    runtime_breakdown: list[dict[str, Any]],
) -> None:
    lines = [
        "# FlowPrim Deployment-Oriented Extra Experiments",
        "",
        "All rows use seed-43 leave-one unknown token corpora unless stated otherwise. The deployment audit uses global benign memory; the token corpora do not contain direct raw IP, absolute timestamp, five-tuple, or service shortcut tokens.",
        "",
        "## Memory Scope Audit",
        "",
        "| Memory scope | Macro-F1 | AUROC | FPR95 | Recall@0.1%FPR | Recall@1%FPR | Val-P99 FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in memory_scope_summary:
        lines.append(
            "| `{mode}` | {macro} +/- {macro_std} | {auroc} +/- {auroc_std} | {fpr95} +/- {fpr95_std} | {r01} +/- {r01_std} | {r1} +/- {r1_std} | {valfpr} +/- {valfpr_std} |".format(
                mode=row["memory_scope"],
                macro=_fmt(row.get("best_macro_f1_mean")),
                macro_std=_fmt(row.get("best_macro_f1_std")),
                auroc=_fmt(row.get("auroc_mean")),
                auroc_std=_fmt(row.get("auroc_std")),
                fpr95=_fmt(row.get("fpr95_mean")),
                fpr95_std=_fmt(row.get("fpr95_std")),
                r01=_fmt(row.get("recall_at_0_1pct_fpr_mean")),
                r01_std=_fmt(row.get("recall_at_0_1pct_fpr_std")),
                r1=_fmt(row.get("recall_at_1pct_fpr_mean")),
                r1_std=_fmt(row.get("recall_at_1pct_fpr_std")),
                valfpr=_fmt(row.get("val_p99_false_positive_rate_mean")),
                valfpr_std=_fmt(row.get("val_p99_false_positive_rate_std")),
            )
        )
    lines.extend(
        [
            "",
            "## Memory-size scaling",
            "",
            "| Benign memory | Macro-F1 | AUROC | Recall@1%FPR | Val-P99 FPR | Score ms/flow |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in memory_summary:
        lines.append(
            "| {n} | {macro} +/- {macro_std} | {auroc} +/- {auroc_std} | {r1} +/- {r1_std} | {valfpr} +/- {valfpr_std} | {ms} +/- {ms_std} |".format(
                n=row["memory_size"],
                macro=_fmt(row.get("best_macro_f1_mean")),
                macro_std=_fmt(row.get("best_macro_f1_std")),
                auroc=_fmt(row.get("auroc_mean")),
                auroc_std=_fmt(row.get("auroc_std")),
                r1=_fmt(row.get("recall_at_1pct_fpr_mean")),
                r1_std=_fmt(row.get("recall_at_1pct_fpr_std")),
                valfpr=_fmt(row.get("val_p99_false_positive_rate_mean")),
                valfpr_std=_fmt(row.get("val_p99_false_positive_rate_std")),
                ms=_fmt(row.get("scoring_ms_per_flow_mean_mean"), 5),
                ms_std=_fmt(row.get("scoring_ms_per_flow_mean_std"), 5),
            )
        )
    lines.extend(
        [
            "",
            "## Calibration stability",
            "",
            "| BENIGN validation size | Macro-F1 | Recall | Test FPR | Threshold std |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in calibration_summary:
        lines.append(
            "| {n} | {macro} +/- {macro_std} | {recall} +/- {recall_std} | {fpr} +/- {fpr_std} | {th_std} +/- {th_std2} |".format(
                n=row["val_benign_n"],
                macro=_fmt(row.get("macro_f1_mean_mean")),
                macro_std=_fmt(row.get("macro_f1_mean_std")),
                recall=_fmt(row.get("attack_recall_mean_mean")),
                recall_std=_fmt(row.get("attack_recall_mean_std")),
                fpr=_fmt(row.get("false_positive_rate_mean_mean")),
                fpr_std=_fmt(row.get("false_positive_rate_mean_std")),
                th_std=_fmt(row.get("threshold_std_mean")),
                th_std2=_fmt(row.get("threshold_std_std")),
            )
        )
    lines.extend(
        [
            "",
            "## Runtime breakdown",
            "",
            "| Stage | Unit | Mean | Std | Range |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in runtime_breakdown:
        lines.append(
            "| {stage} | {unit} | {mean} | {std} | {lo}-{hi} |".format(
                stage=row["stage"],
                unit=row["unit"],
                mean=_fmt(row.get("mean"), 5 if "ms" in row["unit"] else 2),
                std=_fmt(row.get("std"), 5 if "ms" in row["unit"] else 2),
                lo=_fmt(row.get("min"), 5 if "ms" in row["unit"] else 2),
                hi=_fmt(row.get("max"), 5 if "ms" in row["unit"] else 2),
            )
        )
    lines.extend(
        [
            "",
            "## Representative diagnosis cases",
            "",
            "| Setting | Case | Flow | Label | Score/Threshold | Decision | Primitive evidence |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for row in case_rows:
        if row["case_type"] == "benign_tail" and row["unknown_attack_setting"] != "Botnet":
            continue
        lines.append(
            "| {setting} | {case} | `{flow}` | {label} | {score}/{thr} | {decision} | {prim} |".format(
                setting=row["unknown_attack_setting"],
                case=row["case_type"],
                flow=row["short_flow_id"],
                label=row["label"],
                score=_fmt(row["score"], 4),
                thr=_fmt(row["threshold_p99"], 4),
                decision=row["decision"],
                prim=row["primitive_tokens"].replace("|", "/"),
            )
        )
    (out_dir / "deployment_extra_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deployment-oriented FlowPrim extra experiments for the ICDM Applied manuscript.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--memory_sizes", nargs="+", type=int, default=[100, 200, 500, 1000, 1400])
    parser.add_argument("--calibration_sizes", nargs="+", type=int, default=[25, 50, 100, 200])
    parser.add_argument("--calibration_repeats", type=int, default=200)
    parser.add_argument("--latency_repeats", type=int, default=5)
    parser.add_argument("--latency_max_eval", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    memory_scope_rows, memory_scope_summary = run_memory_scope_audit(args.seed, out_dir)
    memory_rows, memory_summary = run_memory_scaling(args.seed, out_dir, args.memory_sizes, args.latency_max_eval, args.latency_repeats)
    calibration_runs, calibration_by_attack, calibration_summary = run_calibration_stability(args.seed, out_dir, args.calibration_sizes, args.calibration_repeats)
    case_rows = run_case_studies(args.seed, out_dir)
    runtime_breakdown = run_runtime_breakdown(out_dir)

    _write_markdown(out_dir, memory_scope_summary, memory_summary, calibration_summary, case_rows, runtime_breakdown)
    with (out_dir / "deployment_extra_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _sanitize(
                {
                    "seed": args.seed,
                    "settings": BEST_SETTINGS,
                    "memory_scope_summary": memory_scope_summary,
                    "memory_summary": memory_summary,
                    "calibration_summary": calibration_summary,
                    "case_studies": case_rows,
                    "runtime_breakdown": runtime_breakdown,
                    "row_counts": {
                        "memory_scope": len(memory_scope_rows),
                        "memory": len(memory_rows),
                        "calibration_runs": len(calibration_runs),
                        "calibration_by_attack": len(calibration_by_attack),
                        "case_studies": len(case_rows),
                    },
                }
            ),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(json.dumps({"out_dir": str(out_dir), "memory_scope_rows": len(memory_scope_rows), "memory_rows": len(memory_rows), "calibration_runs": len(calibration_runs), "case_rows": len(case_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
