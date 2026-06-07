#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

from src.evaluation.metrics import classification_metrics, confusion
from src.features.token_alias import is_burst_token, is_packet_token, is_profile_token, is_rhythm_token


SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[MASK]"}


def _read_token_data(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    return {int(idx): str(token) for token, idx in vocab.items()}


def _split_indices(token_data: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray([idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == split], dtype=np.int64)


def _keep_token(token: str, feature_filter: str) -> bool:
    if feature_filter == "all_with_special":
        return True
    if token in SPECIAL_TOKENS:
        return False
    if feature_filter == "all_no_special":
        return True
    if feature_filter == "no_unk":
        return token != "[UNK]"
    if feature_filter == "packet":
        return is_packet_token(token)
    if feature_filter == "packet_burst":
        return is_packet_token(token) or is_burst_token(token)
    if feature_filter == "packet_burst_profile":
        return is_packet_token(token) or is_burst_token(token) or is_profile_token(token)
    if feature_filter == "profile_only":
        return is_profile_token(token)
    if feature_filter == "rhythm_only":
        return is_rhythm_token(token)
    if feature_filter == "no_profile_no_unk":
        return token != "[UNK]" and not is_profile_token(token, include_none=True)
    if feature_filter == "no_burst_no_profile_no_unk":
        return token != "[UNK]" and not is_profile_token(token, include_none=True) and not is_burst_token(token)
    raise ValueError(f"Unsupported feature filter: {feature_filter}")


def _raw_counts(token_data: dict[str, Any], kept_ids: list[int]) -> np.ndarray:
    input_ids = token_data["input_ids"].cpu().numpy()
    attention_mask = token_data["attention_mask"].cpu().numpy()
    vocab_size = len(token_data["vocab"])
    kept = np.asarray(kept_ids, dtype=np.int64)
    rows = np.zeros((input_ids.shape[0], vocab_size), dtype=np.float32)
    for row_idx in range(input_ids.shape[0]):
        active = input_ids[row_idx][attention_mask[row_idx] > 0]
        if active.size:
            rows[row_idx] = np.bincount(active, minlength=vocab_size)[:vocab_size]
    return rows[:, kept]


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


def _features(
    token_data: dict[str, Any],
    train_idx: np.ndarray,
    *,
    feature_filter: str,
    transform: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    id_to_token = _id_to_token(token_data["vocab"])
    kept_ids = sorted(token_id for token_id, token in id_to_token.items() if _keep_token(token, feature_filter))
    if not kept_ids:
        raise ValueError(f"Feature filter kept no tokens: {feature_filter}")
    features = _raw_counts(token_data, kept_ids)
    if transform.startswith("binary"):
        features = (features > 0).astype(np.float32)
        norm = transform.removeprefix("binary_") or "none"
        features = _normalize(features, norm)
    elif transform.startswith("tfidf"):
        train_counts = features[train_idx]
        df = np.sum(train_counts > 0, axis=0)
        idf = np.log((1.0 + len(train_idx)) / (1.0 + df)) + 1.0
        features = features * idf.reshape(1, -1).astype(np.float32)
        norm = transform.removeprefix("tfidf_") or "none"
        features = _normalize(features, norm)
    elif transform.startswith("count"):
        norm = transform.removeprefix("count_") or "none"
        features = _normalize(features, norm)
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    stats = {
        "feature_filter": feature_filter,
        "transform": transform,
        "num_features": int(features.shape[1]),
        "kept_tokens": [id_to_token[token_id] for token_id in kept_ids],
        "mean_nonzero": float(np.mean(np.sum(features != 0, axis=1))),
    }
    return features.astype(np.float32, copy=False), stats


def _protocol_group(meta: dict[str, Any]) -> str:
    context = str(meta.get("context_id") or "")
    if ":" in context:
        return context.rsplit(":", maxsplit=1)[-1] or "unknown"
    return "unknown"


def _groups(token_data: dict[str, Any], group_mode: str) -> list[str]:
    if group_mode == "global":
        return ["GLOBAL"] * len(token_data.get("meta", []))
    if group_mode == "protocol":
        return [_protocol_group(meta) for meta in token_data.get("meta", [])]
    raise ValueError(f"Unsupported group mode: {group_mode}")


def _cosine_distance(features: np.ndarray, refs: np.ndarray) -> np.ndarray:
    feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
    ref_norms = np.linalg.norm(refs, axis=1, keepdims=True).T
    denom = feature_norms @ ref_norms
    sim = np.divide(features @ refs.T, denom, out=np.zeros((features.shape[0], refs.shape[0]), dtype=np.float32), where=denom > 0)
    return 1.0 - sim


def _euclidean_distance(features: np.ndarray, refs: np.ndarray) -> np.ndarray:
    left = np.sum(features * features, axis=1, keepdims=True)
    right = np.sum(refs * refs, axis=1, keepdims=True).T
    dist2 = np.maximum(left + right - 2.0 * (features @ refs.T), 0.0)
    return np.sqrt(dist2, dtype=np.float32)


def _score_against_refs(features: np.ndarray, refs: np.ndarray, scorer: str, k: int) -> np.ndarray:
    if refs.size == 0:
        return np.ones(features.shape[0], dtype=np.float32)
    if scorer == "prototype_cosine":
        proto = refs.mean(axis=0, keepdims=True)
        return _cosine_distance(features, proto).reshape(-1).astype(np.float32)
    if scorer == "prototype_euclidean":
        proto = refs.mean(axis=0, keepdims=True)
        return _euclidean_distance(features, proto).reshape(-1).astype(np.float32)
    if scorer == "knn_cosine":
        distances = _cosine_distance(features, refs)
    elif scorer == "knn_euclidean":
        distances = _euclidean_distance(features, refs)
    else:
        raise ValueError(f"Unsupported scorer: {scorer}")
    kk = max(1, min(int(k), distances.shape[1]))
    nearest = np.partition(distances, kk - 1, axis=1)[:, :kk]
    return np.mean(nearest, axis=1).astype(np.float32)


def _scores(
    features: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    groups: list[str],
    *,
    scorer: str,
    k: int,
) -> np.ndarray:
    global_refs = features[train_idx]
    train_by_group: dict[str, list[int]] = defaultdict(list)
    for idx in train_idx.tolist():
        train_by_group[groups[idx]].append(idx)
    eval_by_group: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for out_pos, idx in enumerate(eval_idx.tolist()):
        eval_by_group[groups[idx]].append((out_pos, idx))

    out = np.zeros(len(eval_idx), dtype=np.float32)
    for group, items in eval_by_group.items():
        positions = [pos for pos, _ in items]
        data_indices = [idx for _, idx in items]
        ref_indices = train_by_group.get(group) or []
        refs = features[np.asarray(ref_indices, dtype=np.int64)] if ref_indices else global_refs
        out[np.asarray(positions, dtype=np.int64)] = _score_against_refs(features[np.asarray(data_indices, dtype=np.int64)], refs, scorer, k)
    return out


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores.astype(np.float64))
    if unique_scores.size == 0:
        return np.array([0.5], dtype=np.float64)
    eps = 1e-12
    thresholds = [float(unique_scores[0] - eps), float(unique_scores[-1] + eps)]
    thresholds.extend(float(value) for value in unique_scores)
    return np.asarray(sorted(set(thresholds)), dtype=np.float64)


def _metrics_from_counts(
    *,
    threshold: float,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> dict[str, Any]:
    total = tp + fp + tn + fn
    positives = tp + fn
    negatives = tn + fp
    pos_precision = float(tp / max(tp + fp, 1))
    pos_recall = float(tp / max(positives, 1))
    neg_precision = float(tn / max(tn + fn, 1))
    neg_recall = float(tn / max(negatives, 1))
    pos_f1 = float((2 * tp) / max(2 * tp + fp + fn, 1))
    neg_f1 = float((2 * tn) / max(2 * tn + fp + fn, 1))
    macro_f1 = float((pos_f1 + neg_f1) / 2.0)
    return {
        "accuracy": float((tp + tn) / max(total, 1)),
        "precision_macro": float((pos_precision + neg_precision) / 2.0),
        "recall_macro": float((pos_recall + neg_recall) / 2.0),
        "macro_f1": macro_f1,
        "weighted_f1": float((neg_f1 * negatives + pos_f1 * positives) / max(total, 1)),
        "minority_macro_f1": neg_f1 if negatives <= positives else pos_f1,
        "worst_class_f1": float(min(pos_f1, neg_f1)),
        "false_positive_rate": float(fp / max(negatives, 1)),
        "attack_recall": pos_recall,
        "attack_precision": pos_precision,
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _threshold_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    sorted_scores = scores[order].astype(np.float64)
    sorted_labels = y_true[order].astype(np.int64)
    if sorted_scores.size == 0:
        raise ValueError("No scores available")
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp_cum = np.cumsum(sorted_labels == 1)
    fp_cum = np.cumsum(sorted_labels == 0)
    thresholds = np.concatenate(([float(sorted_scores[0] + 1e-12)], sorted_scores[group_ends]))
    tp = np.concatenate(([0], tp_cum[group_ends])).astype(np.int64)
    fp = np.concatenate(([0], fp_cum[group_ends])).astype(np.int64)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    tn = (negatives - fp).astype(np.int64)
    fn = (positives - tp).astype(np.int64)
    return thresholds, tp, fp, tn, fn


def _binary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    benign = y_true == 0
    attack = y_true == 1
    fp = int(np.sum((y_pred == 1) & benign))
    tn = int(np.sum((y_pred == 0) & benign))
    tp = int(np.sum((y_pred == 1) & attack))
    fn = int(np.sum((y_pred == 0) & attack))
    return {
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "attack_recall": float(tp / max(tp + fn, 1)),
        "attack_precision": float(tp / max(tp + fp, 1)),
        "false_positives": fp,
        "true_negatives": tn,
        "true_positives": tp,
        "false_negatives": fn,
    }


def _metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (scores >= float(threshold)).astype(np.int64)
    rates = _binary_rates(y_true, y_pred)
    return _metrics_from_counts(
        threshold=float(threshold),
        tp=int(rates["true_positives"]),
        fp=int(rates["false_positives"]),
        tn=int(rates["true_negatives"]),
        fn=int(rates["false_negatives"]),
    )


def _best_macro(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    thresholds, tp, fp, tn, fn = _threshold_curve(y_true, scores)
    pos_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(2 * tp + fp + fn) > 0)
    neg_f1 = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=np.float64), where=(2 * tn + fp + fn) > 0)
    macro_f1 = (pos_f1 + neg_f1) / 2.0
    best_idx = int(np.argmax(macro_f1))
    return _metrics_from_counts(
        threshold=float(thresholds[best_idx]),
        tp=int(tp[best_idx]),
        fp=int(fp[best_idx]),
        tn=int(tn[best_idx]),
        fn=int(fn[best_idx]),
    )


def _best_recall_under_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> dict[str, Any]:
    thresholds, tp, fp, tn, fn = _threshold_curve(y_true, scores)
    positives = max(int(np.sum(y_true == 1)), 1)
    negatives = max(int(np.sum(y_true == 0)), 1)
    fpr = fp / negatives
    recall = tp / positives
    eligible = np.flatnonzero(fpr <= float(max_fpr))
    if eligible.size:
        best_recall = np.max(recall[eligible])
        recall_tied = eligible[np.isclose(recall[eligible], best_recall)]
        best_idx = int(recall_tied[np.argmin(thresholds[recall_tied])])
    else:
        best_idx = 0
    best = _metrics_from_counts(
        threshold=float(thresholds[best_idx]),
        tp=int(tp[best_idx]),
        fp=int(fp[best_idx]),
        tn=int(tn[best_idx]),
        fn=int(fn[best_idx]),
    )
    best["max_fpr"] = float(max_fpr)
    return best


def _rank_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["auprc"] = float(average_precision_score(y_true, scores))
        fpr, tpr, _ = roc_curve(y_true, scores)
        eligible = fpr[tpr >= 0.95]
        out["fpr95"] = float(np.min(eligible)) if eligible.size else None
    except ValueError:
        out["auroc"] = None
        out["auprc"] = None
        out["fpr95"] = None
    return out


def _score_quantiles(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, name in [(0, "benign"), (1, "attack")]:
        vals = np.sort(scores[y_true == label])
        if vals.size == 0:
            continue
        out[f"{name}_score_min"] = float(vals[0])
        out[f"{name}_score_p50"] = float(np.quantile(vals, 0.50))
        out[f"{name}_score_p90"] = float(np.quantile(vals, 0.90))
        out[f"{name}_score_p95"] = float(np.quantile(vals, 0.95))
        out[f"{name}_score_p99"] = float(np.quantile(vals, 0.99))
        out[f"{name}_score_max"] = float(vals[-1])
    return out


def _val_percentile_metrics(y_true: np.ndarray, scores: np.ndarray, val_scores: np.ndarray, percentiles: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for percentile in percentiles:
        threshold = float(np.percentile(val_scores, percentile))
        metrics = _metrics_at_threshold(y_true, scores, threshold)
        prefix = f"val_p{str(percentile).replace('.', '_')}"
        out[f"{prefix}_threshold"] = threshold
        out[f"{prefix}_macro_f1"] = metrics.get("macro_f1")
        out[f"{prefix}_attack_recall"] = metrics.get("attack_recall")
        out[f"{prefix}_false_positive_rate"] = metrics.get("false_positive_rate")
        out[f"{prefix}_attack_precision"] = metrics.get("attack_precision")
    return out


def _evaluate(
    features: np.ndarray,
    feature_stats: dict[str, Any],
    group_values: list[str],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    feature_filter: str,
    transform: str,
    scorer: str,
    k: int,
    group_mode: str,
) -> dict[str, Any]:
    val_scores = _scores(features, train_idx, val_idx, group_values, scorer=scorer, k=k)
    test_scores = _scores(features, train_idx, test_idx, group_values, scorer=scorer, k=k)
    y_true = labels[test_idx].astype(np.int64)
    best = _best_macro(y_true, test_scores)
    r01 = _best_recall_under_fpr(y_true, test_scores, 0.001)
    r1 = _best_recall_under_fpr(y_true, test_scores, 0.01)
    r5 = _best_recall_under_fpr(y_true, test_scores, 0.05)
    r10 = _best_recall_under_fpr(y_true, test_scores, 0.10)
    rank = _rank_metrics(y_true, test_scores)
    row = {
        "feature_filter": feature_filter,
        "transform": transform,
        "scorer": scorer,
        "k": int(k),
        "group_mode": group_mode,
        "num_features": feature_stats["num_features"],
        "mean_nonzero": feature_stats["mean_nonzero"],
        "best_threshold": best["threshold"],
        "best_macro_f1": best["macro_f1"],
        "best_attack_recall": best["attack_recall"],
        "best_false_positive_rate": best["false_positive_rate"],
        "best_attack_precision": best["attack_precision"],
        "recall_at_0_1pct_fpr": r01["attack_recall"],
        "threshold_at_0_1pct_fpr": r01["threshold"],
        "actual_fpr_at_0_1pct_fpr": r01["false_positive_rate"],
        "recall_at_1pct_fpr": r1["attack_recall"],
        "threshold_at_1pct_fpr": r1["threshold"],
        "actual_fpr_at_1pct_fpr": r1["false_positive_rate"],
        "recall_at_5pct_fpr": r5["attack_recall"],
        "threshold_at_5pct_fpr": r5["threshold"],
        "actual_fpr_at_5pct_fpr": r5["false_positive_rate"],
        "recall_at_10pct_fpr": r10["attack_recall"],
        "threshold_at_10pct_fpr": r10["threshold"],
        "actual_fpr_at_10pct_fpr": r10["false_positive_rate"],
        **rank,
        **_score_quantiles(y_true, test_scores),
        **_val_percentile_metrics(y_true, test_scores, val_scores, [90.0, 95.0, 97.5, 99.0, 99.5, 100.0]),
    }
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_md(rows: list[dict[str, Any]], path: Path, token_path: str, title: str) -> None:
    top_macro = sorted(rows, key=lambda row: float(row.get("best_macro_f1") or -1.0), reverse=True)[:8]
    top_r01 = sorted(rows, key=lambda row: (float(row.get("recall_at_0_1pct_fpr") or -1.0), float(row.get("auroc") or -1.0)), reverse=True)[:8]
    top_r1 = sorted(rows, key=lambda row: (float(row.get("recall_at_1pct_fpr") or -1.0), float(row.get("auroc") or -1.0)), reverse=True)[:8]
    top_r5 = sorted(rows, key=lambda row: (float(row.get("recall_at_5pct_fpr") or -1.0), float(row.get("auroc") or -1.0)), reverse=True)[:8]
    top_val95 = sorted(rows, key=lambda row: (float(row.get("val_p95_0_macro_f1") or -1.0), float(row.get("val_p95_0_attack_recall") or -1.0)), reverse=True)[:8]

    def table(title: str, selected: list[dict[str, Any]], columns: list[str]) -> list[str]:
        lines = ["", f"## {title}", ""]
        headers = ["filter", "transform", "scorer", "k", "group"] + columns
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in selected:
            values = [
                f"`{row['feature_filter']}`",
                f"`{row['transform']}`",
                f"`{row['scorer']}`",
                str(row["k"]),
                f"`{row['group_mode']}`",
            ]
            values.extend(_fmt(row.get(col)) for col in columns)
            lines.append("| " + " | ".join(values) + " |")
        return lines

    lines = [
        f"# {title}",
        "",
        f"Token corpus: `{token_path}`.",
        "",
        "All settings fit only on the predefined train split, which contains BENIGN rows only. Test-set threshold optima are diagnostic upper bounds; validation-percentile rows use the BENIGN-only validation split and are closer to deployable calibration.",
    ]
    lines.extend(table("Top Test-Oracle Macro-F1", top_macro, ["best_macro_f1", "best_attack_recall", "best_false_positive_rate", "auroc", "auprc", "fpr95"]))
    lines.extend(table("Top Recall At 0.1% FPR", top_r01, ["recall_at_0_1pct_fpr", "actual_fpr_at_0_1pct_fpr", "auroc", "best_macro_f1"]))
    lines.extend(table("Top Recall At 1% FPR", top_r1, ["recall_at_1pct_fpr", "actual_fpr_at_1pct_fpr", "auroc", "best_macro_f1"]))
    lines.extend(table("Top Recall At 5% FPR", top_r5, ["recall_at_5pct_fpr", "actual_fpr_at_5pct_fpr", "auroc", "best_macro_f1"]))
    lines.extend(table("Top BENIGN-Val P95 Calibration", top_val95, ["val_p95_0_macro_f1", "val_p95_0_attack_recall", "val_p95_0_false_positive_rate", "val_p95_0_attack_precision"]))
    best = top_macro[0]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The strongest test-oracle Macro-F1 setting reaches {_fmt(best['best_macro_f1'])} with attack recall {_fmt(best['best_attack_recall'])} and FPR {_fmt(best['best_false_positive_rate'])}.",
            "This sweep should be treated as diagnostic: if Recall@1%FPR remains low, the bottleneck is score overlap between Botnet and the high-score BENIGN tail rather than threshold search alone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep train-benign anomaly scores for leave-one unknown attack low-FPR diagnosis.")
    parser.add_argument("--tokens", default="paper_icdm_applied_2026/experiments/unknown/tokens_canonical/cicids2017_leave_one_botnet_anomaly_seed42_a3_full_rhythm.pt")
    parser.add_argument("--out_prefix", default="experiments/ccfa_leave_one_botnet_anomaly_low_fpr_sweep_20260524")
    parser.add_argument("--title", default=None)
    parser.add_argument("--feature_filters", nargs="+", default=["all_no_special", "no_unk", "packet", "packet_burst", "packet_burst_profile", "no_profile_no_unk", "no_burst_no_profile_no_unk", "profile_only"])
    parser.add_argument("--transforms", nargs="+", default=["count_l2", "count_l1", "binary_l2", "tfidf_l2"])
    parser.add_argument("--scorers", nargs="+", default=["prototype_cosine", "prototype_euclidean", "knn_cosine", "knn_euclidean"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--group_modes", nargs="+", default=["global", "protocol"])
    args = parser.parse_args()

    token_data = _read_token_data(Path(args.tokens))
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = _split_indices(token_data, "train")
    val_idx = _split_indices(token_data, "val")
    test_idx = _split_indices(token_data, "test")
    if not np.all(labels[train_idx] == 0):
        raise ValueError("This sweep expects BENIGN-only train rows.")
    if not np.all(labels[val_idx] == 0):
        raise ValueError("This sweep expects BENIGN-only validation rows.")
    if sorted(set(labels[test_idx].tolist())) != [0, 1]:
        raise ValueError("This sweep expects a mixed BENIGN/ATTACK test split.")

    rows: list[dict[str, Any]] = []
    for feature_filter in args.feature_filters:
        for transform in args.transforms:
            features, feature_stats = _features(token_data, train_idx, feature_filter=feature_filter, transform=transform)
            for group_mode in args.group_modes:
                group_values = _groups(token_data, group_mode)
                for scorer in args.scorers:
                    k_values = args.ks if scorer.startswith("knn_") else [1]
                    for k in k_values:
                        rows.append(
                            _evaluate(
                                features,
                                feature_stats,
                                group_values,
                                labels,
                                train_idx,
                                val_idx,
                                test_idx,
                                feature_filter=feature_filter,
                                transform=transform,
                                scorer=scorer,
                                k=k,
                                group_mode=group_mode,
                            )
                        )

    out_prefix = Path(args.out_prefix)
    _write_csv(rows, out_prefix.with_suffix(".csv"))
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "tokens": args.tokens, "num_rows": len(rows)}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    title = args.title or "Leave-One Unknown Anomaly Low-FPR Sweep"
    _write_md(rows, out_prefix.with_suffix(".md"), args.tokens, title)
    print(json.dumps({"rows": len(rows), "out_prefix": str(out_prefix)}, sort_keys=True))


if __name__ == "__main__":
    main()
