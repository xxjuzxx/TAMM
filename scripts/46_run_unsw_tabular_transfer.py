#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import shlex
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.unsw_nb15_adapter import (
    cicids2017_binary_labels,
    cicids2017_label_column,
    cicids2017_shared_features,
    read_csv_paths,
    shared_feature_columns,
    unsw_binary_labels,
    unsw_family_labels,
    unsw_nb15_shared_features,
)
from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.utils.io import write_json
from src.utils.seed import set_seed


def _expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if any(char in item for char in "*?[]"):
            paths.extend(Path(match) for match in sorted(glob.glob(item)))
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*.csv")))
        else:
            paths.append(path)
    return paths


def _label_counts(y: np.ndarray, names: list[str]) -> dict[str, int]:
    counts = Counter(int(item) for item in y.tolist())
    return {names[idx]: int(counts.get(idx, 0)) for idx in range(len(names))}


def _sample_indices(
    y: np.ndarray,
    *,
    fraction: float | None,
    max_rows: int | None,
    balanced: bool,
    seed: int,
) -> np.ndarray:
    indices = np.arange(len(y), dtype=np.int64)
    if fraction is None and max_rows is None:
        return indices
    rng = np.random.default_rng(int(seed))
    selected_parts: list[np.ndarray] = []
    for label in sorted(set(y.tolist())):
        label_idx = indices[y == label]
        rng.shuffle(label_idx)
        if fraction is not None:
            take = int(round(len(label_idx) * float(fraction)))
            if len(label_idx) > 0:
                take = max(1, take)
        else:
            take = len(label_idx)
        selected_parts.append(label_idx[:take])
    selected = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=np.int64)
    if balanced and max_rows is not None and len(selected) > int(max_rows):
        per_label = max(1, int(max_rows) // max(1, len(set(y.tolist()))))
        balanced_parts: list[np.ndarray] = []
        for label in sorted(set(y[selected].tolist())):
            label_idx = selected[y[selected] == label]
            rng.shuffle(label_idx)
            balanced_parts.append(label_idx[:per_label])
        selected = np.concatenate(balanced_parts) if balanced_parts else np.array([], dtype=np.int64)
    if max_rows is not None and len(selected) > int(max_rows):
        rng.shuffle(selected)
        selected = selected[: int(max_rows)]
    return np.sort(selected)


def _make_xgb(num_rounds: int, seed: int, xgb_device: str | None, xgb_tree_method: str | None) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is required for D3 tabular transfer") from exc
    kwargs: dict[str, Any] = {
        "n_estimators": int(num_rounds),
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": int(seed),
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    if xgb_device:
        kwargs["device"] = xgb_device
    if xgb_tree_method:
        kwargs["tree_method"] = xgb_tree_method
    return XGBClassifier(**kwargs)


def _fit_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rounds: int,
    seed: int,
    xgb_device: str | None,
    xgb_tree_method: str | None,
    base_booster: Any | None = None,
) -> tuple[Pipeline, float]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _make_xgb(rounds, seed, xgb_device, xgb_tree_method)),
        ]
    )
    start = time.perf_counter()
    fit_kwargs = {}
    if base_booster is not None:
        fit_kwargs["model__xgb_model"] = base_booster
    pipeline.fit(x, y, **fit_kwargs)
    return pipeline, time.perf_counter() - start


def _aligned_proba(model: Pipeline, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 2:
        return proba
    aligned = np.zeros((proba.shape[0], 2), dtype=float)
    classes = model.named_steps["model"].classes_
    for idx, cls in enumerate(classes):
        aligned[:, int(cls)] = proba[:, idx]
    return aligned


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    for threshold in np.linspace(0.0, 1.0, 101):
        pred = (scores >= threshold).astype(int)
        metrics = classification_metrics(y_true.tolist(), pred.tolist(), np.column_stack([1.0 - scores, scores]))
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics or {}


def _threshold_at_max_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> tuple[float, dict[str, Any]]:
    candidates = sorted(set(float(item) for item in np.concatenate([scores, np.array([0.0, 0.5, 1.0])])), reverse=True)
    best_threshold = 1.0
    best_metrics: dict[str, Any] | None = None
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        tn = int(((y_true == 0) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        denom = tn + fp
        fpr = float(fp / denom) if denom else 0.0
        if fpr > float(max_fpr):
            continue
        metrics = classification_metrics(y_true.tolist(), pred.tolist(), np.column_stack([1.0 - scores, scores]))
        metrics["threshold_fpr"] = fpr
        if best_metrics is None or metrics["recall_macro"] > best_metrics["recall_macro"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        pred = (scores >= 1.0).astype(int)
        best_metrics = classification_metrics(y_true.tolist(), pred.tolist(), np.column_stack([1.0 - scores, scores]))
        best_metrics["threshold_fpr"] = 0.0
    return best_threshold, best_metrics


def _evaluate_binary(model: Pipeline, x: np.ndarray, y: np.ndarray, threshold: float | None = None) -> tuple[dict[str, Any], list[int], np.ndarray]:
    proba = _aligned_proba(model, x)
    pred = (proba[:, 1] >= (0.5 if threshold is None else float(threshold))).astype(int)
    return classification_metrics(y.tolist(), pred.tolist(), proba), pred.tolist(), proba


def _write_run(
    out_dir: Path,
    *,
    run_name: str,
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_pred: list[int],
    target_names: list[str],
    manifest: dict[str, Any],
) -> None:
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics, run_dir / "metrics.json")
    write_json(report_dict(y_true.tolist(), y_pred, target_names=target_names), run_dir / "classification_report.json")
    write_json(confusion(y_true.tolist(), y_pred), run_dir / "confusion_matrix.json")
    write_json(manifest, run_dir / "run_manifest.json")


def _load_cic(
    paths: list[Path],
    *,
    max_rows_per_file: int | None,
    attempted_policy: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, dict[str, Any]]:
    frame = read_csv_paths(paths, max_rows_per_file=max_rows_per_file)
    label_col = cicids2017_label_column(frame)
    y, labels, keep_mask = cicids2017_binary_labels(frame[label_col], attempted_policy=attempted_policy)
    filtered = frame.loc[keep_mask].copy()
    features = cicids2017_shared_features(filtered)
    stats = {
        "csv_files": [str(path) for path in paths],
        "label_column": label_col,
        "attempted_policy": attempted_policy,
        "num_rows_before_policy": int(len(frame)),
        "num_rows_after_policy": int(len(filtered)),
        "dropped_rows_by_label_policy": int((~keep_mask).sum()),
        "raw_label_counts": {str(k): int(v) for k, v in frame[label_col].map(str).value_counts().sort_index().items()},
        "resolved_label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
    }
    return features, y, labels, stats


def _load_unsw(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    features = unsw_nb15_shared_features(frame)
    y = unsw_binary_labels(frame)
    families = unsw_family_labels(frame)
    stats = {
        "csv_file": str(path),
        "num_rows": int(len(frame)),
        "attack_family_counts": dict(sorted(Counter(families).items())),
        "binary_counts": {"BENIGN": int((y == 0).sum()), "ATTACK": int((y == 1).sum())},
        "proto_counts": {str(k): int(v) for k, v in frame.get("proto", pd.Series([], dtype=object)).astype(str).value_counts().sort_index().items()},
        "service_counts": {str(k): int(v) for k, v in frame.get("service", pd.Series([], dtype=object)).astype(str).value_counts().sort_index().items()},
        "state_counts": {str(k): int(v) for k, v in frame.get("state", pd.Series([], dtype=object)).astype(str).value_counts().sort_index().items()},
    }
    return features, y, families, stats


def _domain_gap(
    *,
    cic_x: np.ndarray,
    cic_y: np.ndarray,
    unsw_train_x: np.ndarray,
    unsw_train_y: np.ndarray,
    unsw_test_x: np.ndarray,
    unsw_test_y: np.ndarray,
    seed: int,
    max_points_per_domain: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))

    def sample(x: np.ndarray, y: np.ndarray, domain: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
        take = min(int(max_points_per_domain), len(y))
        idx = np.arange(len(y), dtype=np.int64)
        rng.shuffle(idx)
        idx = np.sort(idx[:take])
        meta = [{"domain": domain, "label": int(y[row]), "row_index": int(row)} for row in idx.tolist()]
        return x[idx], meta

    cic_sample, cic_meta = sample(cic_x, cic_y, "CICIDS2017")
    unsw_train_sample, unsw_train_meta = sample(unsw_train_x, unsw_train_y, "UNSW_train")
    unsw_test_sample, unsw_test_meta = sample(unsw_test_x, unsw_test_y, "UNSW_test")
    x = np.vstack([cic_sample, unsw_train_sample, unsw_test_sample])
    meta = cic_meta + unsw_train_meta + unsw_test_meta
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=2, random_state=int(seed))
    coords = pca.fit_transform(x_scaled)
    for row, coord in zip(meta, coords):
        row["pc1"] = float(coord[0])
        row["pc2"] = float(coord[1])

    domain_names = sorted({row["domain"] for row in meta})
    centroids: dict[str, list[float]] = {}
    for domain in domain_names:
        domain_coords = coords[[idx for idx, row in enumerate(meta) if row["domain"] == domain]]
        centroids[domain] = [float(item) for item in domain_coords.mean(axis=0).tolist()]
    centroid_distances = {
        f"{left}_vs_{right}": float(np.linalg.norm(np.asarray(centroids[left]) - np.asarray(centroids[right])))
        for idx, left in enumerate(domain_names)
        for right in domain_names[idx + 1 :]
    }
    mmd_distances = {}
    for idx, left in enumerate(domain_names):
        left_coords = coords[[row_idx for row_idx, row in enumerate(meta) if row["domain"] == left]]
        for right in domain_names[idx + 1 :]:
            right_coords = coords[[row_idx for row_idx, row in enumerate(meta) if row["domain"] == right]]
            xx = np.exp(-pairwise_distances(left_coords, left_coords, metric="sqeuclidean") / 2.0).mean()
            yy = np.exp(-pairwise_distances(right_coords, right_coords, metric="sqeuclidean") / 2.0).mean()
            xy = np.exp(-pairwise_distances(left_coords, right_coords, metric="sqeuclidean") / 2.0).mean()
            mmd_distances[f"{left}_vs_{right}"] = float(xx + yy - 2.0 * xy)
    return {
        "method": "PCA_on_standardized_shared_tabular_features",
        "explained_variance_ratio": [float(item) for item in pca.explained_variance_ratio_.tolist()],
        "centroids": centroids,
        "centroid_distances": centroid_distances,
        "rbf_mmd_on_pca": mmd_distances,
        "points": meta,
        "note": "UMAP is not installed in the current environment; PCA is used as the first domain-gap diagnostic.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D3 CICIDS2017 -> UNSW-NB15 tabular transfer diagnostics.")
    parser.add_argument("--cic_csv", nargs="+", required=True)
    parser.add_argument("--unsw_train_csv", required=True)
    parser.add_argument("--unsw_test_csv", required=True)
    parser.add_argument("--out", default="outputs/results/ccfa_d3_unsw_tabular_transfer")
    parser.add_argument("--attempted_policy", choices=["keep", "drop", "attack", "benign"], default="drop")
    parser.add_argument("--max_cic_rows_per_file", type=int, default=None)
    parser.add_argument("--cic_train_fraction", type=float, default=None)
    parser.add_argument("--cic_max_train_rows", type=int, default=None)
    parser.add_argument("--balanced_cic_train", action="store_true")
    parser.add_argument("--base_rounds", type=int, default=300)
    parser.add_argument("--few_shot_rounds", type=int, default=80)
    parser.add_argument("--few_shot_scratch_rounds", type=int, default=300)
    parser.add_argument("--unsw_only_rounds", type=int, default=300)
    parser.add_argument("--calibrate_unsw_sanity", action="store_true")
    parser.add_argument("--few_shot_fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--calibrate_few_shot_thresholds", action="store_true")
    parser.add_argument("--calibration_fraction", type=float, default=0.2)
    parser.add_argument("--calibration_max_fpr", type=float, default=0.05)
    parser.add_argument("--pca_max_points_per_domain", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb_device", default=None)
    parser.add_argument("--xgb_tree_method", default="hist")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_names = ["BENIGN", "ATTACK"]

    cic_paths = _expand_inputs(args.cic_csv)
    cic_features, cic_y, cic_labels, cic_stats = _load_cic(
        cic_paths,
        max_rows_per_file=args.max_cic_rows_per_file,
        attempted_policy=args.attempted_policy,
    )
    cic_idx = _sample_indices(
        cic_y,
        fraction=args.cic_train_fraction,
        max_rows=args.cic_max_train_rows,
        balanced=bool(args.balanced_cic_train),
        seed=args.seed,
    )
    cic_x = cic_features.to_numpy(dtype=np.float32)
    cic_train_x = cic_x[cic_idx]
    cic_train_y = cic_y[cic_idx]

    unsw_train_features, unsw_train_y, unsw_train_families, unsw_train_stats = _load_unsw(Path(args.unsw_train_csv))
    unsw_test_features, unsw_test_y, unsw_test_families, unsw_test_stats = _load_unsw(Path(args.unsw_test_csv))
    unsw_train_x = unsw_train_features.to_numpy(dtype=np.float32)
    unsw_test_x = unsw_test_features.to_numpy(dtype=np.float32)

    run_manifest_base = {
        "command": shlex.join(sys.argv),
        "script": Path(__file__).name,
        "feature_columns": shared_feature_columns(),
        "target_names": target_names,
        "seed": int(args.seed),
        "xgb_device": args.xgb_device,
        "xgb_tree_method": args.xgb_tree_method,
    }

    base_model, base_seconds = _fit_pipeline(
        cic_train_x,
        cic_train_y,
        rounds=args.base_rounds,
        seed=args.seed,
        xgb_device=args.xgb_device,
        xgb_tree_method=args.xgb_tree_method,
    )
    base_metrics, base_pred, base_proba = _evaluate_binary(base_model, unsw_test_x, unsw_test_y)
    base_metrics.update(
        {
            "section": "D3_UNSW",
            "setting": "zero_shot_cicids2017_to_unsw",
            "task": "binary",
            "model": "XGBoost",
            "feature_adapter": "shared_tabular_semantics_v1",
            "train_dataset": "CICIDS2017_corrected",
            "test_dataset": "UNSW-NB15",
            "num_train": int(len(cic_train_y)),
            "num_external_test": int(len(unsw_test_y)),
            "num_features": int(cic_train_x.shape[1]),
            "threshold": 0.5,
            "train_seconds": float(base_seconds),
        }
    )
    _write_run(
        out_dir,
        run_name="zero_shot_cicids2017_to_unsw",
        metrics=base_metrics,
        y_true=unsw_test_y,
        y_pred=base_pred,
        target_names=target_names,
        manifest={
            **run_manifest_base,
            "cic_stats": cic_stats,
            "cic_train_selected": {
                "num_rows": int(len(cic_train_y)),
                "binary_counts": _label_counts(cic_train_y, target_names),
                "fraction": args.cic_train_fraction,
                "max_rows": args.cic_max_train_rows,
                "balanced": bool(args.balanced_cic_train),
            },
            "unsw_train_stats": unsw_train_stats,
            "unsw_test_stats": unsw_test_stats,
            "base_rounds": int(args.base_rounds),
        },
    )

    train_threshold, train_threshold_metrics = _best_threshold(unsw_train_y, _aligned_proba(base_model, unsw_train_x)[:, 1])
    calibrated_metrics, calibrated_pred, _ = _evaluate_binary(base_model, unsw_test_x, unsw_test_y, threshold=train_threshold)
    calibrated_metrics.update(
        {
            "section": "D3_UNSW",
            "setting": "zero_shot_cicids2017_to_unsw_unsw_train_threshold",
            "task": "binary",
            "model": "XGBoost",
            "feature_adapter": "shared_tabular_semantics_v1",
            "train_dataset": "CICIDS2017_corrected",
            "threshold_calibration_dataset": "UNSW-NB15_train_labels",
            "test_dataset": "UNSW-NB15",
            "num_train": int(len(cic_train_y)),
            "num_threshold_calibration": int(len(unsw_train_y)),
            "num_external_test": int(len(unsw_test_y)),
            "threshold": float(train_threshold),
            "threshold_calibration_macro_f1": train_threshold_metrics.get("macro_f1"),
            "train_seconds": float(base_seconds),
        }
    )
    _write_run(
        out_dir,
        run_name="zero_shot_cicids2017_to_unsw_train_threshold",
        metrics=calibrated_metrics,
        y_true=unsw_test_y,
        y_pred=calibrated_pred,
        target_names=target_names,
        manifest={**run_manifest_base, "threshold": float(train_threshold), "calibration_metrics": train_threshold_metrics},
    )

    unsw_model, unsw_seconds = _fit_pipeline(
        unsw_train_x,
        unsw_train_y,
        rounds=args.unsw_only_rounds,
        seed=args.seed,
        xgb_device=args.xgb_device,
        xgb_tree_method=args.xgb_tree_method,
    )
    unsw_metrics, unsw_pred, _ = _evaluate_binary(unsw_model, unsw_test_x, unsw_test_y)
    unsw_metrics.update(
        {
            "section": "D3_UNSW",
            "setting": "unsw_train_to_unsw_test_sanity",
            "task": "binary",
            "model": "XGBoost",
            "feature_adapter": "shared_tabular_semantics_v1",
            "train_dataset": "UNSW-NB15_train",
            "test_dataset": "UNSW-NB15_test",
            "num_train": int(len(unsw_train_y)),
            "num_external_test": int(len(unsw_test_y)),
            "num_features": int(unsw_train_x.shape[1]),
            "threshold": 0.5,
            "train_seconds": float(unsw_seconds),
        }
    )
    _write_run(
        out_dir,
        run_name="unsw_train_to_unsw_test_sanity",
        metrics=unsw_metrics,
        y_true=unsw_test_y,
        y_pred=unsw_pred,
        target_names=target_names,
        manifest={**run_manifest_base, "unsw_train_stats": unsw_train_stats, "unsw_test_stats": unsw_test_stats},
    )
    unsw_calibrated_rows: list[dict[str, Any]] = []
    if args.calibrate_unsw_sanity:
        rng = np.random.default_rng(int(args.seed))
        full_train_parts: list[np.ndarray] = []
        full_calibration_parts: list[np.ndarray] = []
        all_unsw_idx = np.arange(len(unsw_train_y), dtype=np.int64)
        for label in sorted(set(unsw_train_y.tolist())):
            label_idx = all_unsw_idx[unsw_train_y == label]
            rng.shuffle(label_idx)
            cal_n = int(round(len(label_idx) * float(args.calibration_fraction)))
            if len(label_idx) >= 2:
                cal_n = max(1, min(cal_n, len(label_idx) - 1))
            full_calibration_parts.append(label_idx[:cal_n])
            full_train_parts.append(label_idx[cal_n:])
        full_train_idx = np.sort(np.concatenate(full_train_parts))
        full_calibration_idx = np.sort(np.concatenate(full_calibration_parts))
        full_model, full_seconds = _fit_pipeline(
            unsw_train_x[full_train_idx],
            unsw_train_y[full_train_idx],
            rounds=args.unsw_only_rounds,
            seed=args.seed,
            xgb_device=args.xgb_device,
            xgb_tree_method=args.xgb_tree_method,
        )
        full_scores = _aligned_proba(full_model, unsw_train_x[full_calibration_idx])[:, 1]
        for policy_name, threshold, cal_metrics in (
            ("cal_macro", *_best_threshold(unsw_train_y[full_calibration_idx], full_scores)),
            (
                f"cal_fpr{int(round(args.calibration_max_fpr * 100)):02d}",
                *_threshold_at_max_fpr(unsw_train_y[full_calibration_idx], full_scores, args.calibration_max_fpr),
            ),
        ):
            eval_metrics, eval_pred, _ = _evaluate_binary(full_model, unsw_test_x, unsw_test_y, threshold=threshold)
            run_name = f"unsw_train_to_unsw_test_sanity_{policy_name}"
            eval_metrics.update(
                {
                    "section": "D3_UNSW",
                    "setting": run_name,
                    "task": "binary",
                    "model": "XGBoost",
                    "feature_adapter": "shared_tabular_semantics_v1",
                    "train_dataset": "UNSW-NB15_train_subset",
                    "threshold_calibration_dataset": "UNSW-NB15_train_subset",
                    "test_dataset": "UNSW-NB15_test",
                    "num_train": int(len(full_train_idx)),
                    "num_threshold_calibration": int(len(full_calibration_idx)),
                    "num_external_test": int(len(unsw_test_y)),
                    "threshold_policy": policy_name,
                    "threshold": float(threshold),
                    "threshold_calibration_macro_f1": cal_metrics.get("macro_f1"),
                    "threshold_calibration_fpr": cal_metrics.get("threshold_fpr"),
                    "num_features": int(unsw_train_x.shape[1]),
                    "train_seconds": float(full_seconds),
                }
            )
            unsw_calibrated_rows.append(
                {
                    "setting": run_name,
                    "num_train": int(len(full_train_idx)),
                    "num_threshold_calibration": int(len(full_calibration_idx)),
                    "num_test": int(len(unsw_test_y)),
                    "threshold": float(threshold),
                    "accuracy": eval_metrics.get("accuracy"),
                    "macro_f1": eval_metrics.get("macro_f1"),
                    "weighted_f1": eval_metrics.get("weighted_f1"),
                    "auroc": eval_metrics.get("auroc"),
                    "auprc": eval_metrics.get("auprc"),
                    "fpr95": eval_metrics.get("fpr95"),
                    "path": str(out_dir / run_name),
                }
            )
            _write_run(
                out_dir,
                run_name=run_name,
                metrics=eval_metrics,
                y_true=unsw_test_y,
                y_pred=eval_pred,
                target_names=target_names,
                manifest={
                    **run_manifest_base,
                    "unsw_train_stats": unsw_train_stats,
                    "unsw_test_stats": unsw_test_stats,
                    "threshold_policy": policy_name,
                    "threshold": float(threshold),
                    "calibration_metrics": cal_metrics,
                    "train_indices": full_train_idx.tolist(),
                    "calibration_indices": full_calibration_idx.tolist(),
                },
            )

    base_booster = base_model.named_steps["model"].get_booster()
    few_rows: list[dict[str, Any]] = []
    for fraction in args.few_shot_fractions:
        few_all_idx = _sample_indices(unsw_train_y, fraction=float(fraction), max_rows=None, balanced=False, seed=args.seed)
        rng = np.random.default_rng(int(args.seed))
        train_parts: list[np.ndarray] = []
        calibration_parts: list[np.ndarray] = []
        for label in sorted(set(unsw_train_y[few_all_idx].tolist())):
            label_idx = few_all_idx[unsw_train_y[few_all_idx] == label]
            rng.shuffle(label_idx)
            cal_n = int(round(len(label_idx) * float(args.calibration_fraction))) if args.calibrate_few_shot_thresholds else 0
            if args.calibrate_few_shot_thresholds and len(label_idx) >= 2:
                cal_n = max(1, min(cal_n, len(label_idx) - 1))
            calibration_parts.append(label_idx[:cal_n])
            train_parts.append(label_idx[cal_n:])
        calibration_idx = np.sort(np.concatenate(calibration_parts)) if calibration_parts else np.array([], dtype=np.int64)
        few_idx = np.sort(np.concatenate(train_parts)) if train_parts else few_all_idx
        few_x = unsw_train_x[few_idx]
        few_y = unsw_train_y[few_idx]
        calibration_x = unsw_train_x[calibration_idx] if len(calibration_idx) else np.empty((0, unsw_train_x.shape[1]), dtype=np.float32)
        calibration_y = unsw_train_y[calibration_idx] if len(calibration_idx) else np.array([], dtype=np.int64)
        few_model, few_seconds = _fit_pipeline(
            few_x,
            few_y,
            rounds=args.few_shot_rounds,
            seed=args.seed,
            xgb_device=args.xgb_device,
            xgb_tree_method=args.xgb_tree_method,
            base_booster=base_booster,
        )
        few_metrics, few_pred, _ = _evaluate_binary(few_model, unsw_test_x, unsw_test_y)
        run_name = f"few_shot_unsw_{int(round(float(fraction) * 100)):02d}pct_warm_start"
        few_metrics.update(
            {
                "section": "D3_UNSW",
                "setting": run_name,
                "task": "binary",
                "model": "XGBoost_warm_start",
                "feature_adapter": "shared_tabular_semantics_v1",
                "base_train_dataset": "CICIDS2017_corrected",
                "few_shot_dataset": "UNSW-NB15_train",
                "test_dataset": "UNSW-NB15_test",
                "few_shot_fraction": float(fraction),
                "num_base_train": int(len(cic_train_y)),
                "num_few_shot_train": int(len(few_y)),
                "num_few_shot_calibration": int(len(calibration_y)),
                "num_external_test": int(len(unsw_test_y)),
                "num_features": int(few_x.shape[1]),
                "threshold": 0.5,
                "base_rounds": int(args.base_rounds),
                "few_shot_rounds": int(args.few_shot_rounds),
                "train_seconds": float(base_seconds + few_seconds),
                "few_shot_seconds": float(few_seconds),
                "num_boosted_rounds": int(few_model.named_steps["model"].get_booster().num_boosted_rounds()),
            }
        )
        few_rows.append(
            {
                "setting": run_name,
                "fraction": float(fraction),
                "num_few_shot_train": int(len(few_y)),
                "num_test": int(len(unsw_test_y)),
                "accuracy": few_metrics.get("accuracy"),
                "macro_f1": few_metrics.get("macro_f1"),
                "weighted_f1": few_metrics.get("weighted_f1"),
                "auroc": few_metrics.get("auroc"),
                "auprc": few_metrics.get("auprc"),
                "fpr95": few_metrics.get("fpr95"),
                "path": str(out_dir / run_name),
            }
        )
        _write_run(
            out_dir,
            run_name=run_name,
            metrics=few_metrics,
            y_true=unsw_test_y,
            y_pred=few_pred,
            target_names=target_names,
            manifest={
                **run_manifest_base,
                "few_shot_fraction": float(fraction),
                "few_shot_indices": few_idx.tolist(),
                "few_shot_calibration_indices": calibration_idx.tolist(),
                "few_shot_binary_counts": _label_counts(few_y, target_names),
                "few_shot_calibration_binary_counts": _label_counts(calibration_y, target_names) if len(calibration_y) else {},
                "base_rounds": int(args.base_rounds),
                "few_shot_rounds": int(args.few_shot_rounds),
                "warm_start": True,
            },
        )
        scratch_model, scratch_seconds = _fit_pipeline(
            few_x,
            few_y,
            rounds=args.few_shot_scratch_rounds,
            seed=args.seed,
            xgb_device=args.xgb_device,
            xgb_tree_method=args.xgb_tree_method,
        )
        scratch_metrics, scratch_pred, _ = _evaluate_binary(scratch_model, unsw_test_x, unsw_test_y)
        scratch_name = f"few_shot_unsw_{int(round(float(fraction) * 100)):02d}pct_scratch"
        scratch_metrics.update(
            {
                "section": "D3_UNSW",
                "setting": scratch_name,
                "task": "binary",
                "model": "XGBoost_scratch",
                "feature_adapter": "shared_tabular_semantics_v1",
                "train_dataset": "UNSW-NB15_train",
                "test_dataset": "UNSW-NB15_test",
                "few_shot_fraction": float(fraction),
                "num_few_shot_train": int(len(few_y)),
                "num_few_shot_calibration": int(len(calibration_y)),
                "num_external_test": int(len(unsw_test_y)),
                "num_features": int(few_x.shape[1]),
                "threshold": 0.5,
                "few_shot_scratch_rounds": int(args.few_shot_scratch_rounds),
                "train_seconds": float(scratch_seconds),
            }
        )
        few_rows.append(
            {
                "setting": scratch_name,
                "fraction": float(fraction),
                "num_few_shot_train": int(len(few_y)),
                "num_test": int(len(unsw_test_y)),
                "accuracy": scratch_metrics.get("accuracy"),
                "macro_f1": scratch_metrics.get("macro_f1"),
                "weighted_f1": scratch_metrics.get("weighted_f1"),
                "auroc": scratch_metrics.get("auroc"),
                "auprc": scratch_metrics.get("auprc"),
                "fpr95": scratch_metrics.get("fpr95"),
                "path": str(out_dir / scratch_name),
            }
        )
        _write_run(
            out_dir,
            run_name=scratch_name,
            metrics=scratch_metrics,
            y_true=unsw_test_y,
            y_pred=scratch_pred,
            target_names=target_names,
            manifest={
                **run_manifest_base,
                "few_shot_fraction": float(fraction),
                "few_shot_indices": few_idx.tolist(),
                "few_shot_calibration_indices": calibration_idx.tolist(),
                "few_shot_binary_counts": _label_counts(few_y, target_names),
                "few_shot_calibration_binary_counts": _label_counts(calibration_y, target_names) if len(calibration_y) else {},
                "few_shot_scratch_rounds": int(args.few_shot_scratch_rounds),
                "warm_start": False,
            },
        )
        if args.calibrate_few_shot_thresholds and len(calibration_y):
            for model_name, model, base_run_name in (
                ("warm_start", few_model, run_name),
                ("scratch", scratch_model, scratch_name),
            ):
                cal_scores = _aligned_proba(model, calibration_x)[:, 1]
                best_macro_threshold, best_macro_cal = _best_threshold(calibration_y, cal_scores)
                fpr_threshold, fpr_cal = _threshold_at_max_fpr(calibration_y, cal_scores, args.calibration_max_fpr)
                for policy_name, threshold, cal_metrics in (
                    ("cal_macro", best_macro_threshold, best_macro_cal),
                    (f"cal_fpr{int(round(args.calibration_max_fpr * 100)):02d}", fpr_threshold, fpr_cal),
                ):
                    eval_metrics, eval_pred, _ = _evaluate_binary(model, unsw_test_x, unsw_test_y, threshold=threshold)
                    calibrated_name = f"{base_run_name}_{policy_name}"
                    eval_metrics.update(
                        {
                            "section": "D3_UNSW",
                            "setting": calibrated_name,
                            "task": "binary",
                            "model": f"XGBoost_{model_name}",
                            "feature_adapter": "shared_tabular_semantics_v1",
                            "threshold_policy": policy_name,
                            "threshold": float(threshold),
                            "threshold_calibration_macro_f1": cal_metrics.get("macro_f1"),
                            "threshold_calibration_fpr": cal_metrics.get("threshold_fpr"),
                            "few_shot_fraction": float(fraction),
                            "num_few_shot_train": int(len(few_y)),
                            "num_few_shot_calibration": int(len(calibration_y)),
                            "num_external_test": int(len(unsw_test_y)),
                        }
                    )
                    few_rows.append(
                        {
                            "setting": calibrated_name,
                            "fraction": float(fraction),
                            "num_few_shot_train": int(len(few_y)),
                            "num_few_shot_calibration": int(len(calibration_y)),
                            "num_test": int(len(unsw_test_y)),
                            "threshold": float(threshold),
                            "accuracy": eval_metrics.get("accuracy"),
                            "macro_f1": eval_metrics.get("macro_f1"),
                            "weighted_f1": eval_metrics.get("weighted_f1"),
                            "auroc": eval_metrics.get("auroc"),
                            "auprc": eval_metrics.get("auprc"),
                            "fpr95": eval_metrics.get("fpr95"),
                            "path": str(out_dir / calibrated_name),
                        }
                    )
                    _write_run(
                        out_dir,
                        run_name=calibrated_name,
                        metrics=eval_metrics,
                        y_true=unsw_test_y,
                        y_pred=eval_pred,
                        target_names=target_names,
                        manifest={
                            **run_manifest_base,
                            "base_run": base_run_name,
                            "threshold_policy": policy_name,
                            "threshold": float(threshold),
                            "calibration_metrics": cal_metrics,
                            "few_shot_fraction": float(fraction),
                            "few_shot_indices": few_idx.tolist(),
                            "few_shot_calibration_indices": calibration_idx.tolist(),
                            "few_shot_binary_counts": _label_counts(few_y, target_names),
                            "few_shot_calibration_binary_counts": _label_counts(calibration_y, target_names),
                        },
                    )

    domain_gap = _domain_gap(
        cic_x=cic_train_x,
        cic_y=cic_train_y,
        unsw_train_x=unsw_train_x,
        unsw_train_y=unsw_train_y,
        unsw_test_x=unsw_test_x,
        unsw_test_y=unsw_test_y,
        seed=args.seed,
        max_points_per_domain=args.pca_max_points_per_domain,
    )
    write_json(domain_gap, out_dir / "domain_gap_pca.json")

    summary_rows = [
        {
            "setting": "zero_shot_cicids2017_to_unsw",
            "num_train": int(len(cic_train_y)),
            "num_test": int(len(unsw_test_y)),
            "accuracy": base_metrics.get("accuracy"),
            "macro_f1": base_metrics.get("macro_f1"),
            "weighted_f1": base_metrics.get("weighted_f1"),
            "auroc": base_metrics.get("auroc"),
            "auprc": base_metrics.get("auprc"),
            "fpr95": base_metrics.get("fpr95"),
            "path": str(out_dir / "zero_shot_cicids2017_to_unsw"),
        },
        {
            "setting": "zero_shot_cicids2017_to_unsw_train_threshold",
            "num_train": int(len(cic_train_y)),
            "num_threshold_calibration": int(len(unsw_train_y)),
            "num_test": int(len(unsw_test_y)),
            "threshold": float(train_threshold),
            "accuracy": calibrated_metrics.get("accuracy"),
            "macro_f1": calibrated_metrics.get("macro_f1"),
            "weighted_f1": calibrated_metrics.get("weighted_f1"),
            "auroc": calibrated_metrics.get("auroc"),
            "auprc": calibrated_metrics.get("auprc"),
            "fpr95": calibrated_metrics.get("fpr95"),
            "path": str(out_dir / "zero_shot_cicids2017_to_unsw_train_threshold"),
        },
        {
            "setting": "unsw_train_to_unsw_test_sanity",
            "num_train": int(len(unsw_train_y)),
            "num_test": int(len(unsw_test_y)),
            "accuracy": unsw_metrics.get("accuracy"),
            "macro_f1": unsw_metrics.get("macro_f1"),
            "weighted_f1": unsw_metrics.get("weighted_f1"),
            "auroc": unsw_metrics.get("auroc"),
            "auprc": unsw_metrics.get("auprc"),
            "fpr95": unsw_metrics.get("fpr95"),
            "path": str(out_dir / "unsw_train_to_unsw_test_sanity"),
        },
        *unsw_calibrated_rows,
        *few_rows,
    ]
    write_json(
        {
            "rows": summary_rows,
            "cic_stats": cic_stats,
            "cic_train_selected_counts": _label_counts(cic_train_y, target_names),
            "unsw_train_stats": unsw_train_stats,
            "unsw_test_stats": unsw_test_stats,
            "domain_gap": {
                key: value
                for key, value in domain_gap.items()
                if key != "points"
            },
        },
        out_dir / "summary.json",
    )

    md_lines = [
        "# D3 UNSW-NB15 Tabular Transfer",
        "",
        "| Setting | Train N | Test N | Accuracy | Macro-F1 | Weighted-F1 | AUROC | AUPRC | FPR95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['setting']}`",
                    str(row.get("num_few_shot_train", row.get("num_train", "-"))),
                    str(row.get("num_test", len(unsw_test_y))),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("macro_f1")),
                    _fmt(row.get("weighted_f1")),
                    _fmt(row.get("auroc")),
                    _fmt(row.get("auprc")),
                    _fmt(row.get("fpr95")),
                ]
            )
            + " |"
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a D3 heterogeneous-transfer diagnostic using shared tabular semantics, not packet-token transfer.",
            "- CSE-CIC-IDS2018 is intentionally not used in this run.",
            "- PCA is used for the domain-gap diagnostic because UMAP is unavailable in the current environment.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "rows": summary_rows}, sort_keys=True))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


if __name__ == "__main__":
    main()
