#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data.label_policy import (
    ATTEMPTED_POLICIES,
    AttemptedPolicy,
    apply_attempted_policy,
    binary_label_for,
    merged_cicids_label,
    normalize_label,
)
from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.training.classifier_trainer import _split_indices, _temporal_stratified_by_group_indices
from src.utils.io import write_json


LABEL_CANDIDATES = {"label", " labels", " Label", "Label"}
DEFAULT_DROP_COLUMNS = {
    "id",
    "flow id",
    "source ip",
    "src ip",
    "destination ip",
    "dst ip",
    "source port",
    "src port",
    "destination port",
    "dst port",
    "timestamp",
    "simillarhttp",
}


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


def _read_csvs(paths: list[Path], max_rows_per_file: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, nrows=max_rows_per_file, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame["__source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No CICFlowMeter CSV files were found")
    return pd.concat(frames, ignore_index=True, sort=False)


def _label_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if column.strip().lower() == "label":
            return column
    raise ValueError(f"Could not find a Label column in columns: {list(frame.columns)[:20]}")


def _labels(raw_labels: pd.Series, task: str) -> tuple[np.ndarray, list[str]]:
    labels = raw_labels.map(normalize_label)
    if task == "binary":
        encoded = labels.map(lambda item: 0 if binary_label_for(item) == "BENIGN" else 1).astype(int).to_numpy()
        return encoded, ["BENIGN", "ATTACK"]
    if task == "multiclass_merged":
        labels = labels.map(merged_cicids_label)
    names = sorted(labels.unique().tolist())
    if "BENIGN" in names:
        names = ["BENIGN"] + [name for name in names if name != "BENIGN"]
    elif "Benign" in names:
        names = ["Benign"] + [name for name in names if name != "Benign"]
    mapping = {name: idx for idx, name in enumerate(names)}
    encoded = labels.map(mapping).astype(int).to_numpy()
    return encoded, names


def _raw_label_group_ids(raw_labels: pd.Series) -> np.ndarray:
    labels = raw_labels.map(normalize_label).astype(str).to_numpy()
    mapping = {label: idx for idx, label in enumerate(sorted(set(labels.tolist())))}
    return np.array([mapping[label] for label in labels], dtype=np.int64)


def _apply_label_policy(frame: pd.DataFrame, label_col: str, attempted_policy: AttemptedPolicy) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    raw_labels = frame[label_col].map(normalize_label)
    resolved = raw_labels.map(lambda item: apply_attempted_policy(item, attempted_policy))
    keep_mask = resolved.notna()
    filtered = frame.loc[keep_mask].copy()
    labels = resolved.loc[keep_mask].map(normalize_label)
    if filtered.empty:
        raise ValueError(f"No rows remain after applying attempted_policy={attempted_policy}")
    stats = {
        "label_column": label_col,
        "attempted_policy": attempted_policy,
        "raw_label_counts": {str(label): int(count) for label, count in raw_labels.value_counts().sort_index().items()},
        "resolved_label_counts": {str(label): int(count) for label, count in labels.value_counts().sort_index().items()},
        "dropped_rows": int((~keep_mask).sum()),
    }
    return filtered, labels, stats


def _feature_frame(frame: pd.DataFrame, label_col: str, keep_identifiers: bool) -> pd.DataFrame:
    drop_columns = {label_col, "__source_file"}
    if not keep_identifiers:
        drop_columns.update(column for column in frame.columns if column.strip().lower() in DEFAULT_DROP_COLUMNS)
    features = frame.drop(columns=[column for column in drop_columns if column in frame.columns], errors="ignore")
    for column in features.columns:
        if features[column].dtype == object:
            features[column] = pd.to_numeric(features[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    features = features.select_dtypes(include=[np.number])
    features = features.replace([np.inf, -np.inf], np.nan)
    if features.empty:
        raise ValueError("No numeric CICFlowMeter feature columns remain after preprocessing")
    return features


def _order_values(frame: pd.DataFrame) -> np.ndarray:
    timestamp_col = None
    for column in frame.columns:
        if column.strip().lower() == "timestamp":
            timestamp_col = column
            break
    if timestamp_col is None:
        return np.arange(len(frame), dtype=float)
    parsed = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if parsed.notna().any():
        values = parsed.astype("int64").to_numpy(dtype=float)
        values[parsed.isna().to_numpy()] = np.arange(len(frame), dtype=float)[parsed.isna().to_numpy()]
        return values
    return pd.to_numeric(frame[timestamp_col], errors="coerce").fillna(np.arange(len(frame))).to_numpy(dtype=float)


def _build_model(kind: str, num_classes: int, seed: int, n_estimators: int, xgb_device: str | None = None, xgb_tree_method: str | None = None) -> Any:
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed; use --model random_forest or install xgboost") from exc
        objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
        kwargs: dict[str, Any] = {
            "n_estimators": n_estimators,
            "random_state": seed,
            "n_jobs": -1,
            "objective": objective,
            "eval_metric": "logloss" if num_classes == 2 else "mlogloss",
        }
        if xgb_device:
            kwargs["device"] = xgb_device
        if xgb_tree_method:
            kwargs["tree_method"] = xgb_tree_method
        if num_classes > 2:
            kwargs["num_class"] = num_classes
        return XGBClassifier(**kwargs)
    raise ValueError(f"Unsupported model: {kind}")


def _aligned_proba(model: Any, x: np.ndarray, num_classes: int) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == num_classes:
        return proba
    aligned = np.zeros((proba.shape[0], num_classes), dtype=float)
    for idx, cls in enumerate(model.classes_):
        aligned[:, int(cls)] = proba[:, idx]
    return aligned


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    if len(set(y_true.tolist())) < 2:
        return classification_metrics(y_true.tolist(), y_pred.tolist(), None)
    return classification_metrics(y_true.tolist(), y_pred.tolist(), y_score)


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    if len(set(y_true.tolist())) < 2:
        pred = (scores >= 0.5).astype(int)
        return 0.5, _safe_metrics(y_true, pred, np.column_stack([1.0 - scores, scores]))
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    for threshold in np.linspace(0.0, 1.0, 101):
        pred = (scores >= threshold).astype(int)
        metrics = _safe_metrics(y_true, pred, np.column_stack([1.0 - scores, scores]))
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics or {}


def _evaluate(model: Any, x: np.ndarray, y: np.ndarray, num_classes: int, threshold: float | None) -> tuple[dict[str, Any], list[int], np.ndarray]:
    proba = _aligned_proba(model, x, num_classes)
    if num_classes == 2 and threshold is not None:
        pred = (proba[:, 1] >= threshold).astype(int)
    else:
        pred = np.argmax(proba, axis=1)
    metrics = _safe_metrics(y, pred, proba)
    return metrics, pred.tolist(), proba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="CICFlowMeter CSV files, directories, or glob patterns")
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", choices=["binary", "multiclass", "multiclass_merged"], default="binary")
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default="random_forest")
    parser.add_argument(
        "--split",
        choices=["stratified", "chronological", "temporal_stratified", "temporal_stratified_raw_label"],
        default="temporal_stratified",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_rows_per_file", type=int, default=None)
    parser.add_argument("--keep_identifiers", action="store_true")
    parser.add_argument("--attempted_policy", choices=ATTEMPTED_POLICIES, default="keep")
    parser.add_argument("--xgb_device", default=None, help="Optional XGBoost device, e.g. cuda or cpu")
    parser.add_argument("--xgb_tree_method", default=None, help="Optional XGBoost tree_method, e.g. hist")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _expand_inputs(args.csv)
    frame = _read_csvs(paths, max_rows_per_file=args.max_rows_per_file)
    label_col = _label_column(frame)
    frame, labels, label_policy_stats = _apply_label_policy(frame, label_col, args.attempted_policy)
    y, target_names = _labels(labels, args.task)
    features = _feature_frame(frame, label_col=label_col, keep_identifiers=args.keep_identifiers)
    order_values = _order_values(frame)
    if args.split == "temporal_stratified_raw_label":
        train_idx, val_idx, test_idx = _temporal_stratified_by_group_indices(
            y,
            _raw_label_group_ids(labels),
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            order_values=order_values,
        )
    else:
        train_idx, val_idx, test_idx = _split_indices(
            y,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            split=args.split,
            order_values=order_values,
        )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _build_model(args.model, len(target_names), args.seed, args.n_estimators, args.xgb_device, args.xgb_tree_method)),
        ]
    )
    start = time.perf_counter()
    x = features.to_numpy(dtype=np.float32)
    pipeline.fit(x[train_idx], y[train_idx])
    train_seconds = time.perf_counter() - start

    threshold = None
    val_metrics: dict[str, Any]
    val_pred: list[int]
    val_proba: np.ndarray
    if len(target_names) == 2:
        val_proba_for_threshold = _aligned_proba(pipeline, x[val_idx], len(target_names))
        threshold, val_metrics = _best_threshold(y[val_idx], val_proba_for_threshold[:, 1])
        val_metrics, val_pred, val_proba = _evaluate(pipeline, x[val_idx], y[val_idx], len(target_names), threshold)
    else:
        val_metrics, val_pred, val_proba = _evaluate(pipeline, x[val_idx], y[val_idx], len(target_names), None)

    test_start = time.perf_counter()
    test_metrics, test_pred, test_proba = _evaluate(pipeline, x[test_idx], y[test_idx], len(target_names), threshold)
    test_seconds = time.perf_counter() - test_start
    test_metrics.update(
        {
            "task": args.task,
            "model": args.model,
            "num_rows": int(len(frame)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "num_features": int(features.shape[1]),
            "target_names": target_names,
            "threshold": threshold,
            "split": args.split,
            "seed": args.seed,
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
            "test_flows_per_second": float(len(test_idx) / test_seconds) if test_seconds > 0 else None,
            "csv_files": [str(path) for path in paths],
            "keep_identifiers": bool(args.keep_identifiers),
            "attempted_policy": args.attempted_policy,
            "dropped_rows_by_label_policy": int(label_policy_stats["dropped_rows"]),
            "xgb_device": args.xgb_device,
            "xgb_tree_method": args.xgb_tree_method,
        }
    )

    write_json(test_metrics, out_dir / "metrics.json")
    write_json(val_metrics, out_dir / "val_metrics.json")
    write_json(report_dict(y[test_idx].tolist(), test_pred, target_names=target_names), out_dir / "classification_report.json")
    write_json(confusion(y[test_idx].tolist(), test_pred), out_dir / "confusion_matrix.json")
    write_json(
        {
            "feature_columns": features.columns.tolist(),
            "label_counts": {target_names[int(idx)]: int(count) for idx, count in zip(*np.unique(y, return_counts=True))},
            "raw_label_counts": {str(label): int(count) for label, count in labels.value_counts().sort_index().items()},
            "label_policy": label_policy_stats,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "xgb_device": args.xgb_device,
            "xgb_tree_method": args.xgb_tree_method,
        },
        out_dir / "run_manifest.json",
    )
    print(test_metrics)


if __name__ == "__main__":
    main()
