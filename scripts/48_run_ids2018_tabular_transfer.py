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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data.unsw_nb15_adapter import (
    cicids2017_binary_labels,
    cicids2017_family_labels,
    cicids2017_label_column,
    cicids2017_shared_features,
    cicids2018_binary_labels,
    cicids2018_family_labels,
    cicids2018_label_column,
    cicids2018_shared_features,
    read_csv_paths,
    shared_feature_columns,
)
from src.evaluation.metrics import classification_metrics, confusion, report_dict
from src.utils.io import write_json
from src.utils.seed import set_seed


COMMON_FAMILIES = ["BENIGN", "Botnet", "BruteForce", "DDoS", "DoS", "Infiltration", "WebAttack"]


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
    if not paths:
        raise ValueError("No CSV files were found")
    return paths


def _target_names(task: str) -> list[str]:
    if task == "binary":
        return ["BENIGN", "ATTACK"]
    if task == "multiclass_common":
        return list(COMMON_FAMILIES)
    raise ValueError(f"Unsupported task: {task}")


def _encode_families(labels: pd.Series, target_names: list[str]) -> tuple[np.ndarray, pd.Series, pd.Series]:
    label_set = set(target_names)
    keep_mask = labels.map(lambda item: str(item) in label_set)
    kept = labels.loc[keep_mask].map(str)
    mapping = {name: idx for idx, name in enumerate(target_names)}
    return kept.map(mapping).astype(int).to_numpy(dtype=np.int64), kept, keep_mask


def _load_cic2017(
    paths: list[Path],
    *,
    task: str,
    attempted_policy: str,
    max_rows_per_file: int | None,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, dict[str, Any]]:
    frame = read_csv_paths(paths, max_rows_per_file=max_rows_per_file)
    label_col = cicids2017_label_column(frame)
    if task == "binary":
        y, labels, keep_mask = cicids2017_binary_labels(frame[label_col], attempted_policy=attempted_policy)
    else:
        _unused_y, families, family_keep_mask = cicids2017_family_labels(frame[label_col], attempted_policy=attempted_policy)
        y, labels, encoded_keep = _encode_families(families, _target_names(task))
        kept_indices = frame.loc[family_keep_mask].index[encoded_keep]
        keep_mask = frame.index.isin(kept_indices)
    filtered = frame.loc[keep_mask].copy()
    features = cicids2017_shared_features(filtered)
    stats = {
        "csv_files": [str(path) for path in paths],
        "dataset": "CICIDS2017_corrected",
        "label_column": label_col,
        "attempted_policy": attempted_policy,
        "num_rows_before_policy": int(len(frame)),
        "num_rows_after_policy": int(len(filtered)),
        "dropped_rows_by_label_policy_or_target_space": int(len(frame) - len(filtered)),
        "raw_label_counts": {str(k): int(v) for k, v in frame[label_col].map(str).value_counts().sort_index().items()},
        "resolved_label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
    }
    return features, y, labels, stats


def _load_cic2018(
    paths: list[Path],
    *,
    task: str,
    attempted_policy: str,
    max_rows_per_file: int | None,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, dict[str, Any]]:
    frame = read_csv_paths(paths, max_rows_per_file=max_rows_per_file)
    label_col = cicids2018_label_column(frame)
    if task == "binary":
        y, labels, keep_mask = cicids2018_binary_labels(frame[label_col], attempted_policy=attempted_policy)
    else:
        _unused_y, families, family_keep_mask = cicids2018_family_labels(frame[label_col], attempted_policy=attempted_policy)
        y, labels, encoded_keep = _encode_families(families, _target_names(task))
        kept_indices = frame.loc[family_keep_mask].index[encoded_keep]
        keep_mask = frame.index.isin(kept_indices)
    filtered = frame.loc[keep_mask].copy()
    features = cicids2018_shared_features(filtered)
    source_counts = {
        str(k): int(v)
        for k, v in filtered["__source_file"].map(lambda item: Path(str(item)).name).value_counts().sort_index().items()
    }
    stats = {
        "csv_files": [str(path) for path in paths],
        "dataset": "CSE-CIC-IDS2018_processed",
        "label_column": label_col,
        "attempted_policy": attempted_policy,
        "num_rows_before_policy": int(len(frame)),
        "num_rows_after_policy": int(len(filtered)),
        "dropped_rows_by_label_policy_or_target_space": int(len(frame) - len(filtered)),
        "raw_label_counts": {str(k): int(v) for k, v in frame[label_col].map(str).value_counts().sort_index().items()},
        "resolved_label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "source_file_counts_after_filter": source_counts,
    }
    return features, y, labels, stats


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
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for label in sorted(set(y.tolist())):
        label_idx = indices[y == label]
        rng.shuffle(label_idx)
        take = len(label_idx)
        if fraction is not None:
            take = max(1, int(round(len(label_idx) * float(fraction))))
        parts.append(label_idx[:take])
    selected = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    if balanced and max_rows is not None and len(selected) > max_rows:
        labels = sorted(set(y[selected].tolist()))
        per_label = max(1, int(max_rows) // max(1, len(labels)))
        balanced_parts = []
        for label in labels:
            label_idx = selected[y[selected] == label]
            rng.shuffle(label_idx)
            balanced_parts.append(label_idx[:per_label])
        selected = np.concatenate(balanced_parts) if balanced_parts else selected
    if max_rows is not None and len(selected) > max_rows:
        rng.shuffle(selected)
        selected = selected[:max_rows]
    return np.sort(selected)


def _split_target_train_calibration(
    y: np.ndarray,
    fraction: float,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_idx = _sample_indices(y, fraction=fraction, max_rows=None, balanced=False, seed=seed)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    cal_parts: list[np.ndarray] = []
    for label in sorted(set(y[all_idx].tolist())):
        label_idx = all_idx[y[all_idx] == label]
        rng.shuffle(label_idx)
        cal_n = int(round(len(label_idx) * calibration_fraction))
        if len(label_idx) >= 2:
            cal_n = max(1, min(cal_n, len(label_idx) - 1))
        train_parts.append(label_idx[cal_n:])
        cal_parts.append(label_idx[:cal_n])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(cal_parts))


def _make_xgb(task: str, num_classes: int, rounds: int, seed: int, xgb_device: str | None, xgb_tree_method: str | None) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is required for D2 tabular transfer") from exc
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    kwargs: dict[str, Any] = {
        "n_estimators": int(rounds),
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": int(seed),
        "n_jobs": -1,
        "objective": objective,
        "eval_metric": "logloss" if task == "binary" else "mlogloss",
    }
    if num_classes > 2:
        kwargs["num_class"] = int(num_classes)
    if xgb_device:
        kwargs["device"] = xgb_device
    if xgb_tree_method:
        kwargs["tree_method"] = xgb_tree_method
    return XGBClassifier(**kwargs)


def _fit_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    num_classes: int,
    rounds: int,
    seed: int,
    xgb_device: str | None,
    xgb_tree_method: str | None,
    base_booster: Any | None = None,
) -> tuple[Pipeline, float]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _make_xgb(task, num_classes, rounds, seed, xgb_device, xgb_tree_method)),
        ]
    )
    start = time.perf_counter()
    fit_kwargs = {}
    if base_booster is not None:
        fit_kwargs["model__xgb_model"] = base_booster
    pipeline.fit(x, y, **fit_kwargs)
    return pipeline, time.perf_counter() - start


def _fit_pipeline_with_all_classes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    num_classes: int,
    rounds: int,
    seed: int,
    xgb_device: str | None,
    xgb_tree_method: str | None,
    base_booster: Any | None = None,
) -> tuple[Pipeline, float]:
    observed = set(int(item) for item in y.tolist())
    if set(range(num_classes)).issubset(observed):
        return _fit_pipeline(
            x,
            y,
            task=task,
            num_classes=num_classes,
            rounds=rounds,
            seed=seed,
            xgb_device=xgb_device,
            xgb_tree_method=xgb_tree_method,
            base_booster=base_booster,
        )
    missing = [idx for idx in range(num_classes) if idx not in observed]
    if x.size == 0:
        raise ValueError("Cannot fit with empty training features")
    filler_x = np.repeat(x[:1], repeats=len(missing), axis=0)
    filler_y = np.array(missing, dtype=y.dtype)
    augmented_x = np.vstack([x, filler_x])
    augmented_y = np.concatenate([y, filler_y])
    return _fit_pipeline(
        augmented_x,
        augmented_y,
        task=task,
        num_classes=num_classes,
        rounds=rounds,
        seed=seed,
        xgb_device=xgb_device,
        xgb_tree_method=xgb_tree_method,
        base_booster=base_booster,
    )


def _aligned_proba(model: Pipeline, x: np.ndarray, num_classes: int) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == num_classes:
        return proba
    aligned = np.zeros((proba.shape[0], num_classes), dtype=float)
    for idx, cls in enumerate(model.named_steps["model"].classes_):
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


def _evaluate(model: Pipeline, x: np.ndarray, y: np.ndarray, num_classes: int, threshold: float | None = None) -> tuple[dict[str, Any], list[int], np.ndarray]:
    proba = _aligned_proba(model, x, num_classes)
    if num_classes == 2 and threshold is not None:
        pred = (proba[:, 1] >= threshold).astype(int)
    else:
        pred = np.argmax(proba, axis=1)
    return classification_metrics(y.tolist(), pred.tolist(), proba), pred.tolist(), proba


def _label_counts(y: np.ndarray, names: list[str]) -> dict[str, int]:
    counts = Counter(int(item) for item in y.tolist())
    return {name: int(counts.get(idx, 0)) for idx, name in enumerate(names)}


def _write_run(
    out_dir: Path,
    run_name: str,
    *,
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


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D2 CICIDS2017 -> CSE-CIC-IDS2018 full-scale tabular transfer.")
    parser.add_argument("--cic2017_csv", nargs="+", required=True)
    parser.add_argument("--ids2018_csv", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/results/ccfa_d2_ids2018_tabular_transfer")
    parser.add_argument("--task", choices=["binary", "multiclass_common"], default="binary")
    parser.add_argument("--attempted_policy", choices=["keep", "drop", "attack", "benign"], default="drop")
    parser.add_argument("--max_cic_rows_per_file", type=int, default=None)
    parser.add_argument("--max_ids2018_rows_per_file", type=int, default=None)
    parser.add_argument("--cic_train_fraction", type=float, default=None)
    parser.add_argument("--cic_max_train_rows", type=int, default=None)
    parser.add_argument("--balanced_cic_train", action="store_true")
    parser.add_argument("--base_rounds", type=int, default=300)
    parser.add_argument("--ids2018_only_rounds", type=int, default=300)
    parser.add_argument("--few_shot_fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--few_shot_rounds", type=int, default=80)
    parser.add_argument("--few_shot_scratch_rounds", type=int, default=300)
    parser.add_argument("--calibration_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb_device", default=None)
    parser.add_argument("--xgb_tree_method", default="hist")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_names = _target_names(args.task)
    num_classes = len(target_names)

    cic_paths = _expand_inputs(args.cic2017_csv)
    ids_paths = _expand_inputs(args.ids2018_csv)
    cic_features, cic_y, cic_labels, cic_stats = _load_cic2017(
        cic_paths,
        task=args.task,
        attempted_policy=args.attempted_policy,
        max_rows_per_file=args.max_cic_rows_per_file,
    )
    ids_features, ids_y, ids_labels, ids_stats = _load_cic2018(
        ids_paths,
        task=args.task,
        attempted_policy=args.attempted_policy,
        max_rows_per_file=args.max_ids2018_rows_per_file,
    )
    cic_x_all = cic_features.to_numpy(dtype=np.float32)
    ids_x = ids_features.to_numpy(dtype=np.float32)
    cic_idx = _sample_indices(
        cic_y,
        fraction=args.cic_train_fraction,
        max_rows=args.cic_max_train_rows,
        balanced=bool(args.balanced_cic_train),
        seed=args.seed,
    )
    cic_x = cic_x_all[cic_idx]
    cic_train_y = cic_y[cic_idx]
    manifest_base = {
        "command": shlex.join(sys.argv),
        "script": Path(__file__).name,
        "task": args.task,
        "feature_columns": shared_feature_columns(),
        "target_names": target_names,
        "seed": int(args.seed),
        "xgb_device": args.xgb_device,
        "xgb_tree_method": args.xgb_tree_method,
    }

    base_model, base_seconds = _fit_pipeline_with_all_classes(
        cic_x,
        cic_train_y,
        task=args.task,
        num_classes=num_classes,
        rounds=args.base_rounds,
        seed=args.seed,
        xgb_device=args.xgb_device,
        xgb_tree_method=args.xgb_tree_method,
    )
    base_metrics, base_pred, _ = _evaluate(base_model, ids_x, ids_y, num_classes)
    base_metrics.update(
        {
            "section": "D2_IDS2018",
            "setting": "zero_shot_cicids2017_to_ids2018_full",
            "task": args.task,
            "model": "XGBoost",
            "feature_adapter": "shared_cicflowmeter_semantics_v2",
            "train_dataset": "CICIDS2017_corrected",
            "test_dataset": "CSE-CIC-IDS2018_processed_full",
            "num_train": int(len(cic_train_y)),
            "num_external_test": int(len(ids_y)),
            "num_features": int(cic_x.shape[1]),
            "threshold": 0.5 if num_classes == 2 else None,
            "train_seconds": float(base_seconds),
        }
    )
    _write_run(
        out_dir,
        "zero_shot_cicids2017_to_ids2018_full",
        metrics=base_metrics,
        y_true=ids_y,
        y_pred=base_pred,
        target_names=target_names,
        manifest={
            **manifest_base,
            "cic_stats": cic_stats,
            "ids2018_stats": ids_stats,
            "cic_train_selected_counts": _label_counts(cic_train_y, target_names),
            "base_rounds": int(args.base_rounds),
        },
    )

    rows: list[dict[str, Any]] = [
        {
            "setting": "zero_shot_cicids2017_to_ids2018_full",
            "num_train": int(len(cic_train_y)),
            "num_test": int(len(ids_y)),
            "accuracy": base_metrics.get("accuracy"),
            "macro_f1": base_metrics.get("macro_f1"),
            "weighted_f1": base_metrics.get("weighted_f1"),
            "auroc": base_metrics.get("auroc"),
            "auprc": base_metrics.get("auprc"),
            "auroc_ovr": base_metrics.get("auroc_ovr"),
            "auprc_ovr": base_metrics.get("auprc_ovr"),
            "fpr95": base_metrics.get("fpr95"),
            "path": str(out_dir / "zero_shot_cicids2017_to_ids2018_full"),
        }
    ]

    threshold = None
    if num_classes == 2:
        few_cal_idx = _sample_indices(ids_y, fraction=0.01, max_rows=None, balanced=False, seed=args.seed)
        threshold, cal_metrics = _best_threshold(ids_y[few_cal_idx], _aligned_proba(base_model, ids_x[few_cal_idx], num_classes)[:, 1])
        calibrated_metrics, calibrated_pred, _ = _evaluate(base_model, ids_x, ids_y, num_classes, threshold=threshold)
        run_name = "zero_shot_cicids2017_to_ids2018_1pct_threshold"
        calibrated_metrics.update(
            {
                "section": "D2_IDS2018",
                "setting": run_name,
                "task": args.task,
                "model": "XGBoost",
                "feature_adapter": "shared_cicflowmeter_semantics_v2",
                "threshold_calibration_dataset": "CSE-CIC-IDS2018_1pct_labels",
                "num_threshold_calibration": int(len(few_cal_idx)),
                "threshold": float(threshold),
                "threshold_calibration_macro_f1": cal_metrics.get("macro_f1"),
            }
        )
        rows.append(
            {
                "setting": run_name,
                "num_train": int(len(cic_train_y)),
                "num_threshold_calibration": int(len(few_cal_idx)),
                "num_test": int(len(ids_y)),
                "threshold": float(threshold),
                "accuracy": calibrated_metrics.get("accuracy"),
                "macro_f1": calibrated_metrics.get("macro_f1"),
                "weighted_f1": calibrated_metrics.get("weighted_f1"),
                "auroc": calibrated_metrics.get("auroc"),
                "auprc": calibrated_metrics.get("auprc"),
                "fpr95": calibrated_metrics.get("fpr95"),
                "path": str(out_dir / run_name),
            }
        )
        _write_run(
            out_dir,
            run_name,
            metrics=calibrated_metrics,
            y_true=ids_y,
            y_pred=calibrated_pred,
            target_names=target_names,
            manifest={**manifest_base, "threshold": float(threshold), "calibration_metrics": cal_metrics},
        )

    ids_model, ids_seconds = _fit_pipeline_with_all_classes(
        ids_x,
        ids_y,
        task=args.task,
        num_classes=num_classes,
        rounds=args.ids2018_only_rounds,
        seed=args.seed,
        xgb_device=args.xgb_device,
        xgb_tree_method=args.xgb_tree_method,
    )
    ids_metrics, ids_pred, _ = _evaluate(ids_model, ids_x, ids_y, num_classes)
    sanity_name = "ids2018_train_to_ids2018_eval_sanity"
    ids_metrics.update(
        {
            "section": "D2_IDS2018",
            "setting": sanity_name,
            "task": args.task,
            "model": "XGBoost",
            "feature_adapter": "shared_cicflowmeter_semantics_v2",
            "train_dataset": "CSE-CIC-IDS2018_processed_full",
            "test_dataset": "CSE-CIC-IDS2018_processed_full",
            "num_train": int(len(ids_y)),
            "num_external_test": int(len(ids_y)),
            "num_features": int(ids_x.shape[1]),
            "train_seconds": float(ids_seconds),
            "note": "same-dataset sanity upper reference; not a leakage-safe train/test split",
        }
    )
    rows.append(
        {
            "setting": sanity_name,
            "num_train": int(len(ids_y)),
            "num_test": int(len(ids_y)),
            "accuracy": ids_metrics.get("accuracy"),
            "macro_f1": ids_metrics.get("macro_f1"),
            "weighted_f1": ids_metrics.get("weighted_f1"),
            "auroc": ids_metrics.get("auroc"),
            "auprc": ids_metrics.get("auprc"),
            "auroc_ovr": ids_metrics.get("auroc_ovr"),
            "auprc_ovr": ids_metrics.get("auprc_ovr"),
            "fpr95": ids_metrics.get("fpr95"),
            "path": str(out_dir / sanity_name),
        }
    )
    _write_run(
        out_dir,
        sanity_name,
        metrics=ids_metrics,
        y_true=ids_y,
        y_pred=ids_pred,
        target_names=target_names,
        manifest={**manifest_base, "ids2018_stats": ids_stats, "ids2018_only_rounds": int(args.ids2018_only_rounds)},
    )

    base_booster = base_model.named_steps["model"].get_booster()
    for fraction in args.few_shot_fractions:
        few_idx, cal_idx = _split_target_train_calibration(ids_y, fraction, args.calibration_fraction, args.seed)
        few_x = ids_x[few_idx]
        few_y = ids_y[few_idx]
        cal_x = ids_x[cal_idx]
        cal_y = ids_y[cal_idx]
        for mode, rounds, booster in (
            ("warm_start", args.few_shot_rounds, base_booster),
            ("scratch", args.few_shot_scratch_rounds, None),
        ):
            model, seconds = _fit_pipeline_with_all_classes(
                few_x,
                few_y,
                task=args.task,
                num_classes=num_classes,
                rounds=rounds,
                seed=args.seed,
                xgb_device=args.xgb_device,
                xgb_tree_method=args.xgb_tree_method,
                base_booster=booster,
            )
            eval_threshold = None
            cal_metrics = None
            if num_classes == 2 and len(cal_y):
                eval_threshold, cal_metrics = _best_threshold(cal_y, _aligned_proba(model, cal_x, num_classes)[:, 1])
            metrics, pred, _ = _evaluate(model, ids_x, ids_y, num_classes, threshold=eval_threshold)
            run_name = f"few_shot_ids2018_{int(round(fraction * 100)):02d}pct_{mode}"
            metrics.update(
                {
                    "section": "D2_IDS2018",
                    "setting": run_name,
                    "task": args.task,
                    "model": f"XGBoost_{mode}",
                    "feature_adapter": "shared_cicflowmeter_semantics_v2",
                    "base_train_dataset": "CICIDS2017_corrected" if mode == "warm_start" else None,
                    "few_shot_dataset": "CSE-CIC-IDS2018_processed",
                    "eval_dataset": "CSE-CIC-IDS2018_processed_full",
                    "few_shot_fraction": float(fraction),
                    "num_base_train": int(len(cic_train_y)) if mode == "warm_start" else 0,
                    "num_few_shot_train": int(len(few_y)),
                    "num_few_shot_calibration": int(len(cal_y)),
                    "num_external_test": int(len(ids_y)),
                    "threshold": None if eval_threshold is None else float(eval_threshold),
                    "threshold_calibration_macro_f1": None if cal_metrics is None else cal_metrics.get("macro_f1"),
                    "train_seconds": float(seconds),
                }
            )
            rows.append(
                {
                    "setting": run_name,
                    "fraction": float(fraction),
                    "num_few_shot_train": int(len(few_y)),
                    "num_few_shot_calibration": int(len(cal_y)),
                    "num_test": int(len(ids_y)),
                    "threshold": None if eval_threshold is None else float(eval_threshold),
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "auroc": metrics.get("auroc"),
                    "auprc": metrics.get("auprc"),
                    "auroc_ovr": metrics.get("auroc_ovr"),
                    "auprc_ovr": metrics.get("auprc_ovr"),
                    "fpr95": metrics.get("fpr95"),
                    "path": str(out_dir / run_name),
                }
            )
            _write_run(
                out_dir,
                run_name,
                metrics=metrics,
                y_true=ids_y,
                y_pred=pred,
                target_names=target_names,
                manifest={
                    **manifest_base,
                    "few_shot_fraction": float(fraction),
                    "few_shot_indices": few_idx.tolist(),
                    "few_shot_calibration_indices": cal_idx.tolist(),
                    "few_shot_counts": _label_counts(few_y, target_names),
                    "few_shot_calibration_counts": _label_counts(cal_y, target_names),
                    "rounds": int(rounds),
                    "warm_start": bool(mode == "warm_start"),
                    "threshold_calibration_metrics": cal_metrics,
                },
            )

    summary = {
        "rows": rows,
        "task": args.task,
        "target_names": target_names,
        "cic2017_stats": cic_stats,
        "cic_train_selected_counts": _label_counts(cic_train_y, target_names),
        "ids2018_stats": ids_stats,
        "ids2018_counts": _label_counts(ids_y, target_names),
    }
    write_json(summary, out_dir / "summary.json")
    md_lines = [
        f"# D2 CSE-CIC-IDS2018 Full-Scale Transfer ({args.task})",
        "",
        "| Setting | Train N | Test N | Accuracy | Macro-F1 | Weighted-F1 | AUROC | AUPRC | FPR95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        train_n = row.get("num_few_shot_train", row.get("num_train", "-"))
        md_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['setting']}`",
                    str(train_n),
                    str(row.get("num_test", "-")),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("macro_f1")),
                    _fmt(row.get("weighted_f1")),
                    _fmt(row.get("auroc", row.get("auroc_ovr"))),
                    _fmt(row.get("auprc", row.get("auprc_ovr"))),
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
            "- This is D2 same-family transfer using shared CICFlowMeter-derived tabular semantics.",
            "- The `ids2018_train_to_ids2018_eval_sanity` row is an upper sanity reference, not a leakage-safe split.",
            "- Few-shot rows evaluate on the full IDS2018 set, including the labeled subset; use them as adaptation diagnostics.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
