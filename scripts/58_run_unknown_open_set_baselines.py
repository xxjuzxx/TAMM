#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


TOKEN_SPECS = [
    {
        "attack": "Botnet",
        "tokens": "paper_icdm_applied_2026/experiments/unknown/tokens_category/cicids2017_leave_one_botnet_anomaly_seed42_a3_full_rhythm.pt",
    },
    {
        "attack": "DDoS",
        "tokens": "paper_icdm_applied_2026/experiments/unknown/tokens_category/cicids2017_leave_one_ddos_anomaly_seed42_a3_full_rhythm.pt",
    },
    {
        "attack": "Probe",
        "tokens": "paper_icdm_applied_2026/experiments/unknown/tokens_category/cicids2017_leave_one_probe_anomaly_seed42_a3_full_rhythm.pt",
    },
    {
        "attack": "WebAttack",
        "tokens": "paper_icdm_applied_2026/experiments/unknown/tokens_category/cicids2017_leave_one_webattack_anomaly_seed42_a3_full_rhythm.pt",
    },
    {
        "attack": "BruteForce",
        "tokens": "paper_icdm_applied_2026/experiments/unknown/tokens_category/cicids2017_leave_one_bruteforce_anomaly_seed42_a3_full_rhythm.pt",
    },
]


def _load_sweep_module() -> Any:
    path = Path(__file__).with_name("52_sweep_anomaly_low_fpr.py")
    spec = importlib.util.spec_from_file_location("sweep_anomaly_low_fpr", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _eval_scores(
    *,
    attack: str,
    method: str,
    feature_filter: str,
    transform: str,
    group_mode: str,
    y_true: np.ndarray,
    test_scores: np.ndarray,
    val_scores: np.ndarray,
    train_n: int,
    val_n: int,
    test_n: int,
) -> dict[str, Any]:
    best = S._best_macro(y_true, test_scores)
    r01 = S._best_recall_under_fpr(y_true, test_scores, 0.001)
    r1 = S._best_recall_under_fpr(y_true, test_scores, 0.01)
    r5 = S._best_recall_under_fpr(y_true, test_scores, 0.05)
    rank = S._rank_metrics(y_true, test_scores)
    val_cal = S._val_percentile_metrics(y_true, test_scores, val_scores, [95.0, 99.0, 99.5, 100.0])
    return {
        "unknown_attack": attack,
        "method": method,
        "feature_filter": feature_filter,
        "transform": transform,
        "group_mode": group_mode,
        "train_n": int(train_n),
        "val_n": int(val_n),
        "test_n": int(test_n),
        "best_macro_f1": best["macro_f1"],
        "best_attack_recall": best["attack_recall"],
        "best_false_positive_rate": best["false_positive_rate"],
        "recall_at_0_1pct_fpr": r01["attack_recall"],
        "actual_fpr_at_0_1pct_fpr": r01["false_positive_rate"],
        "recall_at_1pct_fpr": r1["attack_recall"],
        "actual_fpr_at_1pct_fpr": r1["false_positive_rate"],
        "recall_at_5pct_fpr": r5["attack_recall"],
        "actual_fpr_at_5pct_fpr": r5["false_positive_rate"],
        **rank,
        **val_cal,
    }


def _knn_row(token_data: dict[str, Any], attack: str, feature_filter: str) -> dict[str, Any]:
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = S._split_indices(token_data, "train")
    val_idx = S._split_indices(token_data, "val")
    test_idx = S._split_indices(token_data, "test")
    features, _feature_stats = S._features(token_data, train_idx, feature_filter=feature_filter, transform="binary_l2")
    groups = S._groups(token_data, "protocol")
    val_scores = S._scores(features, train_idx, val_idx, groups, scorer="knn_cosine", k=3)
    test_scores = S._scores(features, train_idx, test_idx, groups, scorer="knn_cosine", k=3)
    return _eval_scores(
        attack=attack,
        method=f"KNN memory ({feature_filter})",
        feature_filter=feature_filter,
        transform="binary_l2",
        group_mode="protocol",
        y_true=labels[test_idx].astype(np.int64),
        test_scores=test_scores,
        val_scores=val_scores,
        train_n=len(train_idx),
        val_n=len(val_idx),
        test_n=len(test_idx),
    )


def _sklearn_rows(token_data: dict[str, Any], attack: str) -> list[dict[str, Any]]:
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = S._split_indices(token_data, "train")
    val_idx = S._split_indices(token_data, "val")
    test_idx = S._split_indices(token_data, "test")
    features, _feature_stats = S._features(token_data, train_idx, feature_filter="packet_burst_profile", transform="binary_l2")
    train_x = features[train_idx]
    val_x = features[val_idx]
    test_x = features[test_idx]
    y_true = labels[test_idx].astype(np.int64)
    baselines: list[tuple[str, Any]] = [
        (
            "IsolationForest",
            IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1),
        ),
        (
            "LOF novelty",
            LocalOutlierFactor(n_neighbors=min(35, max(5, len(train_idx) // 20)), novelty=True),
        ),
        (
            "OneClassSVM nu=0.01",
            OneClassSVM(kernel="rbf", gamma="scale", nu=0.01),
        ),
    ]
    out: list[dict[str, Any]] = []
    for method, model in baselines:
        model.fit(train_x)
        val_scores = -np.asarray(model.score_samples(val_x), dtype=np.float32)
        test_scores = -np.asarray(model.score_samples(test_x), dtype=np.float32)
        out.append(
            _eval_scores(
                attack=attack,
                method=method,
                feature_filter="packet_burst_profile",
                transform="binary_l2",
                group_mode="global",
                y_true=y_true,
                test_scores=test_scores,
                val_scores=val_scores,
                train_n=len(train_idx),
                val_n=len(val_idx),
                test_n=len(test_idx),
            )
        )
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
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def _fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_md(rows: list[dict[str, Any]], path: Path) -> None:
    selected = sorted(rows, key=lambda row: (str(row["unknown_attack"]), -float(row.get("auroc") or -1)))
    lines = [
        "# Unknown Attack Open-Set Baselines",
        "",
        "All baselines fit only on BENIGN train rows from the same leave-one anomaly split. Scores are anomaly scores where larger means more anomalous.",
        "",
        "| Unknown | Method | Feature | AUROC | AUPRC | FPR95 | Recall@0.1%FPR | Recall@1%FPR | Best Macro-F1 | Val-P99 FPR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| {attack} | {method} | {feature} | {auroc} | {auprc} | {fpr95} | {r01} | {r1} | {macro} | {val99_fpr} |".format(
                attack=row["unknown_attack"],
                method=row["method"],
                feature=row["feature_filter"],
                auroc=_fmt(row.get("auroc")),
                auprc=_fmt(row.get("auprc")),
                fpr95=_fmt(row.get("fpr95")),
                r01=_fmt(row.get("recall_at_0_1pct_fpr")),
                r1=_fmt(row.get("recall_at_1pct_fpr")),
                macro=_fmt(row.get("best_macro_f1")),
                val99_fpr=_fmt(row.get("val_p99_0_false_positive_rate")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run open-set baselines on leave-one unknown anomaly token corpora.")
    parser.add_argument("--out_prefix", default="paper_icdm_applied_2026/experiments/baselines/unknown_open_set_baselines")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for spec in TOKEN_SPECS:
        token_path = Path(spec["tokens"])
        if not token_path.exists():
            raise FileNotFoundError(token_path)
        token_data = S._read_token_data(token_path)
        attack = str(spec["attack"])
        rows.append(_knn_row(token_data, attack, "packet"))
        rows.append(_knn_row(token_data, attack, "packet_burst_profile"))
        rows.extend(_sklearn_rows(token_data, attack))
    out_prefix = Path(args.out_prefix)
    _write_csv(rows, out_prefix.with_suffix(".csv"))
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_md(rows, out_prefix.with_suffix(".md"))
    print(json.dumps({"rows": len(rows), "out_prefix": str(out_prefix)}, sort_keys=True))


if __name__ == "__main__":
    main()
