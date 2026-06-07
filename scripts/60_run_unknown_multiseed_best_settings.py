#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np


BEST_SETTINGS = {
    "Botnet": {"feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_euclidean", "k": 3, "group_mode": "protocol"},
    "DDoS": {"feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "protocol"},
    "Probe": {"feature_filter": "all_no_special", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "protocol"},
    "WebAttack": {"feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "global"},
    "BruteForce": {"feature_filter": "packet_burst_profile", "transform": "tfidf_l2", "scorer": "knn_cosine", "k": 3, "group_mode": "protocol"},
}

ATTACK_TO_SLUG = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
}


def _load_sweep_module() -> Any:
    path = Path(__file__).with_name("52_sweep_anomaly_low_fpr.py")
    spec = importlib.util.spec_from_file_location("sweep_anomaly_low_fpr", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _seed42_from_sweep(attack: str, out_dir: Path) -> dict[str, Any]:
    slug = ATTACK_TO_SLUG[attack]
    token_path = out_dir / "tokens_category" / f"cicids2017_leave_one_{slug}_anomaly_seed42_a3_full_rhythm.pt"
    if not token_path.exists():
        raise FileNotFoundError(token_path)
    return _eval_token(token_path, attack, 42)


def _metric_subset(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "best_macro_f1",
        "best_attack_recall",
        "best_false_positive_rate",
        "recall_at_0_1pct_fpr",
        "actual_fpr_at_0_1pct_fpr",
        "recall_at_1pct_fpr",
        "actual_fpr_at_1pct_fpr",
        "recall_at_5pct_fpr",
        "actual_fpr_at_5pct_fpr",
        "auroc",
        "auprc",
        "fpr95",
        "val_p99_0_macro_f1",
        "val_p99_0_attack_recall",
        "val_p99_0_false_positive_rate",
        "val_p99_0_attack_precision",
        "val_p100_0_macro_f1",
        "val_p100_0_attack_recall",
        "val_p100_0_false_positive_rate",
        "val_p100_0_attack_precision",
    ]
    return {key: row.get(key) for key in keys}


def _eval_token(token_path: Path, attack: str, seed: int) -> dict[str, Any]:
    target = BEST_SETTINGS[attack]
    token_data = S._read_token_data(token_path)
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = S._split_indices(token_data, "train")
    val_idx = S._split_indices(token_data, "val")
    test_idx = S._split_indices(token_data, "test")
    features, feature_stats = S._features(token_data, train_idx, feature_filter=target["feature_filter"], transform=target["transform"])
    group_values = S._groups(token_data, target["group_mode"])
    row = S._evaluate(
        features,
        feature_stats,
        group_values,
        labels,
        train_idx,
        val_idx,
        test_idx,
        feature_filter=target["feature_filter"],
        transform=target["transform"],
        scorer=target["scorer"],
        k=int(target["k"]),
        group_mode=target["group_mode"],
    )
    return {
        "unknown_attack": attack,
        "seed": int(seed),
        **target,
        "train_n": len(train_idx),
        "val_n": len(val_idx),
        "test_n": len(test_idx),
        **_metric_subset(row),
    }


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        "best_macro_f1",
        "recall_at_0_1pct_fpr",
        "actual_fpr_at_0_1pct_fpr",
        "recall_at_1pct_fpr",
        "auroc",
        "auprc",
        "fpr95",
        "val_p99_0_macro_f1",
        "val_p99_0_false_positive_rate",
        "val_p100_0_macro_f1",
        "val_p100_0_false_positive_rate",
    ]
    out: list[dict[str, Any]] = []
    for attack in BEST_SETTINGS:
        attack_rows = [row for row in rows if row["unknown_attack"] == attack]
        item: dict[str, Any] = {
            "unknown_attack": attack,
            "seeds": ",".join(str(row["seed"]) for row in sorted(attack_rows, key=lambda r: int(r["seed"]))),
            "num_runs": len(attack_rows),
            "setting": "/".join(str(BEST_SETTINGS[attack][key]) for key in ["feature_filter", "transform", "scorer", "k", "group_mode"]),
        }
        for key in metric_keys:
            vals = [_float(row.get(key)) for row in attack_rows]
            clean = [value for value in vals if value is not None]
            stats = _mean_std(clean) if clean else None
            item[f"{key}_mean"] = None if stats is None else stats["mean"]
            item[f"{key}_std"] = None if stats is None else stats["std"]
        out.append(item)
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    value = _float(value)
    return "-" if value is None else f"{value:.{digits}f}"


def _write_md(agg: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Unknown Attack Best-Setting 3-Seed Summary",
        "",
        "All seeds use regenerated profile/structural category token corpora. The same per-attack setting is locked across seeds.",
        "",
        "| Unknown | Setting | Macro-F1 | AUROC | AUPRC | FPR95 | Recall@0.1%FPR | Recall@1%FPR | Val-P99 FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            "| {attack} | `{setting}` | {macro} +/- {macro_std} | {auroc} +/- {auroc_std} | {auprc} +/- {auprc_std} | {fpr95} +/- {fpr95_std} | {r01} +/- {r01_std} | {r1} +/- {r1_std} | {valfpr} +/- {valfpr_std} |".format(
                attack=row["unknown_attack"],
                setting=row["setting"],
                macro=_fmt(row.get("best_macro_f1_mean")),
                macro_std=_fmt(row.get("best_macro_f1_std")),
                auroc=_fmt(row.get("auroc_mean")),
                auroc_std=_fmt(row.get("auroc_std")),
                auprc=_fmt(row.get("auprc_mean")),
                auprc_std=_fmt(row.get("auprc_std")),
                fpr95=_fmt(row.get("fpr95_mean")),
                fpr95_std=_fmt(row.get("fpr95_std")),
                r01=_fmt(row.get("recall_at_0_1pct_fpr_mean")),
                r01_std=_fmt(row.get("recall_at_0_1pct_fpr_std")),
                r1=_fmt(row.get("recall_at_1pct_fpr_mean")),
                r1_std=_fmt(row.get("recall_at_1pct_fpr_std")),
                valfpr=_fmt(row.get("val_p99_0_false_positive_rate_mean")),
                valfpr_std=_fmt(row.get("val_p99_0_false_positive_rate_std")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate seed-43/44 leave-one unknown best settings and aggregate with seed 42.")
    parser.add_argument("--unknown_dir", default="paper_icdm_applied_2026/experiments/unknown")
    parser.add_argument("--out_prefix", default="paper_icdm_applied_2026/experiments/unknown/unknown_best_settings_3seed")
    args = parser.parse_args()
    unknown_dir = Path(args.unknown_dir)
    rows: list[dict[str, Any]] = []
    for attack in BEST_SETTINGS:
        rows.append(_seed42_from_sweep(attack, unknown_dir))
        slug = ATTACK_TO_SLUG[attack]
        for seed in [43, 44]:
            token_path = unknown_dir / "tokens_category" / f"cicids2017_leave_one_{slug}_anomaly_seed{seed}_a3_full_rhythm.pt"
            if not token_path.exists():
                raise FileNotFoundError(token_path)
            rows.append(_eval_token(token_path, attack, seed))
    agg = _aggregate(rows)
    out_prefix = Path(args.out_prefix)
    _write_csv(rows, out_prefix.with_name(out_prefix.name + "_runs.csv"))
    _write_csv(agg, out_prefix.with_suffix(".csv"))
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": rows, "aggregate": agg}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_md(agg, out_prefix.with_suffix(".md"))
    print(json.dumps({"runs": len(rows), "aggregate": len(agg), "out_prefix": str(out_prefix)}, sort_keys=True))


if __name__ == "__main__":
    main()
