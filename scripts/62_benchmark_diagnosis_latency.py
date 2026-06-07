#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "runtime"
TOKEN_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown" / "tokens_category"
SWEEP_PATH = ROOT / "scripts" / "52_sweep_anomaly_low_fpr.py"

BEST_SETTINGS = {
    "Botnet": {"slug": "botnet", "feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_euclidean", "k": 3, "group_mode": "protocol"},
    "DDoS": {"slug": "ddos", "feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "protocol"},
    "Probe": {"slug": "probe", "feature_filter": "all_no_special", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "protocol"},
    "WebAttack": {"slug": "webattack", "feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "global"},
    "BruteForce": {"slug": "bruteforce", "feature_filter": "packet_burst_profile", "transform": "tfidf_l2", "scorer": "knn_cosine", "k": 3, "group_mode": "protocol"},
}


def _load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("flowprim_latency_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _token_path(attack: str, seed: int) -> Path:
    setting = BEST_SETTINGS[attack]
    return TOKEN_DIR / f"cicids2017_leave_one_{setting['slug']}_anomaly_seed{seed}_a3_full_rhythm.pt"


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


def _quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr) * 1000.0),
        "p50_ms": float(np.quantile(arr, 0.50) * 1000.0),
        "p95_ms": float(np.quantile(arr, 0.95) * 1000.0),
        "p99_ms": float(np.quantile(arr, 0.99) * 1000.0),
        "max_ms": float(np.max(arr) * 1000.0),
    }


def _refs_by_group(features: np.ndarray, train_idx: np.ndarray, groups: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    global_refs = features[train_idx]
    group_indices: dict[str, list[int]] = defaultdict(list)
    for idx in train_idx.tolist():
        group_indices[groups[idx]].append(idx)
    grouped = {
        group: features[np.asarray(indices, dtype=np.int64)]
        for group, indices in group_indices.items()
        if indices
    }
    return global_refs, grouped


def _score_one(feature: np.ndarray, refs: np.ndarray, scorer: str, k: int) -> float:
    return float(S._score_against_refs(feature.reshape(1, -1), refs, scorer, k)[0])


def _benchmark_attack(attack: str, seed: int, max_eval: int, repeats: int) -> dict[str, Any]:
    setting = BEST_SETTINGS[attack]
    token_path = _token_path(attack, seed)

    t0 = time.perf_counter()
    token_data = S._read_token_data(token_path)
    load_seconds = time.perf_counter() - t0

    train_idx = S._split_indices(token_data, "train")
    val_idx = S._split_indices(token_data, "val")
    test_idx = S._split_indices(token_data, "test")

    t0 = time.perf_counter()
    features, feature_stats = S._features(
        token_data,
        train_idx,
        feature_filter=setting["feature_filter"],
        transform=setting["transform"],
    )
    feature_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    groups = S._groups(token_data, setting["group_mode"])
    global_refs, grouped_refs = _refs_by_group(features, train_idx, groups)
    memory_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    val_scores = S._scores(features, train_idx, val_idx, groups, scorer=setting["scorer"], k=int(setting["k"]))
    threshold = float(np.percentile(val_scores, 99.0))
    threshold_seconds = time.perf_counter() - t0

    eval_idx = test_idx[: min(int(max_eval), len(test_idx))]
    if len(eval_idx) == 0:
        raise ValueError(f"No test rows in {token_path}")

    stage_scores: list[float] = []
    stage_report: list[float] = []
    end_to_end: list[float] = []
    alert_count = 0
    warmup = min(32, len(eval_idx))
    for idx in eval_idx[:warmup].tolist():
        refs = grouped_refs.get(groups[idx], global_refs)
        _ = _score_one(features[idx], refs, setting["scorer"], int(setting["k"]))

    for _ in range(int(repeats)):
        for idx in eval_idx.tolist():
            start = time.perf_counter()
            refs = grouped_refs.get(groups[idx], global_refs)
            s0 = time.perf_counter()
            score = _score_one(features[idx], refs, setting["scorer"], int(setting["k"]))
            s1 = time.perf_counter()
            prediction = int(score >= threshold)
            diagnosis_record = {
                "flow_id": token_data["meta"][idx].get("flow_id"),
                "score": score,
                "threshold": threshold,
                "prediction": prediction,
                "group": groups[idx],
            }
            # Force materialization of the small alert record without I/O.
            _ = diagnosis_record["prediction"]
            end = time.perf_counter()
            stage_scores.append(s1 - s0)
            stage_report.append(end - s1)
            end_to_end.append(end - start)
            alert_count += prediction

    scoring_q = _quantiles(stage_scores)
    report_q = _quantiles(stage_report)
    e2e_q = _quantiles(end_to_end)
    num_rows = int(features.shape[0])
    return {
        "attack": attack,
        "seed": seed,
        "token_path": str(token_path.relative_to(ROOT.parent)),
        "feature_filter": setting["feature_filter"],
        "transform": setting["transform"],
        "scorer": setting["scorer"],
        "k": setting["k"],
        "group_mode": setting["group_mode"],
        "num_rows": num_rows,
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        "bench_eval_n": int(len(eval_idx)),
        "repeats": int(repeats),
        "num_features": int(feature_stats["num_features"]),
        "mean_nonzero": float(feature_stats["mean_nonzero"]),
        "load_seconds": float(load_seconds),
        "feature_build_seconds": float(feature_seconds),
        "feature_build_us_per_flow": float(feature_seconds / max(num_rows, 1) * 1_000_000.0),
        "memory_build_seconds": float(memory_seconds),
        "memory_build_us_per_train_flow": float(memory_seconds / max(len(train_idx), 1) * 1_000_000.0),
        "threshold_seconds": float(threshold_seconds),
        "threshold_us_per_val_flow": float(threshold_seconds / max(len(val_idx), 1) * 1_000_000.0),
        "scoring_mean_ms": scoring_q["mean_ms"],
        "scoring_p50_ms": scoring_q["p50_ms"],
        "scoring_p95_ms": scoring_q["p95_ms"],
        "scoring_p99_ms": scoring_q["p99_ms"],
        "record_mean_ms": report_q["mean_ms"],
        "record_p95_ms": report_q["p95_ms"],
        "diagnosis_mean_ms": e2e_q["mean_ms"],
        "diagnosis_p50_ms": e2e_q["p50_ms"],
        "diagnosis_p95_ms": e2e_q["p95_ms"],
        "diagnosis_p99_ms": e2e_q["p99_ms"],
        "diagnosis_max_ms": e2e_q["max_ms"],
        "diagnosis_flows_per_second": float(1.0 / max(e2e_q["mean_ms"] / 1000.0, 1e-12)),
        "alert_count": int(alert_count),
        "timed_calls": int(len(end_to_end)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "feature_build_us_per_flow",
        "memory_build_us_per_train_flow",
        "threshold_us_per_val_flow",
        "scoring_mean_ms",
        "scoring_p95_ms",
        "diagnosis_mean_ms",
        "diagnosis_p50_ms",
        "diagnosis_p95_ms",
        "diagnosis_p99_ms",
        "diagnosis_flows_per_second",
    ]
    out: list[dict[str, Any]] = []
    for key in keys:
        values = [float(row[key]) for row in rows]
        out.append(
            {
                "metric": key,
                "mean": float(statistics.fmean(values)),
                "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                "min": float(min(values)),
                "max": float(max(values)),
            }
        )
    return out


def _write_md(rows: list[dict[str, Any]], agg: list[dict[str, Any]], path: Path) -> None:
    def fmt(value: float, digits: int = 4) -> str:
        return f"{float(value):.{digits}f}"

    lines = [
        "# FlowPrim Diagnosis Latency Benchmark",
        "",
        "Scope: algorithmic latency after a flow record has already been completed and behavior tokens are available. PCAP capture, Zeek parsing, and flow-timeout waiting are deployment ingestion costs and are not included.",
        "",
        "| Attack | Feature view | Scorer | Features | Build us/flow | Score p95 ms | Diagnosis p50 ms | Diagnosis p95 ms | Diagnosis p99 ms | FPS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {attack} | `{view}` | `{scorer}` | {features} | {build} | {score95} | {p50} | {p95} | {p99} | {fps} |".format(
                attack=row["attack"],
                view=row["feature_filter"],
                scorer=row["scorer"],
                features=row["num_features"],
                build=fmt(row["feature_build_us_per_flow"], 2),
                score95=fmt(row["scoring_p95_ms"], 4),
                p50=fmt(row["diagnosis_p50_ms"], 4),
                p95=fmt(row["diagnosis_p95_ms"], 4),
                p99=fmt(row["diagnosis_p99_ms"], 4),
                fps=fmt(row["diagnosis_flows_per_second"], 1),
            )
        )
    lookup = {row["metric"]: row for row in agg}
    lines.extend(
        [
            "",
            "Aggregate over attacks:",
            "",
            f"- Mean feature construction cost: {fmt(lookup['feature_build_us_per_flow']['mean'], 2)} us/flow.",
            f"- Mean per-flow diagnosis latency: {fmt(lookup['diagnosis_mean_ms']['mean'], 4)} ms; p95 range {fmt(lookup['diagnosis_p95_ms']['min'], 4)}-{fmt(lookup['diagnosis_p95_ms']['max'], 4)} ms.",
            f"- Mean throughput from per-flow diagnosis calls: {fmt(lookup['diagnosis_flows_per_second']['mean'], 1)} flows/s.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FlowPrim memory-diagnosis latency after behavior tokens are available.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max_eval", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = [_benchmark_attack(attack, args.seed, args.max_eval, args.repeats) for attack in BEST_SETTINGS]
    agg = _aggregate(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "diagnosis_latency_by_attack.csv")
    _write_csv(agg, out_dir / "diagnosis_latency_summary.csv")
    _write_md(rows, agg, out_dir / "diagnosis_latency_summary.md")
    with (out_dir / "diagnosis_latency_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "aggregate": agg, "settings": BEST_SETTINGS}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"rows": len(rows), "out_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
