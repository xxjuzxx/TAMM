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

from run_memory_optimization_experiments import (
    ATTACK_SLUG,
    DEFAULT_TOKEN_DIR,
    ArtifactData,
    StructuralPrimitiveConfig,
    _cosine_distances,
    _evt_threshold,
    _features_uniform,
    _load_artifact,
    _safe_float,
    _standard_metrics,
    _token_path,
    _topk_from_distances,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "online_replay_knn"


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


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _query_one(query: np.ndarray, memory: np.ndarray, k: int) -> tuple[float, np.ndarray, np.ndarray]:
    distances = _cosine_distances(query.reshape(1, -1), memory)
    nn, vals = _topk_from_distances(distances, k)
    return float(np.mean(vals[0])), nn[0].astype(np.int64), vals[0].astype(np.float32)


def _calibration_scores(features: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, k: int) -> tuple[np.ndarray, list[float]]:
    memory = features[train_idx]
    scores = np.zeros(len(val_idx), dtype=np.float32)
    query_ms: list[float] = []
    for pos, idx in enumerate(val_idx.tolist()):
        start = time.perf_counter()
        score, _, _ = _query_one(features[int(idx)], memory, k)
        query_ms.append((time.perf_counter() - start) * 1000.0)
        scores[pos] = score
    return scores, query_ms


def _ordered_test_indices(data: ArtifactData, replay_order: str) -> np.ndarray:
    if replay_order == "artifact_order":
        return np.asarray(sorted(data.test_idx.tolist()), dtype=np.int64)
    if replay_order == "stable_hash":
        return np.asarray(
            sorted(data.test_idx.tolist(), key=lambda idx: str(data.path) + ":" + str(data.rows[int(idx)][:5]) + ":" + str(idx)),
            dtype=np.int64,
        )
    raise ValueError(f"Unsupported replay order: {replay_order}")


def _sliding_fpr(rows: list[dict[str, Any]], windows: list[int]) -> dict[str, float]:
    benign = [row for row in rows if int(row["binary_label_id"]) == 0]
    out: dict[str, float] = {}
    for window in windows:
        if not benign:
            out[f"sliding_fpr_window_{window}"] = 0.0
            continue
        max_rate = 0.0
        left = 0
        fp = 0
        for right, row in enumerate(benign):
            if row["is_false_positive"]:
                fp += 1
            while right - left + 1 > int(window):
                if benign[left]["is_false_positive"]:
                    fp -= 1
                left += 1
            denom = max(1, right - left + 1)
            max_rate = max(max_rate, fp / denom)
        out[f"sliding_fpr_window_{window}"] = float(max_rate)
    return out


def _alert_delay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attack_rows = [row for row in rows if int(row["binary_label_id"]) == 1]
    if not attack_rows:
        return {
            "first_attack_pos": "",
            "first_attack_alert_pos": "",
            "delay_to_first_attack_alert_flows": "",
            "attack_alert_coverage": 0.0,
        }
    first_attack = int(attack_rows[0]["stream_pos"])
    first_alert = next((int(row["stream_pos"]) for row in attack_rows if row["prediction"] == 1), None)
    return {
        "first_attack_pos": first_attack,
        "first_attack_alert_pos": "" if first_alert is None else first_alert,
        "delay_to_first_attack_alert_flows": "" if first_alert is None else int(max(0, first_alert - first_attack)),
        "attack_alert_coverage": float(sum(1 for row in attack_rows if row["prediction"] == 1) / max(len(attack_rows), 1)),
    }


def _time_to_first_fp(rows: list[dict[str, Any]]) -> Any:
    first = next((int(row["stream_pos"]) for row in rows if row["is_false_positive"]), None)
    return "" if first is None else first


def _neighbor_turnover(neighbors: list[np.ndarray]) -> dict[str, float]:
    if len(neighbors) < 2:
        return {"neighbor_turnover_mean": 0.0, "neighbor_turnover_p95": 0.0}
    vals: list[float] = []
    for left, right in zip(neighbors, neighbors[1:]):
        a = set(int(x) for x in left.tolist())
        b = set(int(x) for x in right.tolist())
        union = max(len(a | b), 1)
        vals.append(1.0 - len(a & b) / union)
    return {
        "neighbor_turnover_mean": float(statistics.mean(vals)),
        "neighbor_turnover_p95": float(np.quantile(np.asarray(vals, dtype=np.float32), 0.95)),
    }


def _memory_snapshot(memory: np.ndarray) -> dict[str, Any]:
    nonzero = np.sum(memory != 0, axis=1)
    return {
        "memory_size": int(memory.shape[0]),
        "vocab_size": int(memory.shape[1]),
        "memory_bytes_estimate": int(memory.nbytes),
        "memory_mib_estimate": float(memory.nbytes / (1024.0 * 1024.0)),
        "memory_mean_nonzero": float(np.mean(nonzero)) if nonzero.size else 0.0,
        "memory_p95_nonzero": float(np.quantile(nonzero, 0.95)) if nonzero.size else 0.0,
    }


def _make_memory(
    features: np.ndarray,
    train_idx: np.ndarray,
    *,
    policy: str,
    ratio: float,
    seed: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    memory_full = features[train_idx]
    n = memory_full.shape[0]
    if policy == "full":
        selected = np.arange(n, dtype=np.int64)
    elif policy == "tail_preserving_coreset":
        keep = max(k, min(n, int(round(n * float(ratio)))))
        centroid = memory_full.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(centroid, axis=1, keepdims=True)
        centroid = np.divide(centroid, denom, out=np.zeros_like(centroid), where=denom > 0).reshape(-1)
        tail_score = _cosine_distances(memory_full, centroid.reshape(1, -1)).reshape(-1)
        tail_keep = max(1, keep // 2)
        tail_idx = np.argsort(-tail_score)[:tail_keep]
        remaining = np.setdiff1d(np.arange(n), tail_idx, assume_unique=False)
        rng = np.random.default_rng(seed)
        fill = rng.choice(remaining, size=max(0, keep - len(tail_idx)), replace=False) if keep > len(tail_idx) else np.empty(0, dtype=np.int64)
        selected = np.sort(np.concatenate([tail_idx, fill]).astype(np.int64))
    else:
        raise ValueError(f"Unsupported memory policy: {policy}")
    memory = memory_full[selected]
    metadata = {
        "memory_policy": policy,
        "memory_ratio": float(ratio) if policy != "full" else 1.0,
        "selected_memory_rows": int(len(selected)),
        "original_memory_rows": int(n),
        "attack_labels_used_for_memory": False,
    }
    metadata.update(_memory_snapshot(memory))
    return memory, selected, metadata


def _replay_one(
    data: ArtifactData,
    *,
    k: int,
    threshold_mode: str,
    replay_order: str,
    memory_policy: str,
    memory_ratio: float,
    sliding_windows: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features = _features_uniform(data.raw_matrix)
    memory, selected_local, memory_meta = _make_memory(
        features,
        data.train_idx,
        policy=memory_policy,
        ratio=memory_ratio,
        seed=data.seed * 1000 + len(data.attack),
        k=k,
    )
    val_scores, val_query_ms = _calibration_scores(features, data.train_idx[selected_local] if memory_policy != "full" else data.train_idx, data.val_idx, k)
    if threshold_mode == "p99":
        threshold = float(np.percentile(val_scores, 99.0))
        threshold_status = "empirical_p99"
    elif threshold_mode == "evt_p99":
        threshold, threshold_status = _evt_threshold(val_scores)
    else:
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")

    labels = data.labels
    stream_idx = _ordered_test_indices(data, replay_order)
    rows: list[dict[str, Any]] = []
    test_scores = np.zeros(len(stream_idx), dtype=np.float32)
    query_ms: list[float] = []
    neighbor_sets: list[np.ndarray] = []
    start_replay = time.perf_counter()
    for pos, idx in enumerate(stream_idx.tolist()):
        start = time.perf_counter()
        score, nn, distances = _query_one(features[int(idx)], memory, k)
        elapsed = (time.perf_counter() - start) * 1000.0
        query_ms.append(elapsed)
        test_scores[pos] = score
        neighbor_sets.append(nn)
        label_id = int(labels[int(idx)])
        pred = int(score >= threshold)
        meta = {
            "flow_id": data.path.name.removesuffix(".pt") + f":{idx}",
            "artifact_row": int(idx),
            "split": "test",
            "label": data.attack if label_id == 1 else "BENIGN",
            "binary_label": "ATTACK" if label_id == 1 else "BENIGN",
            "binary_label_id": label_id,
        }
        source_meta = {}
        # ArtifactData does not keep raw meta to avoid carrying fields into tokens;
        # flow ids in score output are audit-only and do not affect scoring.
        rows.append(
            {
                "seed": data.seed,
                "heldout_attack": data.attack,
                "stream_pos": int(pos),
                **meta,
                "score": float(score),
                "threshold": float(threshold),
                "prediction": pred,
                "is_false_positive": bool(label_id == 0 and pred == 1),
                "is_true_positive": bool(label_id == 1 and pred == 1),
                "nearest_benign_indices": ";".join(str(int(x)) for x in nn.tolist()),
                "nearest_benign_distances": ";".join(f"{float(x):.6f}" for x in distances.tolist()),
                "query_ms": float(elapsed),
                "memory_policy": memory_policy,
                "threshold_mode": threshold_mode,
                "replay_clock": replay_order,
            }
        )
    replay_seconds = time.perf_counter() - start_replay
    y_true = labels[stream_idx].astype(np.int64)
    batch_metrics = _standard_metrics(y_true=y_true, test_scores=test_scores, val_scores=val_scores, threshold=threshold, threshold_type=threshold_status)
    preds = (test_scores >= threshold).astype(np.int64)
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    benign = int(np.sum(y_true == 0))
    attack = int(np.sum(y_true == 1))
    q = np.asarray(query_ms, dtype=np.float32)
    val_q = np.asarray(val_query_ms, dtype=np.float32)
    summary = {
        "seed": data.seed,
        "heldout_attack": data.attack,
        "token_path": str(data.path),
        "replay_clock": replay_order,
        "threshold_mode": threshold_mode,
        "threshold_status": threshold_status,
        "threshold": float(threshold),
        "k": int(k),
        "test_flows": int(len(stream_idx)),
        "test_benign": benign,
        "test_attack": attack,
        "false_positives": fp,
        "true_positives": tp,
        "false_positive_rate": float(fp / max(benign, 1)),
        "false_alerts_per_10k_benign": float(fp / max(benign, 1) * 10000.0),
        "attack_recall_online": float(tp / max(attack, 1)),
        "total_alerts": int(np.sum(preds == 1)),
        "replay_seconds": float(replay_seconds),
        "throughput_flows_per_second": float(len(stream_idx) / max(replay_seconds, 1e-12)),
        "query_ms_mean": float(np.mean(q)) if q.size else 0.0,
        "query_ms_p50": float(np.quantile(q, 0.50)) if q.size else 0.0,
        "query_ms_p95": float(np.quantile(q, 0.95)) if q.size else 0.0,
        "query_ms_p99": float(np.quantile(q, 0.99)) if q.size else 0.0,
        "calibration_query_ms_mean": float(np.mean(val_q)) if val_q.size else 0.0,
        "calibration_scores_p50": float(np.quantile(val_scores, 0.50)) if val_scores.size else 0.0,
        "calibration_scores_p99": float(np.quantile(val_scores, 0.99)) if val_scores.size else 0.0,
        "test_score_p50": float(np.quantile(test_scores, 0.50)) if test_scores.size else 0.0,
        "test_score_p95": float(np.quantile(test_scores, 0.95)) if test_scores.size else 0.0,
        "test_score_p99": float(np.quantile(test_scores, 0.99)) if test_scores.size else 0.0,
        "score_std": float(np.std(test_scores)) if test_scores.size else 0.0,
        "attack_labels_used_for_threshold": False,
        "attack_labels_used_for_memory": False,
        "raw_ip_used_as_token": False,
        "absolute_time_used_as_token": False,
        "five_tuple_used_as_token": False,
        "protocol_or_service_used_as_behavior": False,
        **memory_meta,
        **_sliding_fpr(rows, sliding_windows),
        **_alert_delay(rows),
        **_neighbor_turnover(neighbor_sets),
    }
    summary.update({f"batch_{key}": value for key, value in batch_metrics.items() if key in {"auroc", "auprc", "fpr95", "p99_macro_f1", "p99_attack_precision"}})
    return summary, rows


def _aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    metrics = [
        "false_positive_rate",
        "false_alerts_per_10k_benign",
        "attack_recall_online",
        "throughput_flows_per_second",
        "query_ms_mean",
        "query_ms_p50",
        "query_ms_p95",
        "query_ms_p99",
        "memory_mib_estimate",
        "memory_size",
        "neighbor_turnover_mean",
        "score_std",
        "delay_to_first_attack_alert_flows",
    ]
    out: list[dict[str, Any]] = []
    for key_values, items in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, key_values)}
        row["run_count"] = len(items)
        for metric in metrics:
            vals = [_safe_float(item.get(metric)) for item in items]
            vals = [val for val in vals if not math.isnan(val)]
            if not vals:
                continue
            row[f"{metric}_mean"] = float(statistics.mean(vals))
            row[f"{metric}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def _write_report(out_dir: Path, summary_rows: list[dict[str, Any]], missing: list[str]) -> None:
    agg = _aggregate(summary_rows, ["memory_policy", "threshold_mode"])
    lines = [
        "# FlowPrim KNN Online Replay Report",
        "",
        "This is an offline order-only replay over verified leave-one unknown token artifacts. The artifacts do not expose reliable absolute timestamps, so `replay_clock=artifact_order`; timestamp fields are not used as behavior tokens.",
        "",
        "## Aggregate Metrics",
        "",
        "| memory policy | threshold | runs | FPR | false alerts/10k | attack recall | q ms p50 | q ms p95 | throughput flows/s | memory MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in agg:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("memory_policy")),
                    str(row.get("threshold_mode")),
                    str(row.get("run_count")),
                    f"{float(row.get('false_positive_rate_mean', 0.0)):.4f}",
                    f"{float(row.get('false_alerts_per_10k_benign_mean', 0.0)):.2f}",
                    f"{float(row.get('attack_recall_online_mean', 0.0)):.4f}",
                    f"{float(row.get('query_ms_p50_mean', 0.0)):.4f}",
                    f"{float(row.get('query_ms_p95_mean', 0.0)):.4f}",
                    f"{float(row.get('throughput_flows_per_second_mean', 0.0)):.2f}",
                    f"{float(row.get('memory_mib_estimate_mean', 0.0)):.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Leakage Controls",
            "",
            "- Memory uses train BENIGN rows only.",
            "- Thresholds use val BENIGN scores only.",
            "- Test labels are used only after replay for metrics.",
            "- Raw IP, absolute timestamp, complete five-tuple, protocol, and service are not behavior tokens or memory grouping keys.",
            "",
            "## Limitations",
            "",
            "- This is not live packet capture; it replays finalized flow artifacts.",
            "- Because verified artifacts do not include usable timestamps, delay is measured in number of replayed flows rather than seconds.",
            f"- Missing corpora: {len(missing)}.",
        ]
    )
    (out_dir / "online_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(agg, out_dir / "online_replay_aggregate.csv")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    token_dir = Path(args.token_dir)
    cfg = StructuralPrimitiveConfig(
        enabled=True,
        enable_packet_shape_primitives=True,
        enable_burst_shape_primitives=True,
        enable_timing_rhythm_primitives=True,
        enable_direction_transition_primitives=True,
        enable_composite_primitives=True,
        min_support=int(args.structural_min_support),
        max_structural_primitives_per_family=int(args.max_structural_per_family),
    )
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in args.seeds:
        for attack in args.attacks:
            path = _token_path(token_dir, attack, int(seed))
            if not path.exists():
                missing.append(str(path))
                print(f"[WARN] missing token corpus: {path}", file=sys.stderr)
                continue
            print(f"[replay] loading {attack} seed {seed}", flush=True)
            data = _load_artifact(path, attack, int(seed), cfg)
            for threshold_mode in args.threshold_modes:
                for memory_policy in args.memory_policies:
                    ratio = float(args.memory_ratio if memory_policy != "full" else 1.0)
                    summary, rows = _replay_one(
                        data,
                        k=int(args.k),
                        threshold_mode=str(threshold_mode),
                        replay_order=str(args.replay_order),
                        memory_policy=str(memory_policy),
                        memory_ratio=ratio,
                        sliding_windows=[int(item) for item in args.sliding_windows],
                    )
                    summary_rows.append(summary)
                    if args.write_scores:
                        score_rows.extend(rows)
            _write_csv(summary_rows, out_dir / "online_replay_summary.csv")
            if args.write_scores:
                _write_csv(score_rows, out_dir / "online_replay_scores.csv")
    _write_csv(summary_rows, out_dir / "online_replay_summary.csv")
    if args.write_scores:
        _write_csv(score_rows, out_dir / "online_replay_scores.csv")
    _write_json({"missing": missing, "runs": len(summary_rows), "attacks": args.attacks, "seeds": args.seeds}, out_dir / "online_replay_manifest.json")
    _write_report(out_dir, summary_rows, missing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run behavior-only FlowPrim KNN online replay over verified token artifacts.")
    parser.add_argument("--token-dir", default=str(DEFAULT_TOKEN_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threshold-modes", nargs="+", default=["p99", "evt_p99"])
    parser.add_argument("--memory-policies", nargs="+", default=["full", "tail_preserving_coreset"])
    parser.add_argument("--memory-ratio", type=float, default=0.75)
    parser.add_argument("--replay-order", choices=["artifact_order", "stable_hash"], default="artifact_order")
    parser.add_argument("--sliding-windows", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--structural-min-support", type=int, default=5)
    parser.add_argument("--max-structural-per-family", type=int, default=24)
    parser.add_argument("--write-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
