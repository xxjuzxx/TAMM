#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

import run_memory_optimization_experiments as M


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT.parent / "paper" / "flowprim_motif_memory_icdm"
TABLE_DIR = PAPER_DIR / "tables"


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or value == "":
            return "--"
        x = float(value)
        if not np.isfinite(x):
            return "--"
        return f"{x:.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def _summarize(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, items in grouped.items():
        row = {key: value for key, value in zip(keys, key_values)}
        row["num_runs"] = len(items)
        for metric in metrics:
            vals: list[float] = []
            for item in items:
                try:
                    value = float(item.get(metric, ""))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    vals.append(value)
            mean, std = _mean_std(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    return out


def _score_one(query: np.ndarray, memory: np.ndarray, k: int) -> float:
    if memory.size == 0:
        return 1.0
    distances = M._cosine_distances(query.reshape(1, -1), memory).reshape(-1)
    kk = max(1, min(int(k), distances.shape[0]))
    return float(np.mean(np.partition(distances, kk - 1)[:kk]))


def _time_queries(
    features: np.ndarray,
    memory: np.ndarray,
    eval_idx: np.ndarray,
    *,
    k: int,
    max_eval: int,
    repeats: int,
) -> dict[str, float]:
    chosen = eval_idx[: min(int(max_eval), len(eval_idx))]
    if chosen.size == 0:
        return {
            "eval_flows": 0,
            "query_ms_mean": float("nan"),
            "query_ms_std": float("nan"),
            "query_ms_p50": float("nan"),
            "query_ms_p95": float("nan"),
            "query_ms_p99": float("nan"),
        }
    warmup = chosen[: min(32, len(chosen))]
    for idx in warmup.tolist():
        _score_one(features[int(idx)], memory, k)
    timings: list[float] = []
    for _ in range(int(repeats)):
        for idx in chosen.tolist():
            start = time.perf_counter()
            _score_one(features[int(idx)], memory, k)
            timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings, dtype=np.float64)
    return {
        "eval_flows": int(chosen.size),
        "query_ms_mean": float(np.mean(arr)),
        "query_ms_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "query_ms_p50": float(np.quantile(arr, 0.50)),
        "query_ms_p95": float(np.quantile(arr, 0.95)),
        "query_ms_p99": float(np.quantile(arr, 0.99)),
    }


def _time_batch_scores(
    features: np.ndarray,
    memory: np.ndarray,
    eval_idx: np.ndarray,
    *,
    k: int,
    max_eval: int,
    repeats: int,
) -> dict[str, float]:
    chosen = eval_idx[: min(int(max_eval), len(eval_idx))]
    if chosen.size == 0:
        return {
            "eval_flows": 0,
            "batch_ms_per_flow_mean": float("nan"),
            "batch_ms_per_flow_std": float("nan"),
            "batch_ms_per_flow_p50": float("nan"),
            "batch_ms_per_flow_p95": float("nan"),
        }
    eval_x = features[chosen]
    _ = M._exact_scores_and_neighbors(eval_x, memory, k)
    timings: list[float] = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        _ = M._exact_scores_and_neighbors(eval_x, memory, k)
        timings.append((time.perf_counter() - start) * 1000.0 / max(len(chosen), 1))
    arr = np.asarray(timings, dtype=np.float64)
    return {
        "eval_flows": int(chosen.size),
        "batch_ms_per_flow_mean": float(np.mean(arr)),
        "batch_ms_per_flow_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "batch_ms_per_flow_p50": float(np.quantile(arr, 0.50)),
        "batch_ms_per_flow_p95": float(np.quantile(arr, 0.95)),
    }


def _metrics_for_memory(data: M.ArtifactData, features: np.ndarray, memory: np.ndarray, k: int) -> dict[str, Any]:
    val_scores, _ = M._exact_scores_and_neighbors(features[data.val_idx], memory, k)
    test_scores, _ = M._exact_scores_and_neighbors(features[data.test_idx], memory, k)
    return M._standard_metrics(y_true=data.labels[data.test_idx].astype(np.int64), test_scores=test_scores, val_scores=val_scores)


def _memory_by_strategy(
    data: M.ArtifactData,
    features: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
    k: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    train_local = np.arange(len(data.train_idx), dtype=np.int64)
    full_memory = features[data.train_idx]
    if strategy == "uniform_exact":
        return full_memory, {"memory_size": int(full_memory.shape[0]), "source": "full benign memory"}
    if strategy == "coreset_tail_preserving_0.5":
        n = full_memory.shape[0]
        keep = max(k, min(n, int(round(n * 0.5))))
        centroid = M._normalize_l2(full_memory.mean(axis=0, keepdims=True)).reshape(-1)
        tail_score = M._cosine_distances(full_memory, centroid.reshape(1, -1)).reshape(-1)
        tail_keep = max(1, keep // 2)
        tail_idx = np.argsort(-tail_score)[:tail_keep]
        remaining = np.setdiff1d(np.arange(n), tail_idx, assume_unique=False)
        fill = max(0, keep - len(tail_idx))
        fill_idx = rng.choice(remaining, size=fill, replace=False) if fill > 0 else np.empty(0, dtype=np.int64)
        selected = np.sort(np.concatenate([tail_idx, fill_idx]).astype(np.int64))
        return full_memory[selected], {"memory_size": int(selected.size), "source": "tail-preserving 50% coreset"}
    if strategy == "coreset_random_0.5":
        n = full_memory.shape[0]
        keep = max(k, min(n, int(round(n * 0.5))))
        selected = np.sort(rng.choice(np.arange(n), size=keep, replace=False))
        return full_memory[selected], {"memory_size": int(selected.size), "source": "random 50% coreset"}
    if strategy == "tfidf_train_only":
        tfidf_features = M._features_idf(data.raw_matrix, data.train_idx)
        return tfidf_features[data.train_idx], {
            "memory_size": int(len(data.train_idx)),
            "source": "train-only TF-IDF memory",
            "features_override": tfidf_features,
        }
    if strategy == "evt_tail_p99":
        return None, {"memory_size": "", "source": "threshold-only EVT diagnostic"}

    shuffled = np.arange(len(data.train_idx), dtype=np.int64)
    rng.shuffle(shuffled)
    core_size = max(k, int(round(len(shuffled) * 0.75)))
    core = np.sort(shuffled[:core_size])
    candidates = np.sort(shuffled[core_size:])
    if strategy == "tail_aware_update":
        if candidates.size:
            core_scores, _ = M._exact_scores_and_neighbors(full_memory[candidates], full_memory[core], k)
            low_cut = float(np.percentile(core_scores, 50.0))
            p95_cut = float(np.percentile(core_scores, 95.0))
            low_add = candidates[core_scores <= low_cut]
            tail_add = candidates[core_scores >= p95_cut]
            tail_keep = tail_add[: max(1, len(candidates) // 20)] if tail_add.size else np.empty(0, dtype=np.int64)
            selected = np.sort(np.concatenate([core, low_add, tail_keep]))
        else:
            selected = core
        return full_memory[selected], {"memory_size": int(selected.size), "source": "tail-aware benign update"}
    if strategy == "random_benign_update":
        if candidates.size:
            random_add = np.sort(rng.choice(candidates, size=max(1, len(candidates) // 2), replace=False))
            selected = np.sort(np.concatenate([core, random_add]))
        else:
            selected = core
        return full_memory[selected], {"memory_size": int(selected.size), "source": "random benign update"}
    if strategy == "oracle_pollution_5pct_attack_diagnostic":
        attack_test = data.test_idx[data.labels[data.test_idx] == 1]
        if attack_test.size:
            pollute_n = max(1, int(round(len(train_local) * 0.05)))
            sample = attack_test[: min(pollute_n, len(attack_test))]
            memory = np.vstack([full_memory, features[sample]])
        else:
            memory = full_memory
        return memory, {"memory_size": int(memory.shape[0]), "source": "oracle attack-pollution diagnostic"}
    raise ValueError(f"Unknown strategy: {strategy}")


def _latex_table(headers: list[str], rows: list[list[str]]) -> str:
    spec = "l" * len(headers)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def rebuild(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    token_dir = Path(args.token_dir)
    cfg = M.StructuralPrimitiveConfig(
        enabled=True,
        enable_packet_shape_primitives=True,
        enable_burst_shape_primitives=True,
        enable_timing_rhythm_primitives=True,
        enable_direction_transition_primitives=True,
        enable_composite_primitives=True,
        min_support=int(args.structural_min_support),
        max_structural_primitives_per_family=int(args.max_structural_per_family),
    )
    attacks = [str(x) for x in args.attacks]
    seeds = [int(x) for x in args.seeds]
    k = int(args.k)
    strategy_labels = [
        ("uniform_exact", "Exact full memory"),
        ("coreset_tail_preserving_0.5", "Tail coreset 50\\%"),
        ("coreset_random_0.5", "Random coreset 50\\%"),
        ("tfidf_train_only", "Train-only TF-IDF"),
        ("evt_tail_p99", "EVT tail P99"),
        ("tail_aware_update", "Tail-aware update"),
        ("random_benign_update", "Random benign update"),
        ("oracle_pollution_5pct_attack_diagnostic", "5\\% attack pollution"),
    ]

    original_summary_rows = _read_csv(ROOT / "results" / "memory_optimization" / "summary_by_setting.csv")
    original_by_setting = {
        (row.get("experiment_group"), row.get("setting")): row
        for row in original_summary_rows
    }
    strategy_source = {
        "uniform_exact": ("baseline", "uniform_exact"),
        "coreset_tail_preserving_0.5": ("indexed_retrieval", "coreset_tail_preserving_0.5"),
        "coreset_random_0.5": ("indexed_retrieval", "coreset_random_0.5"),
        "tfidf_train_only": ("token_weighting", "tfidf_train_only"),
        "evt_tail_p99": ("calibration", "evt_tail_p99"),
        "tail_aware_update": ("memory_governance", "tail_aware_update"),
        "random_benign_update": ("memory_governance", "random_benign_update"),
        "oracle_pollution_5pct_attack_diagnostic": ("memory_governance", "oracle_pollution_5pct_attack_diagnostic"),
    }

    governance_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    state_cache: dict[tuple[str, int], tuple[M.ArtifactData, np.ndarray]] = {}
    for seed in seeds:
        for attack in attacks:
            path = M._token_path(token_dir, attack, seed)
            if not path.exists():
                print(f"[WARN] missing {path}")
                continue
            data = M._load_artifact(path, attack, seed, cfg)
            features = M._features_uniform(data.raw_matrix)
            state_cache[(attack, seed)] = (data, features)
            for strategy, label in strategy_labels:
                stable_offset = sum(ord(ch) for ch in strategy) % 997
                rng = np.random.default_rng(seed * 10000 + len(attack) * 100 + stable_offset)
                memory, meta = _memory_by_strategy(data, features, strategy, rng, k)
                if memory is None:
                    row = {
                        "strategy": strategy,
                        "label": label,
                        "attack": attack,
                        "seed": seed,
                        "batch_ms_per_flow_mean": float("nan"),
                        "memory_size": "",
                        "source": meta["source"],
                    }
                else:
                    feature_source = meta.get("features_override", features)
                    timing = _time_batch_scores(feature_source, memory, np.concatenate([data.val_idx, data.test_idx]), k=k, max_eval=args.max_eval, repeats=args.repeats)
                    row = {
                        "strategy": strategy,
                        "label": label,
                        "attack": attack,
                        "seed": seed,
                        "memory_size": meta["memory_size"],
                        "source": meta["source"],
                        **timing,
                    }
                governance_rows.append(row)

    for attack in ["Botnet", "DDoS", "Probe"]:
        if (attack, args.scaling_seed) not in state_cache:
            path = M._token_path(token_dir, attack, args.scaling_seed)
            if not path.exists():
                continue
            data = M._load_artifact(path, attack, args.scaling_seed, cfg)
            features = M._features_uniform(data.raw_matrix)
            state_cache[(attack, args.scaling_seed)] = (data, features)
        data, features = state_cache[(attack, args.scaling_seed)]
        train_idx = data.train_idx
        for size in args.scaling_sizes:
            size = int(size)
            if len(train_idx) >= size:
                refs_idx = train_idx[:size]
                source = "real subsample"
            else:
                reps = int(math.ceil(size / max(len(train_idx), 1)))
                refs_idx = np.tile(train_idx, reps)[:size]
                source = "duplicated memory"
            memory = features[refs_idx]
            timing = _time_batch_scores(features, memory, data.test_idx, k=k, max_eval=args.max_eval, repeats=args.repeats)
            scaling_rows.append(
                {
                    "attack": attack,
                    "seed": args.scaling_seed,
                    "memory_size": size,
                    "source": source,
                    "memory_mb": float(memory.nbytes / (1024.0 * 1024.0)),
                    **timing,
                }
            )

    gov_summary = _summarize(
        governance_rows,
        ["strategy", "label"],
        [
            "batch_ms_per_flow_mean",
            "batch_ms_per_flow_p50",
            "batch_ms_per_flow_p95",
            "memory_size",
        ],
    )
    scaling_summary = _summarize(
        scaling_rows,
        ["memory_size", "source"],
        ["batch_ms_per_flow_mean", "batch_ms_per_flow_p50", "batch_ms_per_flow_p95", "memory_mb"],
    )
    _write_csv(governance_rows, out_dir / "memory_governance_runtime_unified_runs.csv")
    _write_csv(gov_summary, out_dir / "memory_governance_runtime_unified_summary.csv")
    _write_csv(scaling_rows, out_dir / "knn_scaling_runtime_unified_runs.csv")
    _write_csv(scaling_summary, out_dir / "knn_scaling_runtime_unified_summary.csv")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_scope": "batch exact-KNN scoring ms/flow over motif transaction vectors with a fixed evaluation cap",
                "max_eval": int(args.max_eval),
                "repeats": int(args.repeats),
                "attacks": attacks,
                "seeds": seeds,
                "scaling_attacks": ["Botnet", "DDoS", "Probe"],
                "scaling_seed": int(args.scaling_seed),
                "note": "Quality metrics in the paper table are read from the original memory-governance experiment; runtime is regenerated under one batch exact-KNN scoring protocol.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    order = {strategy: i for i, (strategy, _) in enumerate(strategy_labels)}
    gov_summary = sorted(gov_summary, key=lambda row: order.get(str(row.get("strategy")), 999))
    table_rows: list[list[str]] = []
    for row in gov_summary:
        source_key = strategy_source.get(str(row.get("strategy")))
        original = original_by_setting.get(source_key, {}) if source_key else {}
        table_rows.append(
            [
                str(row["label"]),
                _fmt(original.get("auroc_mean")),
                _fmt(original.get("recall_at_0_1pct_fpr_mean")),
                _fmt(original.get("recall_at_1pct_fpr_mean")),
                _fmt(original.get("p99_realized_fpr_mean")),
                _fmt(original.get("false_alerts_per_10k_benign_mean"), 1),
                _fmt(row.get("batch_ms_per_flow_mean_mean"), 3),
                _fmt(original.get("memory_size_mean") if original.get("memory_size_mean", "") != "" else row.get("memory_size_mean"), 0),
            ]
        )
    (TABLE_DIR / "table_memory_governance_optimization.tex").write_text(
        _latex_table(["Setting", "AUROC", "R@0.1\\%", "R@1\\%", "P99 FPR", "Alerts/10k", "Batch ms/flow", "Memory"], table_rows),
        encoding="utf-8",
    )

    scaling_summary = sorted(scaling_summary, key=lambda row: float(row["memory_size"]))
    scaling_lookup = {int(float(row["memory_size"])): row for row in scaling_summary}
    runtime_lines = [
        r"\begin{tabularx}{\columnwidth}{lXcc}",
        r"\toprule",
        r"Category & Setting & Throughput / latency & Scope \\",
        r"\midrule",
        r"Artifact load & processed flow JSONL & 40,797 flows/s & existing Zeek-aligned flows \\",
        r"Artifact load & profile motif JSONL & 58,489 flows/s & cached motif artifacts \\",
        r"Diagnosis & KNN scoring to records & 8,710 flows/s & first 512 test flows \\",
        r"\midrule",
    ]
    for size in [1000, 10000, 50000]:
        row = scaling_lookup.get(size)
        if row is None:
            continue
        source = "real subsample" if "real" in str(row.get("source")) else "duplicated memory"
        runtime_lines.append(
            rf"KNN scaling & {size // 1000}k benign memory & {_fmt(row.get('batch_ms_per_flow_mean_mean'), 3)} ms/flow & exact KNN, {source} \\"
        )
    runtime_lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    (TABLE_DIR / "table_runtime_scaling_compact.tex").write_text("\n".join(runtime_lines), encoding="utf-8")

    print(json.dumps({"output": str(out_dir), "governance_rows": len(governance_rows), "scaling_rows": len(scaling_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild TAMM runtime tables with one per-query exact-KNN timing protocol.")
    parser.add_argument("--token-dir", default=str(M.DEFAULT_TOKEN_DIR))
    parser.add_argument("--output", default=str(ROOT / "results" / "runtime_unified"))
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-eval", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scaling-seed", type=int, default=43)
    parser.add_argument("--scaling-sizes", nargs="+", type=int, default=[1000, 10000, 50000])
    parser.add_argument("--structural-min-support", type=int, default=5)
    parser.add_argument("--max-structural-per-family", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    rebuild(parse_args())


if __name__ == "__main__":
    main()
