#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNSW = "data/raw/UNSW-NB15/CSV Files/UNSW-NB15_*.csv"
DEFAULT_OUT = ROOT / "results" / "unsw_nb15_behavior_knn_existing"

UNSW_COLUMNS = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "Sload",
    "Dload",
    "Spkts",
    "Dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smeansz",
    "dmeansz",
    "trans_depth",
    "res_bdy_len",
    "Sjit",
    "Djit",
    "Stime",
    "Ltime",
    "Sintpkt",
    "Dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "Label",
]

# Behavior-summary columns only. IPs, ports, protocol, service, and absolute
# timestamps are excluded from transaction tokens and memory keys.
COUNT_COLUMNS = ["Spkts", "Dpkts", "sloss", "dloss"]
BYTE_COLUMNS = ["sbytes", "dbytes", "Sload", "Dload"]
LENGTH_COLUMNS = ["smeansz", "dmeansz"]
TIMING_COLUMNS = ["dur", "Sintpkt", "Dintpkt", "Sjit", "Djit", "tcprtt", "synack", "ackdat"]
STATE_COLUMNS = ["state"]
TCP_SHAPE_COLUMNS = ["swin", "dwin"]
CONTENT_SUMMARY_COLUMNS = ["trans_depth", "res_bdy_len", "ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd"]

NUMERIC_BEHAVIOR_COLUMNS = (
    COUNT_COLUMNS
    + BYTE_COLUMNS
    + LENGTH_COLUMNS
    + TIMING_COLUMNS
    + TCP_SHAPE_COLUMNS
    + CONTENT_SUMMARY_COLUMNS
)

FEATURE_VIEWS = {
    "flow_summary_all": NUMERIC_BEHAVIOR_COLUMNS + STATE_COLUMNS,
    "packet_burst_proxy": COUNT_COLUMNS + BYTE_COLUMNS + LENGTH_COLUMNS + TIMING_COLUMNS,
    "count_byte_timing": COUNT_COLUMNS + BYTE_COLUMNS + TIMING_COLUMNS,
    "count_byte_only": COUNT_COLUMNS + BYTE_COLUMNS,
    "timing_only": TIMING_COLUMNS,
    "length_only": LENGTH_COLUMNS,
    "state_free_all": NUMERIC_BEHAVIOR_COLUMNS,
}

FAMILY_MAP = {
    "normal": "BENIGN",
    "generic": "Generic",
    "exploits": "Exploits",
    "fuzzers": "Fuzzers",
    "dos": "DoS",
    "reconnaissance": "Reconnaissance",
    "analysis": "Analysis",
    "backdoor": "Backdoor",
    "backdoors": "Backdoor",
    "shellcode": "Shellcode",
    "worms": "Worms",
}


def _expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        if any(char in item for char in "*?[]"):
            paths.extend(Path(path) for path in sorted(glob.glob(item)))
        elif Path(item).is_dir():
            paths.extend(sorted(Path(item).glob("*.csv")))
        else:
            paths.append(Path(item))
    return [path for path in paths if path.exists()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _family(value: object, label: object = None) -> str:
    if label is not None:
        try:
            if int(float(label)) == 0:
                return "BENIGN"
        except Exception:
            pass
    raw = str(value).strip().lower()
    if raw in {"", "nan", "none", "-", "normal"}:
        return "BENIGN"
    return FAMILY_MAP.get(raw, str(value).strip() or "OtherAttack")


def _load_rows(paths: list[Path], *, chunksize: int, max_rows_per_file: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    raw_counts: Counter[str] = Counter()
    rows_seen = 0
    for path in paths:
        file_rows = 0
        for chunk in pd.read_csv(path, header=None, names=UNSW_COLUMNS, chunksize=chunksize, low_memory=False):
            if max_rows_per_file is not None:
                remaining = int(max_rows_per_file) - file_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()
            file_rows += len(chunk)
            chunk["family"] = [_family(cat, lab) for cat, lab in zip(chunk["attack_cat"], chunk["Label"])]
            chunk["__source_file"] = str(path)
            raw_counts.update(chunk["family"].tolist())
            keep_cols = list(dict.fromkeys(["family", "__source_file", *FEATURE_VIEWS["flow_summary_all"]]))
            frames.append(chunk[keep_cols].copy())
            rows_seen += len(chunk)
    if not frames:
        raise RuntimeError("No UNSW rows were loaded.")
    return pd.concat(frames, ignore_index=True, sort=False), {"family_counts": dict(sorted(raw_counts.items())), "rows_seen": rows_seen}


def _sample_frame(frame: pd.DataFrame, caps: dict[str, int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    parts: list[pd.DataFrame] = []
    for family, cap in caps.items():
        subset = frame.loc[frame["family"] == family]
        if subset.empty or cap <= 0:
            continue
        take = min(len(subset), int(cap))
        idx = subset.index.to_numpy(dtype="int64").copy()
        rng.shuffle(idx)
        parts.append(frame.loc[np.sort(idx[:take])].copy())
    if not parts:
        raise RuntimeError("Sampling produced no rows.")
    return pd.concat(parts, ignore_index=True, sort=False)


def _log_transform(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.clip(values, 0.0, None))


def _fit_edges(train_frame: pd.DataFrame, columns: list[str], bins: int) -> dict[str, list[float]]:
    edges: dict[str, list[float]] = {}
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    for column in columns:
        if column in STATE_COLUMNS:
            continue
        values = _log_transform(pd.to_numeric(train_frame[column], errors="coerce").fillna(0.0).to_numpy(dtype="float32"))
        qs = np.quantile(values[np.isfinite(values)], quantiles) if values.size else []
        edges[column] = sorted({float(q) for q in qs if math.isfinite(float(q))})
    return edges


def _tokenize_row(row: pd.Series, columns: list[str], edges: dict[str, list[float]], view: str) -> list[str]:
    tokens: list[str] = []
    for column in columns:
        safe = column.upper().replace("/", "_").replace(" ", "_").replace("-", "_")
        if column in STATE_COLUMNS:
            state = str(row.get(column, "none")).strip().lower() or "none"
            if state not in {"fin", "con", "int", "req", "rst"}:
                state = "other"
            tokens.append(f"UNSW_FLOWSTAT_{view.upper()}_STATE_{state.upper()}")
            continue
        value = float(pd.to_numeric(row.get(column, 0.0), errors="coerce") or 0.0)
        lv = float(np.log1p(max(0.0, value)))
        bucket = int(np.searchsorted(np.asarray(edges.get(column, []), dtype="float32"), lv, side="right"))
        tokens.append(f"UNSW_FLOWSTAT_{view.upper()}_{safe}_BIN_{bucket}")
    spkts = float(pd.to_numeric(row.get("Spkts", 0.0), errors="coerce") or 0.0)
    dpkts = float(pd.to_numeric(row.get("Dpkts", 0.0), errors="coerce") or 0.0)
    total = spkts + dpkts
    if total <= 4:
        tokens.append("UNSW_FLOWSTAT_SHORT_FLOW")
    if total > 0:
        frac = spkts / total
        if frac >= 0.85:
            tokens.append("UNSW_FLOWSTAT_SRC_DOMINANT")
        elif frac <= 0.15:
            tokens.append("UNSW_FLOWSTAT_DST_DOMINANT")
        else:
            tokens.append("UNSW_FLOWSTAT_BIDIRECTIONAL")
    return tokens


def _build_matrix(frame: pd.DataFrame, columns: list[str], view: str, edges: dict[str, list[float]], vocab: dict[str, int] | None) -> tuple[np.ndarray, dict[str, int], list[list[str]]]:
    token_rows = [_tokenize_row(row, columns, edges, view) for _idx, row in frame.iterrows()]
    if vocab is None:
        vocab = {token: idx for idx, token in enumerate(sorted({token for tokens in token_rows for token in tokens}))}
    matrix = np.zeros((len(token_rows), len(vocab)), dtype="float32")
    for row_idx, tokens in enumerate(token_rows):
        for token in tokens:
            idx = vocab.get(token)
            if idx is not None:
                matrix[row_idx, idx] += 1.0
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0), vocab, token_rows


def _score_knn(query: np.ndarray, memory: np.ndarray, k: int, batch_size: int) -> np.ndarray:
    out = np.zeros(query.shape[0], dtype="float32")
    kk = max(1, min(int(k), memory.shape[0]))
    for start in range(0, query.shape[0], batch_size):
        end = min(start + batch_size, query.shape[0])
        distances = (1.0 - np.clip(query[start:end] @ memory.T, -1.0, 1.0)).astype("float32")
        part = np.argpartition(distances, kk - 1, axis=1)[:, :kk]
        vals = np.take_along_axis(distances, part, axis=1)
        out[start:end] = np.mean(vals, axis=1)
    return out


def _macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for cls in (0, 1):
        tp = int(np.sum((y_true == cls) & (pred == cls)))
        fp = int(np.sum((y_true != cls) & (pred == cls)))
        fn = int(np.sum((y_true == cls) & (pred != cls)))
        vals.append(float((2 * tp) / max(2 * tp + fp + fn, 1)))
    return float(sum(vals) / 2.0)


def _metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype("int64")
    benign = y_true == 0
    attack = y_true == 1
    fp = int(np.sum((pred == 1) & benign))
    tn = int(np.sum((pred == 0) & benign))
    tp = int(np.sum((pred == 1) & attack))
    fn = int(np.sum((pred == 0) & attack))
    return {
        "macro_f1": _macro_f1(y_true, pred),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "attack_recall": float(tp / max(tp + fn, 1)),
        "attack_precision": float(tp / max(tp + fp, 1)),
        "false_positives": fp,
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def _best_recall_under_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in np.unique(scores):
        m = _metrics_at_threshold(y_true, scores, float(threshold))
        if float(m["false_positive_rate"]) <= target_fpr:
            if best is None or float(m["attack_recall"]) > float(best["attack_recall"]):
                best = {**m, "threshold": float(threshold)}
    if best is None:
        threshold = float(np.max(scores) + 1e-6)
        best = {**_metrics_at_threshold(y_true, scores, threshold), "threshold": threshold}
    return best


def _evaluate(y_true: np.ndarray, test_scores: np.ndarray, val_scores: np.ndarray) -> dict[str, Any]:
    p99_threshold = float(np.percentile(val_scores, 99.0))
    p99 = _metrics_at_threshold(y_true, test_scores, p99_threshold)
    r001 = _best_recall_under_fpr(y_true, test_scores, 0.001)
    r01 = _best_recall_under_fpr(y_true, test_scores, 0.01)
    row: dict[str, Any] = {}
    try:
        row["auroc"] = float(roc_auc_score(y_true, test_scores))
        row["auprc"] = float(average_precision_score(y_true, test_scores))
        fpr, tpr, _thresholds = roc_curve(y_true, test_scores)
        eligible = fpr[tpr >= 0.95]
        row["fpr95"] = float(np.min(eligible)) if eligible.size else ""
    except ValueError:
        row.update({"auroc": "", "auprc": "", "fpr95": ""})
    row.update(
        {
            "recall_at_0_1pct_fpr": r001["attack_recall"],
            "recall_at_1pct_fpr": r01["attack_recall"],
            "oracle_1pct_threshold": r01["threshold"],
            "p99_threshold": p99_threshold,
            "p99_realized_fpr": p99["false_positive_rate"],
            "false_alerts_per_10k_benign": float(p99["false_positive_rate"] * 10000.0),
            "p99_attack_recall": p99["attack_recall"],
            "p99_macro_f1": p99["macro_f1"],
        }
    )
    return row


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[Any, Any]:
    vals = []
    for row in rows:
        try:
            val = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return "", ""
    return float(np.mean(vals)), float(np.std(vals))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["heldout_attack"]), str(row["feature_view"]))].append(row)
    metrics = ["auroc", "auprc", "fpr95", "recall_at_0_1pct_fpr", "recall_at_1pct_fpr", "p99_realized_fpr", "false_alerts_per_10k_benign", "p99_attack_recall", "p99_macro_f1", "vocab_size", "memory_size", "query_ms_per_flow"]
    out: list[dict[str, Any]] = []
    for (attack, view), items in sorted(groups.items()):
        row: dict[str, Any] = {"dataset": "UNSW-NB15", "heldout_attack": attack, "feature_view": view, "runs": len(items), "seed_count": len({item["seed"] for item in items})}
        for metric in metrics:
            mean, std = _mean_std(items, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    for view in sorted({str(row["feature_view"]) for row in rows}):
        items = [row for row in rows if str(row["feature_view"]) == view]
        row = {"dataset": "UNSW-NB15", "heldout_attack": "Aggregate", "feature_view": view, "runs": len(items), "seed_count": len({item["seed"] for item in items})}
        for metric in metrics:
            mean, std = _mean_std(items, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _expand_inputs(args.unsw_csv)
    if not paths:
        raise FileNotFoundError("No UNSW-NB15 CSV files were found.")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, load_meta = _load_rows(paths, chunksize=args.chunksize, max_rows_per_file=args.max_rows_per_file)
    caps = {"BENIGN": int(args.benign_train + args.benign_val + args.benign_test)}
    for attack in args.attacks:
        caps[attack] = int(args.attack_test_cap)
    sampled = _sample_frame(frame, caps, args.sample_seed)
    label_counts = {str(k): int(v) for k, v in Counter(sampled["family"].tolist()).items()}
    attacks = [attack for attack in args.attacks if label_counts.get(attack, 0) > 0]
    missing_attacks = [attack for attack in args.attacks if label_counts.get(attack, 0) <= 0]
    metrics_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        benign_idx = sampled.index[sampled["family"] == "BENIGN"].to_numpy(dtype="int64").copy()
        rng.shuffle(benign_idx)
        benign_cap = int(args.benign_train + args.benign_val + args.benign_test)
        if len(benign_idx) < benign_cap:
            raise RuntimeError(f"Need {benign_cap} benign samples, found {len(benign_idx)}.")
        train_idx = benign_idx[: args.benign_train]
        val_idx = benign_idx[args.benign_train : args.benign_train + args.benign_val]
        test_benign_idx = benign_idx[args.benign_train + args.benign_val : benign_cap]
        for attack in attacks:
            attack_idx = sampled.index[sampled["family"] == attack].to_numpy(dtype="int64").copy()
            rng.shuffle(attack_idx)
            test_attack_idx = attack_idx[: min(len(attack_idx), int(args.attack_test_cap))]
            if len(test_attack_idx) == 0:
                continue
            y_true = np.concatenate([np.zeros(len(test_benign_idx), dtype="int64"), np.ones(len(test_attack_idx), dtype="int64")])
            test_idx = np.concatenate([test_benign_idx, test_attack_idx])
            split_rows.append({"seed": seed, "heldout_attack": attack, "train_benign": len(train_idx), "val_benign": len(val_idx), "test_benign": len(test_benign_idx), "test_attack": len(test_attack_idx), "status": "ok"})
            for view in args.views:
                columns = [column for column in FEATURE_VIEWS[view] if column in sampled.columns]
                train_frame = sampled.loc[train_idx, columns]
                val_frame = sampled.loc[val_idx, columns]
                test_frame = sampled.loc[test_idx, columns]
                fit_start = time.perf_counter()
                edges = _fit_edges(train_frame, columns, args.bins)
                train_x, vocab, _train_tokens = _build_matrix(train_frame, columns, view, edges, vocab=None)
                val_x, _vocab, _val_tokens = _build_matrix(val_frame, columns, view, edges, vocab=vocab)
                test_x, _vocab, test_tokens = _build_matrix(test_frame, columns, view, edges, vocab=vocab)
                fit_seconds = time.perf_counter() - fit_start
                score_start = time.perf_counter()
                val_scores = _score_knn(val_x, train_x, args.k, args.score_batch_size)
                test_scores = _score_knn(test_x, train_x, args.k, args.score_batch_size)
                score_seconds = time.perf_counter() - score_start
                metric = _evaluate(y_true, test_scores, val_scores)
                counts = [len(tokens) for tokens in test_tokens]
                metric.update(
                    {
                        "dataset": "UNSW-NB15",
                        "source_type": "full_csv_flow_summary",
                        "evidence_granularity": "flow_summary_behavior_tokens",
                        "heldout_attack": attack,
                        "seed": int(seed),
                        "feature_view": view,
                        "primitive_version": "not_available_without_packet_sequence",
                        "threshold_type": "benign_validation_p99",
                        "train_benign": int(len(train_idx)),
                        "val_benign": int(len(val_idx)),
                        "test_benign_count": int(len(test_benign_idx)),
                        "test_attack_count": int(len(test_attack_idx)),
                        "memory_size": int(train_x.shape[0]),
                        "vocab_size": int(train_x.shape[1]),
                        "avg_token_count_per_flow": float(np.mean(counts)) if counts else 0.0,
                        "fit_seconds": float(fit_seconds),
                        "score_seconds": float(score_seconds),
                        "query_ms_per_flow": float((score_seconds * 1000.0) / max(len(val_idx) + len(test_idx), 1)),
                        "attack_labels_used_for_threshold": False,
                        "attack_labels_used_for_memory": False,
                        "raw_ip_used_as_token": False,
                        "absolute_time_used_as_token": False,
                        "five_tuple_used_as_token": False,
                        "protocol_or_service_used_as_behavior": False,
                        "port_used_as_behavior": False,
                    }
                )
                metrics_rows.append(metric)
                print(json.dumps({k: metric[k] for k in ("seed", "heldout_attack", "feature_view", "auroc", "recall_at_1pct_fpr", "p99_realized_fpr")}, sort_keys=True))
    summary_rows = _summarize(metrics_rows)
    _write_csv(out_dir / "unsw_behavior_knn_metrics.csv", metrics_rows)
    _write_csv(out_dir / "unsw_behavior_knn_summary.csv", summary_rows)
    _write_csv(out_dir / "unsw_behavior_knn_splits.csv", split_rows)
    _write_json(
        out_dir / "unsw_behavior_knn_manifest.json",
        {
            "unsw_csv": [str(path) for path in paths],
            "sample_seed": int(args.sample_seed),
            "seeds": list(map(int, args.seeds)),
            "attacks_requested": args.attacks,
            "attacks_run": attacks,
            "missing_attacks": missing_attacks,
            "load_meta": load_meta,
            "sampled_label_counts": label_counts,
            "views": args.views,
            "benign_train": int(args.benign_train),
            "benign_val": int(args.benign_val),
            "benign_test": int(args.benign_test),
            "attack_test_cap": int(args.attack_test_cap),
            "scan_scope": "limited" if args.max_rows_per_file is not None else "full_csv_scan",
            "leakage_controls": {
                "vocabulary_edges_fit_on": "train_benign_only",
                "memory": "train_benign_only",
                "threshold": "validation_benign_only",
                "test_labels_used_for": "metrics_only",
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
                "protocol_or_service_used_as_behavior": False,
                "port_used_as_behavior": False,
            },
            "elapsed_seconds": float(time.perf_counter() - start_all),
        },
    )
    lines = [
        "# UNSW-NB15 Existing-data Behavior-KNN Diagnostic",
        "",
        "This experiment uses the full UNSW-NB15 CSV flow-summary artifacts currently present locally. It is not claimed as an IDS2017-equivalent packet/burst/primitive PCAP rebuild.",
        "",
        "## Leakage Controls",
        "",
        "- Train-only discretization edges and token vocabulary.",
        "- Benign train only for KNN memory.",
        "- Benign validation only for P99 threshold calibration.",
        "- Test labels are used only for metrics.",
        "- Raw IP, absolute timestamp, complete five-tuple, protocol, service, and ports are not behavior tokens or memory grouping keys.",
        "",
        "## Main Summary",
        "",
        "| Held-out | View | Runs | AUROC | R@1%FPR | P99 FPR | Alerts/10k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [r for r in summary_rows if r.get("feature_view") == args.main_view]:
        lines.append(
            f"| {row['heldout_attack']} | {row['feature_view']} | {row['runs']} | "
            f"{float(row['auroc_mean']):.4f} | {float(row['recall_at_1pct_fpr_mean']):.4f} | "
            f"{float(row['p99_realized_fpr_mean']):.4f} | {float(row['false_alerts_per_10k_benign_mean']):.1f} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"out": str(out_dir), "metrics_rows": len(metrics_rows), "summary_rows": len(summary_rows), "attacks_run": attacks, "missing_attacks": missing_attacks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UNSW-NB15 existing CSV behavior-summary benign-memory KNN diagnostics.")
    parser.add_argument("--unsw-csv", nargs="+", default=[DEFAULT_UNSW])
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--attacks", nargs="+", default=["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"])
    parser.add_argument("--views", nargs="+", default=["flow_summary_all", "packet_burst_proxy", "state_free_all", "count_byte_timing"])
    parser.add_argument("--main-view", default="packet_burst_proxy")
    parser.add_argument("--benign-train", type=int, default=3000)
    parser.add_argument("--benign-val", type=int, default=1000)
    parser.add_argument("--benign-test", type=int, default=3000)
    parser.add_argument("--attack-test-cap", type=int, default=3000)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260605)
    parser.add_argument("--score-batch-size", type=int, default=1024)
    args = parser.parse_args()
    unknown = [view for view in args.views if view not in FEATURE_VIEWS]
    if unknown:
        raise ValueError(f"Unsupported feature views: {unknown}")
    if args.main_view not in args.views:
        raise ValueError("--main-view must be included in --views")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
