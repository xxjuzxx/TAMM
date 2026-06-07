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

from src.data.dataset_adapter import normalize_attack_family


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS2018 = "data/raw/CSE-CIC-IDS2018/Processed Traffic Data for ML Algorithms/*.csv"
DEFAULT_OUT = ROOT / "results" / "ids2018_behavior_knn"

CONTEXT_COLUMNS = {
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",
    "__source_file",
    "__row_id",
}

COUNT_COLUMNS = [
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "Subflow Fwd Pkts",
    "Subflow Bwd Pkts",
    "Fwd Act Data Pkts",
]
BYTE_COLUMNS = [
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Byts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
]
LENGTH_COLUMNS = [
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
]
TIMING_COLUMNS = [
    "Flow Duration",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]
DIRECTION_COLUMNS = ["Down/Up Ratio"]
FLAG_COLUMNS = [
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
]
TCP_SHAPE_COLUMNS = [
    "Fwd Header Len",
    "Bwd Header Len",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Seg Size Min",
]

BEHAVIOR_COLUMNS = (
    COUNT_COLUMNS
    + BYTE_COLUMNS
    + LENGTH_COLUMNS
    + TIMING_COLUMNS
    + DIRECTION_COLUMNS
    + FLAG_COLUMNS
    + TCP_SHAPE_COLUMNS
)

FEATURE_VIEWS = {
    "flow_summary_all": BEHAVIOR_COLUMNS,
    "flow_summary_no_timing": [c for c in BEHAVIOR_COLUMNS if c not in TIMING_COLUMNS],
    "flow_summary_no_length": [c for c in BEHAVIOR_COLUMNS if c not in LENGTH_COLUMNS],
    "flow_summary_no_direction": [c for c in BEHAVIOR_COLUMNS if c not in DIRECTION_COLUMNS],
    "flow_summary_counts_only": COUNT_COLUMNS + BYTE_COLUMNS,
    "flow_summary_timing_only": TIMING_COLUMNS,
    "flow_summary_length_only": LENGTH_COLUMNS,
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


def _canonical_family(label: object) -> str:
    family = normalize_attack_family(label)
    if family == "BENIGN":
        return "BENIGN"
    return family


def _to_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            out[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            out[column] = 0.0
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype("float32")


def _label_counts(paths: list[Path], chunksize: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in paths:
        for chunk in pd.read_csv(path, usecols=["Label"], chunksize=chunksize, low_memory=False):
            labels = chunk["Label"].map(_canonical_family)
            counts.update(labels.tolist())
    return dict(sorted((k, int(v)) for k, v in counts.items()))


def _reservoir_add(
    samples: dict[str, list[tuple[float, dict[str, Any]]]],
    family: str,
    key: float,
    row: dict[str, Any],
    cap: int,
) -> None:
    bucket = samples[family]
    if len(bucket) < cap:
        bucket.append((key, row))
        if len(bucket) == cap:
            bucket.sort(key=lambda item: item[0])
        return
    if key < bucket[-1][0]:
        bucket[-1] = (key, row)
        bucket.sort(key=lambda item: item[0])


def _sample_rows(
    paths: list[Path],
    *,
    caps: dict[str, int],
    seed: int,
    chunksize: int,
    max_chunks_per_file: int | None,
    max_rows_per_file: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples: dict[str, pd.DataFrame] = {}
    source_counts: Counter[str] = Counter()
    raw_label_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    usecols = list(dict.fromkeys(["Label", *BEHAVIOR_COLUMNS]))
    global_row = 0
    for path in paths:
        rows_seen_in_file = 0
        for chunk_no, chunk in enumerate(pd.read_csv(path, usecols=lambda col: col in usecols, chunksize=chunksize, low_memory=False), start=1):
            if max_chunks_per_file is not None and chunk_no > int(max_chunks_per_file):
                break
            if max_rows_per_file is not None:
                remaining = int(max_rows_per_file) - rows_seen_in_file
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()
            rows_seen_in_file += len(chunk)
            if "Label" not in chunk.columns:
                continue
            raw_labels = chunk["Label"].astype(str)
            families = raw_labels.map(_canonical_family)
            raw_label_counts.update(raw_labels.tolist())
            family_counts.update(families.tolist())
            keys = pd.Series(rng.random(len(chunk)), index=chunk.index, name="__sample_key")
            row_ordinals = pd.Series(np.arange(global_row, global_row + len(chunk), dtype=np.int64), index=chunk.index)
            global_row += len(chunk)
            for family, cap in caps.items():
                cap = int(cap)
                if cap <= 0:
                    continue
                family_mask = families == family
                if not bool(family_mask.any()):
                    continue
                family_keys = keys.loc[family_mask]
                local_keep = min(cap, len(family_keys))
                candidate_index = family_keys.nsmallest(local_keep).index
                family_numeric = _to_numeric_frame(chunk.loc[candidate_index], BEHAVIOR_COLUMNS)
                family_part = family_numeric.copy()
                family_part["family"] = family
                family_part["binary_label"] = "BENIGN" if family == "BENIGN" else "ATTACK"
                family_part["dataset"] = "CSE-CIC-IDS2018"
                family_part["source_file"] = path.name
                family_part["source_row_ordinal"] = row_ordinals.loc[candidate_index].to_numpy(dtype=np.int64)
                family_part["source_scan_limited"] = bool(max_chunks_per_file is not None or max_rows_per_file is not None)
                family_part["__sample_key"] = keys.loc[candidate_index].to_numpy(dtype=np.float64)
                existing = samples.get(family)
                if existing is None:
                    combined = family_part
                else:
                    combined = pd.concat([existing, family_part], ignore_index=True, sort=False)
                if len(combined) > cap:
                    combined = combined.nsmallest(cap, "__sample_key")
                samples[family] = combined.reset_index(drop=True)
                source_counts[path.name] += int(family_mask.sum())
    rows: list[dict[str, Any]] = []
    for family in sorted(samples):
        selected = samples[family].sort_values("__sample_key").drop(columns=["__sample_key"])
        rows.extend(selected.to_dict(orient="records"))
    meta = {
        "raw_label_counts": dict(sorted((k, int(v)) for k, v in raw_label_counts.items())),
        "family_counts": dict(sorted((k, int(v)) for k, v in family_counts.items())),
        "selected_counts": dict(sorted((family, int(len(items))) for family, items in samples.items())),
        "source_files": [str(path) for path in paths],
        "source_rows_seen": int(sum(family_counts.values())),
    }
    return rows, meta


def _log_transform(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    return np.log1p(values)


def _fit_edges(train_frame: pd.DataFrame, columns: list[str], bins: int) -> dict[str, list[float]]:
    edges: dict[str, list[float]] = {}
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    for column in columns:
        values = _log_transform(train_frame[column].to_numpy(dtype="float32"))
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            edges[column] = []
            continue
        qs = np.quantile(vals, quantiles)
        uniq = sorted({float(q) for q in qs if math.isfinite(float(q))})
        edges[column] = uniq
    return edges


def _tokenize_row(row: pd.Series, columns: list[str], edges: dict[str, list[float]], view: str) -> list[str]:
    tokens: list[str] = []
    for column in columns:
        value = float(row.get(column, 0.0))
        lv = float(np.log1p(max(0.0, value)))
        edge = edges.get(column, [])
        bucket = int(np.searchsorted(np.asarray(edge, dtype="float32"), lv, side="right"))
        safe_column = column.upper().replace("/", "_").replace(" ", "_").replace("-", "_")
        tokens.append(f"IDS2018_FLOWSTAT_{view.upper()}_{safe_column}_BIN_{bucket}")
        if column in FLAG_COLUMNS and value > 0:
            tokens.append(f"IDS2018_FLOWSTAT_FLAG_PRESENT_{safe_column}")
    fwd_pkts = float(row.get("Tot Fwd Pkts", 0.0))
    bwd_pkts = float(row.get("Tot Bwd Pkts", 0.0))
    total_pkts = fwd_pkts + bwd_pkts
    if total_pkts <= 4:
        tokens.append("IDS2018_FLOWSTAT_SHORT_FLOW")
    if total_pkts > 0:
        frac = fwd_pkts / total_pkts
        if frac >= 0.85:
            tokens.append("IDS2018_FLOWSTAT_FWD_DOMINANT")
        elif frac <= 0.15:
            tokens.append("IDS2018_FLOWSTAT_BWD_DOMINANT")
        else:
            tokens.append("IDS2018_FLOWSTAT_BIDIRECTIONAL")
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
    matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0)
    return matrix, vocab, token_rows


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
    thresholds = np.unique(scores)
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        m = _metrics_at_threshold(y_true, scores, float(threshold))
        if float(m["false_positive_rate"]) <= target_fpr:
            if best is None or float(m["attack_recall"]) > float(best["attack_recall"]):
                best = {**m, "threshold": float(threshold)}
    if best is None:
        threshold = float(np.max(scores) + 1e-6)
        best = {**_metrics_at_threshold(y_true, scores, threshold), "threshold": threshold}
    return best


def _rank_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {}
    try:
        row["auroc"] = float(roc_auc_score(y_true, scores))
        row["auprc"] = float(average_precision_score(y_true, scores))
        fpr, tpr, _thresholds = roc_curve(y_true, scores)
        eligible = fpr[tpr >= 0.95]
        row["fpr95"] = float(np.min(eligible)) if eligible.size else ""
    except ValueError:
        row["auroc"] = ""
        row["auprc"] = ""
        row["fpr95"] = ""
    return row


def _evaluate(y_true: np.ndarray, test_scores: np.ndarray, val_scores: np.ndarray) -> dict[str, Any]:
    p99_threshold = float(np.percentile(val_scores, 99.0))
    p99 = _metrics_at_threshold(y_true, test_scores, p99_threshold)
    r001 = _best_recall_under_fpr(y_true, test_scores, 0.001)
    r01 = _best_recall_under_fpr(y_true, test_scores, 0.01)
    r05 = _best_recall_under_fpr(y_true, test_scores, 0.05)
    out = _rank_metrics(y_true, test_scores)
    out.update(
        {
            "recall_at_0_1pct_fpr": r001["attack_recall"],
            "recall_at_1pct_fpr": r01["attack_recall"],
            "recall_at_5pct_fpr": r05["attack_recall"],
            "oracle_1pct_macro_f1": r01["macro_f1"],
            "oracle_1pct_threshold": r01["threshold"],
            "p99_threshold": p99_threshold,
            "p99_realized_fpr": p99["false_positive_rate"],
            "false_alerts_per_10k_benign": float(p99["false_positive_rate"] * 10000.0),
            "p99_attack_recall": p99["attack_recall"],
            "p99_macro_f1": p99["macro_f1"],
            "p99_attack_precision": p99["attack_precision"],
        }
    )
    return out


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
    metrics = [
        "auroc",
        "auprc",
        "fpr95",
        "recall_at_0_1pct_fpr",
        "recall_at_1pct_fpr",
        "recall_at_5pct_fpr",
        "p99_realized_fpr",
        "false_alerts_per_10k_benign",
        "p99_attack_recall",
        "p99_macro_f1",
        "vocab_size",
        "memory_size",
        "query_ms_per_flow",
    ]
    out: list[dict[str, Any]] = []
    for (attack, view), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "dataset": "CSE-CIC-IDS2018",
            "heldout_attack": attack,
            "feature_view": view,
            "runs": len(items),
            "seed_count": len({item["seed"] for item in items}),
        }
        for metric in metrics:
            mean, std = _mean_std(items, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    view_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        view_groups[str(row["feature_view"])].append(row)
    for view, items in sorted(view_groups.items()):
        row = {
            "dataset": "CSE-CIC-IDS2018",
            "heldout_attack": "Aggregate",
            "feature_view": view,
            "runs": len(items),
            "seed_count": len({item["seed"] for item in items}),
        }
        for metric in metrics:
            mean, std = _mean_std(items, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    return out


def _tex_escape(text: Any) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


def _format_pm(mean: Any, std: Any, digits: int = 4) -> str:
    try:
        m = float(mean)
        s = float(std)
    except (TypeError, ValueError):
        return "-"
    return f"${m:.{digits}f}\\pm{s:.{digits}f}$"


def _write_tex_table(summary_rows: list[dict[str, Any]], path: Path, feature_view: str) -> None:
    rows = [row for row in summary_rows if row.get("feature_view") == feature_view]
    order = ["Botnet", "DDoS", "DoS", "BruteForce", "WebAttack", "Infiltration", "Aggregate"]
    rows = sorted(rows, key=lambda row: order.index(str(row["heldout_attack"])) if str(row["heldout_attack"]) in order else 99)
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Held-out & Runs & AUROC & R@0.1\%FPR & R@1\%FPR & P99 FPR & Alerts/10k \\",
        r"\midrule",
    ]
    for row in rows:
        attack = str(row["heldout_attack"])
        label = "Aggregate" if attack == "Aggregate" else attack
        lines.append(
            " & ".join(
                [
                    _tex_escape(label),
                    str(row.get("runs", "")),
                    _format_pm(row.get("auroc_mean"), row.get("auroc_std")),
                    _format_pm(row.get("recall_at_0_1pct_fpr_mean"), row.get("recall_at_0_1pct_fpr_std")),
                    _format_pm(row.get("recall_at_1pct_fpr_mean"), row.get("recall_at_1pct_fpr_std")),
                    _format_pm(row.get("p99_realized_fpr_mean"), row.get("p99_realized_fpr_std")),
                    _format_pm(row.get("false_alerts_per_10k_benign_mean"), row.get("false_alerts_per_10k_benign_std"), digits=1),
                ]
            )
            + r" \\"
        )
        if attack == "Infiltration":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _expand_inputs(args.ids2018_csv)
    if not paths:
        raise FileNotFoundError("No IDS2018 CSV files were found.")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    benign_cap = int(args.benign_train + args.benign_val + args.benign_test)
    caps = {"BENIGN": benign_cap}
    for attack in args.attacks:
        caps[attack] = int(args.attack_test_cap)
    rows, sample_meta = _sample_rows(
        paths,
        caps=caps,
        seed=args.sample_seed,
        chunksize=args.chunksize,
        max_chunks_per_file=args.max_chunks_per_file,
        max_rows_per_file=args.max_rows_per_file,
    )
    label_counts = {str(k): int(v) for k, v in sample_meta.get("family_counts", {}).items()}
    attacks = [attack for attack in args.attacks if label_counts.get(attack, 0) > 0]
    missing_attacks = [attack for attack in args.attacks if label_counts.get(attack, 0) <= 0]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("IDS2018 sampling produced no rows.")
    for column in BEHAVIOR_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0

    metrics_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        benign_idx = frame.index[frame["family"] == "BENIGN"].to_numpy(dtype="int64").copy()
        rng.shuffle(benign_idx)
        if len(benign_idx) < benign_cap:
            raise RuntimeError(f"Need {benign_cap} benign samples, found {len(benign_idx)}.")
        train_idx = benign_idx[: args.benign_train]
        val_idx = benign_idx[args.benign_train : args.benign_train + args.benign_val]
        test_benign_idx = benign_idx[args.benign_train + args.benign_val : benign_cap]
        for attack in attacks:
            attack_idx = frame.index[frame["family"] == attack].to_numpy(dtype="int64").copy()
            rng.shuffle(attack_idx)
            attack_take = min(len(attack_idx), int(args.attack_test_cap))
            test_attack_idx = attack_idx[:attack_take]
            if attack_take == 0:
                split_rows.append({"seed": seed, "heldout_attack": attack, "status": "skipped_no_attack_rows"})
                continue
            test_idx = np.concatenate([test_benign_idx, test_attack_idx])
            y_true = np.concatenate([np.zeros(len(test_benign_idx), dtype="int64"), np.ones(len(test_attack_idx), dtype="int64")])
            split_rows.append(
                {
                    "seed": seed,
                    "heldout_attack": attack,
                    "status": "ok",
                    "train_benign": len(train_idx),
                    "val_benign": len(val_idx),
                    "test_benign": len(test_benign_idx),
                    "test_attack": len(test_attack_idx),
                    "attack_available_sampled": len(attack_idx),
                    "memory_uses_attack_labels": False,
                    "threshold_uses_attack_labels": False,
                }
            )
            for view in args.views:
                columns = [column for column in FEATURE_VIEWS[view] if column in frame.columns and column not in CONTEXT_COLUMNS]
                train_frame = frame.loc[train_idx, columns]
                val_frame = frame.loc[val_idx, columns]
                test_frame = frame.loc[test_idx, columns]
                fit_start = time.perf_counter()
                edges = _fit_edges(train_frame, columns, args.bins)
                train_x, vocab, train_tokens = _build_matrix(train_frame, columns, view, edges, vocab=None)
                val_x, _vocab, _val_tokens = _build_matrix(val_frame, columns, view, edges, vocab=vocab)
                test_x, _vocab, test_tokens = _build_matrix(test_frame, columns, view, edges, vocab=vocab)
                fit_seconds = time.perf_counter() - fit_start
                score_start = time.perf_counter()
                val_scores = _score_knn(val_x, train_x, args.k, args.score_batch_size)
                test_scores = _score_knn(test_x, train_x, args.k, args.score_batch_size)
                score_seconds = time.perf_counter() - score_start
                metric = _evaluate(y_true, test_scores, val_scores)
                p2_counts = [len(tokens) for tokens in test_tokens]
                metric.update(
                    {
                        "dataset": "CSE-CIC-IDS2018",
                        "source_type": "processed_cicflowmeter_csv",
                        "evidence_granularity": "flow_summary_behavior_tokens",
                        "heldout_attack": attack,
                        "seed": int(seed),
                        "feature_view": view,
                        "context_mode": "none",
                        "primitive_version": "not_available_without_packet_sequence",
                        "threshold_type": "benign_validation_p99",
                        "train_benign": int(len(train_idx)),
                        "val_benign": int(len(val_idx)),
                        "test_benign_count": int(len(test_benign_idx)),
                        "test_attack_count": int(len(test_attack_idx)),
                        "memory_size": int(train_x.shape[0]),
                        "vocab_size": int(train_x.shape[1]),
                        "avg_token_count_per_flow": float(np.mean(p2_counts)) if p2_counts else 0.0,
                        "median_token_count_per_flow": float(np.median(p2_counts)) if p2_counts else 0.0,
                        "p95_token_count_per_flow": float(np.quantile(p2_counts, 0.95)) if p2_counts else 0.0,
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
    _write_csv(out_dir / "ids2018_behavior_knn_metrics.csv", metrics_rows)
    _write_csv(out_dir / "ids2018_behavior_knn_summary.csv", summary_rows)
    _write_csv(out_dir / "ids2018_behavior_knn_splits.csv", split_rows)
    _write_csv(out_dir / "ids2018_dataset_summary.csv", [{"family": k, "rows": v} for k, v in label_counts.items()])
    _write_json(
        out_dir / "ids2018_behavior_knn_manifest.json",
        {
            "ids2018_csv": [str(path) for path in paths],
            "output": str(out_dir),
            "sample_seed": int(args.sample_seed),
            "seeds": list(map(int, args.seeds)),
            "attacks_requested": args.attacks,
            "attacks_run": attacks,
            "missing_attacks": missing_attacks,
            "label_counts": label_counts,
            "sample_meta": sample_meta,
            "views": args.views,
            "benign_train": int(args.benign_train),
            "benign_val": int(args.benign_val),
            "benign_test": int(args.benign_test),
            "attack_test_cap": int(args.attack_test_cap),
            "max_chunks_per_file": args.max_chunks_per_file,
            "max_rows_per_file": args.max_rows_per_file,
            "scan_scope": "limited" if args.max_chunks_per_file is not None or args.max_rows_per_file is not None else "full_csv_scan",
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
    _write_tex_table(summary_rows, out_dir / "table_ids2018_behavior_knn.tex", args.main_view)
    if not args.no_paper_sync:
        paper_table = ROOT.parent / "paper" / "tables" / "table_ids2018_behavior_knn.tex"
        _write_tex_table(summary_rows, paper_table, args.main_view)

    lines = [
        "# IDS2018 Behavior-KNN External Diagnostic",
        "",
        "This experiment uses CSE-CIC-IDS2018 processed CICFlowMeter CSV files. It is not a packet/burst/primitive full-path rebuild because the available CSV rows do not expose packet order, burst spans, or per-packet timing.",
        "",
        "## Leakage Controls",
        "",
        "- Train-only discretization edges and token vocabulary.",
        "- Benign train only for KNN memory.",
        "- Benign validation only for P99 threshold calibration.",
        "- Test labels are used only for metrics.",
        "- Raw IP, absolute timestamp, complete five-tuple, protocol, service, and ports are not behavior tokens or memory grouping keys.",
        "",
        "## Missing / Skipped",
        "",
        f"- Missing requested attack families in IDS2018 labels: {', '.join(missing_attacks) if missing_attacks else 'none'}.",
        "- Packet/burst structural primitives are unsupported from processed CSV and are therefore not claimed.",
        f"- CSV scan scope: {'limited' if args.max_chunks_per_file is not None or args.max_rows_per_file is not None else 'full'}; max_chunks_per_file={args.max_chunks_per_file}, max_rows_per_file={args.max_rows_per_file}.",
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
    parser = argparse.ArgumentParser(description="Run IDS2018 behavior-summary benign-memory KNN diagnostics.")
    parser.add_argument("--ids2018-csv", nargs="+", default=[DEFAULT_IDS2018])
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce", "DoS", "Infiltration"])
    parser.add_argument("--views", nargs="+", default=["flow_summary_all", "flow_summary_no_timing", "flow_summary_no_length", "flow_summary_no_direction"])
    parser.add_argument("--main-view", default="flow_summary_all")
    parser.add_argument("--benign-train", type=int, default=2000)
    parser.add_argument("--benign-val", type=int, default=1000)
    parser.add_argument("--benign-test", type=int, default=2000)
    parser.add_argument("--attack-test-cap", type=int, default=3000)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--max-chunks-per-file", type=int, default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260530)
    parser.add_argument("--score-batch-size", type=int, default=1024)
    parser.add_argument("--no-paper-sync", action="store_true", help="Do not update paper/tables outputs.")
    args = parser.parse_args()
    unknown_views = [view for view in args.views if view not in FEATURE_VIEWS]
    if unknown_views:
        raise ValueError(f"Unsupported feature views: {unknown_views}")
    if args.main_view not in args.views:
        raise ValueError("--main-view must be included in --views")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
