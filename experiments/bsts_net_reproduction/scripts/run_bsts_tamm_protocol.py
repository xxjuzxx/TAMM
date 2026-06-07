#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def load_bsts_module() -> Any:
    module_path = Path(__file__).with_name("run_bsts_adapted.py")
    spec = importlib.util.spec_from_file_location("bsts_adapted_core", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_bsts_module()


ATTACKS = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_flows_by_id(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    rows: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for idx, row in B.iter_flows(path):
        flow_id = str(row.get("flow_id") or row.get("_row_id"))
        if not flow_id:
            flow_id = B.stable_id(row, idx)
        row["_row_id"] = flow_id
        row["_attack_family"] = B.attack_family(row)
        rows[flow_id] = row
        counts[row["_attack_family"]] += 1
    return rows, counts


def rows_for_split(flow_by_id: dict[str, dict[str, Any]], split_payload: dict[str, Any], split: str) -> list[dict[str, Any]]:
    ids = split_payload["splits"][split]
    missing = [flow_id for flow_id in ids if flow_id not in flow_by_id]
    if missing:
        raise KeyError(f"{len(missing)} split ids are missing from flow source; first={missing[:3]}")
    return [flow_by_id[flow_id] for flow_id in ids]


def labels_for(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([0 if B.is_benign(row) else 1 for row in rows], dtype=np.int32)


def fpr_at_recall(labels: np.ndarray, scores: np.ndarray, target_recall: float = 0.95) -> float:
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return float("nan")
    fpr, tpr, _thresholds = roc_curve(labels, scores)
    candidates = fpr[tpr >= target_recall]
    return float(np.min(candidates)) if candidates.size else 1.0


def metric_bundle(labels: np.ndarray, scores: np.ndarray, val_scores: np.ndarray | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    has_both = labels.size > 0 and len(set(labels.tolist())) == 2
    out["auroc"] = float(roc_auc_score(labels, scores)) if has_both else float("nan")
    out["auprc"] = float(average_precision_score(labels, scores)) if has_both else float("nan")
    out["fpr95"] = fpr_at_recall(labels, scores, 0.95)
    for target in (0.001, 0.01, 0.05):
        threshold = B.fixed_fpr_threshold(scores, labels, target)
        metrics = B.metrics_at(labels, scores, threshold)
        key = str(target)
        out[f"recall_at_{key}_fpr"] = float(metrics.get("recall", float("nan")))
        out[f"precision_at_{key}_fpr"] = float(metrics.get("precision", float("nan")))
        out[f"actual_fpr_at_{key}_fpr"] = float(metrics.get("fpr", float("nan")))
        out[f"macro_f1_at_{key}_fpr"] = float(f1_score(labels, (scores >= threshold).astype(np.int32), average="macro", zero_division=0))
    if val_scores is not None and val_scores.size:
        p99 = B.threshold_at(val_scores, 0.99)
        metrics = B.metrics_at(labels, scores, p99)
        pred = (scores >= p99).astype(np.int32)
        out["val_p99_threshold"] = float(p99)
        out["val_p99_realized_fpr"] = float(metrics.get("fpr", float("nan")))
        out["val_p99_attack_recall"] = float(metrics.get("recall", float("nan")))
        out["val_p99_precision"] = float(precision_score(labels, pred, zero_division=0))
        out["val_p99_f1"] = float(f1_score(labels, pred, zero_division=0))
        out["val_p99_macro_f1"] = float(f1_score(labels, pred, average="macro", zero_division=0))
        out["false_alerts_per_10k_benign"] = float(out["val_p99_realized_fpr"] * 10000.0)
    return out


def knn_scores(train_emb: np.ndarray, query_emb: np.ndarray, k: int, chunk_size: int = 1024) -> np.ndarray:
    if query_emb.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    k = min(max(1, k), train_emb.shape[0])
    scores: list[np.ndarray] = []
    for start in range(0, query_emb.shape[0], chunk_size):
        q = query_emb[start : start + chunk_size]
        diff = q[:, None, :] - train_emb[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        kth = np.partition(dist, kth=k - 1, axis=1)[:, :k]
        scores.append(np.mean(kth, axis=1).astype(np.float32))
    return np.concatenate(scores, axis=0)


def centroid_scores(train_emb: np.ndarray, query_emb: np.ndarray, class_nums: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    k = min(max(1, class_nums), train_emb.shape[0])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(train_emb)
    diff = query_emb[:, None, :] - km.cluster_centers_[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(dist, axis=1).astype(np.float32), {"embedding_centroids": int(k)}


def slug_for_attack(attack: str) -> str:
    return ATTACKS[attack]


def split_file_for(base_dir: Path, attack: str, seed: int) -> Path:
    return base_dir / f"splits_leave_one_{slug_for_attack(attack)}_anomaly_seed{seed}.json"


def evaluate_split(
    *,
    flow_by_id: dict[str, dict[str, Any]],
    source_counts: Counter[str],
    split_path: Path,
    output_dir: Path,
    max_triplets: int,
    epochs: int,
    batch_size: int,
    lr: float,
    win_size: int,
    class_nums: int,
    delta1: int,
    delta2: float,
    delta3: int,
    knn_k: int,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    payload = read_json(split_path)
    seed = int(payload["seed"])
    heldout = str(payload["leave_label"])
    train = rows_for_split(flow_by_id, payload, "train")
    val = rows_for_split(flow_by_id, payload, "val")
    test = rows_for_split(flow_by_id, payload, "test")
    run_dir = output_dir / f"ids2017_leave_one_{slug_for_attack(heldout)}_seed{seed}"
    model, train_meta = B.train_model(
        train,
        run_dir / f"bsts_adapted_seed{seed}.pt",
        seed,
        max_triplets,
        epochs,
        batch_size,
        lr,
    )

    train_emb = B.embed_rows(model, train, batch_size)
    val_emb = B.embed_rows(model, val, batch_size)
    test_emb = B.embed_rows(model, test, batch_size)
    val_labels_flow = labels_for(val)
    test_labels_flow = labels_for(test)

    flow_knn_val_scores = knn_scores(train_emb, val_emb, knn_k)
    flow_knn_test_scores = knn_scores(train_emb, test_emb, knn_k)
    flow_centroid_val_scores, centroid_meta = centroid_scores(train_emb, val_emb, class_nums, seed)
    flow_centroid_test_scores, _ = centroid_scores(train_emb, test_emb, class_nums, seed)

    centroids, window_centroid_meta = B.train_window_centroids(model, train, win_size, class_nums, batch_size)
    val_x, val_window_labels_raw, _val_metas = B.window_features(val, val_emb, win_size, max(1, win_size // 2))
    test_x, test_window_labels_raw, test_metas = B.window_features(test, test_emb, win_size, max(1, win_size // 2))
    val_window_labels = np.asarray(val_window_labels_raw, dtype=np.int32)
    test_window_labels = np.asarray(test_window_labels_raw, dtype=np.int32)
    val_window_scores = B.score_windows(val_x, centroids)
    test_window_scores = B.score_windows(test_x, centroids)
    entropy_pred, entropy_scores, entropy_meta = B.entropy_rule_predict(
        test_x,
        test_window_labels,
        test_metas,
        class_nums,
        delta1,
        delta2,
        delta3,
    )

    rows: list[dict[str, Any]] = []
    base = {
        "dataset": "CICIDS2017",
        "heldout_attack": heldout,
        "seed": seed,
        "split_file": str(split_path),
        "source_rows": int(sum(source_counts.values())),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "test_attack_flows": int(np.sum(test_labels_flow == 1)),
        "test_benign_flows": int(np.sum(test_labels_flow == 0)),
        "train_triplets": train_meta["triplets"],
        "train_device": train_meta["device"],
        "epochs": epochs,
        "max_triplets": max_triplets,
        "elapsed_sec": time.perf_counter() - started,
        "official_reproduction_status": "tamm_protocol_adapted_core",
    }
    rows.append(
        {
            **base,
            "method": "BSTS-Net embedding-KNN",
            "eval_unit": "flow",
            "knn_k": knn_k,
            **metric_bundle(test_labels_flow, flow_knn_test_scores, flow_knn_val_scores),
        }
    )
    rows.append(
        {
            **base,
            "method": "BSTS-Net embedding-centroid",
            "eval_unit": "flow",
            **centroid_meta,
            **metric_bundle(test_labels_flow, flow_centroid_test_scores, flow_centroid_val_scores),
        }
    )
    rows.append(
        {
            **base,
            "method": "BSTS-Net relation-window distance",
            "eval_unit": "srcip-port-window",
            "val_windows": int(len(val_window_labels)),
            "test_windows": int(len(test_window_labels)),
            "test_attack_windows": int(np.sum(test_window_labels == 1)),
            "test_benign_windows": int(np.sum(test_window_labels == 0)),
            **window_centroid_meta,
            **metric_bundle(test_window_labels, test_window_scores, val_window_scores),
        }
    )
    entropy_metrics = B.metrics_from_pred(test_window_labels, entropy_pred)
    entropy_has_both = len(set(test_window_labels.tolist())) == 2
    rows.append(
        {
            **base,
            "method": "BSTS-Net native entropy rule",
            "eval_unit": "srcip-port-window",
            "val_windows": int(len(val_window_labels)),
            "test_windows": int(len(test_window_labels)),
            "test_attack_windows": int(np.sum(test_window_labels == 1)),
            "test_benign_windows": int(np.sum(test_window_labels == 0)),
            "auroc": float(roc_auc_score(test_window_labels, entropy_scores)) if entropy_has_both else float("nan"),
            "auprc": float(average_precision_score(test_window_labels, entropy_scores)) if entropy_has_both else float("nan"),
            "fpr95": fpr_at_recall(test_window_labels, entropy_scores, 0.95),
            "direct_rule_fpr": entropy_metrics.get("fpr", float("nan")),
            "direct_rule_precision": entropy_metrics.get("precision", float("nan")),
            "direct_rule_recall": entropy_metrics.get("recall", float("nan")),
            "direct_rule_f1": entropy_metrics.get("f1", float("nan")),
            "false_alerts_per_10k_benign": entropy_metrics.get("false_alerts_per_10k_benign", float("nan")),
            **{f"entropy_{key}": value for key, value in entropy_meta.items()},
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, run_dir / "metrics.csv")
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "auroc",
        "auprc",
        "fpr95",
        "recall_at_0.001_fpr",
        "recall_at_0.01_fpr",
        "recall_at_0.05_fpr",
        "actual_fpr_at_0.01_fpr",
        "val_p99_realized_fpr",
        "val_p99_attack_recall",
        "false_alerts_per_10k_benign",
        "direct_rule_fpr",
        "direct_rule_recall",
    ]
    out: list[dict[str, Any]] = []
    keys = sorted({(row["method"], row["eval_unit"], row["heldout_attack"]) for row in rows})
    for method, eval_unit, heldout in keys:
        group = [row for row in rows if row["method"] == method and row["eval_unit"] == eval_unit and row["heldout_attack"] == heldout]
        item: dict[str, Any] = {"method": method, "eval_unit": eval_unit, "heldout_attack": heldout, "runs": len(group)}
        for metric in metrics:
            vals = []
            for row in group:
                try:
                    vals.append(float(row.get(metric, "nan")))
                except (TypeError, ValueError):
                    pass
            mean, std = mean_std(vals)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        out.append(item)
    for method, eval_unit in sorted({(row["method"], row["eval_unit"]) for row in rows}):
        group = [row for row in rows if row["method"] == method and row["eval_unit"] == eval_unit]
        item = {"method": method, "eval_unit": eval_unit, "heldout_attack": "Aggregate", "runs": len(group)}
        for metric in metrics:
            vals = []
            for row in group:
                try:
                    vals.append(float(row.get(metric, "nan")))
                except (TypeError, ValueError):
                    pass
            mean, std = mean_std(vals)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        out.append(item)
    return out


def fmt_pm(mean: Any, std: Any, digits: int = 4) -> str:
    try:
        m = float(mean)
        s = float(std)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(m):
        return "-"
    return f"${m:.{digits}f}\\pm{s:.{digits}f}$"


def write_latex_summary(agg_rows: list[dict[str, Any]], path: Path, methods: list[str]) -> None:
    selected = [row for row in agg_rows if row["heldout_attack"] == "Aggregate" and row["method"] in methods]
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Method & Unit & Runs & AUROC & FPR95 & R@0.1\\%FPR & R@1\\%FPR \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['method']} & {row['eval_unit']} & {row['runs']} & "
            f"{fmt_pm(row['auroc_mean'], row['auroc_std'])} & "
            f"{fmt_pm(row['fpr95_mean'], row['fpr95_std'])} & "
            f"{fmt_pm(row['recall_at_0.001_fpr_mean'], row['recall_at_0.001_fpr_std'])} & "
            f"{fmt_pm(row['recall_at_0.01_fpr_mean'], row['recall_at_0.01_fpr_std'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BSTS-Net adapted core under TAMM leave-one split protocol.")
    parser.add_argument("--flow-file", default="outputs/processed/ccfa/cicids2017_interim_labeled_flows.jsonl")
    parser.add_argument("--split-dir", default="paper_icdm_applied_2026/experiments/unknown")
    parser.add_argument("--output-dir", default="experiments/bsts_net_reproduction/tamm_protocol")
    parser.add_argument("--attacks", nargs="+", default=list(ATTACKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-triplets", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--win-size", type=int, default=10)
    parser.add_argument("--class-nums", type=int, default=6)
    parser.add_argument("--delta1", type=int, default=3)
    parser.add_argument("--delta2", type=float, default=1.0)
    parser.add_argument("--delta3", type=int, default=3)
    parser.add_argument("--knn-k", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    flow_by_id, source_counts = read_flows_by_id(Path(args.flow_file))
    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for attack in args.attacks:
            split_path = split_file_for(Path(args.split_dir), attack, seed)
            print(f"running {attack} seed={seed} split={split_path}")
            rows = evaluate_split(
                flow_by_id=flow_by_id,
                source_counts=source_counts,
                split_path=split_path,
                output_dir=output_dir,
                max_triplets=args.max_triplets,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                win_size=args.win_size,
                class_nums=args.class_nums,
                delta1=args.delta1,
                delta2=args.delta2,
                delta3=args.delta3,
                knn_k=args.knn_k,
            )
            all_rows.extend(rows)
            primary = next(row for row in rows if row["method"] == "BSTS-Net embedding-KNN")
            print(
                f"{attack} seed={seed} embedding-KNN auroc={primary['auroc']:.4f} "
                f"r@1%fpr={primary['recall_at_0.01_fpr']:.4f} "
                f"p99fpr={primary['val_p99_realized_fpr']:.4f}"
            )

    write_csv(all_rows, output_dir / "bsts_tamm_protocol_runs.csv")
    agg_rows = aggregate(all_rows)
    write_csv(agg_rows, output_dir / "bsts_tamm_protocol_aggregate.csv")
    write_latex_summary(
        agg_rows,
        output_dir / "table_bsts_tamm_protocol_summary.tex",
        ["BSTS-Net embedding-KNN", "BSTS-Net relation-window distance", "BSTS-Net native entropy rule"],
    )
    print(f"wrote {output_dir / 'bsts_tamm_protocol_runs.csv'}")
    print(f"wrote {output_dir / 'bsts_tamm_protocol_aggregate.csv'}")


if __name__ == "__main__":
    main()
