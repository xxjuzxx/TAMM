#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

from run_motif_selection_experiments import (
    ATTACK_SLUG,
    DEFAULT_TOKEN_DIR,
    _augment_structural_rows,
    _features_from_rows,
    _read_token_data,
    _split_indices,
    _token_rows,
)
from src.features.motif_selection import DEFAULT_MOTIF_PREFIXES, extract_candidate_motifs
from src.features.token_alias import SPECIAL_TOKENS, is_packet_burst_token


ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "scripts" / "52_sweep_anomaly_low_fpr.py"
DEFAULT_OUT = ROOT / "results" / "explanation_contrast"


def _load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("tamm_low_fpr_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sweep module from {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tamm_low_fpr_sweep_explain"] = module
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _token_path(token_dir: Path, attack: str, seed: int) -> Path:
    return token_dir / f"cicids2017_leave_one_{ATTACK_SLUG[attack]}_anomaly_seed{seed}_a3_full_rhythm.pt"


def _nearest_indices(features: np.ndarray, train_idx: np.ndarray, query_idx: np.ndarray, *, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    refs = features[train_idx]
    queries = features[query_idx]
    distances = S._cosine_distance(queries, refs)
    kk = max(1, min(int(k), distances.shape[1]))
    local = np.argpartition(distances, kk - 1, axis=1)[:, :kk]
    nearest_dist = np.take_along_axis(distances, local, axis=1)
    order = np.argsort(nearest_dist, axis=1)
    sorted_local = np.take_along_axis(local, order, axis=1)
    sorted_dist = np.take_along_axis(nearest_dist, order, axis=1)
    return train_idx[sorted_local], sorted_dist


def _score_one_against_train(features: np.ndarray, train_idx: np.ndarray, row: np.ndarray, *, k: int) -> float:
    distances = S._cosine_distance(row.reshape(1, -1).astype(np.float32), features[train_idx])
    kk = max(1, min(int(k), distances.shape[1]))
    return float(np.partition(distances, kk - 1, axis=1)[:, :kk].mean())


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def run_one(token_path: Path, *, attack: str, seed: int, feature_view: str, transform: str, k: int, max_alerts: int) -> list[dict[str, Any]]:
    token_data = _read_token_data(token_path)
    rows, _ = _augment_structural_rows(_token_rows(token_data), _split_indices(token_data, "train"))
    train_idx = _split_indices(token_data, "train")
    val_idx = _split_indices(token_data, "val")
    test_idx = _split_indices(token_data, "test")
    labels = token_data["binary_labels"].cpu().numpy().astype(np.int64)
    motifs = set(extract_candidate_motifs([{"tokens": r} for r in rows], DEFAULT_MOTIF_PREFIXES))
    packet_burst = {token for toks in rows for token in toks if is_packet_burst_token(token)}
    if feature_view == "selected_or_all_motifs_only":
        keep_tokens = motifs
    elif feature_view == "packet_burst_plus_motifs":
        keep_tokens = packet_burst | motifs
    else:
        raise ValueError(f"Unsupported feature view: {feature_view}")
    features, stats = _features_from_rows(rows, keep_tokens, transform=transform, train_idx=train_idx)
    del stats
    groups = ["GLOBAL"] * len(rows)
    val_scores = S._scores(features, train_idx, val_idx, groups, scorer="knn_cosine", k=k)
    test_scores = S._scores(features, train_idx, test_idx, groups, scorer="knn_cosine", k=k)
    threshold = float(np.quantile(val_scores, 0.99))
    alert_local = np.flatnonzero(test_scores >= threshold)
    alert_local = alert_local[np.argsort(-test_scores[alert_local])][:max_alerts]
    if alert_local.size == 0:
        return []
    alert_idx = test_idx[alert_local]
    nearest, nearest_dist = _nearest_indices(features, train_idx, alert_idx, k=k)
    col_tokens = sorted(token for token in keep_tokens if token not in SPECIAL_TOKENS)
    out: list[dict[str, Any]] = []
    motif_cols = [idx for idx, token in enumerate(col_tokens) if any(token.startswith(prefix) for prefix in DEFAULT_MOTIF_PREFIXES)]
    for pos, idx in enumerate(alert_idx.tolist()):
        nn = int(nearest[pos, 0])
        active_alert = set(np.flatnonzero(features[idx] > 0).tolist())
        active_nn = set(np.flatnonzero(features[nn] > 0).tolist())
        alert_only_motif_cols = sorted((active_alert - active_nn).intersection(motif_cols), key=lambda c: col_tokens[c])
        nn_only_motif_cols = sorted((active_nn - active_alert).intersection(motif_cols), key=lambda c: col_tokens[c])
        row_removed = features[idx].copy()
        for c in alert_only_motif_cols[:3]:
            row_removed[c] = 0.0
        denom = np.linalg.norm(row_removed)
        if denom > 0 and transform.endswith("_l2"):
            row_removed = row_removed / denom
        cf_score = _score_one_against_train(features, train_idx, row_removed, k=k)
        score = float(test_scores[alert_local[pos]])
        out.append(
            {
                "seed": seed,
                "heldout_attack": attack,
                "feature_view": feature_view,
                "row_index": idx,
                "label": int(labels[idx]),
                "score": score,
                "threshold": threshold,
                "nearest_train_row": nn,
                "nearest_distance": float(nearest_dist[pos, 0]),
                "alert_only_motif_count": len(alert_only_motif_cols),
                "neighbor_only_motif_count": len(nn_only_motif_cols),
                "top_alert_only_motifs": ";".join(col_tokens[c] for c in alert_only_motif_cols[:8]),
                "top_neighbor_only_motifs": ";".join(col_tokens[c] for c in nn_only_motif_cols[:8]),
                "score_after_removing_top3_alert_motifs": cf_score,
                "score_drop_top3": score - cf_score,
                "has_motif_contrast": bool(alert_only_motif_cols or nn_only_motif_cols),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["feature_view"]), []).append(row)
    out = []
    for view, vals in sorted(groups.items()):
        runs = len({(v["heldout_attack"], v["seed"]) for v in vals})
        alerts = len(vals)
        coverage = sum(1 for v in vals if str(v.get("has_motif_contrast")) == "True" or v.get("has_motif_contrast") is True) / max(alerts, 1)
        for_metric = ["alert_only_motif_count", "neighbor_only_motif_count", "score_drop_top3"]
        row: dict[str, Any] = {"feature_view": view, "runs": runs, "alerts": alerts, "contrast_coverage": coverage}
        for col in for_metric:
            nums = [float(v[col]) for v in vals if v.get(col) not in ("", None)]
            row[f"mean_{col}"] = float(sum(nums) / len(nums)) if nums else 0.0
            row[f"std_{col}"] = float(statistics.pstdev(nums)) if len(nums) > 1 else 0.0
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nearest-neighbor motif contrast records for TAMM diagnosis.")
    parser.add_argument("--token_dir", default=str(DEFAULT_TOKEN_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--feature_views", nargs="+", default=["selected_or_all_motifs_only", "packet_burst_plus_motifs"])
    parser.add_argument("--transform", default="binary_l2")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max_alerts", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    attacks = ["DDoS"] if args.quick else args.attacks
    seeds = [43] if args.quick else args.seeds
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for attack in attacks:
        for seed in seeds:
            path = _token_path(Path(args.token_dir), attack, seed)
            if not path.exists():
                skipped.append({"heldout_attack": attack, "seed": seed, "reason": f"missing token corpus: {path}"})
                continue
            for view in args.feature_views:
                rows.extend(run_one(path, attack=attack, seed=seed, feature_view=view, transform=args.transform, k=args.k, max_alerts=args.max_alerts))
    out = Path(args.out_dir)
    _write_csv(rows, out / "explanation_contrast_cases.csv")
    _write_csv(summarize(rows), out / "explanation_contrast_summary.csv")
    (out / "explanation_contrast_manifest.json").write_text(
        json.dumps(
            {
                "case_rows": len(rows),
                "skipped": skipped,
                "leakage_control": "nearest-neighbor contrast is post-hoc audit over fixed benign-memory scores; no attack labels used for threshold calibration",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out), "case_rows": len(rows), "skipped": len(skipped)}, sort_keys=True))


if __name__ == "__main__":
    main()
