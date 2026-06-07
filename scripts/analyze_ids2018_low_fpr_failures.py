#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.features.token_alias import canonical_tokens, is_packet_burst_token, is_profile_token


ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "scripts" / "52_sweep_anomaly_low_fpr.py"
ATTACK_SLUG = {
    "Botnet": "botnet",
    "BruteForce": "bruteforce",
    "DDoS": "ddos",
    "DoS": "dos",
    "Infiltration": "infiltration",
    "WebAttack": "webattack",
}


def _load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("flowprim_low_fpr_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sweep module from {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["flowprim_low_fpr_sweep"] = module
    spec.loader.exec_module(module)
    return module


S = _load_sweep_module()


def _read_token_data(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _split_indices(token_data: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray([idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == split], dtype=np.int64)


def _id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    return {int(idx): token for token, idx in vocab.items()}


def _row_tokens(token_data: dict[str, Any], row_idx: int) -> list[str]:
    id_to_tok = _id_to_token(token_data["vocab"])
    ids = token_data["input_ids"][row_idx].cpu().numpy()
    mask = token_data["attention_mask"][row_idx].cpu().numpy() > 0
    return canonical_tokens(id_to_tok.get(int(token_id), "[UNK]") for token_id in ids[mask])


def _keep_token(token: str, view: str) -> bool:
    if token in {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}:
        return False
    if view == "packet_burst_only":
        return is_packet_burst_token(token)
    if view == "packet_burst_plus_profile":
        return is_packet_burst_token(token) or is_profile_token(token)
    raise ValueError(f"Unsupported view: {view}")


def _features(token_rows: list[list[str]], view: str, transform: str, train_idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
    kept = sorted({tok for row in token_rows for tok in row if _keep_token(tok, view)})
    col = {tok: idx for idx, tok in enumerate(kept)}
    mat = np.zeros((len(token_rows), len(kept)), dtype=np.float32)
    for i, row in enumerate(token_rows):
        counts = Counter(tok for tok in row if tok in col)
        for tok, count in counts.items():
            mat[i, col[tok]] = float(count)
    if transform.startswith("binary"):
        mat = (mat > 0).astype(np.float32)
    elif transform.startswith("tfidf"):
        df = np.sum(mat[train_idx] > 0, axis=0)
        idf = np.log((1.0 + len(train_idx)) / (1.0 + df)) + 1.0
        mat = mat * idf.reshape(1, -1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    norm = "l2" if transform.endswith("_l2") else "none"
    if norm == "l2":
        denom = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = np.divide(mat, denom, out=np.zeros_like(mat, dtype=np.float32), where=denom > 0)
    return mat.astype(np.float32, copy=False), kept


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {key: float("nan") for key in ["min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"]}
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _token_stats(tokens: list[str]) -> dict[str, int]:
    active = [tok for tok in tokens if tok not in {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}]
    return {
        "active_tokens": len(active),
        "unique_active_tokens": len(set(active)),
        "packet_burst_tokens": sum(1 for tok in active if is_packet_burst_token(tok)),
        "profile_tokens": sum(1 for tok in active if is_profile_token(tok)),
        "flow_tokens": sum(1 for tok in active if tok.startswith("FLOW_")),
        "burst_tokens": sum(1 for tok in active if tok.startswith("BURST_")),
        "pkt_tokens": sum(1 for tok in active if tok.startswith("PKT_")),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    token_dir = Path(args.token_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []
    token_rows_out: list[dict[str, Any]] = []

    for attack in args.attacks:
        slug = ATTACK_SLUG[attack]
        for seed in args.seeds:
            token_path = token_dir / f"{args.artifact_prefix}_leave_one_{slug}_anomaly_seed{seed}_a3_full_rhythm.pt"
            if not token_path.exists():
                summary_rows.append({"attack": attack, "seed": seed, "status": "missing", "token_path": str(token_path)})
                continue
            data = _read_token_data(token_path)
            labels = data["binary_labels"].cpu().numpy().astype(np.int64)
            train_idx = _split_indices(data, "train")
            val_idx = _split_indices(data, "val")
            test_idx = _split_indices(data, "test")
            token_rows = [_row_tokens(data, idx) for idx in range(len(data["meta"]))]
            features, kept = _features(token_rows, args.feature_view, args.transform, train_idx)
            groups = ["GLOBAL"] * len(labels)
            val_scores = S._scores(features, train_idx, val_idx, groups, scorer=args.scorer, k=args.k)
            test_scores = S._scores(features, train_idx, test_idx, groups, scorer=args.scorer, k=args.k)
            test_labels = labels[test_idx]
            benign_scores = test_scores[test_labels == 0]
            attack_scores = test_scores[test_labels == 1]
            val_threshold = float(np.quantile(val_scores, args.val_quantile))
            oracle_threshold = float(np.quantile(benign_scores, 0.99)) if len(benign_scores) else float("nan")
            detected_val = float(np.mean(attack_scores >= val_threshold)) if len(attack_scores) else float("nan")
            detected_oracle = float(np.mean(attack_scores >= oracle_threshold)) if len(attack_scores) else float("nan")
            row: dict[str, Any] = {
                "attack": attack,
                "seed": seed,
                "status": "ok",
                "feature_view": args.feature_view,
                "transform": args.transform,
                "scorer": args.scorer,
                "k": args.k,
                "vocab_size": len(kept),
                "val_threshold": val_threshold,
                "oracle_1pct_threshold": oracle_threshold,
                "test_benign_count": int(np.sum(test_labels == 0)),
                "test_attack_count": int(np.sum(test_labels == 1)),
                "attack_recall_val_p99": detected_val,
                "attack_recall_oracle_1pct": detected_oracle,
                "attack_above_val_threshold": int(np.sum(attack_scores >= val_threshold)),
                "attack_above_oracle_threshold": int(np.sum(attack_scores >= oracle_threshold)),
            }
            for prefix, scores in [("val_benign", val_scores), ("test_benign", benign_scores), ("test_attack", attack_scores)]:
                for key, value in _quantiles(np.asarray(scores, dtype=np.float64)).items():
                    row[f"{prefix}_{key}"] = value
            summary_rows.append(row)

            attack_global = test_idx[test_labels == 1]
            order = np.argsort(attack_scores)
            for rank in range(min(args.low_k, len(order))):
                local_pos = int(order[rank])
                global_idx = int(attack_global[local_pos])
                tokens = token_rows[global_idx]
                meta = data["meta"][global_idx]
                stats = _token_stats(tokens)
                low_rows.append(
                    {
                        "attack": attack,
                        "seed": seed,
                        "rank_low_score": rank + 1,
                        "score": float(attack_scores[local_pos]),
                        "val_threshold": val_threshold,
                        "oracle_1pct_threshold": oracle_threshold,
                        "flow_id": meta.get("flow_id"),
                        "context_id": meta.get("context_id"),
                        "token_len": meta.get("token_len"),
                        "truncated": meta.get("truncated"),
                        **stats,
                    }
                )
            attack_counter = Counter()
            benign_counter = Counter()
            for idx in attack_global:
                attack_counter.update(tok for tok in set(token_rows[int(idx)]) if _keep_token(tok, args.feature_view))
            benign_global = test_idx[test_labels == 0]
            for idx in benign_global:
                benign_counter.update(tok for tok in set(token_rows[int(idx)]) if _keep_token(tok, args.feature_view))
            for tok, attack_support in attack_counter.most_common(args.top_tokens):
                attack_rate = attack_support / max(len(attack_global), 1)
                benign_rate = benign_counter.get(tok, 0) / max(len(benign_global), 1)
                token_rows_out.append(
                    {
                        "attack": attack,
                        "seed": seed,
                        "token": tok,
                        "attack_support": attack_support,
                        "attack_rate": attack_rate,
                        "benign_support": benign_counter.get(tok, 0),
                        "benign_rate": benign_rate,
                        "attack_minus_benign": attack_rate - benign_rate,
                    }
                )

    _write_csv(out_dir / "score_distribution_summary.csv", summary_rows)
    _write_csv(out_dir / "lowest_scoring_attack_flows.csv", low_rows)
    _write_csv(out_dir / "attack_token_support.csv", token_rows_out)
    _write_report(out_dir, summary_rows, args)
    return {"output": str(out_dir), "summary_rows": len(summary_rows), "low_rows": len(low_rows)}


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        if value == "" or value is None:
            return "-"
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_report(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# IDS2018 Low-FPR Failure Diagnostics",
        "",
        f"Feature view: `{args.feature_view}`; transform: `{args.transform}`; scorer: `{args.scorer}`; k={args.k}.",
        "",
        "## Score Distribution Summary",
        "",
        "| Attack | Seed | Test attacks | Val P99 thr | Test benign P99 | Attack p50 | Attack p90 | Attack p99 | Recall@valP99 | Recall@oracle1% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ok_rows:
        lines.append(
            "| {attack} | {seed} | {n} | {val_thr} | {ben_p99} | {a50} | {a90} | {a99} | {rv} | {ro} |".format(
                attack=row["attack"],
                seed=row["seed"],
                n=row["test_attack_count"],
                val_thr=_fmt(row["val_threshold"]),
                ben_p99=_fmt(row["test_benign_p99"]),
                a50=_fmt(row["test_attack_p50"]),
                a90=_fmt(row["test_attack_p90"]),
                a99=_fmt(row["test_attack_p99"]),
                rv=_fmt(row["attack_recall_val_p99"]),
                ro=_fmt(row["attack_recall_oracle_1pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Hints",
            "",
            "- If attack p99 is below the test benign p99, the family cannot reach Recall@1%FPR regardless of benign-validation threshold tuning.",
            "- If attack p50/p90 are close to benign center/tail, the issue is score ranking rather than only calibration.",
            "- Raw IP, absolute timestamp, complete five-tuple, protocol, and service are not used by this diagnostic scorer.",
            "",
            "## Outputs",
            "",
            "- `score_distribution_summary.csv`",
            "- `lowest_scoring_attack_flows.csv`",
            "- `attack_token_support.csv`",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze IDS2018 low-FPR score failures.")
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-prefix", default="ids2018_official_victim")
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "BruteForce"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--feature-view", default="packet_burst_only")
    parser.add_argument("--transform", default="binary_l2")
    parser.add_argument("--scorer", default="knn_euclidean")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--val-quantile", type=float, default=0.99)
    parser.add_argument("--low-k", type=int, default=50)
    parser.add_argument("--top-tokens", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(analyze(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
