#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.features.token_alias import canonical_tokens, is_packet_burst_token, is_profile_token, is_structural_token


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
    return {int(idx): str(token) for token, idx in vocab.items()}


def _row_tokens(token_data: dict[str, Any], row_idx: int) -> list[str]:
    id_to_tok = _id_to_token(token_data["vocab"])
    ids = token_data["input_ids"][row_idx].cpu().numpy()
    mask = token_data["attention_mask"][row_idx].cpu().numpy() > 0
    return canonical_tokens(id_to_tok.get(int(token_id), "[UNK]") for token_id in ids[mask])


def _keep_token_for_view(token: str, feature_view: str) -> bool:
    if token in {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}:
        return False
    is_profile = is_profile_token(token)
    is_structural = is_structural_token(token)
    is_packet_burst = is_packet_burst_token(token)
    if feature_view == "packet_burst_only":
        return is_packet_burst
    if feature_view == "packet_burst_plus_profile":
        return is_packet_burst or is_profile
    if feature_view == "packet_burst_plus_structural":
        return is_packet_burst or is_structural
    if feature_view == "packet_burst_plus_profile_structural":
        return is_packet_burst or is_profile or is_structural
    raise ValueError(f"Unsupported feature view: {feature_view}")


def _features_for_view(
    token_data: dict[str, Any],
    train_idx: np.ndarray,
    *,
    feature_view: str,
    transform: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    token_rows = [_row_tokens(token_data, idx) for idx in range(len(token_data["meta"]))]
    kept = sorted({token for row in token_rows for token in row if _keep_token_for_view(token, feature_view)})
    if not kept:
        raise ValueError(f"Feature view kept no tokens: {feature_view}")
    col = {token: idx for idx, token in enumerate(kept)}
    features = np.zeros((len(token_rows), len(kept)), dtype=np.float32)
    for row_idx, tokens in enumerate(token_rows):
        counts = Counter(token for token in tokens if token in col)
        for token, count in counts.items():
            features[row_idx, col[token]] = float(count)
    if transform.startswith("binary"):
        features = (features > 0).astype(np.float32)
        norm = transform.removeprefix("binary_") or "none"
        features = _normalize(features, norm)
    elif transform.startswith("tfidf"):
        train_counts = features[train_idx]
        df = np.sum(train_counts > 0, axis=0)
        idf = np.log((1.0 + len(train_idx)) / (1.0 + df)) + 1.0
        features = features * idf.reshape(1, -1).astype(np.float32)
        norm = transform.removeprefix("tfidf_") or "none"
        features = _normalize(features, norm)
    elif transform.startswith("count"):
        norm = transform.removeprefix("count_") or "none"
        features = _normalize(features, norm)
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    return features.astype(np.float32, copy=False), {
        "feature_view": feature_view,
        "transform": transform,
        "num_features": int(features.shape[1]),
        "kept_tokens": kept,
    }


def _normalize(features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return features.astype(np.float32, copy=False)
    if mode == "l1":
        denom = np.sum(np.abs(features), axis=1, keepdims=True)
    elif mode == "l2":
        denom = np.linalg.norm(features, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported normalization: {mode}")
    return np.divide(features, denom, out=np.zeros_like(features, dtype=np.float32), where=denom > 0)


def _token_path(token_dir: Path, artifact_prefix: str, attack: str, seed: int) -> Path:
    return token_dir / f"{artifact_prefix}_leave_one_{ATTACK_SLUG[attack]}_anomaly_seed{seed}_a3_full_rhythm.pt"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def _collect_needed_ids(token_dir: Path, artifact_prefix: str, attacks: list[str], seeds: list[int]) -> set[str]:
    ids: set[str] = set()
    for attack in attacks:
        for seed in seeds:
            path = _token_path(token_dir, artifact_prefix, attack, seed)
            if not path.exists():
                continue
            data = _read_token_data(path)
            ids.update(str(meta.get("flow_id")) for meta in data.get("meta", []) if meta.get("flow_id"))
    return ids


def _build_flow_cache(flow_jsonl: Path, needed_ids: set[str], cache_path: Path) -> dict[str, dict[str, Any]]:
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_ids = set(payload.get("flows", {}))
        if needed_ids.issubset(cached_ids):
            return payload["flows"]
    flows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(flow_jsonl):
        flow_id = str(row.get("flow_id") or "")
        if flow_id not in needed_ids:
            continue
        lens = row.get("lens") or []
        flows[flow_id] = {
            "flow_id": flow_id,
            "attack_family": row.get("attack_family") or row.get("label"),
            "binary_label": row.get("binary_label"),
            "packet_count": int(row.get("packet_count") or len(lens) or 0),
            "byte_count": int(row.get("byte_count") or sum(lens) or 0),
            "duration": float(row.get("duration") or 0.0),
            "lens_head": [int(x) for x in lens[:12]],
            "dirs_head": [int(x) for x in (row.get("dirs") or [])[:12]],
            "raw_ip_used_as_token": bool(row.get("raw_ip_used_as_token")),
            "absolute_time_used_as_token": bool(row.get("absolute_time_used_as_token")),
            "five_tuple_used_as_token": bool(row.get("five_tuple_used_as_token")),
            "protocol_service_used_as_memory_key": bool(row.get("protocol_service_used_as_memory_key")),
        }
        if len(flows) == len(needed_ids):
            break
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"flows": flows}, sort_keys=True) + "\n", encoding="utf-8")
    return flows


def _stratum(meta: dict[str, Any] | None) -> str:
    if not meta:
        return "missing_meta"
    pkt = int(meta.get("packet_count") or 0)
    byt = int(meta.get("byte_count") or 0)
    dur = float(meta.get("duration") or 0.0)
    if pkt <= 2 and byt <= 0:
        return "degenerate_p2_zero_byte"
    if pkt <= 2:
        return "short_p2_nonzero"
    if pkt <= 10 and dur <= 0.02:
        return "short_p10_fast"
    if pkt <= 20:
        return "medium_p20"
    return "long_gt20"


def _gate_keep(meta: dict[str, Any] | None, gate: str) -> bool:
    if gate == "all":
        return True
    stratum = _stratum(meta)
    if gate == "exclude_degenerate_p2_zero_byte":
        return stratum != "degenerate_p2_zero_byte"
    if gate == "exclude_all_p2":
        return stratum not in {"degenerate_p2_zero_byte", "short_p2_nonzero"}
    if gate == "exclude_short_p10_fast":
        return stratum not in {"degenerate_p2_zero_byte", "short_p2_nonzero", "short_p10_fast"}
    raise ValueError(f"Unsupported gate: {gate}")


def _metrics_for_subset(y_true: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y = y_true[mask]
    s = scores[mask]
    if len(y) == 0:
        return {
            "test_benign": 0,
            "test_attack": 0,
            "auroc": "",
            "auprc": "",
            "recall_at_0_1pct_fpr": "",
            "recall_at_1pct_fpr": "",
            "recall_at_5pct_fpr": "",
        }
    rank = S._rank_metrics(y, s)
    r01 = S._best_recall_under_fpr(y, s, 0.001)
    r1 = S._best_recall_under_fpr(y, s, 0.01)
    r5 = S._best_recall_under_fpr(y, s, 0.05)
    return {
        "test_benign": int(np.sum(y == 0)),
        "test_attack": int(np.sum(y == 1)),
        "auroc": rank.get("auroc"),
        "auprc": rank.get("auprc"),
        "recall_at_0_1pct_fpr": r01.get("attack_recall"),
        "recall_at_1pct_fpr": r1.get("attack_recall"),
        "recall_at_5pct_fpr": r5.get("attack_recall"),
    }


def _thresholded(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    pred = (scores >= thresholds).astype(np.int64)
    benign = y_true == 0
    attack = y_true == 1
    fp = int(np.sum((pred == 1) & benign))
    tn = int(np.sum((pred == 0) & benign))
    tp = int(np.sum((pred == 1) & attack))
    fn = int(np.sum((pred == 0) & attack))
    pos_f1 = float((2 * tp) / max(2 * tp + fp + fn, 1))
    neg_f1 = float((2 * tn) / max(2 * tn + fp + fn, 1))
    return {
        "p99_realized_fpr": float(fp / max(fp + tn, 1)),
        "p99_attack_recall": float(tp / max(tp + fn, 1)),
        "p99_macro_f1": float((pos_f1 + neg_f1) / 2.0),
        "false_alerts_per_10k_benign": float(fp / max(fp + tn, 1) * 10000.0),
    }


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[Any, Any]:
    vals = []
    for row in rows:
        value = row.get(key)
        if value == "" or value is None:
            continue
        value = float(value)
        if not math.isnan(value):
            vals.append(value)
    if not vals:
        return "", ""
    return float(np.mean(vals)), float(np.std(vals))


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


def audit(args: argparse.Namespace) -> dict[str, Any]:
    token_dir = Path(args.token_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    needed_ids = _collect_needed_ids(token_dir, args.artifact_prefix, args.attacks, args.seeds)
    flow_meta = _build_flow_cache(Path(args.flow_jsonl), needed_ids, out_dir / "flow_behavior_metadata_cache.json")

    run_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    missing_meta = 0
    stratum_counts: Counter[str] = Counter()

    for attack in args.attacks:
        for seed in args.seeds:
            token_path = _token_path(token_dir, args.artifact_prefix, attack, seed)
            if not token_path.exists():
                run_rows.append({"heldout_attack": attack, "seed": seed, "status": "missing", "token_path": str(token_path)})
                continue
            data = _read_token_data(token_path)
            labels = data["binary_labels"].cpu().numpy().astype(np.int64)
            train_idx = _split_indices(data, "train")
            val_idx = _split_indices(data, "val")
            test_idx = _split_indices(data, "test")
            features, _ = _features_for_view(data, train_idx, feature_view=args.feature_view, transform=args.transform)
            groups = ["GLOBAL"] * len(labels)
            val_scores = S._scores(features, train_idx, val_idx, groups, scorer=args.scorer, k=args.k)
            test_scores = S._scores(features, train_idx, test_idx, groups, scorer=args.scorer, k=args.k)
            test_labels = labels[test_idx]
            val_threshold = float(np.quantile(val_scores, args.val_quantile))

            row_meta = [flow_meta.get(str(meta.get("flow_id") or "")) for meta in data.get("meta", [])]
            missing_meta += sum(1 for meta in row_meta if meta is None)
            strata = np.asarray([_stratum(row_meta[idx]) for idx in range(len(row_meta))], dtype=object)
            for s in strata[test_idx]:
                stratum_counts[str(s)] += 1

            base = {
                "heldout_attack": attack,
                "seed": seed,
                "status": "ok",
                "feature_view": args.feature_view,
                "transform": args.transform,
                "scorer": args.scorer,
                "k": args.k,
                "val_p99_threshold": val_threshold,
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
                "protocol_service_used_as_memory_key": False,
            }

            for gate in args.gates:
                mask = np.asarray([_gate_keep(row_meta[int(idx)], gate) for idx in test_idx], dtype=bool)
                m = _metrics_for_subset(test_labels, test_scores, mask)
                kept_scores = test_scores[mask]
                kept_labels = test_labels[mask]
                p99 = _thresholded(kept_labels, kept_scores, np.full(len(kept_scores), val_threshold, dtype=np.float32)) if len(kept_scores) else {}
                run_rows.append(
                    {
                        **base,
                        "evaluation_mode": "evidence_quality_gate",
                        "gate": gate,
                        "kept_test_flows": int(np.sum(mask)),
                        "removed_test_flows": int(len(mask) - np.sum(mask)),
                        "removed_attack_flows": int(np.sum((~mask) & (test_labels == 1))),
                        "removed_benign_flows": int(np.sum((~mask) & (test_labels == 0))),
                        **m,
                        **p99,
                    }
                )

            test_strata = strata[test_idx]
            for stratum in sorted(set(test_strata.tolist())):
                mask = test_strata == stratum
                m = _metrics_for_subset(test_labels, test_scores, mask)
                p99 = _thresholded(test_labels[mask], test_scores[mask], np.full(int(np.sum(mask)), val_threshold, dtype=np.float32))
                stratum_rows.append({**base, "stratum": stratum, **m, **p99})

            stratum_thresholds: dict[str, float] = {}
            val_strata = strata[val_idx]
            for stratum in sorted(set(strata.tolist())):
                vals = val_scores[val_strata == stratum]
                if len(vals) >= args.min_val_per_stratum:
                    stratum_thresholds[stratum] = float(np.quantile(vals, args.val_quantile))
                else:
                    stratum_thresholds[stratum] = val_threshold
            per_row_threshold = np.asarray([stratum_thresholds[str(s)] for s in test_strata], dtype=np.float32)
            conservative_threshold = np.asarray(
                [max(val_threshold, stratum_thresholds[str(s)]) for s in test_strata],
                dtype=np.float32,
            )
            threshold_rows.append(
                {
                    **base,
                    "threshold_mode": "behavior_stratified_p99",
                    "min_val_per_stratum": args.min_val_per_stratum,
                    **_metrics_for_subset(test_labels, test_scores, np.ones(len(test_idx), dtype=bool)),
                    **_thresholded(test_labels, test_scores, per_row_threshold),
                }
            )
            threshold_rows.append(
                {
                    **base,
                    "threshold_mode": "conservative_max_global_stratum_p99",
                    "min_val_per_stratum": args.min_val_per_stratum,
                    **_metrics_for_subset(test_labels, test_scores, np.ones(len(test_idx), dtype=bool)),
                    **_thresholded(test_labels, test_scores, conservative_threshold),
                }
            )
            threshold_rows.append(
                {
                    **base,
                    "threshold_mode": "global_p99",
                    "min_val_per_stratum": "",
                    **_metrics_for_subset(test_labels, test_scores, np.ones(len(test_idx), dtype=bool)),
                    **_thresholded(test_labels, test_scores, np.full(len(test_idx), val_threshold, dtype=np.float32)),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for source, rows, group_keys in [
        ("gate", run_rows, ["evaluation_mode", "gate"]),
        ("threshold", threshold_rows, ["threshold_mode"]),
        ("stratum", stratum_rows, ["stratum"]),
    ]:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("status") == "ok":
                grouped[tuple(row.get(k) for k in group_keys)].append(row)
        for key_values, items in sorted(grouped.items()):
            out = {"source": source, "runs": len(items)}
            out.update({k: v for k, v in zip(group_keys, key_values)})
            for metric in [
                "auroc",
                "auprc",
                "recall_at_0_1pct_fpr",
                "recall_at_1pct_fpr",
                "recall_at_5pct_fpr",
                "p99_realized_fpr",
                "p99_attack_recall",
                "p99_macro_f1",
                "false_alerts_per_10k_benign",
                "test_benign",
                "test_attack",
                "removed_attack_flows",
                "removed_benign_flows",
            ]:
                mean, std = _mean_std(items, metric)
                out[metric] = mean
                out[f"{metric}_std"] = std
            summary_rows.append(out)

    _write_csv(out_dir / "behavior_strata_run_metrics.csv", run_rows)
    _write_csv(out_dir / "behavior_strata_metrics.csv", stratum_rows)
    _write_csv(out_dir / "behavior_stratified_threshold_metrics.csv", threshold_rows)
    _write_csv(out_dir / "behavior_strata_summary.csv", summary_rows)
    _write_report(out_dir, summary_rows, stratum_counts, missing_meta, args)
    return {"output": str(out_dir), "runs": len(run_rows), "threshold_rows": len(threshold_rows), "strata_rows": len(stratum_rows)}


def _fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return "-"
    try:
        if math.isnan(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_report(out_dir: Path, summary_rows: list[dict[str, Any]], counts: Counter[str], missing_meta: int, args: argparse.Namespace) -> None:
    lines = [
        "# IDS2018 Behavior-evidence Strata Diagnostic",
        "",
        "This diagnostic keeps the same behavior-token KNN scoring path and tests whether behavior-only evidence-quality strata explain the IDS2018 low-FPR failures.",
        "",
        "Leakage controls: raw IP, absolute timestamp, five-tuple, protocol, and service are not used as behavior tokens, memory keys, or threshold strata.",
        "",
        f"Feature view: `{args.feature_view}`; transform: `{args.transform}`; scorer: `{args.scorer}`; k={args.k}.",
        f"Missing joined flow metadata rows: {missing_meta}.",
        "",
        "## Test Stratum Counts",
        "",
        "| Stratum | Rows |",
        "|---|---:|",
    ]
    for name, count in sorted(counts.items()):
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## Aggregate Summary", "", "| Source | Setting | Runs | AUROC | R@1%FPR | P99 FPR | P99 attack recall | Removed attack | Removed benign |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary_rows:
        setting = row.get("gate") or row.get("threshold_mode") or row.get("stratum")
        lines.append(
            "| {source} | {setting} | {runs} | {auroc} | {r1} | {p99} | {p99r} | {ra} | {rb} |".format(
                source=row.get("source"),
                setting=setting,
                runs=row.get("runs"),
                auroc=_fmt(row.get("auroc")),
                r1=_fmt(row.get("recall_at_1pct_fpr")),
                p99=_fmt(row.get("p99_realized_fpr")),
                p99r=_fmt(row.get("p99_attack_recall")),
                ra=_fmt(row.get("removed_attack_flows"), 1),
                rb=_fmt(row.get("removed_benign_flows"), 1),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `behavior_stratified_p99` is a benign-validation-only calibration diagnostic, not a test-oracle threshold.",
            "- Evidence-quality gates are diagnostics unless the paper explicitly defines low-evidence flows as insufficient-evidence records rather than high-confidence anomaly alerts.",
            "- Any improvement after removing short/degenerate flows should be interpreted together with removed attack/benign coverage.",
            "",
            "## Outputs",
            "",
            "- `behavior_strata_run_metrics.csv`",
            "- `behavior_strata_metrics.csv`",
            "- `behavior_stratified_threshold_metrics.csv`",
            "- `behavior_strata_summary.csv`",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit IDS2018 behavior-only evidence strata.")
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--flow-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-prefix", default="ids2018_official_victim")
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "BruteForce", "DDoS", "DoS", "Infiltration", "WebAttack"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--feature-view", default="packet_burst_only")
    parser.add_argument("--transform", default="binary_l2")
    parser.add_argument("--scorer", default="knn_euclidean")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--val-quantile", type=float, default=0.99)
    parser.add_argument("--min-val-per-stratum", type=int, default=100)
    parser.add_argument(
        "--gates",
        nargs="+",
        default=["all", "exclude_degenerate_p2_zero_byte", "exclude_all_p2", "exclude_short_p10_fast"],
    )
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
