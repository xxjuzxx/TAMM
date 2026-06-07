#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.utils.io import write_json


METRIC_ALIASES = {
    "auroc": ("auroc", "auroc_ovr"),
    "auprc": ("auprc", "auprc_ovr"),
    "fpr95": ("fpr95", "fpr_at_95_tpr"),
    "accuracy": ("accuracy",),
    "macro_f1": ("macro_f1",),
    "weighted_f1": ("weighted_f1",),
    "threshold": ("threshold",),
}


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _metric(metrics: dict[str, Any], name: str) -> float | None:
    for key in METRIC_ALIASES[name]:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _score_summary(scores_path: Path | None) -> dict[str, Any]:
    if scores_path is None or not scores_path.exists():
        return {"available": False}
    rows = _read_jsonl(scores_path)
    labels = sorted({int(row.get("binary_label", 0)) for row in rows})
    summary: dict[str, Any] = {"available": True, "num_scores": len(rows)}

    def summarize(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "std": None, "min": None, "q25": None, "median": None, "q75": None, "q95": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "median": float(np.quantile(arr, 0.50)),
            "q75": float(np.quantile(arr, 0.75)),
            "q95": float(np.quantile(arr, 0.95)),
            "max": float(np.max(arr)),
        }

    summary["overall"] = summarize([float(row.get("anomaly_score", 0.0)) for row in rows])
    by_label: dict[str, Any] = {}
    for label in labels:
        by_label[str(label)] = summarize([float(row.get("anomaly_score", 0.0)) for row in rows if int(row.get("binary_label", 0)) == label])
    summary["by_binary_label"] = by_label
    return summary


def _service_key_from_meta(meta: dict[str, Any] | None) -> str:
    if not meta:
        return "UNKNOWN"
    key = meta.get("service_key")
    if isinstance(key, (list, tuple)):
        return "|".join(str(item) for item in key)
    if key:
        return str(key)
    return "UNKNOWN"


def _per_service_fpr(scores_path: Path | None, token_path: str | Path | None) -> dict[str, Any]:
    if scores_path is None or not scores_path.exists() or token_path is None:
        return {"available": False}
    token_file = Path(token_path)
    if not token_file.exists():
        return {"available": False, "reason": f"token file not found: {token_file}"}
    rows = _read_jsonl(scores_path)
    token_data = torch.load(token_file, map_location="cpu", weights_only=False)
    meta_rows = token_data.get("meta", [])
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"benign": 0, "fp": 0, "total": 0})
    for row in rows:
        data_idx = int(row.get("index", -1))
        meta = meta_rows[data_idx] if 0 <= data_idx < len(meta_rows) else None
        key = _service_key_from_meta(meta)
        bucket = buckets[key]
        bucket["total"] += 1
        if int(row.get("binary_label", 0)) == 0:
            bucket["benign"] += 1
            if int(row.get("prediction", 0)) == 1:
                bucket["fp"] += 1
    fprs = []
    service_rows = []
    for key, counts in buckets.items():
        benign = counts["benign"]
        fpr = float(counts["fp"] / benign) if benign else 0.0
        if benign:
            fprs.append(fpr)
        service_rows.append({"service_key": key, **counts, "fpr": fpr})
    service_rows.sort(key=lambda item: (-float(item["fpr"]), -int(item["benign"]), item["service_key"]))
    return {
        "available": True,
        "num_services": len(buckets),
        "num_services_with_benign": int(sum(1 for row in service_rows if row["benign"] > 0)),
        "mean_service_fpr": float(np.mean(fprs)) if fprs else None,
        "max_service_fpr": float(max(fprs)) if fprs else None,
        "services_with_fp": int(sum(1 for row in service_rows if row["fp"] > 0)),
        "top_service_fpr": service_rows[:10],
    }


def _manifest_for(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "run_manifest.json"
    return _read_json(path) if path.exists() else {}


def _scores_path(result_dir: Path) -> Path | None:
    path = result_dir / "scores.jsonl"
    return path if path.exists() else None


def _canonical_encoder_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state_dict)!r}")
    canonical: dict[str, torch.Tensor] = {}
    prefixes = (
        "encoder.token_embedding.",
        "encoder.position_embedding.",
        "encoder.type_embedding.",
        "encoder.norm.",
        "encoder.encoder.",
        "encoder.class_attention.",
        "encoder.class_queries",
        "token_embedding.",
        "position_embedding.",
        "type_embedding.",
        "norm.",
        "encoder.",
        "class_attention.",
        "class_queries",
    )
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        if key.startswith("encoder."):
            key = key.removeprefix("encoder.")
        if key.startswith("classifier.") or key.startswith("class_logit_"):
            continue
        if torch.is_tensor(value):
            canonical[key] = value.detach().cpu()
    return canonical


def _hash_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compare_encoder_checkpoints(baseline_checkpoint: str | Path | None, modified_checkpoint: str | Path | None) -> dict[str, Any]:
    if not baseline_checkpoint or not modified_checkpoint:
        return {"available": False}
    base = _canonical_encoder_state(baseline_checkpoint)
    mod = _canonical_encoder_state(modified_checkpoint)
    base_keys = set(base)
    mod_keys = set(mod)
    shared = sorted(base_keys & mod_keys)
    missing = sorted(base_keys - mod_keys)
    added = sorted(mod_keys - base_keys)
    changed = []
    max_abs_diff = 0.0
    l2_diff_sq = 0.0
    compared_params = 0
    for key in shared:
        if tuple(base[key].shape) != tuple(mod[key].shape):
            changed.append({"key": key, "reason": "shape", "baseline_shape": list(base[key].shape), "modified_shape": list(mod[key].shape)})
            continue
        if _hash_tensor(base[key]) != _hash_tensor(mod[key]):
            diff = (base[key].float() - mod[key].float()).abs()
            changed.append({"key": key, "reason": "value", "max_abs_diff": float(diff.max().item())})
            max_abs_diff = max(max_abs_diff, float(diff.max().item()))
            l2_diff_sq += float(torch.sum(diff * diff).item())
        compared_params += int(base[key].numel())
    identical = not missing and not added and not changed
    return {
        "available": True,
        "identical": bool(identical),
        "num_baseline_encoder_tensors": len(base),
        "num_modified_encoder_tensors": len(mod),
        "num_shared_encoder_tensors": len(shared),
        "num_missing_tensors": len(missing),
        "num_added_tensors": len(added),
        "num_changed_tensors": len(changed),
        "num_compared_params": compared_params,
        "max_abs_diff": float(max_abs_diff),
        "l2_diff": float(math.sqrt(l2_diff_sq)),
        "missing_tensors": missing[:20],
        "added_tensors": added[:20],
        "changed_tensors": changed[:20],
    }


def evaluate_gate(
    *,
    experiment: str,
    baseline_dir: str | Path,
    modified_dir: str | Path,
    classification_delta_macro_f1: float | None,
    baseline_checkpoint: str | Path | None,
    modified_checkpoint: str | Path | None,
    auroc_drop_limit: float,
    auprc_drop_limit: float,
    fpr95_worsen_limit: float,
    min_classification_gain: float,
) -> dict[str, Any]:
    baseline_dir = Path(baseline_dir)
    modified_dir = Path(modified_dir)
    base_metrics = _read_json(baseline_dir / "metrics.json")
    mod_metrics = _read_json(modified_dir / "metrics.json")
    base_manifest = _manifest_for(baseline_dir)
    mod_manifest = _manifest_for(modified_dir)

    deltas: dict[str, float | None] = {}
    values: dict[str, float | None] = {}
    for name in METRIC_ALIASES:
        base_value = _metric(base_metrics, name)
        mod_value = _metric(mod_metrics, name)
        values[f"baseline_{name}"] = base_value
        values[f"modified_{name}"] = mod_value
        deltas[f"delta_{name}"] = None if base_value is None or mod_value is None else float(mod_value - base_value)

    encoder_compare = compare_encoder_checkpoints(baseline_checkpoint, modified_checkpoint)
    base_scores = _score_summary(_scores_path(baseline_dir))
    mod_scores = _score_summary(_scores_path(modified_dir))
    token_path = mod_manifest.get("tokens") or base_manifest.get("tokens")
    base_service = _per_service_fpr(_scores_path(baseline_dir), token_path) if base_metrics.get("baseline") == "service_memory" else {"available": False}
    mod_service = _per_service_fpr(_scores_path(modified_dir), token_path) if mod_metrics.get("baseline") == "service_memory" else {"available": False}

    auroc_drop = -(deltas["delta_auroc"] or 0.0)
    auprc_drop = -(deltas["delta_auprc"] or 0.0)
    fpr95_worsen = deltas["delta_fpr95"]
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    if deltas["delta_auroc"] is not None and auroc_drop > auroc_drop_limit:
        fail_reasons.append(f"AUROC dropped by {auroc_drop:.6f} > {auroc_drop_limit:.6f}")
    if deltas["delta_auprc"] is not None and auprc_drop > auprc_drop_limit:
        fail_reasons.append(f"AUPRC dropped by {auprc_drop:.6f} > {auprc_drop_limit:.6f}")
    if fpr95_worsen is not None and fpr95_worsen > fpr95_worsen_limit:
        fail_reasons.append(f"FPR@95TPR worsened by {fpr95_worsen:.6f} > {fpr95_worsen_limit:.6f}")
    if classification_delta_macro_f1 is not None and classification_delta_macro_f1 < min_classification_gain and fail_reasons:
        fail_reasons.append(
            f"classification Macro-F1 gain {classification_delta_macro_f1:.6f} < {min_classification_gain:.6f} while anomaly regressed"
        )
    if encoder_compare.get("available") and not encoder_compare.get("identical"):
        warn_reasons.append("encoder checkpoint weights differ; anomaly evaluation is required for this comparison")
    if not base_scores.get("available") or not mod_scores.get("available"):
        warn_reasons.append("score distribution summary unavailable because scores.jsonl is missing")

    status = "FAIL" if fail_reasons else ("WARN" if warn_reasons else "PASS")
    return {
        "experiment": experiment,
        "status": status,
        "fail_reasons": fail_reasons,
        "warn_reasons": warn_reasons,
        "classification_delta_macro_f1": classification_delta_macro_f1,
        "baseline_dir": str(baseline_dir),
        "modified_dir": str(modified_dir),
        "baseline_checkpoint": str(baseline_checkpoint) if baseline_checkpoint else base_manifest.get("checkpoint"),
        "modified_checkpoint": str(modified_checkpoint) if modified_checkpoint else mod_manifest.get("checkpoint"),
        "baseline_feature": base_metrics.get("feature"),
        "modified_feature": mod_metrics.get("feature"),
        "baseline_method": base_metrics.get("baseline"),
        "modified_method": mod_metrics.get("baseline"),
        "score_method": mod_metrics.get("score_method"),
        "normalize": mod_metrics.get("normalize"),
        **values,
        **deltas,
        "encoder_compare": encoder_compare,
        "baseline_score_summary": base_scores,
        "modified_score_summary": mod_scores,
        "baseline_service_fpr": base_service,
        "modified_service_fpr": mod_service,
        "baseline_command": base_manifest.get("command"),
        "modified_command": mod_manifest.get("command"),
    }


def _flatten_row(result: dict[str, Any]) -> dict[str, Any]:
    base_service = result.get("baseline_service_fpr", {}) or {}
    mod_service = result.get("modified_service_fpr", {}) or {}
    enc = result.get("encoder_compare", {}) or {}
    return {
        "experiment": result["experiment"],
        "status": result["status"],
        "baseline_method": result.get("baseline_method"),
        "feature": result.get("modified_feature"),
        "baseline_auroc": result.get("baseline_auroc"),
        "modified_auroc": result.get("modified_auroc"),
        "delta_auroc": result.get("delta_auroc"),
        "baseline_auprc": result.get("baseline_auprc"),
        "modified_auprc": result.get("modified_auprc"),
        "delta_auprc": result.get("delta_auprc"),
        "baseline_fpr95": result.get("baseline_fpr95"),
        "modified_fpr95": result.get("modified_fpr95"),
        "delta_fpr95": result.get("delta_fpr95"),
        "baseline_accuracy": result.get("baseline_accuracy"),
        "modified_accuracy": result.get("modified_accuracy"),
        "delta_accuracy": result.get("delta_accuracy"),
        "baseline_macro_f1": result.get("baseline_macro_f1"),
        "modified_macro_f1": result.get("modified_macro_f1"),
        "delta_macro_f1": result.get("delta_macro_f1"),
        "baseline_threshold": result.get("baseline_threshold"),
        "modified_threshold": result.get("modified_threshold"),
        "classification_delta_macro_f1": result.get("classification_delta_macro_f1"),
        "encoder_identical": enc.get("identical") if enc.get("available") else None,
        "encoder_changed_tensors": enc.get("num_changed_tensors") if enc.get("available") else None,
        "encoder_max_abs_diff": enc.get("max_abs_diff") if enc.get("available") else None,
        "baseline_mean_service_fpr": base_service.get("mean_service_fpr"),
        "modified_mean_service_fpr": mod_service.get("mean_service_fpr"),
        "baseline_max_service_fpr": base_service.get("max_service_fpr"),
        "modified_max_service_fpr": mod_service.get("max_service_fpr"),
        "fail_reasons": "; ".join(result.get("fail_reasons", [])),
        "warn_reasons": "; ".join(result.get("warn_reasons", [])),
        "baseline_dir": result.get("baseline_dir"),
        "modified_dir": result.get("modified_dir"),
    }


def write_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    rows = [_flatten_row(result) for result in results]
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="anomaly_regression")
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--modified_dir", required=True)
    parser.add_argument("--baseline_checkpoint", default=None)
    parser.add_argument("--modified_checkpoint", default=None)
    parser.add_argument("--classification_delta_macro_f1", type=float, default=None)
    parser.add_argument("--auroc_drop_limit", type=float, default=0.003)
    parser.add_argument("--auprc_drop_limit", type=float, default=0.005)
    parser.add_argument("--fpr95_worsen_limit", type=float, default=0.01)
    parser.add_argument("--min_classification_gain", type=float, default=0.005)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    result = evaluate_gate(
        experiment=args.experiment,
        baseline_dir=args.baseline_dir,
        modified_dir=args.modified_dir,
        classification_delta_macro_f1=args.classification_delta_macro_f1,
        baseline_checkpoint=args.baseline_checkpoint,
        modified_checkpoint=args.modified_checkpoint,
        auroc_drop_limit=args.auroc_drop_limit,
        auprc_drop_limit=args.auprc_drop_limit,
        fpr95_worsen_limit=args.fpr95_worsen_limit,
        min_classification_gain=args.min_classification_gain,
    )
    result["command"] = shlex.join(sys.argv)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    write_json(result, args.out_json)
    if args.out_csv:
        write_csv([result], args.out_csv)
    print(json.dumps(_flatten_row(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
