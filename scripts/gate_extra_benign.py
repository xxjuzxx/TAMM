#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extra_benign_common import (
    EXTRA_ARTIFACT_DIR,
    EXTRA_RESULT_DIR,
    dominant_token_group,
    load_extra_metadata,
    load_extra_token_rows,
    load_state,
    raw_count_matrix_from_token_rows,
    score_external,
    transform_extra_counts,
    write_csv,
)


def _bucket(score: float, p95: float, p99: float, p99_5: float, p100: float) -> str:
    if score < p95:
        return "below_p95"
    if score < p99:
        return "p95_p99"
    if score < p99_5:
        return "p99_p99_5"
    if score <= p100:
        return "p99_5_p100"
    return "above_p100"


def _status(bucket: str) -> str:
    if bucket in {"below_p95", "p95_p99", "p99_p99_5"}:
        return "memory_candidate"
    if bucket == "p99_5_p100":
        return "tail_test_candidate"
    return "quarantine"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate extra benign candidates with old FlowPrim benign memory and benign-validation thresholds.")
    parser.add_argument("--old-benign-memory-artifact", default=None, help="Optional manifest path; current implementation recomputes from token corpus for reproducibility.")
    parser.add_argument("--old-calibration-thresholds", default=None, help="Optional old threshold artifact; current implementation recomputes from benign validation scores.")
    parser.add_argument("--extra-benign-histograms", default=str(EXTRA_ARTIFACT_DIR / "extra_benign_histograms.npz"), help="Accepted for interface compatibility; token JSONL is used to map into split-specific vocab.")
    parser.add_argument("--extra-benign-tokens", default=str(EXTRA_ARTIFACT_DIR / "extra_benign_tokens.jsonl"))
    parser.add_argument("--extra-benign-metadata", default=str(EXTRA_ARTIFACT_DIR / "extra_benign_metadata.csv"))
    parser.add_argument("--gate-policy", choices=["below_p95", "below_p99", "below_p99_5", "below_p100"], default="below_p99_5")
    parser.add_argument("--attack", default="Botnet")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-dir", default=str(EXTRA_RESULT_DIR))
    args = parser.parse_args()

    del args.old_benign_memory_artifact, args.old_calibration_thresholds, args.extra_benign_histograms, args.gate_policy
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(args.attack, args.seed)
    token_rows = load_extra_token_rows(args.extra_benign_tokens)
    metadata = load_extra_metadata(args.extra_benign_metadata)
    counts, flow_ids, extra_groups = raw_count_matrix_from_token_rows(token_rows, state["token_data"])
    extra_features, _kept_ids, kept_tokens = transform_extra_counts(
        counts,
        state["token_data"],
        state["train_idx"],
        state["setting"]["feature_filter"],
        state["setting"]["transform"],
    )
    ref_features = state["features"][state["train_idx"]]
    ref_groups = [state["groups"][idx] for idx in state["train_idx"].tolist()]
    if state["setting"]["group_mode"] == "global":
        extra_groups = ["GLOBAL"] * len(extra_groups)
        ref_groups = ["GLOBAL"] * len(ref_groups)
    scores = score_external(
        extra_features,
        extra_groups,
        ref_features,
        ref_groups,
        scorer=state["setting"]["scorer"],
        k=int(state["setting"]["k"]),
    )
    val_scores = score_external(
        state["features"][state["val_idx"]],
        [state["groups"][idx] for idx in state["val_idx"].tolist()],
        ref_features,
        ref_groups,
        scorer=state["setting"]["scorer"],
        k=int(state["setting"]["k"]),
    )
    p95, p99, p99_5, p100 = [float(np.percentile(val_scores, q)) for q in [95.0, 99.0, 99.5, 100.0]]
    rows = []
    for pos, flow_id in enumerate(flow_ids):
        meta = metadata.get(flow_id, {})
        score = float(scores[pos])
        bucket = _bucket(score, p95, p99, p99_5, p100)
        rows.append(
            {
                "flow_id": flow_id,
                "old_score": score,
                "old_threshold_p95": p95,
                "old_threshold_p99": p99,
                "old_threshold_p99_5": p99_5,
                "old_threshold_p100": p100,
                "gate_bucket": bucket,
                "admission_status": _status(bucket),
                "dominant_token_group": dominant_token_group(extra_features[pos], kept_tokens),
                "primitive_SHORT": meta.get("primitive_SHORT", ""),
                "primitive_SAME": meta.get("primitive_SAME", ""),
                "primitive_PKT": meta.get("primitive_PKT", ""),
                "primitive_LOCAL": meta.get("primitive_LOCAL", ""),
                "primitive_REPEAT": meta.get("primitive_REPEAT", ""),
                "primitive_DUP": meta.get("primitive_DUP", ""),
                "protocol": meta.get("protocol", ""),
                "service": meta.get("service", ""),
                "source_dataset": meta.get("source_dataset", ""),
                "capability_level": meta.get("capability_level", ""),
                "raw_ip_used_as_token": "false",
                "absolute_time_used_as_token": "false",
                "five_tuple_used_as_token": "false",
                "gate_reference_attack": args.attack,
                "gate_reference_seed": args.seed,
            }
        )
    summary = []
    total = max(len(rows), 1)
    for status in ["memory_candidate", "tail_test_candidate", "quarantine"]:
        subset = [row for row in rows if row["admission_status"] == status]
        summary.append({"admission_status": status, "count": len(subset), "fraction": len(subset) / total})
    for bucket in ["below_p95", "p95_p99", "p99_p99_5", "p99_5_p100", "above_p100"]:
        subset = [row for row in rows if row["gate_bucket"] == bucket]
        summary.append({"admission_status": f"bucket:{bucket}", "count": len(subset), "fraction": len(subset) / total})
    write_csv(rows, out_dir / "extra_benign_gate_scores.csv")
    write_csv(summary, out_dir / "extra_benign_gate_summary.csv")
    print(json.dumps({"gate_scores": len(rows), "summary_rows": len(summary), "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
