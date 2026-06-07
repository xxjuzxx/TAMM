#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extra_benign_common import (
    EXTRA_ARTIFACT_DIR,
    EXTRA_SPLIT_DIR,
    ROOT,
    command_used,
    load_extra_token_rows,
    load_state,
    now_utc,
    raw_count_matrix_from_token_rows,
    read_csv,
    rel,
    save_memory_artifact,
    transform_extra_counts,
    write_json,
)


def _select(rows: list[dict[str, str]], strategy: str, max_n: int | None, seed: int) -> list[dict[str, str]]:
    candidates = [row for row in rows if row.get("admission_status") == "memory_candidate"]
    if strategy == "low_risk_only":
        candidates = [row for row in candidates if row.get("gate_bucket") in {"below_p95", "p95_p99"}]
    rng = np.random.default_rng(seed)
    ordered = list(candidates)
    if strategy == "random":
        rng.shuffle(ordered)
    if max_n is not None:
        ordered = ordered[: max(0, int(max_n))]
    return ordered


def _select_token_coreset(
    rows: list[dict[str, str]],
    token_rows: list[dict],
    state: dict,
    max_n: int | None,
    seed: int,
) -> list[dict[str, str]]:
    candidates = [row for row in rows if row.get("admission_status") == "memory_candidate"]
    if max_n is None or len(candidates) <= max_n:
        return candidates
    candidate_ids = {row["flow_id"] for row in candidates}
    selected_token_rows = [row for row in token_rows if row["flow_id"] in candidate_ids]
    if len(selected_token_rows) <= max_n:
        selected_ids = {row["flow_id"] for row in selected_token_rows}
        return [row for row in candidates if row["flow_id"] in selected_ids]
    counts, flow_ids, _groups = raw_count_matrix_from_token_rows(selected_token_rows, state["token_data"])
    feats, _kept_ids, _kept_tokens = transform_extra_counts(
        counts,
        state["token_data"],
        state["train_idx"],
        state["setting"]["feature_filter"],
        state["setting"]["transform"],
    )
    rng = np.random.default_rng(seed)
    centroid = feats.mean(axis=0, keepdims=True)
    first = int(np.argmin(np.sum((feats - centroid) ** 2, axis=1)))
    selected = [first]
    min_dist = np.sum((feats - feats[first]) ** 2, axis=1)
    min_dist[first] = -1.0
    while len(selected) < max_n and len(selected) < feats.shape[0]:
        max_dist = float(np.max(min_dist))
        ties = np.flatnonzero(np.isclose(min_dist, max_dist))
        nxt = int(rng.choice(ties)) if len(ties) > 1 else int(ties[0])
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.sum((feats - feats[nxt]) ** 2, axis=1))
        min_dist[selected] = -1.0
    selected_ids = {flow_ids[idx] for idx in selected}
    return [row for row in candidates if row["flow_id"] in selected_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an old+extra benign memory artifact for FlowPrim KNN scoring.")
    parser.add_argument("--extra-memory-split", default=str(EXTRA_SPLIT_DIR / "extra_benign_memory.csv"))
    parser.add_argument("--extra-benign-tokens", default=str(EXTRA_ARTIFACT_DIR / "extra_benign_tokens.jsonl"))
    parser.add_argument("--memory-strategy", choices=["random", "token_coreset", "low_risk_only"], default="random")
    parser.add_argument("--max-extra-memory", type=int, default=1000)
    parser.add_argument("--attack", default="Botnet")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-memory", default=str(ROOT / "artifacts" / "memory" / "knn_memory_old_plus_extra.pkl"))
    parser.add_argument("--output-manifest", default=str(ROOT / "artifacts" / "memory" / "memory_manifest.json"))
    args = parser.parse_args()

    state = load_state(args.attack, args.seed)
    split_rows = read_csv(args.extra_memory_split)
    all_token_rows = load_extra_token_rows(args.extra_benign_tokens)
    if args.memory_strategy == "token_coreset":
        selected = _select_token_coreset(split_rows, all_token_rows, state, args.max_extra_memory, args.seed)
    else:
        selected = _select(split_rows, args.memory_strategy, args.max_extra_memory, args.seed)
    selected_ids = {row["flow_id"] for row in selected}
    token_rows = [row for row in all_token_rows if row["flow_id"] in selected_ids]
    counts, flow_ids, groups = raw_count_matrix_from_token_rows(token_rows, state["token_data"])
    extra_features, kept_ids, kept_tokens = transform_extra_counts(
        counts,
        state["token_data"],
        state["train_idx"],
        state["setting"]["feature_filter"],
        state["setting"]["transform"],
    )
    if state["setting"]["group_mode"] == "global":
        groups = ["GLOBAL"] * len(groups)
    payload = {
        "attack": args.attack,
        "seed": args.seed,
        "strategy": args.memory_strategy,
        "base_features": state["features"][state["train_idx"]],
        "base_groups": [state["groups"][idx] for idx in state["train_idx"].tolist()],
        "extra_features": extra_features,
        "extra_groups": groups,
        "extra_flow_ids": flow_ids,
        "feature_filter": state["setting"]["feature_filter"],
        "transform": state["setting"]["transform"],
        "scorer": state["setting"]["scorer"],
        "k": int(state["setting"]["k"]),
        "group_mode": state["setting"]["group_mode"],
        "kept_ids": kept_ids,
        "kept_tokens": kept_tokens,
    }
    save_memory_artifact(payload, args.output_memory)
    manifest = {
        "base_memory_size": int(len(state["train_idx"])),
        "extra_memory_size": int(extra_features.shape[0]),
        "total_memory_size": int(len(state["train_idx"]) + extra_features.shape[0]),
        "strategy": args.memory_strategy,
        "gate_policy": "below_p99_5_default",
        "source_datasets": sorted({row.get("source_dataset", "") for row in selected}),
        "raw_ip_used_as_token": False,
        "absolute_time_used_as_token": False,
        "five_tuple_used_as_token": False,
        "created_at": now_utc(),
        "command_used": command_used(),
        "output_memory": rel(args.output_memory),
    }
    write_json(manifest, args.output_manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
