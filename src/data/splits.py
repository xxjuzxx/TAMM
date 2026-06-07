from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from src.data.dataset_adapter import split_coverage_report
from src.data.dataset_adapter import infer_day


SplitMap = dict[str, list[str]]


def _label(flow: dict[str, Any]) -> str:
    return str(flow.get("attack_family") or flow.get("label") or flow.get("binary_label") or "")


def _flow_id(flow: dict[str, Any], idx: int) -> str:
    return str(flow.get("flow_id") or f"idx:{idx}")


def _sorted_indices(flows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        sorted(
            range(len(flows)),
            key=lambda idx: (
                float(flows[idx].get("start_ts") or 0.0),
                _label(flows[idx]),
                _flow_id(flows[idx], idx),
            ),
        ),
        dtype=np.int64,
    )


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _cut_indices(indices: np.ndarray, val_ratio: float, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(indices))
    test_n = int(round(n * float(test_ratio)))
    val_n = int(round(n * float(val_ratio)))
    if n >= 3:
        test_n = max(1, min(test_n, n - 2))
        val_n = max(1, min(val_n, n - test_n - 1))
    else:
        test_n = 0
        val_n = 0
    train_n = max(0, n - val_n - test_n)
    return indices[:train_n], indices[train_n : train_n + val_n], indices[train_n + val_n :]


def _cut_indices_train_val(indices: np.ndarray, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(indices))
    val_n = int(round(n * float(val_ratio)))
    if n >= 2:
        val_n = max(1, min(val_n, n - 1))
    else:
        val_n = 0
    return indices[: n - val_n], indices[n - val_n :]


def _random_split(flows: list[dict[str, Any]], val_ratio: float, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(flows), dtype=np.int64)
    _rng(seed).shuffle(indices)
    return _cut_indices(indices, val_ratio, test_ratio)


def _stratified_random_split(
    flows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, flow in enumerate(flows):
        label_to_indices[_label(flow)].append(idx)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    rng = _rng(seed)
    for label in sorted(label_to_indices):
        indices = np.array(label_to_indices[label], dtype=np.int64)
        rng.shuffle(indices)
        train_idx, val_idx, test_idx = _cut_indices(indices, val_ratio, test_ratio)
        train_parts.append(train_idx)
        val_parts.append(val_idx)
        test_parts.append(test_idx)
    return (
        np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64),
        np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64),
        np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64),
    )


def _temporal_split(flows: list[dict[str, Any]], val_ratio: float, test_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cut_indices(_sorted_indices(flows), val_ratio, test_ratio)


def _temporal_stratified_split(
    flows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, flow in enumerate(flows):
        label_to_indices[_label(flow)].append(idx)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(label_to_indices):
        ordered = np.array(
            sorted(
                label_to_indices[label],
                key=lambda idx: (float(flows[idx].get("start_ts") or 0.0), _flow_id(flows[idx], idx)),
            ),
            dtype=np.int64,
        )
        train_idx, val_idx, test_idx = _cut_indices(ordered, val_ratio, test_ratio)
        train_parts.append(train_idx)
        val_parts.append(val_idx)
        test_parts.append(test_idx)
    return (
        np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64),
        np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64),
        np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64),
    )


def _small_debug_split(
    flows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_per_label: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, flow in enumerate(flows):
        label_to_indices[_label(flow)].append(idx)
    rng = _rng(seed)
    selected: list[int] = []
    sampled_counts: dict[str, int] = {}
    for label in sorted(label_to_indices):
        indices = np.array(label_to_indices[label], dtype=np.int64)
        rng.shuffle(indices)
        chosen = sorted(indices[: max(1, int(max_per_label))].tolist())
        selected.extend(chosen)
        sampled_counts[label] = len(chosen)
    selected_set = set(selected)
    sampled_flows = [flow for idx, flow in enumerate(flows) if idx in selected_set]
    train_local, val_local, test_local = _stratified_random_split(sampled_flows, val_ratio, test_ratio, seed)
    local_to_global = np.array([idx for idx, _flow in enumerate(flows) if idx in selected_set], dtype=np.int64)
    return local_to_global[train_local], local_to_global[val_local], local_to_global[test_local], {"max_per_label": int(max_per_label), "sampled_counts": sampled_counts}


def _day_wise_split(
    flows: list[dict[str, Any]],
    train_days: list[str] | None,
    val_days: list[str] | None,
    test_days: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_days_norm = {day.lower() for day in (train_days or ["Monday", "Tuesday"])}
    val_days_norm = {day.lower() for day in (val_days or ["Wednesday"])}
    test_days_norm = {day.lower() for day in (test_days or ["Thursday", "Friday"])}
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    unknown_idx: list[int] = []
    for idx, flow in enumerate(flows):
        day = infer_day(flow).lower()
        if day in train_days_norm:
            train_idx.append(idx)
        elif day in val_days_norm:
            val_idx.append(idx)
        elif day in test_days_norm:
            test_idx.append(idx)
        else:
            unknown_idx.append(idx)
    extra = {
        "train_days": sorted(train_days_norm),
        "val_days": sorted(val_days_norm),
        "test_days": sorted(test_days_norm),
        "dropped_unknown_day_count": len(unknown_idx),
    }
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64), extra


def _leave_one_attack_out_split(
    flows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    leave_label: str | None,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]]:
    labels = sorted({_label(flow) for flow in flows if _label(flow).upper() != "BENIGN"})
    if not labels:
        labels = sorted({_label(flow) for flow in flows})
    selected = leave_label or labels[0]
    rng = _rng(seed)
    benign_idx = np.array([idx for idx, flow in enumerate(flows) if _label(flow).upper() == "BENIGN"], dtype=np.int64)
    held_idx = np.array([idx for idx, flow in enumerate(flows) if _label(flow) == selected], dtype=np.int64)
    known_attack_idx = np.array(
        [idx for idx, flow in enumerate(flows) if _label(flow).upper() != "BENIGN" and _label(flow) != selected],
        dtype=np.int64,
    )
    rng.shuffle(benign_idx)
    benign_train, benign_val, benign_test = _cut_indices(benign_idx, val_ratio, test_ratio)
    if mode == "anomaly":
        train_idx = benign_train
        val_idx = benign_val
    elif mode == "classification":
        rng.shuffle(known_attack_idx)
        known_train, known_val = _cut_indices_train_val(known_attack_idx, val_ratio)
        train_idx = np.concatenate([benign_train, known_train])
        val_idx = np.concatenate([benign_val, known_val])
    else:
        raise ValueError(f"Unsupported leave_one mode: {mode}")
    test_idx = np.concatenate([benign_test, held_idx])
    extra = {
        "leave_label": selected,
        "leave_one_mode": mode,
        "held_out_count": int(len(held_idx)),
        "test_benign_count": int(len(benign_test)),
    }
    return train_idx, val_idx, test_idx, selected, extra


def _few_label_split(
    flows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    train_per_label: int,
    train_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = _stratified_random_split(flows, val_ratio, test_ratio, seed)
    labels_by_idx = np.array([_label(flow) for flow in flows])
    rng = _rng(seed)
    selected_train: list[np.ndarray] = []
    for label in sorted(set(labels_by_idx[train_idx].tolist())):
        label_indices = train_idx[labels_by_idx[train_idx] == label]
        rng.shuffle(label_indices)
        if train_fraction is not None:
            keep = max(1, int(np.ceil(label_indices.size * float(train_fraction))))
        else:
            keep = max(1, int(train_per_label))
        selected_train.append(np.sort(label_indices[: min(keep, label_indices.size)]))
    return (
        np.concatenate(selected_train) if selected_train else train_idx,
        val_idx,
        test_idx,
    )


def _to_ids(flows: list[dict[str, Any]], indices: np.ndarray) -> list[str]:
    return [_flow_id(flows[int(idx)], int(idx)) for idx in sorted(indices.tolist())]


def build_split(
    flows: list[dict[str, Any]],
    split: str,
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    leave_label: str | None = None,
    train_per_label: int = 8,
    max_per_label: int = 1000,
    train_days: list[str] | None = None,
    val_days: list[str] | None = None,
    test_days: list[str] | None = None,
    leave_one_mode: str = "classification",
    train_fraction: float | None = None,
) -> dict[str, Any]:
    split = str(split)
    canonical_split = {
        "random_stratified": "stratified_random",
        "temporal_chronological": "temporal_stratified",
    }.get(split, split)
    if canonical_split == "small_debug":
        train_idx, val_idx, test_idx, extra = _small_debug_split(flows, val_ratio, test_ratio, seed, max_per_label)
    elif canonical_split == "random":
        train_idx, val_idx, test_idx = _random_split(flows, val_ratio, test_ratio, seed)
        extra: dict[str, Any] = {}
    elif canonical_split == "stratified_random":
        train_idx, val_idx, test_idx = _stratified_random_split(flows, val_ratio, test_ratio, seed)
        extra = {}
    elif canonical_split == "temporal":
        train_idx, val_idx, test_idx = _temporal_split(flows, val_ratio, test_ratio)
        extra = {}
    elif canonical_split == "temporal_stratified":
        train_idx, val_idx, test_idx = _temporal_stratified_split(flows, val_ratio, test_ratio)
        extra = {}
    elif canonical_split == "day_wise":
        train_idx, val_idx, test_idx, extra = _day_wise_split(flows, train_days, val_days, test_days)
    elif canonical_split == "leave_one_attack_out":
        train_idx, val_idx, test_idx, _selected, extra = _leave_one_attack_out_split(
            flows,
            val_ratio,
            test_ratio,
            seed,
            leave_label,
            leave_one_mode,
        )
    elif canonical_split == "few_label":
        train_idx, val_idx, test_idx = _few_label_split(flows, val_ratio, test_ratio, seed, train_per_label, train_fraction)
        extra = {"train_per_label": int(train_per_label), "train_fraction": train_fraction}
    else:
        raise ValueError(f"Unsupported split: {split}")

    splits = {"train": _to_ids(flows, train_idx), "val": _to_ids(flows, val_idx), "test": _to_ids(flows, test_idx)}
    validate_splits(splits)
    label_counts = split_label_counts(flows, splits)
    payload = {
        "format": "flowprim_split_v1",
        "split": split,
        "canonical_split": canonical_split,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "splits": splits,
        "counts": {key: len(value) for key, value in splits.items()},
        "label_counts": label_counts,
        "coverage": split_coverage_report(flows, splits),
        "flow_count": int(len(flows)),
        "flow_id_digest": hashlib.sha1("\n".join(_flow_id(flow, idx) for idx, flow in enumerate(flows)).encode("utf-8")).hexdigest(),
    }
    payload.update(extra)
    return payload


def split_label_counts(flows: list[dict[str, Any]], splits: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    by_id = {_flow_id(flow, idx): flow for idx, flow in enumerate(flows)}
    out: dict[str, dict[str, int]] = {}
    for split_name, ids in splits.items():
        counts = Counter(_label(by_id[flow_id]) for flow_id in ids if flow_id in by_id)
        out[split_name] = dict(sorted(counts.items()))
    return out


def validate_splits(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for flow_id in splits.get(split_name, []):
            if flow_id in seen:
                raise ValueError(f"flow_id {flow_id} appears in both {seen[flow_id]} and {split_name}")
            seen[flow_id] = split_name


def split_lookup(split_payload: dict[str, Any]) -> dict[str, str]:
    splits = split_payload.get("splits", split_payload)
    out: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for flow_id in splits.get(split_name, []):
            out[str(flow_id)] = split_name
    return out


def flows_for_split(flows: list[dict[str, Any]], split_payload: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    ids = set(split_payload.get("splits", split_payload).get(split_name, []))
    return [flow for idx, flow in enumerate(flows) if _flow_id(flow, idx) in ids]


def attach_split_to_flows(flows: list[dict[str, Any]], split_payload: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = split_lookup(split_payload)
    out: list[dict[str, Any]] = []
    for idx, flow in enumerate(flows):
        row = dict(flow)
        row["split"] = lookup.get(_flow_id(flow, idx))
        out.append(row)
    return out
