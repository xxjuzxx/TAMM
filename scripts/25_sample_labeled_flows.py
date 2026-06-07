#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import random
from collections import Counter, defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.data.label_policy import binary_label_for
from src.utils.io import iter_jsonl, write_json, write_jsonl


def _expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        if any(char in item for char in "*?[]"):
            paths.extend(Path(match) for match in sorted(glob.glob(item)))
        else:
            paths.append(Path(item))
    return paths


def _source_tag(path: Path) -> str:
    stem = path.stem
    for suffix in ("_labeled_flows_smoke", "_labeled_flows_expanded_drop", "_labeled_flows"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.replace(":", "_")


def _prepare_row(row: dict[str, Any], source: Path, row_idx: int) -> dict[str, Any]:
    prepared = dict(row)
    label = str(prepared.get("label", "UNKNOWN"))
    prepared["label"] = label
    prepared["binary_label"] = str(prepared.get("binary_label") or binary_label_for(label))
    prepared.setdefault("dataset_file", str(source))
    prepared.setdefault("source_file", str(source))
    original_flow_id = str(prepared.get("flow_id") or row_idx)
    prepared["source_flow_id"] = original_flow_id
    prepared["flow_id"] = f"{_source_tag(source)}:{original_flow_id}"
    return prepared


def _cap_binary_rows_by_label(rows: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("label", "UNKNOWN"))].append(row)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)

    selected: list[dict[str, Any]] = []
    labels = sorted(by_label)
    while len(selected) < limit and labels:
        next_labels: list[str] = []
        for label in labels:
            label_rows = by_label[label]
            if not label_rows:
                continue
            selected.append(label_rows.pop())
            if len(selected) >= limit:
                break
            if label_rows:
                next_labels.append(label)
        labels = next_labels
    return selected


def load_labeled_flows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row_idx, row in enumerate(iter_jsonl(path)):
            rows.append(_prepare_row(row, path, row_idx))
    return rows


def _parse_label_source_rules(items: list[str]) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for item in items:
        if "::" not in item:
            raise ValueError("label/source filter rules must use LABEL::SOURCE_GLOB")
        label, source_glob = item.split("::", 1)
        rules.append((label, source_glob))
    return rules


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(fnmatch(value, pattern) for pattern in patterns)


def filter_labeled_flows(
    rows: list[dict[str, Any]],
    exclude_label_from_source: list[str] | None = None,
    include_label: list[str] | None = None,
) -> list[dict[str, Any]]:
    rules = _parse_label_source_rules(exclude_label_from_source or [])
    include_patterns = include_label or []
    if not rules and not include_patterns:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label", "UNKNOWN"))
        if include_patterns and not _matches_any(label, include_patterns):
            continue
        source = str(row.get("source_file", row.get("dataset_file", "")))
        if any(label == rule_label and fnmatch(source, source_glob) for rule_label, source_glob in rules):
            continue
        filtered.append(row)
    return filtered


def _semantic_key(row: dict[str, Any]) -> tuple[Any, ...]:
    service_key = row.get("service_key")
    if isinstance(service_key, (list, tuple)):
        service_key_value = tuple(str(item) for item in service_key)
    else:
        service_key_value = str(service_key)
    start_ts = row.get("start_ts")
    try:
        start_value: float | str | None = round(float(start_ts), 6)
    except (TypeError, ValueError):
        start_value = str(start_ts)
    return (
        str(row.get("label", "UNKNOWN")),
        start_value,
        service_key_value,
        int(row.get("packet_count") or len(row.get("lens", []))),
        tuple(row.get("lens", [])),
        tuple(row.get("dirs", [])),
    )


def dedupe_semantic_flows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    dropped = 0
    for row in rows:
        key = _semantic_key(row)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, dropped


def sample_labeled_flows(
    rows: list[dict[str, Any]],
    per_label_limit: int | None = None,
    per_binary_label_limit: int | None = None,
    binary_cap_strategy: str = "label_balanced",
    seed: int = 42,
    sort_by_time: bool = True,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("label", "UNKNOWN"))].append(row)

    selected: list[dict[str, Any]] = []
    for label in sorted(by_label):
        label_rows = list(by_label[label])
        if per_label_limit is not None and len(label_rows) > per_label_limit:
            label_rows = rng.sample(label_rows, per_label_limit)
        selected.extend(label_rows)

    if per_binary_label_limit is not None:
        by_binary: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_binary[str(row.get("binary_label") or binary_label_for(row.get("label", "UNKNOWN")))].append(row)
        selected = []
        for label in sorted(by_binary):
            label_rows = list(by_binary[label])
            if len(label_rows) > per_binary_label_limit:
                if binary_cap_strategy == "random":
                    label_rows = rng.sample(label_rows, per_binary_label_limit)
                elif binary_cap_strategy == "label_balanced":
                    label_rows = _cap_binary_rows_by_label(label_rows, per_binary_label_limit, rng)
                else:
                    raise ValueError(f"unsupported binary cap strategy: {binary_cap_strategy}")
            selected.extend(label_rows)

    if sort_by_time:
        selected.sort(key=lambda row: (float(row.get("start_ts") or 0.0), str(row.get("flow_id", ""))))
    return selected


def summarize(rows: list[dict[str, Any]], selected: list[dict[str, Any]], sources: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    flow_ids = [str(row.get("flow_id", "")) for row in selected]
    selected_by_source = Counter(str(row.get("source_file", row.get("dataset_file", "UNKNOWN"))) for row in selected)
    selected_binary_by_source = Counter(
        (
            str(row.get("source_file", row.get("dataset_file", "UNKNOWN"))),
            str(row.get("binary_label", "UNKNOWN")),
        )
        for row in selected
    )
    return {
        "sources": [str(path) for path in sources],
        "seed": int(args.seed),
        "per_label_limit": args.per_label_limit,
        "per_binary_label_limit": args.per_binary_label_limit,
        "binary_cap_strategy": args.binary_cap_strategy,
        "exclude_label_from_source": args.exclude_label_from_source,
        "include_label": getattr(args, "include_label", []),
        "dedupe_semantic": bool(getattr(args, "dedupe_semantic", False)),
        "rows_before_semantic_dedup": getattr(args, "rows_before_semantic_dedup", len(rows)),
        "semantic_duplicate_rows_dropped": getattr(args, "semantic_duplicate_rows_dropped", 0),
        "sort_by_time": bool(args.sort_by_time),
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "input_label_counts": dict(sorted(Counter(str(row.get("label", "UNKNOWN")) for row in rows).items())),
        "selected_label_counts": dict(sorted(Counter(str(row.get("label", "UNKNOWN")) for row in selected).items())),
        "selected_binary_counts": dict(sorted(Counter(str(row.get("binary_label", "UNKNOWN")) for row in selected).items())),
        "selected_source_counts": dict(sorted(selected_by_source.items())),
        "selected_binary_counts_by_source": {
            f"{source}|{label}": count for (source, label), count in sorted(selected_binary_by_source.items())
        },
        "duplicate_flow_ids": len(flow_ids) - len(set(flow_ids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats_out", default=None)
    parser.add_argument("--per_label_limit", type=int, default=None)
    parser.add_argument("--per_binary_label_limit", type=int, default=None)
    parser.add_argument("--binary_cap_strategy", choices=["label_balanced", "random"], default="label_balanced")
    parser.add_argument(
        "--exclude_label_from_source",
        action="append",
        default=[],
        help="Exclude rows matching LABEL::SOURCE_GLOB, e.g. 'BENIGN::*ddos*'. Can be repeated.",
    )
    parser.add_argument(
        "--include_label",
        action="append",
        default=[],
        help="Only keep labels matching this glob. Can be repeated.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sort_by_time", action="store_true")
    parser.add_argument(
        "--dedupe_semantic",
        action="store_true",
        help="Drop duplicate flows with the same label, timestamp, service key, packet count, lengths, and directions before sampling.",
    )
    args = parser.parse_args()

    sources = _expand_inputs(args.flows)
    rows = filter_labeled_flows(
        load_labeled_flows(sources),
        exclude_label_from_source=args.exclude_label_from_source,
        include_label=args.include_label,
    )
    args.rows_before_semantic_dedup = len(rows)
    args.semantic_duplicate_rows_dropped = 0
    if args.dedupe_semantic:
        rows, args.semantic_duplicate_rows_dropped = dedupe_semantic_flows(rows)
    selected = sample_labeled_flows(
        rows,
        per_label_limit=args.per_label_limit,
        per_binary_label_limit=args.per_binary_label_limit,
        binary_cap_strategy=args.binary_cap_strategy,
        seed=args.seed,
        sort_by_time=args.sort_by_time,
    )
    write_jsonl(selected, args.out)
    stats = summarize(rows, selected, sources, args)
    if args.stats_out:
        write_json(stats, args.stats_out)
    print(stats)


if __name__ == "__main__":
    main()
