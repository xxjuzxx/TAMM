#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from fnmatch import fnmatch
from typing import Any

import _bootstrap  # noqa: F401

from src.data.label_policy import binary_label_for, merged_cicids_label
from src.utils.io import iter_jsonl, write_json, write_jsonl


def _matches(value: object, patterns: list[str]) -> bool:
    if not patterns:
        return True
    text = str(value or "")
    return any(fnmatch(text.lower(), pattern.lower()) for pattern in patterns)


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    label_globs: list[str] | None = None,
    raw_label_globs: list[str] | None = None,
    merged_label_globs: list[str] | None = None,
    binary_label_globs: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        label = row.get("label")
        raw_label = row.get("raw_label", label)
        binary_label = row.get("binary_label") or binary_label_for(label)
        merged_label = merged_cicids_label(label)
        if not _matches(label, label_globs or []):
            continue
        if not _matches(raw_label, raw_label_globs or []):
            continue
        if not _matches(merged_label, merged_label_globs or []):
            continue
        if not _matches(binary_label, binary_label_globs or []):
            continue
        selected.append(row)
    return selected


def summarize(rows: list[dict[str, Any]], selected: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input": args.input,
        "output": args.out,
        "label_glob": args.label_glob,
        "raw_label_glob": args.raw_label_glob,
        "merged_label_glob": args.merged_label_glob,
        "binary_label_glob": args.binary_label_glob,
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "input_label_counts": dict(sorted(Counter(str(row.get("label", "UNKNOWN")) for row in rows).items())),
        "selected_label_counts": dict(sorted(Counter(str(row.get("label", "UNKNOWN")) for row in selected).items())),
        "selected_raw_label_counts": dict(
            sorted(Counter(str(row.get("raw_label", row.get("label", "UNKNOWN"))) for row in selected).items())
        ),
        "selected_merged_label_counts": dict(
            sorted(Counter(merged_cicids_label(row.get("label", "UNKNOWN")) for row in selected).items())
        ),
        "selected_binary_label_counts": dict(
            sorted(Counter(str(row.get("binary_label") or binary_label_for(row.get("label", "UNKNOWN"))) for row in selected).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter labeled flow JSONL rows by label/raw_label/merged/binary globs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats_out", default=None)
    parser.add_argument("--label_glob", action="append", default=[])
    parser.add_argument("--raw_label_glob", action="append", default=[])
    parser.add_argument("--merged_label_glob", action="append", default=[])
    parser.add_argument("--binary_label_glob", action="append", default=[])
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input))
    selected = filter_rows(
        rows,
        label_globs=args.label_glob,
        raw_label_globs=args.raw_label_glob,
        merged_label_globs=args.merged_label_glob,
        binary_label_globs=args.binary_label_glob,
    )
    write_jsonl(selected, args.out)
    stats = summarize(rows, selected, args)
    if args.stats_out:
        write_json(stats, args.stats_out)
    print(stats)


if __name__ == "__main__":
    main()
