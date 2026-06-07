#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401


def _is_degenerate_p2_zero_byte(row: dict[str, Any]) -> bool:
    pkt = int(row.get("packet_count") or 0)
    byte_count = int(row.get("byte_count") or 0)
    lens = row.get("lens") or []
    if byte_count == 0 and lens:
        byte_count = int(sum(int(x) for x in lens))
    return pkt <= 2 and byte_count <= 0


def _family(row: dict[str, Any]) -> str:
    return str(row.get("attack_family") or row.get("label") or row.get("binary_label") or "UNKNOWN")


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


def filter_flows(args: argparse.Namespace) -> dict[str, Any]:
    src = Path(args.input)
    dst = Path(args.output)
    report_dir = Path(args.report_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    kept_counts: Counter[str] = Counter()
    removed_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    total = 0
    kept = 0
    removed = 0

    with src.open("r", encoding="utf-8") as in_handle, dst.open("w", encoding="utf-8") as out_handle:
        for line_no, line in enumerate(in_handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            family = _family(row)
            total += 1
            total_counts[family] += 1
            if _is_degenerate_p2_zero_byte(row):
                removed += 1
                removed_counts[family] += 1
                continue
            kept += 1
            kept_counts[family] += 1
            out_handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    rows = []
    for family in sorted(set(total_counts) | set(kept_counts) | set(removed_counts)):
        total_n = total_counts[family]
        removed_n = removed_counts[family]
        rows.append(
            {
                "family": family,
                "total": total_n,
                "kept": kept_counts[family],
                "removed": removed_n,
                "removed_rate": float(removed_n / max(total_n, 1)),
            }
        )
    _write_csv(report_dir / "removed_degenerate_p2_zero_byte_counts.csv", rows)
    summary = {
        "input": str(src),
        "output": str(dst),
        "policy": "remove flows with packet_count <= 2 and byte_count <= 0",
        "total_flows": total,
        "kept_flows": kept,
        "removed_flows": removed,
        "removed_rate": float(removed / max(total, 1)),
        "counts_csv": str(report_dir / "removed_degenerate_p2_zero_byte_counts.csv"),
        "behavior_only_policy": True,
        "raw_ip_used_as_filter": False,
        "absolute_time_used_as_filter": False,
        "five_tuple_used_as_filter": False,
        "protocol_service_used_as_filter": False,
    }
    (report_dir / "filter_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# IDS2018 Degenerate-flow Filter",
        "",
        f"Input: `{src}`",
        f"Output: `{dst}`",
        "",
        "Policy: remove flows with `packet_count <= 2` and `byte_count <= 0`.",
        "",
        "This is a behavior-evidence quality diagnostic. It does not use raw IP, absolute timestamp, five-tuple, protocol, or service.",
        "",
        f"Total flows: {total}",
        f"Removed flows: {removed}",
        f"Removed rate: {summary['removed_rate']:.6f}",
        "",
        "| Family | Total | Kept | Removed | Removed rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['total']} | {row['kept']} | {row['removed']} | {row['removed_rate']:.6f} |"
        )
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter IDS2018 degenerate 2-packet zero-byte flows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(filter_flows(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
