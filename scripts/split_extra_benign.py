#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from extra_benign_common import EXTRA_ARTIFACT_DIR, EXTRA_RESULT_DIR, EXTRA_SPLIT_DIR, read_csv, write_csv


def _pkt_bucket(value: str) -> str:
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return "pkt_unknown"
    if n <= 3:
        return "pkt_1_3"
    if n <= 10:
        return "pkt_4_10"
    if n <= 50:
        return "pkt_11_50"
    return "pkt_gt50"


def _stable_sort(rows: list[dict[str, str]], mode: str, seed: int) -> list[dict[str, str]]:
    if mode == "temporal" and any(str(row.get("timestamp_start") or "") for row in rows):
        return sorted(rows, key=lambda row: (float(row.get("timestamp_start") or 0.0), row.get("flow_id", "")))
    if mode == "stratified":
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            key = f"{row.get('gate_bucket','unknown')}:{_pkt_bucket(row.get('packet_count',''))}"
            grouped[key].append(row)
        out: list[dict[str, str]] = []
        for key in sorted(grouped):
            items = sorted(grouped[key], key=lambda row: row.get("flow_id", ""))
            out.extend(items)
        return out
    rng = np.random.default_rng(seed)
    items = list(rows)
    rng.shuffle(items)
    return items


def _cut(rows: list[dict[str, str]], memory_ratio: float, calibration_ratio: float, tail_ratio: float) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    n = len(rows)
    mem_n = int(round(n * memory_ratio))
    cal_n = int(round(n * calibration_ratio))
    tail_n = int(round(n * tail_ratio))
    if mem_n + cal_n + tail_n > n:
        tail_n = max(0, n - mem_n - cal_n)
    return rows[:mem_n], rows[mem_n : mem_n + cal_n], rows[mem_n + cal_n : mem_n + cal_n + tail_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split gated extra benign data into memory/calibration/tail/quarantine pools.")
    parser.add_argument("--gate-scores", default=str(EXTRA_RESULT_DIR / "extra_benign_gate_scores.csv"))
    parser.add_argument("--extra-benign-metadata", default=str(EXTRA_ARTIFACT_DIR / "extra_benign_metadata.csv"))
    parser.add_argument("--split-mode", choices=["temporal", "stratified", "random"], default="temporal")
    parser.add_argument("--memory-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--tail-test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-dir", default=str(EXTRA_SPLIT_DIR))
    parser.add_argument("--results-dir", default=str(EXTRA_RESULT_DIR))
    args = parser.parse_args()

    gate_rows = read_csv(args.gate_scores)
    meta = {row["flow_id"]: row for row in read_csv(args.extra_benign_metadata)}
    enriched = []
    quarantine = []
    for row in gate_rows:
        merged = dict(meta.get(row["flow_id"], {}))
        merged.update(row)
        if row.get("admission_status") == "quarantine":
            quarantine.append(merged)
        else:
            enriched.append(merged)
    memory_source = [row for row in enriched if row.get("admission_status") == "memory_candidate"]
    tail_source = [row for row in enriched if row.get("admission_status") != "memory_candidate"]
    ordered = _stable_sort(memory_source, args.split_mode, args.seed)
    memory, calibration, tail_from_memory = _cut(ordered, args.memory_ratio, args.calibration_ratio, args.tail_test_ratio)
    tail = tail_from_memory + _stable_sort(tail_source, args.split_mode, args.seed)

    out_dir = Path(args.output_dir)
    res_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    write_csv(memory, out_dir / "extra_benign_memory.csv")
    write_csv(calibration, out_dir / "extra_benign_calibration.csv")
    write_csv(tail, out_dir / "extra_benign_tail_test.csv")
    write_csv(quarantine, out_dir / "extra_benign_quarantine.csv")
    summary = [
        {"split": "memory", "count": len(memory), "quarantine_included": 0},
        {"split": "calibration", "count": len(calibration), "quarantine_included": 0},
        {"split": "tail_test", "count": len(tail), "quarantine_included": 0},
        {"split": "quarantine", "count": len(quarantine), "quarantine_included": len(quarantine)},
    ]
    write_csv(summary, res_dir / "extra_benign_split_summary.csv")
    print(json.dumps({"memory": len(memory), "calibration": len(calibration), "tail_test": len(tail), "quarantine": len(quarantine)}, indent=2))


if __name__ == "__main__":
    main()
