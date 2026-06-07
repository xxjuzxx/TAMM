#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data.ids2018_schedule_labeler import label_ids2018_flows_by_schedule, load_attack_windows
from src.data.zeek_parser import aggregate_zeek_logs
from src.utils.io import write_json, write_jsonl


def _expand_inputs(items: list[str], suffixes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if any(char in item for char in "*?[]"):
            paths.extend(Path(match) for match in sorted(glob.glob(item)))
        elif path.is_dir():
            paths.extend(sorted(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in suffixes))
        else:
            paths.append(path)
    return [path for path in paths if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Label IDS2018 PCAP-derived Zeek flows using official schedule/IP windows.")
    parser.add_argument("--zeek_logs", nargs="+", required=True)
    parser.add_argument("--schedule", default="configs/ids2018_attack_schedule.yaml")
    parser.add_argument("--day", default=None, help="Optional IDS2018 day directory, e.g. Friday-02-03-2018.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quarantine_out", default=None)
    parser.add_argument("--stats_out", default=None)
    parser.add_argument("--label_alignment_report", default=None)
    parser.add_argument("--dataset_report", default=None)
    parser.add_argument("--tolerance_seconds", type=float, default=0.0)
    parser.add_argument("--ambiguous_policy", choices=["quarantine", "first", "drop", "benign"], default="quarantine")
    args = parser.parse_args()

    zeek_paths = _expand_inputs(args.zeek_logs, {".log", ".tsv", ".json", ".jsonl", ".ndjson"})
    if not zeek_paths:
        raise FileNotFoundError(f"No Zeek logs matched: {args.zeek_logs}")
    windows, manifest = load_attack_windows(args.schedule)
    flows = aggregate_zeek_logs(zeek_paths)
    if args.day:
        for flow in flows:
            flow["day"] = args.day
            flow["ids2018_day"] = args.day
    labeled, quarantine, alignment, report = label_ids2018_flows_by_schedule(
        flows,
        windows,
        day=args.day,
        tolerance_seconds=args.tolerance_seconds,
        ambiguous_policy=args.ambiguous_policy,
    )
    alignment["zeek_logs"] = [str(path) for path in zeek_paths]
    alignment["schedule"] = str(args.schedule)
    alignment["schedule_source_url"] = manifest.get("source_url")
    alignment["schedule_to_zeek_offset_seconds"] = manifest.get("schedule_to_zeek_offset_seconds")
    write_jsonl(labeled, args.out)
    if args.quarantine_out:
        write_jsonl(quarantine, args.quarantine_out)
    if args.stats_out:
        write_json(alignment, args.stats_out)
    if args.label_alignment_report:
        write_json(alignment, args.label_alignment_report)
    if args.dataset_report:
        write_json(report, args.dataset_report)
    print(alignment)


if __name__ == "__main__":
    main()
