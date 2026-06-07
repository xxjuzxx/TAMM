#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data.cicids2017_adapter import align_and_adapt_cicids2017
from src.data.cicids2017_labeler import read_cicids_flow_csvs
from src.data.label_policy import ATTEMPTED_POLICIES
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
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zeek_logs", "--zeek_json", dest="zeek_logs", nargs="+", required=True)
    parser.add_argument("--label_csv", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--unmatched_out", default=None)
    parser.add_argument("--stats_out", default=None)
    parser.add_argument("--label_alignment_report", default=None)
    parser.add_argument("--dataset_report", default=None)
    parser.add_argument("--tolerance_seconds", type=float, default=2.0)
    parser.add_argument("--attempted_policy", choices=ATTEMPTED_POLICIES, default="keep")
    args = parser.parse_args()

    zeek_paths = _expand_inputs(args.zeek_logs, {".log", ".tsv", ".json", ".jsonl", ".ndjson"})
    label_paths = _expand_inputs(args.label_csv, {".csv"})
    flows = aggregate_zeek_logs(zeek_paths)
    label_rows = read_cicids_flow_csvs(label_paths, attempted_policy=args.attempted_policy)
    labeled, unmatched, alignment, report = align_and_adapt_cicids2017(
        flows,
        label_rows,
        tolerance_seconds=args.tolerance_seconds,
    )
    stats = dict(alignment)
    stats["zeek_logs"] = [str(path) for path in zeek_paths]
    stats["label_csvs"] = [str(path) for path in label_paths]
    stats["attempted_policy"] = args.attempted_policy
    write_jsonl(labeled, args.out)
    if args.unmatched_out:
        write_jsonl(unmatched, args.unmatched_out)
    if args.stats_out:
        write_json(stats, args.stats_out)
    if args.label_alignment_report:
        write_json(alignment, args.label_alignment_report)
    if args.dataset_report:
        write_json(report, args.dataset_report)
    print(stats)


if __name__ == "__main__":
    main()
