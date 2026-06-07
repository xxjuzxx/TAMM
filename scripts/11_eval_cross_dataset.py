#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401

from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default="experiments/crossnet_all_results.csv")
    parser.add_argument("--out", default="outputs/results/cross_dataset_summary.json")
    args = parser.parse_args()
    path = Path(args.results_csv)
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    summary = {"source": str(path), "num_rows": len(rows), "rows": rows[:50]}
    write_json(summary, args.out)
    print({"out": args.out, "num_rows": len(rows)})


if __name__ == "__main__":
    main()
