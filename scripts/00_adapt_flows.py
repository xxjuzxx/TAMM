#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.data.dataset_adapter import dataset_report, normalize_flows
from src.utils.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="CICIDS2017")
    parser.add_argument("--dataset_report", required=True)
    args = parser.parse_args()

    flows = read_jsonl(args.flows)
    normalized = normalize_flows(flows, dataset=args.dataset)
    write_jsonl(normalized, args.out)
    write_json(dataset_report(normalized, dataset=args.dataset, source=args.flows), args.dataset_report)
    print({"out": args.out, "dataset_report": args.dataset_report, "num_flows": len(normalized)})


if __name__ == "__main__":
    main()
