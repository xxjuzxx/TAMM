#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.data.packet_csv import build_flows_from_csvs
from src.utils.io import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats_out", default=None)
    parser.add_argument("--max_flows_per_label", type=int, default=None)
    parser.add_argument("--max_packets_per_flow", type=int, default=512)
    args = parser.parse_args()

    flows, stats = build_flows_from_csvs(
        args.data_root,
        max_flows_per_label=args.max_flows_per_label,
        max_packets_per_flow=args.max_packets_per_flow,
    )
    write_jsonl(flows, args.out)
    if args.stats_out:
        write_json(stats, args.stats_out)
    print(stats)


if __name__ == "__main__":
    main()
