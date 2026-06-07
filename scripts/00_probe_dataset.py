#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data.packet_csv import discover_packet_csvs
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    for path in discover_packet_csvs(args.data_root):
        line_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        rows.append({"label": path.parent.name, "path": str(path), "rows": line_count, "size_bytes": path.stat().st_size})
    result = {"data_root": str(Path(args.data_root)), "files": rows, "total_rows": sum(item["rows"] for item in rows)}
    if args.out:
        write_json(result, args.out)
    print(result)


if __name__ == "__main__":
    main()
