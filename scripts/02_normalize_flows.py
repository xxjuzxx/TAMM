#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import pandas as pd

from src.pipeline.common import ROOT, command_record, ensure_dirs, read_csv, write_csv, write_json, write_md


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_record(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    lens = [int(x) for x in row.get("lens", [])][:512]
    dirs = [int(bool(x)) for x in row.get("dirs", [])][:512]
    iats = [float(x) for x in row.get("iats", [])][:512]
    return {
        "flow_id": str(row.get("flow_id")),
        "dataset": str(row.get("dataset") or "CICIDS2017"),
        "label": str(row.get("label") or row.get("attack_family") or ""),
        "attack_family": str(row.get("attack_family") or row.get("label") or ""),
        "binary_label": str(row.get("binary_label") or ("BENIGN" if str(row.get("label", "")).upper() == "BENIGN" else "ATTACK")),
        "protocol": str(row.get("protocol") or row.get("proto") or "").upper(),
        "service": str(row.get("service") or row.get("appinfo") or ""),
        "start_ts": row.get("start_ts"),
        "duration": float(row.get("duration") or 0.0),
        "packet_count": int(row.get("packet_count") or len(lens)),
        "byte_count": int(sum(lens)),
        "lens": lens,
        "dirs": dirs,
        "iats": iats,
        "source_path": source,
        "raw_ip_used_as_token": False,
        "absolute_time_used_as_token": False,
        "five_tuple_used_as_token": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize FlowPrim flow JSONL records into canonical parquet files.")
    parser.add_argument("--input", default="outputs/processed/ccfa/cicids2017_interim_labeled_flows.jsonl")
    parser.add_argument("--output", default="data/interim/normalized_flows/cicids2017/cicids2017_interim_normalized.parquet")
    parser.add_argument("--summary", default="data/manifests/normalizer_summary.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    input_path = ROOT / args.input
    records = []
    for idx, row in enumerate(_iter_jsonl(input_path)):
        if args.max_rows is not None and idx >= args.max_rows:
            break
        records.append(_normalize_record(row, source=str(input_path)))
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out, index=False)
    summary_row = {
        "input": str(input_path),
        "output": str(out),
        "rows": len(records),
        "avg_packet_count": float(sum(r["packet_count"] for r in records) / max(len(records), 1)),
        "avg_duration": float(sum(r["duration"] for r in records) / max(len(records), 1)),
        "length_bin_rule": "existing FlowPrim log/count bins in configs/cicids2017.yaml",
        "iat_bin_rule": "existing FlowPrim iat bins in configs/cicids2017.yaml",
        "command": command_record(sys.argv)["command"],
    }
    write_csv(ROOT / args.summary, [summary_row])
    write_json(ROOT / "data/manifests/normalizer_manifest.json", {"command": command_record(sys.argv), "summary": summary_row})
    write_md(ROOT / "reports/normalizer_summary.md", ["# Normalizer Summary", "", f"- Rows normalized: {len(records)}", f"- Output: `{out}`"])
    print(out)


if __name__ == "__main__":
    main()

