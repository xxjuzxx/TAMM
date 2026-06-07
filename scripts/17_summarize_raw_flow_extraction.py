#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, write_csv, write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize day-level raw PCAP -> Zeek -> corrected-label flow extraction outputs.")
    parser.add_argument("--input-dir", default="data/interim/flows/cicids2017")
    args = parser.parse_args()
    ensure_dirs()
    rows: list[dict[str, Any]] = []
    for stats_path in sorted((ROOT / args.input_dir).glob("*/raw_cicids2017_*_label_stats.json")):
        day = stats_path.parent.name
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        labeled = stats_path.with_name(f"raw_cicids2017_{day}_labeled_flows.jsonl")
        unmatched = stats_path.with_name(f"raw_cicids2017_{day}_unmatched_flows.jsonl")
        rows.append(
            {
                "day": day,
                "total_zeek_flows": stats.get("total_zeek_flows"),
                "matched_flows": stats.get("matched_flows"),
                "unmatched_flows": stats.get("unmatched_flows"),
                "match_rate": stats.get("match_rate"),
                "ambiguous_matches": stats.get("ambiguous_matches"),
                "ambiguous_rate": stats.get("ambiguous_rate"),
                "matched_attack_families": json.dumps(stats.get("match_by_attack_family", {}), sort_keys=True),
                "labeled_flows_path": str(labeled),
                "labeled_flows_size_bytes": labeled.stat().st_size if labeled.exists() else "",
                "unmatched_flows_path": str(unmatched),
                "unmatched_flows_size_bytes": unmatched.stat().st_size if unmatched.exists() else "",
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
            }
        )
    total_zeek = sum(int(row.get("total_zeek_flows") or 0) for row in rows)
    total_matched = sum(int(row.get("matched_flows") or 0) for row in rows)
    total_unmatched = sum(int(row.get("unmatched_flows") or 0) for row in rows)
    write_csv(ROOT / "data/manifests/raw_flow_extraction_summary.csv", rows)
    write_json(
        ROOT / "data/manifests/raw_flow_extraction_summary.json",
        {
            "command": command_record(sys.argv),
            "total_zeek_flows": total_zeek,
            "matched_flows": total_matched,
            "unmatched_flows": total_unmatched,
            "match_rate": total_matched / max(total_zeek, 1),
            "rows": rows,
        },
    )
    write_md(
        ROOT / "reports/raw_flow_extraction_summary.md",
        [
            "# Raw PCAP-to-Flow Extraction Summary",
            "",
            f"- Total Zeek flows: {total_zeek}",
            f"- Matched labeled flows: {total_matched}",
            f"- Unmatched flows: {total_unmatched}",
            f"- Overall match rate: {total_matched / max(total_zeek, 1):.4f}",
            "- Raw IP, absolute timestamp, and full five-tuple fields were used for parsing/joining/audit only, not emitted as behavior tokens.",
            "",
            "| Day | Zeek flows | Matched | Unmatched | Match rate | Families |",
            "|---|---:|---:|---:|---:|---|",
            *[
                f"| {row['day']} | {row['total_zeek_flows']} | {row['matched_flows']} | {row['unmatched_flows']} | {float(row['match_rate']):.4f} | `{row['matched_attack_families']}` |"
                for row in rows
            ],
        ],
    )
    print(ROOT / "reports/raw_flow_extraction_summary.md")


if __name__ == "__main__":
    main()
