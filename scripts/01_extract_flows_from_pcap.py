#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, read_csv, run_command, write_csv, write_json, write_md


def _day_slug(day: str) -> str:
    return str(day).lower().replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract canonical labeled flows from raw PCAP via the existing Zeek-first FlowPrim pipeline.")
    parser.add_argument("--manifest", default="data/manifests/pcap_manifest.csv")
    parser.add_argument("--output-dir", default="data/interim/flows/cicids2017")
    parser.add_argument("--log-dir", default="logs/extraction")
    parser.add_argument("--mode", choices=["smoke", "full", "dry_run"], default="dry_run")
    parser.add_argument("--stage", choices=["flows_only", "full_token_pipeline"], default="flows_only")
    parser.add_argument("--limit-days", nargs="*", default=None)
    parser.add_argument("--skip-zeek", action="store_true", help="Reuse existing Zeek output if --zeek-out-dir is supplied per direct command use.")
    parser.add_argument("--ignore-checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip a day when expected labeled-flow and stats outputs already exist.")
    args = parser.parse_args()

    ensure_dirs()
    rows = read_csv(ROOT / args.manifest)
    if args.limit_days:
        wanted = {_day_slug(day) for day in args.limit_days}
        rows = [row for row in rows if _day_slug(row.get("day", "")) in wanted]

    output_dir = ROOT / args.output_dir
    log_dir = ROOT / args.log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        day = _day_slug(row["day"])
        pcap = row["pcap_path"]
        csv_path = row["corrected_csv_path"]
        prefix = f"raw_cicids2017_{day}"
        out_flow = output_dir / day / f"{prefix}_labeled_flows.jsonl"
        out_stats = output_dir / day / f"{prefix}_label_stats.json"
        zeek_out = output_dir / day / f"{prefix}_zeek"
        if args.stage == "flows_only":
            zeek_cmd = [
                "scripts/00_run_zeek.sh",
                "--input",
                pcap,
                "--out_dir",
                str(zeek_out),
            ]
            if args.ignore_checksums:
                zeek_cmd.append("--ignore_checksums")
            label_cmd = [
                "python",
                "scripts/01_label_flows.py",
                "--zeek_logs",
                str(zeek_out),
                "--label_csv",
                csv_path,
                "--attempted_policy",
                "drop",
                "--tolerance_seconds",
                "2.0",
                "--out",
                str(out_flow),
                "--unmatched_out",
                str(output_dir / day / f"{prefix}_unmatched_flows.jsonl"),
                "--stats_out",
                str(out_stats),
                "--label_alignment_report",
                str(output_dir / day / f"{prefix}_label_alignment_report.json"),
            ]
            cmd = zeek_cmd if not args.skip_zeek else []
            full_cmd = [zeek_cmd, label_cmd] if not args.skip_zeek else [label_cmd]
        else:
            cmd = [
                "python",
                "scripts/26_run_zeek_pipeline.py",
                "--input",
                pcap,
                "--label_csv",
                csv_path,
                "--prefix",
                prefix,
                "--processed_dir",
                str(output_dir / day),
                "--token_dir",
                str(ROOT / "data" / "interim" / "tokens" / "cicids2017" / day),
                "--profile_mode",
                "full",
                "--max_len",
                "512",
            ]
            if args.ignore_checksums:
                cmd.append("--ignore_checksums")
            if args.skip_zeek:
                cmd.append("--skip_zeek")
            if args.mode == "dry_run":
                cmd.append("--dry_run")
            full_cmd = [cmd]
        status = "planned"
        reason = ""
        try:
            if args.resume and out_flow.exists() and out_stats.exists():
                status = "existing"
                reason = "expected labeled-flow and stats outputs already exist"
            elif args.resume and args.stage == "flows_only" and zeek_out.exists() and any(zeek_out.rglob("conn.log")):
                label_only_cmd = [full_cmd[-1]]
                for step_idx, step_cmd in enumerate(label_only_cmd):
                    run_command(step_cmd, log_path=log_dir / f"01_extract_{day}_resume_step{step_idx + 1}.log")
                status = "ok" if out_flow.exists() else "completed_missing_expected_output"
            elif args.mode == "full":
                for step_idx, step_cmd in enumerate(full_cmd):
                    run_command(step_cmd, log_path=log_dir / f"01_extract_{day}_step{step_idx + 1}.log")
                status = "ok" if out_flow.exists() else "completed_missing_expected_output"
            elif args.mode == "smoke":
                status = "skipped_full_pcap"
                reason = "smoke mode records command only; full PCAP Zeek parsing is heavyweight"
                write_md(log_dir / f"01_extract_{day}_smoke.log", ["# Planned commands", "", *[" ".join(step_cmd) for step_cmd in full_cmd]])
            else:
                status = "dry_run"
                write_md(log_dir / f"01_extract_{day}_dry_run.log", ["# Planned commands", "", *[" ".join(step_cmd) for step_cmd in full_cmd]])
        except Exception as exc:
            status = "failed"
            reason = str(exc)
        manifest_rows.append(
            {
                "dataset": "CICIDS2017",
                "day": day,
                "pcap_path": pcap,
                "corrected_csv_path": csv_path,
                "output_labeled_flows": str(out_flow),
                "output_label_stats": str(out_stats),
                "mode": args.mode,
                "stage": args.stage,
                "status": status,
                "skipped_reason": reason,
                "command": " && ".join(" ".join(step_cmd) for step_cmd in full_cmd),
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
            }
        )

    write_csv(ROOT / "data/manifests/flow_extraction_manifest.csv", manifest_rows)
    write_json(ROOT / "data/manifests/flow_extraction_summary.json", {"command": command_record(sys.argv), "rows": manifest_rows})
    write_md(
        ROOT / "reports/flow_extraction_report.md",
        [
            "# Flow Extraction Report",
            "",
            f"Mode: `{args.mode}`",
            "",
            "The script uses the existing Zeek-first FlowPrim pipeline. In `smoke` or `dry_run` mode, full PCAP parsing is not claimed as completed.",
            "",
            *[f"- {row['day']}: {row['status']} {row['skipped_reason']}".rstrip() for row in manifest_rows],
        ],
    )
    print(ROOT / "data/manifests/flow_extraction_manifest.csv")


if __name__ == "__main__":
    main()
