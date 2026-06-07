#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from extra_benign_common import EXTRA_ARTIFACT_DIR, EXTRA_REPORT_DIR, EXTRA_RESULT_DIR, EXTRA_SPLIT_DIR, read_csv, read_json, write_csv
from extra_benign_tables import write_extra_benign_data_summary, write_throughput_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build extra benign data summary and optional artifact-path throughput table.")
    parser.add_argument("--artifact-dir", default=str(EXTRA_ARTIFACT_DIR))
    parser.add_argument("--results-dir", default=str(EXTRA_RESULT_DIR))
    parser.add_argument("--split-dir", default=str(EXTRA_SPLIT_DIR))
    parser.add_argument("--include-throughput", action="store_true")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    results_dir = Path(args.results_dir)
    split_dir = Path(args.split_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    EXTRA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prepare = read_json(artifact_dir / "extra_benign_prepare_summary.json")
    meta = read_csv(artifact_dir / "extra_benign_metadata.csv")
    gate = read_csv(results_dir / "extra_benign_gate_scores.csv")
    split_summary = {row["split"]: int(float(row["count"])) for row in read_csv(results_dir / "extra_benign_split_summary.csv")}
    pass_count = sum(1 for row in gate if row.get("admission_status") != "quarantine")
    quarantine = sum(1 for row in gate if row.get("admission_status") == "quarantine")
    time_values = [float(row["timestamp_start"]) for row in meta if row.get("timestamp_start")]
    time_span = f"{min(time_values):.0f}-{max(time_values):.0f}" if time_values else "not available"
    summary_rows = [
        {
            "Source": "CIC-IDS2017 extra benign PCAP slices",
            "Input type": "pcap/zeek artifact",
            "Capability level": "full_behavior",
            "Raw flows": prepare.get("raw_flows_seen", 0),
            "Dedup flows": prepare.get("dedup_benign_flows", 0),
            "Admission pass": pass_count,
            "Quarantine": quarantine,
            "Memory candidates": split_summary.get("memory", 0),
            "Calibration candidates": split_summary.get("calibration", 0),
            "Tail-test candidates": split_summary.get("tail_test", 0),
            "Time span": time_span,
            "Use in paper": "memory/calibration only",
        }
    ]
    write_csv(summary_rows, results_dir / "extra_benign_data_summary.csv")
    write_csv(summary_rows, EXTRA_REPORT_DIR / "extra_benign_data_summary.csv")
    write_extra_benign_data_summary(results_dir / "extra_benign_data_summary.csv")

    if args.include_throughput:
        stages = []
        pcap_zeek_path = results_dir / "extra_benign_pcap_zeek_throughput.csv"
        if pcap_zeek_path.exists():
            pcap_rows = read_csv(pcap_zeek_path)
            write_csv(pcap_rows, EXTRA_REPORT_DIR / "extra_benign_pcap_zeek_throughput.csv")
            input_count = sum(int(float(row.get("input_count") or 0)) for row in pcap_rows)
            output_count = sum(int(float(row.get("output_count") or 0)) for row in pcap_rows)
            wall_time = sum(float(row.get("wall_time_seconds") or 0.0) for row in pcap_rows)
            file_size = sum(int(float(row.get("file_size_bytes") or 0)) for row in pcap_rows)
            stages.append(
                {
                    "stage": "PCAP -> Zeek logs",
                    "input_count": input_count,
                    "output_count": output_count,
                    "wall_time_seconds": wall_time,
                    "throughput_per_second": input_count / max(wall_time, 1e-9),
                    "notes": f"Measured Zeek parse on {len(pcap_rows)} slices; {file_size} input bytes.",
                }
            )
        t0 = time.perf_counter()
        meta_rows = read_csv(artifact_dir / "extra_benign_metadata.csv")
        t_meta = time.perf_counter() - t0
        stages.append({"stage": "Metadata CSV load", "input_count": len(meta_rows), "output_count": len(meta_rows), "wall_time_seconds": t_meta, "throughput_per_second": len(meta_rows) / max(t_meta, 1e-9), "notes": "Loads prepared metadata after measured Zeek parsing."})
        t0 = time.perf_counter()
        token_lines = sum(1 for _ in (artifact_dir / "extra_benign_tokens.jsonl").open("r", encoding="utf-8"))
        t_tok = time.perf_counter() - t0
        stages.append({"stage": "Token JSONL scan", "input_count": token_lines, "output_count": token_lines, "wall_time_seconds": t_tok, "throughput_per_second": token_lines / max(t_tok, 1e-9), "notes": "Scans generated behavior-token JSONL."})
        for name in ["extra_benign_memory.csv", "extra_benign_calibration.csv", "extra_benign_tail_test.csv"]:
            t0 = time.perf_counter()
            rows = read_csv(split_dir / name)
            elapsed = time.perf_counter() - t0
            stages.append({"stage": f"Split load {name}", "input_count": len(rows), "output_count": len(rows), "wall_time_seconds": elapsed, "throughput_per_second": len(rows) / max(elapsed, 1e-9), "notes": "Loads gated split after measured Zeek parsing."})
        write_csv(stages, results_dir / "extra_benign_e2e_throughput.csv")
        write_csv(stages, EXTRA_REPORT_DIR / "extra_benign_e2e_throughput.csv")
        write_throughput_table(results_dir / "extra_benign_e2e_throughput.csv")
    print(json.dumps({"summary_rows": len(summary_rows), "throughput": bool(args.include_throughput)}, indent=2))


if __name__ == "__main__":
    main()
