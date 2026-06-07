#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from extra_benign_common import EXTRA_REPORT_DIR, EXTRA_RESULT_DIR, ROOT, write_csv


DEFAULT_PCAPS = [
    ROOT / "outputs" / "pcap_smoke_inputs" / "Benign_0500001_0505000.pcap",
    ROOT / "outputs" / "pcap_smoke_inputs" / "Benign_1000001_1005000.pcap",
    ROOT / "outputs" / "pcap_smoke_inputs" / "Benign_1500001_1505000.pcap",
    ROOT / "outputs" / "pcap_smoke_inputs" / "Benign_2000001_2005000.pcap",
    ROOT / "outputs" / "pcap_smoke_inputs" / "Benign_2500001_2505000.pcap",
]


def _capinfos(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["capinfos", "-Tm", str(path)],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    if not rows:
        raise RuntimeError(f"capinfos returned no rows for {path}")
    row = rows[0]
    return {
        "packet_count": int(float(row.get("Number of packets") or 0)),
        "file_size_bytes": int(float(row.get("File size (bytes)") or path.stat().st_size)),
        "data_size_bytes": int(float(row.get("Data size (bytes)") or 0)),
        "capture_duration_seconds": float(row.get("Capture duration (seconds)") or 0.0),
    }


def _count_zeek_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def _run_zeek(pcap: Path, out_dir: Path, *, zeek_script: str, ignore_checksums: bool, overwrite: bool) -> dict[str, Any]:
    run_dir = out_dir / pcap.name.removesuffix(".pcap")
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        "scripts/00_run_zeek.sh",
        "--input",
        str(pcap),
        "--out_dir",
        str(out_dir),
        "--zeek_script",
        zeek_script,
    ]
    if ignore_checksums:
        cmd.append("--ignore_checksums")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Zeek failed for {pcap}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    stats = _capinfos(pcap)
    conn_rows = _count_zeek_rows(run_dir / "conn.log")
    feature_rows = _count_zeek_rows(run_dir / "Features.log")
    packet_count = int(stats["packet_count"])
    file_size = int(stats["file_size_bytes"])
    return {
        "stage": "PCAP -> Zeek logs",
        "pcap": str(pcap.relative_to(ROOT)),
        "input_count": packet_count,
        "output_count": conn_rows,
        "feature_rows": feature_rows,
        "wall_time_seconds": wall,
        "throughput_per_second": packet_count / max(wall, 1e-9),
        "file_size_bytes": file_size,
        "bytes_per_second": file_size / max(wall, 1e-9),
        "capture_duration_seconds": stats["capture_duration_seconds"],
        "zeek_out_dir": str(run_dir.relative_to(ROOT)),
        "ignore_checksums": bool(ignore_checksums),
        "notes": "Measured by rerunning Zeek on the extra-benign PCAP slice.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure PCAP-to-Zeek throughput for extra benign PCAP slices.")
    parser.add_argument("--pcaps", nargs="+", default=[str(path) for path in DEFAULT_PCAPS])
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "extra_benign_zeek_throughput_allfeas"))
    parser.add_argument("--results-dir", default=str(EXTRA_RESULT_DIR))
    parser.add_argument("--zeek-script", default="../third_party/BSTS-Net/AllFeas.zeek")
    parser.add_argument("--ignore-checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.pcaps:
        pcap = Path(item)
        if not pcap.is_absolute():
            pcap = ROOT / pcap
        if not pcap.exists():
            raise FileNotFoundError(pcap)
        rows.append(
            _run_zeek(
                pcap,
                out_dir,
                zeek_script=args.zeek_script,
                ignore_checksums=bool(args.ignore_checksums),
                overwrite=bool(args.overwrite),
            )
        )
    write_csv(rows, result_dir / "extra_benign_pcap_zeek_throughput.csv")
    EXTRA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, EXTRA_REPORT_DIR / "extra_benign_pcap_zeek_throughput.csv")
    print(json.dumps({"pcaps": len(rows), "result": str(result_dir / "extra_benign_pcap_zeek_throughput.csv")}, indent=2))


if __name__ == "__main__":
    main()
