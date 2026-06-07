#!/usr/bin/env python3
"""Organize CSE-CIC-IDS2018 into copied and extracted artifacts.

The script intentionally keeps raw archives and extracted files separate:

  processed_csv/                 copied CICFlowMeter CSV files
  source_archive_manifest.csv     source pcap/log archive references
  extracted_pcaps/<day>/          extracted packet captures
  extracted_logs/<day>/           extracted Windows event logs
  manifests/                      reproducibility manifests

It can resume safely: existing regular CSV files with matching sizes are reused,
and extraction success markers prevent repeated full extraction. By default the
large PCAP/log archives are not copied; they are referenced in a manifest and
extracted directly into the organized directory. Use --archive-mode copy only if
you explicitly want duplicate raw archive files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterable
from zipfile import ZipFile


DATASET = "CSE-CIC-IDS2018"


def sha256_prefix(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    """Return a bounded checksum for audit without scanning huge files fully."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], log_path: Path | None = None) -> None:
    """Run a command and optionally append stdout/stderr to a log file."""
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def copy_file(src: Path, dst: Path) -> dict[str, object]:
    """Copy src to dst unless an existing regular file already matches size."""
    ensure_dir(dst.parent)
    src_size = src.stat().st_size
    if dst.exists() and not dst.is_symlink() and dst.stat().st_size == src_size:
        return {
            "source": str(src),
            "destination": str(dst),
            "size_bytes": src_size,
            "status": "already_copied",
            "sha256_prefix": sha256_prefix(dst),
        }

    tmp = dst.with_name(dst.name + ".copying")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return {
        "source": str(src),
        "destination": str(dst),
        "size_bytes": src_size,
        "status": "copied",
        "sha256_prefix": sha256_prefix(dst),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_source_files(source_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    processed_root = source_root / "Processed Traffic Data for ML Algorithms"
    raw_root = source_root / "Original Network Traffic and Log data"
    csv_files = sorted(processed_root.glob("*TrafficForML_CICFlowMeter.csv"))
    pcap_archives = sorted(raw_root.glob("*/pcap.zip")) + sorted(raw_root.glob("*/pcap.rar"))
    log_archives = sorted(raw_root.glob("*/logs.zip"))
    return csv_files, sorted(pcap_archives), log_archives


def day_from_archive(path: Path) -> str:
    return path.parent.name


def zip_entry_summary(path: Path) -> tuple[int, int, int]:
    with ZipFile(path) as zf:
        infos = zf.infolist()
    entry_count = len(infos)
    pcap_like = sum(1 for x in infos if x.filename.lower().endswith((".pcap", ".pcapng", ".cap")))
    total_uncompressed = sum(x.file_size for x in infos)
    return entry_count, pcap_like, total_uncompressed


def rar_entry_summary(path: Path, unrar_bin: Path) -> tuple[int, int, int | None]:
    proc = subprocess.run(
        [str(unrar_bin), "lb", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    entries = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    pcap_like = sum(1 for x in entries if x.lower().endswith((".pcap", ".pcapng", ".cap")))
    return len(entries), pcap_like, None


def extract_zip(path: Path, out_dir: Path, marker: Path, log_path: Path) -> dict[str, object]:
    ensure_dir(out_dir)
    entry_count, pcap_like, total_uncompressed = zip_entry_summary(path)
    if marker.exists():
        return {
            "archive": str(path),
            "extract_dir": str(out_dir),
            "entry_count": entry_count,
            "pcap_like_entry_count": pcap_like,
            "uncompressed_bytes": total_uncompressed,
            "status": "already_extracted",
            "reason": "",
        }
    started = time.time()
    run(["unzip", "-oq", str(path), "-d", str(out_dir)], log_path=log_path)
    marker.write_text(
        json.dumps(
            {
                "archive": str(path),
                "entry_count": entry_count,
                "pcap_like_entry_count": pcap_like,
                "uncompressed_bytes": total_uncompressed,
                "elapsed_sec": time.time() - started,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "archive": str(path),
        "extract_dir": str(out_dir),
        "entry_count": entry_count,
        "pcap_like_entry_count": pcap_like,
        "uncompressed_bytes": total_uncompressed,
        "status": "extracted",
        "reason": "",
    }


def extract_rar(
    path: Path, out_dir: Path, marker: Path, log_path: Path, unrar_bin: Path
) -> dict[str, object]:
    ensure_dir(out_dir)
    entry_count, pcap_like, total_uncompressed = rar_entry_summary(path, unrar_bin)
    if marker.exists():
        return {
            "archive": str(path),
            "extract_dir": str(out_dir),
            "entry_count": entry_count,
            "pcap_like_entry_count": pcap_like,
            "uncompressed_bytes": total_uncompressed or "",
            "status": "already_extracted",
            "reason": "",
        }
    started = time.time()
    run([str(unrar_bin), "x", "-o+", str(path), str(out_dir) + "/"], log_path=log_path)
    marker.write_text(
        json.dumps(
            {
                "archive": str(path),
                "entry_count": entry_count,
                "pcap_like_entry_count": pcap_like,
                "uncompressed_bytes": total_uncompressed,
                "elapsed_sec": time.time() - started,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "archive": str(path),
        "extract_dir": str(out_dir),
        "entry_count": entry_count,
        "pcap_like_entry_count": pcap_like,
        "uncompressed_bytes": total_uncompressed or "",
        "status": "extracted",
        "reason": "",
    }


def count_files(
    root: Path, suffixes: Iterable[str] | None = None, *, exclude_names: set[str] | None = None
) -> tuple[int, int]:
    suffix_tuple = tuple(s.lower() for s in suffixes) if suffixes else None
    exclude_names = exclude_names or set()
    count = 0
    size = 0
    if not root.exists():
        return 0, 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in exclude_names:
            continue
        if suffix_tuple and not path.name.lower().endswith(suffix_tuple):
            continue
        count += 1
        size += path.stat().st_size
    return count, size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/raw/CSE-CIC-IDS2018"),
    )
    parser.add_argument(
        "--organized-root",
        type=Path,
        default=Path("data/raw/CSE-CIC-IDS2018_organized"),
    )
    parser.add_argument(
        "--unrar-bin",
        type=Path,
        default=Path("data/raw/.tools/unrar-bin"),
    )
    parser.add_argument(
        "--archive-mode",
        choices=["reference", "copy"],
        default="reference",
        help="reference source archives directly, or copy archives before extraction",
    )
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root
    root = args.organized_root
    manifests = root / "manifests"
    logs = root / "logs"
    ensure_dir(root)
    for sub in [
        "processed_csv",
        "raw_pcap_archives" if args.archive_mode == "copy" else "source_archives",
        "raw_log_archives" if args.archive_mode == "copy" else "source_archives",
        "extracted_pcaps",
        "extracted_logs",
        "manifests",
        "notes",
        "logs",
    ]:
        ensure_dir(root / sub)

    csv_files, pcap_archives, log_archives = list_source_files(source_root)
    if not csv_files or not pcap_archives:
        raise SystemExit(f"Missing expected IDS2018 files under {source_root}")

    copy_rows: list[dict[str, object]] = []
    processed_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []

    for src in csv_files:
        dst = root / "processed_csv" / src.name
        info = copy_file(src, dst)
        copy_rows.append({"kind": "processed_csv", **info})
        processed_rows.append(
            {
                "dataset": DATASET,
                "file": src.name,
                "organized_path": str(dst),
                "original_path": str(src),
                "size_bytes": dst.stat().st_size,
                "status": "copied_regular_file" if dst.is_file() and not dst.is_symlink() else "invalid",
            }
        )

    for src in pcap_archives:
        day = day_from_archive(src)
        if args.archive_mode == "copy":
            dst = root / "raw_pcap_archives" / day / src.name
            info = copy_file(src, dst)
            copy_rows.append({"kind": "pcap_archive", **info})
            organized_path = dst
            status = "copied_regular_file" if dst.is_file() and not dst.is_symlink() else "invalid"
        else:
            organized_path = src
            status = "referenced_source_archive"
        raw_rows.append(
            {
                "dataset": DATASET,
                "day": day,
                "archive_type": "pcap_archive",
                "organized_path": str(organized_path),
                "original_path": str(src),
                "size_bytes": src.stat().st_size,
                "status": status,
            }
        )

    for src in log_archives:
        day = day_from_archive(src)
        if args.archive_mode == "copy":
            dst = root / "raw_log_archives" / day / src.name
            info = copy_file(src, dst)
            copy_rows.append({"kind": "log_archive", **info})
            organized_path = dst
            status = "copied_regular_file" if dst.is_file() and not dst.is_symlink() else "invalid"
        else:
            organized_path = src
            status = "referenced_source_archive"
        raw_rows.append(
            {
                "dataset": DATASET,
                "day": day,
                "archive_type": "log_archive",
                "organized_path": str(organized_path),
                "original_path": str(src),
                "size_bytes": src.stat().st_size,
                "status": status,
            }
        )

    write_csv(
        manifests / "copy_manifest.csv",
        copy_rows,
        ["kind", "source", "destination", "size_bytes", "status", "sha256_prefix"],
    )
    write_csv(
        manifests / "processed_csv_manifest.csv",
        processed_rows,
        ["dataset", "file", "organized_path", "original_path", "size_bytes", "status"],
    )
    write_csv(
        manifests / ("raw_archive_copy_manifest.csv" if args.archive_mode == "copy" else "source_archive_manifest.csv"),
        raw_rows,
        [
            "dataset",
            "day",
            "archive_type",
            "organized_path",
            "original_path",
            "size_bytes",
            "status",
        ],
    )

    extraction_rows: list[dict[str, object]] = []
    if not args.skip_extract:
        for row in raw_rows:
            archive = Path(str(row["organized_path"]))
            day = str(row["day"])
            if row["archive_type"] == "pcap_archive":
                out_dir = root / "extracted_pcaps" / day
            else:
                out_dir = root / "extracted_logs" / day
            marker = out_dir / ".flowprim_extract_complete.json"
            log_path = logs / "extraction" / f"{day}_{archive.name}.log"
            try:
                if archive.suffix.lower() == ".zip":
                    result = extract_zip(archive, out_dir, marker, log_path)
                elif archive.suffix.lower() == ".rar":
                    result = extract_rar(archive, out_dir, marker, log_path, args.unrar_bin)
                else:
                    result = {
                        "archive": str(archive),
                        "extract_dir": str(out_dir),
                        "entry_count": "",
                        "pcap_like_entry_count": "",
                        "uncompressed_bytes": "",
                        "status": "skipped",
                        "reason": f"unsupported suffix {archive.suffix}",
                    }
            except Exception as exc:  # keep the inventory complete even if one archive fails
                result = {
                    "archive": str(archive),
                    "extract_dir": str(out_dir),
                    "entry_count": "",
                    "pcap_like_entry_count": "",
                    "uncompressed_bytes": "",
                    "status": "failed",
                    "reason": repr(exc),
                }
            result.update({"dataset": DATASET, "day": day, "archive_type": row["archive_type"]})
            extraction_rows.append(result)

    write_csv(
        manifests / "extraction_manifest.csv",
        extraction_rows,
        [
            "dataset",
            "day",
            "archive_type",
            "archive",
            "extract_dir",
            "entry_count",
            "pcap_like_entry_count",
            "uncompressed_bytes",
            "status",
            "reason",
        ],
    )

    pcap_file_count, pcap_file_bytes = count_files(
        root / "extracted_pcaps", exclude_names={".flowprim_extract_complete.json"}
    )
    log_file_count, log_file_bytes = count_files(
        root / "extracted_logs", exclude_names={".flowprim_extract_complete.json"}
    )
    if args.archive_mode == "copy":
        archive_count, archive_bytes = count_files(root / "raw_pcap_archives")
        log_archive_count, log_archive_bytes = count_files(root / "raw_log_archives")
    else:
        archive_count = len(pcap_archives)
        archive_bytes = sum(p.stat().st_size for p in pcap_archives)
        log_archive_count = len(log_archives)
        log_archive_bytes = sum(p.stat().st_size for p in log_archives)
    csv_count, csv_bytes = count_files(root / "processed_csv", [".csv"])
    regular_symlinks = [str(p) for p in root.rglob("*") if p.is_symlink()]

    summary = {
        "dataset": DATASET,
        "status": "organized_copied_extracted",
        "source_root": str(source_root),
        "organized_root": str(root),
        "unrar_bin": str(args.unrar_bin),
        "archive_mode": args.archive_mode,
        "processed_csv_files": csv_count,
        "processed_csv_bytes": csv_bytes,
        "pcap_archives": archive_count,
        "pcap_archive_bytes": archive_bytes,
        "log_archives": log_archive_count,
        "log_archive_bytes": log_archive_bytes,
        "extracted_pcap_like_files": pcap_file_count,
        "extracted_pcap_like_bytes": pcap_file_bytes,
        "extracted_log_files": log_file_count,
        "extracted_log_bytes": log_file_bytes,
        "remaining_symlinks": regular_symlinks,
        "notes": [
            "Processed CSV files are copied into this organized tree.",
            "When archive_mode=reference, raw archives stay in the original dataset tree and only extracted files are materialized here.",
            "PCAP archives are extracted under extracted_pcaps by day.",
            "Log archives are extracted under extracted_logs by day.",
            "FlowPrim behavior tokens must not use raw IP, absolute timestamp, complete five-tuple, protocol, service, or ports as behavior tokens.",
        ],
    }
    (manifests / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    readme = f"""# CSE-CIC-IDS2018 Organized Dataset

Status: **organized, CSV-copied, and extracted**.

Original dataset:

```text
{source_root}
```

Organized dataset:

```text
{root}
```

## Layout

```text
processed_csv/                 copied CICFlowMeter processed flow CSV files
source_archive_manifest.csv     source pcap/log archive references
extracted_pcaps/<day>/          extracted PCAP/CAP files for packet-level FlowPrim rebuild
extracted_logs/<day>/           extracted Windows event logs
manifests/                      copy and extraction manifests
notes/                          downstream FlowPrim notes
logs/                           copy/extraction logs
```

## Inventory

- Processed CSV files: {csv_count}
- Source PCAP archives referenced: {archive_count}
- Source log archives referenced: {log_archive_count}
- Extracted PCAP-like files: {pcap_file_count}
- Extracted log files: {log_file_count}

## FlowPrim Notes

Use `extracted_pcaps/` for packet/burst/structural-primitive reconstruction.
The processed CSV files are aggregate CICFlowMeter summaries and do not preserve
packet order, packet direction-length motifs, burst spans, or primitive trigger
positions.

Raw IP addresses, absolute timestamps, ports, protocol, and complete five-tuples
may only be used for joining, splitting, deduplication, audit, or label alignment.
They must not be used as behavior tokens or benign-memory grouping keys.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
