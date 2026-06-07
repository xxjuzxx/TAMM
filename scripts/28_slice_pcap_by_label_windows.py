#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.data.cicids2017_labeler import _parse_timestamp_seconds
from src.data.label_policy import ATTEMPTED_POLICIES, apply_attempted_policy, normalize_label
from src.utils.io import write_json


@dataclass(frozen=True)
class LabelRow:
    source_file: str
    row_index: int
    label: str
    raw_label: str
    start_ts: float
    end_ts: float


def _expand_inputs(items: list[str], suffixes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if any(char in item for char in "*?[]"):
            paths.extend(Path(match) for match in sorted(glob.glob(item)))
        elif path.is_dir():
            paths.extend(sorted(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in suffixes))
        else:
            paths.append(path)
    return paths


def _row_value(row: dict[str, str], *names: str, default: str = "") -> str:
    lower = {key.lower().strip(): value for key, value in row.items()}
    for name in names:
        if name in row and row[name] != "":
            return row[name]
        value = lower.get(name.lower())
        if value:
            return value
    return default


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _duration_seconds(row: dict[str, str]) -> float:
    raw = _to_float(_row_value(row, "Flow Duration", "duration", default="0"))
    return raw / 1_000_000.0 if raw > 10000 else raw


def read_label_rows(paths: list[Path], attempted_policy: str = "drop") -> tuple[list[LabelRow], dict[str, Any]]:
    rows: list[LabelRow] = []
    stats: dict[str, Any] = {
        "input_csvs": [str(path) for path in paths],
        "attempted_policy": attempted_policy,
        "input_rows": 0,
        "skipped_attempted_rows": 0,
        "skipped_bad_timestamp_rows": 0,
        "raw_label_counts": Counter(),
        "label_counts": Counter(),
    }
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                stats["input_rows"] += 1
                raw_label = normalize_label(_row_value(row, "Label", "label", default=path.stem))
                stats["raw_label_counts"][raw_label] += 1
                label = apply_attempted_policy(raw_label, attempted_policy)
                if label is None:
                    stats["skipped_attempted_rows"] += 1
                    continue
                timestamp_raw = _row_value(row, "Timestamp", "Flow Start", "start_ts", "frame.time_epoch", default="")
                start_ts = _parse_timestamp_seconds(timestamp_raw)
                if start_ts is None:
                    stats["skipped_bad_timestamp_rows"] += 1
                    continue
                normalized_label = normalize_label(label)
                duration = max(_duration_seconds(row), 0.0)
                rows.append(
                    LabelRow(
                        source_file=str(path),
                        row_index=row_index,
                        label=normalized_label,
                        raw_label=raw_label,
                        start_ts=start_ts,
                        end_ts=start_ts + duration,
                    )
                )
                stats["label_counts"][normalized_label] += 1

    stats["raw_label_counts"] = dict(sorted(stats["raw_label_counts"].items()))
    stats["label_counts"] = dict(sorted(stats["label_counts"].items()))
    return rows, stats


def _matches_any_glob(value: str, patterns: list[str]) -> bool:
    value_lower = value.lower()
    return any(fnmatch(value_lower, pattern.lower()) for pattern in patterns)


def filter_rows(rows: list[LabelRow], exact_labels: list[str], label_globs: list[str]) -> list[LabelRow]:
    exact = {normalize_label(label).lower() for label in exact_labels}
    selected: list[LabelRow] = []
    for row in rows:
        label = row.label
        if exact and label.lower() in exact:
            selected.append(row)
            continue
        if label_globs and _matches_any_glob(label, label_globs):
            selected.append(row)
    return selected


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "label"


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _compact_time(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%S")


def _window_to_dict(
    *,
    input_pcap: Path,
    out_dir: Path,
    label: str,
    window_index: int,
    start_ts: float,
    end_ts: float,
    padded_start_ts: float,
    padded_end_ts: float,
    row_count: int,
    source_files: list[str],
    raw_label_counts: Counter[str],
) -> dict[str, Any]:
    label_slug = _safe_slug(label)
    output_name = (
        f"{input_pcap.stem}_{label_slug}_{window_index:03d}_"
        f"{_compact_time(padded_start_ts)}_{_compact_time(padded_end_ts)}.pcap"
    )
    return {
        "label": label,
        "label_slug": label_slug,
        "row_count": row_count,
        "source_files": sorted(source_files),
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "padded_start_ts": padded_start_ts,
        "padded_end_ts": padded_end_ts,
        "start_utc": _iso_utc(start_ts),
        "end_utc": _iso_utc(end_ts),
        "padded_start_utc": _iso_utc(padded_start_ts),
        "padded_end_utc": _iso_utc(padded_end_ts),
        "output_pcap": str(out_dir / output_name),
    }


def build_windows(
    rows: list[LabelRow],
    input_pcap: Path,
    out_dir: Path,
    *,
    padding_seconds: float = 5.0,
    merge_gap_seconds: float = 60.0,
    min_rows_per_window: int = 1,
    max_windows_per_label: int | None = None,
    window_time_source: str = "start",
    max_rows_per_window: int | None = None,
) -> list[dict[str, Any]]:
    if window_time_source not in {"start", "flow_end"}:
        raise ValueError(f"unsupported window_time_source: {window_time_source}")
    if max_rows_per_window is not None and max_rows_per_window <= 0:
        raise ValueError("max_rows_per_window must be positive")

    by_label: dict[str, list[LabelRow]] = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)

    windows: list[dict[str, Any]] = []
    for label in sorted(by_label):
        label_rows = sorted(by_label[label], key=lambda item: (item.start_ts, item.end_ts, item.row_index))
        label_windows: list[dict[str, Any]] = []
        current_start = label_rows[0].start_ts
        first_end = label_rows[0].end_ts if window_time_source == "flow_end" else label_rows[0].start_ts
        current_end = first_end
        row_count = 0
        source_files: set[str] = set()
        raw_counts: Counter[str] = Counter()

        def add_current() -> None:
            if row_count < min_rows_per_window:
                return
            padded_start = max(current_start - padding_seconds, 0.0)
            padded_end = current_end + padding_seconds
            label_windows.append(
                _window_to_dict(
                    input_pcap=input_pcap,
                    out_dir=out_dir,
                    label=label,
                    window_index=len(label_windows) + 1,
                    start_ts=current_start,
                    end_ts=current_end,
                    padded_start_ts=padded_start,
                    padded_end_ts=padded_end,
                    row_count=row_count,
                    source_files=sorted(source_files),
                    raw_label_counts=raw_counts,
                )
            )

        for row in label_rows:
            row_end = row.end_ts if window_time_source == "flow_end" else row.start_ts
            reached_row_cap = max_rows_per_window is not None and row_count >= max_rows_per_window
            reached_gap = row_count > 0 and row.start_ts > current_end + merge_gap_seconds
            if reached_gap or reached_row_cap:
                add_current()
                current_start = row.start_ts
                current_end = row_end
                row_count = 0
                source_files = set()
                raw_counts = Counter()
            current_end = max(current_end, row_end)
            row_count += 1
            source_files.add(row.source_file)
            raw_counts[row.raw_label] += 1
        add_current()

        if max_windows_per_label is not None and len(label_windows) > max_windows_per_label:
            label_windows = sorted(label_windows, key=lambda item: (-int(item["row_count"]), item["start_ts"]))[
                :max_windows_per_label
            ]
            label_windows = sorted(label_windows, key=lambda item: item["start_ts"])
            for index, window in enumerate(label_windows, start=1):
                old_output = Path(window["output_pcap"])
                new_name = re.sub(r"_(\d{3})_", f"_{index:03d}_", old_output.name, count=1)
                window["output_pcap"] = str(old_output.with_name(new_name))

        windows.extend(label_windows)

    return sorted(windows, key=lambda item: (item["padded_start_ts"], item["label"], item["output_pcap"]))


def editcap_command(
    window: dict[str, Any],
    input_pcap: Path,
    *,
    editcap_bin: str = "editcap",
    output_type: str = "pcap",
) -> list[str]:
    return [
        editcap_bin,
        "-F",
        output_type,
        "-A",
        str(window["padded_start_utc"]),
        "-B",
        str(window["padded_end_utc"]),
        str(input_pcap),
        str(window["output_pcap"]),
    ]


def execute_windows(
    windows: list[dict[str, Any]],
    input_pcap: Path,
    *,
    editcap_bin: str = "editcap",
    output_type: str = "pcap",
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for window in windows:
        output_path = Path(str(window["output_pcap"]))
        command = editcap_command(window, input_pcap, editcap_bin=editcap_bin, output_type=output_type)
        result = {"output_pcap": str(output_path), "command": command, "status": "planned"}
        if output_path.exists() and not overwrite:
            result["status"] = "skipped_existing"
            results.append(result)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print("+ " + " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        result["returncode"] = int(completed.returncode)
        result["status"] = "ok" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        results.append(result)
    return results


def build_manifest(args: argparse.Namespace, read_stats: dict[str, Any], selected: list[LabelRow], windows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_pcap": str(args.input_pcap),
        "out_dir": str(args.out_dir),
        "label_csv": [str(path) for path in args.label_csv],
        "label": args.label,
        "label_glob": args.label_glob,
        "attempted_policy": args.attempted_policy,
        "padding_seconds": float(args.padding_seconds),
        "merge_gap_seconds": float(args.merge_gap_seconds),
        "window_time_source": args.window_time_source,
        "max_rows_per_window": args.max_rows_per_window,
        "min_rows_per_window": int(args.min_rows_per_window),
        "max_windows_per_label": args.max_windows_per_label,
        "read_stats": read_stats,
        "selected_rows": len(selected),
        "selected_label_counts": dict(sorted(Counter(row.label for row in selected).items())),
        "windows": windows,
        "commands": [
            editcap_command(window, Path(args.input_pcap), editcap_bin=args.editcap_bin, output_type=args.output_type)
            for window in windows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build optional editcap slices from a full-day PCAP using corrected CICIDS label time windows."
    )
    parser.add_argument("--input_pcap", required=True)
    parser.add_argument("--label_csv", nargs="+", required=True)
    parser.add_argument("--label", action="append", default=[], help="Exact post-policy label to include. Can repeat.")
    parser.add_argument("--label_glob", action="append", default=[], help="Case-insensitive glob for post-policy labels.")
    parser.add_argument("--attempted_policy", choices=ATTEMPTED_POLICIES, default="drop")
    parser.add_argument("--padding_seconds", type=float, default=5.0)
    parser.add_argument("--merge_gap_seconds", type=float, default=60.0)
    parser.add_argument("--window_time_source", choices=["start", "flow_end"], default="start")
    parser.add_argument("--max_rows_per_window", type=int, default=None)
    parser.add_argument("--min_rows_per_window", type=int, default=1)
    parser.add_argument("--max_windows_per_label", type=int, default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--manifest_out", default=None)
    parser.add_argument("--execute", action="store_true", help="Run editcap. Without this flag, only writes a plan.")
    parser.add_argument("--editcap_bin", default="editcap")
    parser.add_argument("--output_type", default="pcap")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.label and not args.label_glob:
        raise SystemExit("At least one --label or --label_glob is required.")

    input_pcap = Path(args.input_pcap)
    label_csvs = _expand_inputs(args.label_csv, {".csv"})
    args.label_csv = label_csvs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, read_stats = read_label_rows(label_csvs, attempted_policy=args.attempted_policy)
    selected = filter_rows(rows, args.label, args.label_glob)
    windows = build_windows(
        selected,
        input_pcap,
        out_dir,
        padding_seconds=args.padding_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
        min_rows_per_window=args.min_rows_per_window,
        max_windows_per_label=args.max_windows_per_label,
        window_time_source=args.window_time_source,
        max_rows_per_window=args.max_rows_per_window,
    )
    manifest = build_manifest(args, read_stats, selected, windows)

    if args.execute:
        manifest["execution"] = execute_windows(
            windows,
            input_pcap,
            editcap_bin=args.editcap_bin,
            output_type=args.output_type,
            overwrite=args.overwrite,
        )
    else:
        manifest["execution"] = [{"output_pcap": window["output_pcap"], "status": "planned"} for window in windows]

    manifest_out = Path(args.manifest_out) if args.manifest_out else out_dir / "slice_plan.json"
    write_json(manifest, manifest_out)
    print(
        {
            "selected_rows": len(selected),
            "windows": len(windows),
            "manifest": str(manifest_out),
            "execute": bool(args.execute),
        }
    )


if __name__ == "__main__":
    main()
