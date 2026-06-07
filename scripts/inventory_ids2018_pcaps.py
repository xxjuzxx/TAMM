#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PCAP_ROOT = Path("data/raw/CSE-CIC-IDS2018_organized/extracted_pcaps")
DEFAULT_SCHEDULE = ROOT / "configs" / "ids2018_attack_schedule.yaml"
DEFAULT_OUT_DIR = ROOT / "data" / "manifests"
DEFAULT_REPORT = ROOT / "reports" / "ids2018_pcap_integrity_report.md"
DEFAULT_CANDIDATES = ROOT / "data" / "manifests" / "ids2018_allfeas_candidate_manifest.csv"
DEFAULT_SUMMARY = ROOT / "data" / "manifests" / "ids2018_pcap_integrity_summary.json"


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def _read_schedule(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _schedule_by_day(schedule: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in schedule.get("windows", []):
        by_day[str(window.get("day_dir") or "")].append(window)
    return dict(by_day)


def _extract_ips_from_name(name: str) -> list[str]:
    return sorted(set(re.findall(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", name)))


def _day_from_path(path: Path, pcap_root: Path) -> str:
    try:
        rel = path.relative_to(pcap_root)
        return rel.parts[0] if rel.parts else "unknown"
    except ValueError:
        return "unknown"


def _parse_capinfos_csv(line: str) -> list[str]:
    # capinfos -T -m emits comma-separated fields without quoting for these fields.
    return [part.strip() for part in line.strip().split(",")]


def _capinfos(path: Path) -> dict[str, Any]:
    proc = _run(["capinfos", "-T", "-m", "-r", "-c", "-s", "-u", str(path)], timeout=120)
    info: dict[str, Any] = {
        "capinfos_exit_code": proc.returncode,
        "capinfos_stderr": proc.stderr.strip(),
        "packet_count": "",
        "file_size_bytes_capinfos": "",
        "duration_seconds": "",
    }
    if proc.returncode == 0 and proc.stdout.strip():
        parts = _parse_capinfos_csv(proc.stdout.strip().splitlines()[-1])
        if len(parts) >= 4:
            info["packet_count"] = _to_int(parts[1])
            info["file_size_bytes_capinfos"] = _to_int(parts[2])
            info["duration_seconds"] = _to_float(parts[3])
    return info


def _to_int(value: Any) -> int | str:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return ""


def _to_float(value: Any) -> float | str:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return ""


def _file_type(path: Path) -> str:
    proc = _run(["file", "-b", str(path)], timeout=30)
    return proc.stdout.strip() if proc.returncode == 0 else f"file_error: {proc.stderr.strip()}"


def _integrity_status(file_type: str, capinfo: dict[str, Any]) -> tuple[str, str]:
    lowered_type = file_type.lower()
    stderr = str(capinfo.get("capinfos_stderr") or "").lower()
    if "pcap" not in lowered_type:
        return "unsupported", "file command did not identify pcap/pcapng"
    if capinfo.get("capinfos_exit_code") != 0:
        if "cut short" in stderr or "truncated" in stderr:
            return "truncated", capinfo.get("capinfos_stderr") or "capinfos failed"
        return "capinfos_error", capinfo.get("capinfos_stderr") or "capinfos failed"
    if "cut short" in stderr or "truncated" in stderr:
        return "truncated_readable", capinfo.get("capinfos_stderr") or "capinfos warning"
    return "ok", ""


def _candidate_rank(row: dict[str, Any]) -> tuple[int, int, int, str]:
    status = str(row.get("integrity_status") or "")
    status_rank = 0 if status == "ok" else 1 if status == "truncated_readable" else 2
    matched = int(_truthy(row.get("matches_official_endpoint_ip")))
    packets = int(row.get("packet_count") or 0)
    size = int(row.get("file_size_bytes") or 0)
    return (status_rank, -matched, -packets, -size, str(row.get("pcap_path") or ""))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _refresh_derived_fields(
    row: dict[str, Any],
    *,
    pcap_root: Path,
    all_endpoint_ips_by_day: dict[str, set[str]],
    families_by_day: dict[str, set[str]],
    window_count_by_day: dict[str, int],
) -> dict[str, Any]:
    refreshed = dict(row)
    path = Path(str(refreshed.get("pcap_path") or ""))
    day = _day_from_path(path, pcap_root)
    if day == "unknown":
        day = str(refreshed.get("day_dir") or "unknown")
    ips_in_name = _extract_ips_from_name(path.name or str(refreshed.get("pcap_name") or ""))
    official_ips = all_endpoint_ips_by_day.get(day, set())
    matched_ips = sorted(set(ips_in_name).intersection(official_ips))
    refreshed.update(
        {
            "day_dir": day,
            "pcap_name": path.name or str(refreshed.get("pcap_name") or ""),
            "schedule_window_count_for_day": window_count_by_day.get(day, 0),
            "official_attack_families_for_day": ";".join(sorted(families_by_day.get(day, set()))),
            "pcap_name_ips_count": len(ips_in_name),
            "matches_official_endpoint_ip": bool(matched_ips),
            "matched_official_endpoint_ips_count": len(matched_ips),
            "audit_only_endpoint_ips_in_name": ";".join(ips_in_name),
            "audit_only_matched_official_endpoint_ips": ";".join(matched_ips),
            "raw_ip_used_as_token": False,
            "absolute_time_used_as_token": False,
            "five_tuple_used_as_token": False,
            "protocol_service_used_as_memory_key": False,
        }
    )
    return refreshed


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    pcap_root = Path(args.pcap_root)
    schedule = _read_schedule(Path(args.schedule))
    windows_by_day = _schedule_by_day(schedule)
    all_endpoint_ips_by_day: dict[str, set[str]] = {}
    families_by_day: dict[str, set[str]] = {}
    window_count_by_day: dict[str, int] = {}
    for day, windows in windows_by_day.items():
        ips: set[str] = set()
        families: set[str] = set()
        for window in windows:
            ips.update(str(ip) for ip in window.get("attacker_ips", []))
            ips.update(str(ip) for ip in window.get("victim_ips", []))
            families.add(str(window.get("attack_family") or "UNKNOWN"))
        all_endpoint_ips_by_day[day] = ips
        families_by_day[day] = families
        window_count_by_day[day] = len(windows)

    pcap_paths = sorted(path for path in pcap_root.glob("*/*/*") if path.is_file())
    if args.limit and args.limit > 0:
        pcap_paths = pcap_paths[: args.limit]

    out_manifest = Path(args.output)
    rows: list[dict[str, Any]] = []
    processed_paths: set[str] = set()
    if args.resume and out_manifest.exists():
        with out_manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                refreshed = _refresh_derived_fields(
                    dict(row),
                    pcap_root=pcap_root,
                    all_endpoint_ips_by_day=all_endpoint_ips_by_day,
                    families_by_day=families_by_day,
                    window_count_by_day=window_count_by_day,
                )
                rows.append(refreshed)
                processed_paths.add(str(row.get("pcap_path") or ""))
        if rows:
            print(f"resuming from {len(rows)} existing manifest rows in {out_manifest}", flush=True)

    total_paths = len(pcap_paths)
    for idx, path in enumerate(pcap_paths, start=1):
        if str(path) in processed_paths:
            if args.progress_every and idx % args.progress_every == 0:
                print(f"scanned {idx}/{total_paths} pcaps ({len(rows)} rows retained)", flush=True)
            continue
        day = _day_from_path(path, pcap_root)
        file_type = _file_type(path)
        capinfo = _capinfos(path)
        status, reason = _integrity_status(file_type, capinfo)
        ips_in_name = _extract_ips_from_name(path.name)
        official_ips = all_endpoint_ips_by_day.get(day, set())
        matched_ips = sorted(set(ips_in_name).intersection(official_ips))
        rows.append(
            {
                "dataset": "CSE-CIC-IDS2018",
                "day_dir": day,
                "pcap_path": str(path),
                "pcap_name": path.name,
                "file_size_bytes": path.stat().st_size,
                "file_type": file_type,
                "integrity_status": status,
                "integrity_reason": reason,
                "packet_count": capinfo.get("packet_count", ""),
                "duration_seconds": capinfo.get("duration_seconds", ""),
                "capinfos_exit_code": capinfo.get("capinfos_exit_code", ""),
                "capinfos_stderr": capinfo.get("capinfos_stderr", ""),
                "schedule_window_count_for_day": window_count_by_day.get(day, 0),
                "official_attack_families_for_day": ";".join(sorted(families_by_day.get(day, set()))),
                "pcap_name_ips_count": len(ips_in_name),
                "matches_official_endpoint_ip": bool(matched_ips),
                "matched_official_endpoint_ips_count": len(matched_ips),
                "audit_only_endpoint_ips_in_name": ";".join(ips_in_name),
                "audit_only_matched_official_endpoint_ips": ";".join(matched_ips),
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
                "protocol_service_used_as_memory_key": False,
            }
        )
        if args.checkpoint_every and len(rows) % args.checkpoint_every == 0:
            _write_csv(out_manifest, rows)
        if args.progress_every and idx % args.progress_every == 0:
            print(f"scanned {idx}/{total_paths} pcaps ({len(rows)} rows retained)", flush=True)

    rows_sorted = sorted(rows, key=lambda r: (str(r["day_dir"]), _candidate_rank(r)))
    candidate_rows: list[dict[str, Any]] = []
    by_day_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_sorted:
        by_day_rows[str(row["day_dir"])].append(row)
    for day, day_rows in sorted(by_day_rows.items()):
        top = sorted(day_rows, key=_candidate_rank)[: args.candidates_per_day]
        for rank, row in enumerate(top, start=1):
            candidate_rows.append(
                {
                    "candidate_rank": rank,
                    "day_dir": day,
                    "attack_families_for_day": row["official_attack_families_for_day"],
                    "pcap_path": row["pcap_path"],
                    "pcap_name": row["pcap_name"],
                    "integrity_status": row["integrity_status"],
                    "file_size_bytes": row["file_size_bytes"],
                    "packet_count": row["packet_count"],
                    "duration_seconds": row["duration_seconds"],
                    "matches_official_endpoint_ip": _truthy(row["matches_official_endpoint_ip"]),
                    "matched_official_endpoint_ips_count": row["matched_official_endpoint_ips_count"],
                    "selection_policy": "prefer_ok_then_official_endpoint_ip_match_then_packet_count_then_size",
                    "label_alignment_use_only": True,
                    "raw_ip_used_as_token": False,
                    "absolute_time_used_as_token": False,
                    "five_tuple_used_as_token": False,
                    "protocol_service_used_as_memory_key": False,
                }
            )

    _write_csv(out_manifest, rows)
    _write_csv(Path(args.candidates_out), candidate_rows)
    summary = _summary(rows, candidate_rows, schedule)
    _write_json(Path(args.summary_out), summary)
    _write_report(Path(args.report), rows, candidate_rows, summary, out_manifest, Path(args.candidates_out), Path(args.summary_out))
    return summary


def _summary(rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], schedule: dict[str, Any]) -> dict[str, Any]:
    by_status = Counter(str(row["integrity_status"]) for row in rows)
    by_day = Counter(str(row["day_dir"]) for row in rows)
    by_day_status: dict[str, Counter[str]] = defaultdict(Counter)
    size_by_day: dict[str, int] = defaultdict(int)
    packets_by_day: dict[str, int] = defaultdict(int)
    endpoint_match_by_day: dict[str, int] = defaultdict(int)
    for row in rows:
        day = str(row["day_dir"])
        by_day_status[day][str(row["integrity_status"])] += 1
        size_by_day[day] += int(row.get("file_size_bytes") or 0)
        packets_by_day[day] += int(row.get("packet_count") or 0)
        endpoint_match_by_day[day] += int(_truthy(row.get("matches_official_endpoint_ip")))
    candidate_by_day = Counter(str(row["day_dir"]) for row in candidate_rows)
    family_days: dict[str, set[str]] = defaultdict(set)
    for window in schedule.get("windows", []):
        family_days[str(window.get("attack_family") or "UNKNOWN")].add(str(window.get("day_dir") or ""))
    return {
        "schema_version": "flowprim_ids2018_pcap_integrity_summary_v1",
        "dataset": "CSE-CIC-IDS2018",
        "pcap_count": len(rows),
        "total_size_bytes": sum(int(row.get("file_size_bytes") or 0) for row in rows),
        "integrity_status_counts": dict(sorted(by_status.items())),
        "day_counts": dict(sorted(by_day.items())),
        "day_status_counts": {day: dict(sorted(counts.items())) for day, counts in sorted(by_day_status.items())},
        "day_size_bytes": dict(sorted(size_by_day.items())),
        "day_packet_counts": dict(sorted(packets_by_day.items())),
        "day_official_endpoint_name_match_counts": dict(sorted(endpoint_match_by_day.items())),
        "candidate_counts_by_day": dict(sorted(candidate_by_day.items())),
        "attack_family_days": {family: sorted(days) for family, days in sorted(family_days.items())},
        "token_safety": {
            "raw_ip_used_as_token": False,
            "absolute_time_used_as_token": False,
            "five_tuple_used_as_token": False,
            "protocol_service_used_as_memory_key": False,
        },
        "notes": [
            "IP addresses parsed from PCAP filenames are audit/selection metadata only.",
            "The candidate manifest is for scheduling Zeek/AllFeas extraction and label alignment; it is not a behavior-token vocabulary.",
            "Integrity is measured with file(1) and capinfos; repaired PCAP generation is a separate explicit step.",
        ],
    }


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    manifest_path: Path,
    candidate_path: Path,
    summary_path: Path,
) -> None:
    lines = [
        "# IDS2018 PCAP Integrity Inventory",
        "",
        "This report inventories local CSE-CIC-IDS2018 PCAP-like files before any full AllFeas rerun.",
        "",
        "## Outputs",
        "",
        f"- Full manifest: `{manifest_path}`",
        f"- AllFeas candidate manifest: `{candidate_path}`",
        f"- JSON summary: `{summary_path}`",
        "",
        "## Overall Counts",
        "",
        f"- PCAP-like files scanned: {_fmt_int(summary['pcap_count'])}",
        f"- Total bytes: {_fmt_int(summary['total_size_bytes'])}",
        "",
        "### Integrity Status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in summary["integrity_status_counts"].items():
        lines.append(f"| {status} | {_fmt_int(count)} |")
    lines.extend(["", "## Day Coverage", "", "| Day | Files | OK | Truncated/Errors | Endpoint-name matches | Bytes | Packets |", "|---|---:|---:|---:|---:|---:|---:|"])
    for day, count in summary["day_counts"].items():
        status_counts = summary["day_status_counts"].get(day, {})
        ok = int(status_counts.get("ok", 0))
        bad = int(count) - ok
        lines.append(
            f"| {day} | {_fmt_int(count)} | {_fmt_int(ok)} | {_fmt_int(bad)} | "
            f"{_fmt_int(summary['day_official_endpoint_name_match_counts'].get(day, 0))} | "
            f"{_fmt_int(summary['day_size_bytes'].get(day, 0))} | {_fmt_int(summary['day_packet_counts'].get(day, 0))} |"
        )
    lines.extend(["", "## Attack Family Days", "", "| Family | Days |", "|---|---|"])
    for family, days in summary["attack_family_days"].items():
        lines.append(f"| {family} | {', '.join(days)} |")
    lines.extend(["", "## Top AllFeas Candidates Per Day", "", "| Day | Rank | Status | Endpoint match | Packets | Bytes | PCAP |", "|---|---:|---|---:|---:|---:|---|"])
    for row in candidate_rows:
        lines.append(
            f"| {row['day_dir']} | {row['candidate_rank']} | {row['integrity_status']} | "
            f"{int(_truthy(row['matches_official_endpoint_ip']))} | {_fmt_int(row['packet_count'])} | "
            f"{_fmt_int(row['file_size_bytes'])} | `{row['pcap_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "- Raw IP addresses parsed from filenames are audit and extraction-planning metadata only.",
            "- Absolute timestamps, raw IPs, complete five-tuples, protocol, and service are not behavior tokens, structural primitives, KNN memory grouping keys, or threshold features.",
            "- The manifest is intended to drive subsequent Zeek/AllFeas extraction jobs and to document truncated/repaired input provenance.",
            "- Repaired PCAPs, if generated later, should be kept in a separate directory and marked with repaired provenance.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory local IDS2018 PCAP integrity and AllFeas extraction candidates.")
    parser.add_argument("--pcap-root", default=str(DEFAULT_PCAP_ROOT))
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    parser.add_argument("--output", default=str(DEFAULT_OUT_DIR / "ids2018_pcap_integrity_manifest.csv"))
    parser.add_argument("--candidates-out", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--candidates-per-day", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit; 0 scans all files.")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    summary = build_inventory(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
