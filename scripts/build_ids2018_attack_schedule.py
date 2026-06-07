#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any
from urllib.request import urlopen

import _bootstrap  # noqa: F401
import yaml

from src.data.ids2018_schedule_labeler import (
    DAY_DIR_BY_ISO_DATE,
    canonical_ids2018_family,
    extract_ipv4s,
    iso_date_from_official,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://www.unb.ca/cic/datasets/ids-2018.html"
DEFAULT_YAML = ROOT / "configs" / "ids2018_attack_schedule.yaml"
DEFAULT_CSV = ROOT / "data" / "manifests" / "ids2018_attack_schedule.csv"
DEFAULT_JSON = ROOT / "configs" / "ids2018_attack_schedule.json"
DEFAULT_CONFIG_CSV = ROOT / "configs" / "ids2018_attack_schedule.csv"
DEFAULT_REPORT = ROOT / "reports" / "ids2018_attack_schedule_manifest.md"


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_cell = False
        self._rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
            self._rows = []
        elif self._in_table and tag == "tr":
            self._row = []
        elif self._in_table and tag in {"td", "th"}:
            self._in_cell = True
            self._cell = []
        elif self._in_cell and tag == "br":
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._in_cell = False
        elif tag == "tr" and self._in_table:
            if self._row:
                self._rows.append(self._row)
        elif tag == "table" and self._in_table:
            self.tables.append(self._rows)
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data)


def _fetch_html(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _official_schedule_rows(html: str) -> list[dict[str, str]]:
    parser = _TableParser()
    parser.feed(html)
    for table in parser.tables:
        if not table:
            continue
        header = [cell.strip() for cell in table[0]]
        if header == ["Attacker", "Victim", "Attack Name", "Date", "Attack Start Time", "Attack Finish Time"]:
            rows = []
            for raw in table[1:]:
                if len(raw) != len(header):
                    continue
                rows.append(dict(zip(header, raw)))
            return rows
    raise RuntimeError("Could not locate IDS2018 official attack schedule table on the UNB page")


def _window_id(idx: int, attack_name: str, iso_date: str, start: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", attack_name.lower()).strip("_")
    return f"ids2018_{iso_date}_{start.replace(':', '')}_{idx:02d}_{slug}"


def build_manifest(url: str) -> dict[str, Any]:
    rows = _official_schedule_rows(_fetch_html(url))
    windows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    duplicate_rows = 0
    for idx, row in enumerate(rows, start=1):
        iso_date = iso_date_from_official(row["Date"])
        attacker_ips = extract_ipv4s(row["Attacker"])
        victim_ips = extract_ipv4s(row["Victim"])
        key = (
            iso_date,
            row["Attack Name"],
            row["Attack Start Time"],
            row["Attack Finish Time"],
            tuple(attacker_ips),
            tuple(victim_ips),
        )
        if key in seen:
            duplicate_rows += 1
            continue
        seen.add(key)
        windows.append(
            {
                "window_id": _window_id(idx, row["Attack Name"], iso_date, row["Attack Start Time"]),
                "attack_name": row["Attack Name"],
                "attack_family": canonical_ids2018_family(row["Attack Name"]),
                "date": row["Date"],
                "iso_date": iso_date,
                "day_dir": DAY_DIR_BY_ISO_DATE.get(iso_date, iso_date),
                "start_time": row["Attack Start Time"],
                "finish_time": row["Attack Finish Time"],
                "attacker": row["Attacker"],
                "victim": row["Victim"],
                "attacker_ips": attacker_ips,
                "victim_ips": victim_ips,
                "source_url": url,
                "confidence": "official_schedule",
                "label_alignment_use_only": True,
            }
        )
    return {
        "schema_version": "flowprim_ids2018_attack_schedule_v1",
        "dataset": "CSE-CIC-IDS2018",
        "source_url": url,
        "source_table": "UNB CSE-CIC-IDS2018 Table 2: List of daily attacks, Machine IPs, Start and finish time of attack(s)",
        "schedule_to_zeek_offset_seconds": 14400,
        "offset_basis": "Matches the existing Tuesday exact five-tuple IDS2018 CSV-to-Zeek alignment audit.",
        "label_protocol": "schedule_ip_time_window",
        "label_alignment_use_only": True,
        "token_safety": {
            "raw_ip_used_as_token": False,
            "absolute_time_used_as_token": False,
            "five_tuple_used_as_token": False,
            "protocol_service_used_as_memory_key": False,
        },
        "duplicate_rows_collapsed": duplicate_rows,
        "windows": windows,
    }


def _write_csv(path: Path, windows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "window_id",
        "attack_name",
        "attack_family",
        "iso_date",
        "day_dir",
        "start_time",
        "finish_time",
        "attacker_ips",
        "victim_ips",
        "source_url",
        "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in windows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fields},
                    "attacker_ips": json.dumps(row.get("attacker_ips", []), sort_keys=True),
                    "victim_ips": json.dumps(row.get("victim_ips", []), sort_keys=True),
                }
            )


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    yaml_path: Path,
    csv_path: Path,
    json_path: Path,
    config_csv_path: Path,
) -> None:
    rows = manifest["windows"]
    by_family: dict[str, int] = {}
    by_day: dict[str, int] = {}
    for row in rows:
        by_family[row["attack_family"]] = by_family.get(row["attack_family"], 0) + 1
        by_day[row["day_dir"]] = by_day.get(row["day_dir"], 0) + 1
    lines = [
        "# IDS2018 Official Attack Schedule Manifest",
        "",
        f"Source: `{manifest['source_url']}`",
        "",
        "This manifest is used only for label alignment and audit metadata. Raw IP",
        "addresses, absolute timestamps, complete five-tuples, protocol, and service",
        "remain excluded from FlowPrim behavior tokens and KNN benign-memory grouping.",
        "",
        "## Outputs",
        "",
        f"- YAML: `{yaml_path}`",
        f"- CSV: `{csv_path}`",
        f"- JSON copy: `{json_path}`",
        f"- Config CSV copy: `{config_csv_path}`",
        "",
        "## Labeling Protocol",
        "",
        "- Tuesday IDS2018 can also be labeled by exact complete five-tuple + timestamp CSV joining.",
        "- Non-Tuesday IDS2018 days use official attacker/victim IP and attack-time-window labeling.",
        "- The schedule-to-Zeek offset is 14,400 seconds, consistent with the existing Tuesday exact-join audit.",
        "- Schedule/IP labels have weaker provenance than exact CSV joins and must be reported separately.",
        "",
        "## Window Counts",
        "",
        f"- Windows: {len(rows)}",
        f"- Duplicate official rows collapsed: {manifest.get('duplicate_rows_collapsed', 0)}",
        f"- By family: `{json.dumps(dict(sorted(by_family.items())), sort_keys=True)}`",
        f"- By day: `{json.dumps(dict(sorted(by_day.items())), sort_keys=True)}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FlowPrim IDS2018 official attack schedule manifest.")
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--out-yaml", default=str(DEFAULT_YAML))
    parser.add_argument("--out-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out-json", default=str(DEFAULT_JSON))
    parser.add_argument("--config-csv-copy", default=str(DEFAULT_CONFIG_CSV))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    manifest = build_manifest(args.source_url)
    yaml_path = Path(args.out_yaml)
    csv_path = Path(args.out_csv)
    json_path = Path(args.out_json)
    config_csv_path = Path(args.config_csv_copy)
    report_path = Path(args.report)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False), encoding="utf-8")
    _write_csv(csv_path, manifest["windows"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(config_csv_path, manifest["windows"])
    _write_report(report_path, manifest, yaml_path, csv_path, json_path, config_csv_path)
    print(
        json.dumps(
            {
                "windows": len(manifest["windows"]),
                "yaml": str(yaml_path),
                "csv": str(csv_path),
                "json": str(json_path),
                "config_csv": str(config_csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
