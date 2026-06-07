#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import write_json


PACKET_STYLE_REQUIRED = {
    "frame.time_epoch",
    "frame.len",
    "ip.src",
    "ip.dst",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
}

CICFLOWMETER_HINTS = {
    "flow id",
    "source ip",
    "source port",
    "destination ip",
    "destination port",
    "protocol",
    "timestamp",
    "flow duration",
    "total fwd packets",
    "total backward packets",
    "label",
}


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def _count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        count = sum(1 for _ in handle)
    return max(0, count - 1)


def _classify_header(header: list[str]) -> dict[str, Any]:
    normalized = [item.strip() for item in header]
    lower = {item.lower() for item in normalized}
    packet_style = PACKET_STYLE_REQUIRED.issubset(lower)
    cicflowmeter_like = len(CICFLOWMETER_HINTS & lower) >= 6 and "label" in lower
    return {
        "column_count": len(normalized),
        "has_label_column": "label" in lower,
        "packet_style_required_columns_present": packet_style,
        "cicflowmeter_like_labelled_flow_csv": cicflowmeter_like,
        "columns": normalized,
    }


def audit(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root)
    files = []
    for path in sorted(root.glob("*/*.csv")):
        header = _read_header(path)
        header_info = _classify_header(header)
        pcap_path = path.with_suffix(".pcap")
        files.append(
            {
                "label_from_parent_directory": path.parent.name,
                "path": str(path),
                "rows": _count_rows(path),
                "size_bytes": path.stat().st_size,
                "pcap_sibling_exists": pcap_path.exists(),
                "pcap_sibling_path": str(pcap_path) if pcap_path.exists() else None,
                **header_info,
            }
        )

    all_packet_style = bool(files) and all(item["packet_style_required_columns_present"] for item in files)
    any_label_column = any(item["has_label_column"] for item in files)
    any_cicflowmeter_like = any(item["cicflowmeter_like_labelled_flow_csv"] for item in files)

    if all_packet_style and not any_label_column:
        dataset_kind = "packet_csv_split_by_label_directory"
        label_provenance = "derived_from_parent_directory_names"
        corrected_label_status = "not_official_or_corrected_flow_labels"
    elif any_cicflowmeter_like:
        dataset_kind = "cicflowmeter_labelled_flow_csv"
        label_provenance = "csv_label_column"
        corrected_label_status = "unknown_without_source_metadata"
    else:
        dataset_kind = "unknown_csv_layout"
        label_provenance = "unknown"
        corrected_label_status = "unknown"

    return {
        "data_root": str(root),
        "dataset_kind": dataset_kind,
        "label_provenance": label_provenance,
        "corrected_label_status": corrected_label_status,
        "total_csv_files": len(files),
        "total_rows": sum(int(item["rows"]) for item in files),
        "label_row_counts": {
            item["label_from_parent_directory"]: int(item["rows"])
            for item in sorted(files, key=lambda row: row["label_from_parent_directory"])
        },
        "has_any_label_column": any_label_column,
        "has_any_cicflowmeter_like_labelled_flow_csv": any_cicflowmeter_like,
        "files": files,
        "external_corrected_sources": [
            {
                "name": "DistriNet CNS2022 improved CIC-IDS-2017",
                "url": "https://intrusion-detection.distrinet-research.be/CNS2022/CICIDS2017.html",
                "dataset_url": "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip",
                "notes": "Improved ground-truth labelling, additional features, and public labelling code.",
            },
            {
                "name": "DistriNet WTMC2021 fixed CICFlowMeter and improved regenerated CICIDS2017",
                "url": "https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html",
                "notes": "Earlier fixed CICFlowMeter route; the page points to the CNS2022 release as the most recent version.",
            },
        ],
        "recommendation": (
            "Use this local packet CSV split only as a directory-label derived working dataset. "
            "For final A0 baselines or paper-grade reported IDS2017 results, use the corrected "
            "DistriNet/CNS2022 labelled flow dataset or rerun fixed CICFlowMeter plus the public "
            "labelling logic, then document how Attempted labels are handled."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/ids2017")
    parser.add_argument("--out", default="outputs/processed/ids2017_label_provenance.json")
    args = parser.parse_args()

    result = audit(args.data_root)
    write_json(result, args.out)
    print(result)


if __name__ == "__main__":
    main()
