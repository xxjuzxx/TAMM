#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.pipeline.common import ROOT, command_record, ensure_dirs, read_yaml, write_csv, write_json, write_md


DAY_KEYS = {
    "monday": "monday",
    "tuesday": "tuesday",
    "wednesday": "wednesday",
    "thursday": "thursday",
    "friday": "friday",
}


def _day_from_name(name: str) -> str:
    lower = name.lower()
    for key, value in DAY_KEYS.items():
        if key in lower:
            return value
    return "unknown"


def _sha1_prefix(path: Path, limit: int = 1_048_576) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def _csv_summary(path: Path) -> dict[str, Any]:
    label_counter: Counter[str] = Counter()
    protocol_counter: Counter[str] = Counter()
    rows = 0
    first_timestamp = ""
    last_timestamp = ""
    header: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        for row in reader:
            rows += 1
            label = str(row.get("Label") or row.get("label") or "").strip() or "UNKNOWN"
            proto = str(row.get("Protocol") or row.get("protocol") or "").strip() or "UNKNOWN"
            ts = str(row.get("Timestamp") or row.get("timestamp") or "").strip()
            if ts and not first_timestamp:
                first_timestamp = ts
            if ts:
                last_timestamp = ts
            label_counter[label] += 1
            protocol_counter[proto] += 1
    return {
        "csv_rows": rows,
        "csv_columns": len(header),
        "label_counts": dict(sorted(label_counter.items())),
        "protocol_counts": dict(sorted(protocol_counter.items())),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def _artifact_status(root: Path) -> dict[str, Any]:
    unknown_dir = root / "paper_icdm_applied_2026" / "experiments" / "unknown"
    token_dir = unknown_dir / "tokens_category"
    split_files = sorted(unknown_dir.glob("splits_leave_one_*_seed*.json"))
    token_files = sorted(token_dir.glob("*.pt"))
    vocab_files = sorted(token_dir.glob("*_vocab.json"))
    result_dir = root / "results" / "primitive_categories"
    return {
        "existing_flow_jsonl": str(root / "outputs" / "processed" / "ccfa" / "cicids2017_interim_labeled_flows.jsonl"),
        "existing_zeek_aligned_flow_jsonl": str(root / "outputs" / "processed" / "ccfa" / "cicids2017_zeek_aligned_labeled_flows.jsonl"),
        "unknown_split_count": len(split_files),
        "category_token_corpus_count": len(token_files),
        "category_vocab_count": len(vocab_files),
        "primitive_category_result_csv_count": len(list(result_dir.glob("*.csv"))) if result_dir.exists() else 0,
        "missing_expected_token_corpora": _missing_expected_tokens(token_dir),
    }


def _missing_expected_tokens(token_dir: Path) -> list[str]:
    attacks = ["botnet", "ddos", "probe", "webattack", "bruteforce"]
    seeds = [42, 43, 44]
    missing: list[str] = []
    for attack in attacks:
        for seed in seeds:
            path = token_dir / f"cicids2017_leave_one_{attack}_anomaly_seed{seed}_a3_full_rhythm.pt"
            if not path.exists():
                missing.append(str(path))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory FlowPrim raw data, labels, splits, token corpora, and existing results.")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--hash-prefix", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ensure_dirs()
    cfg = read_yaml(ROOT / args.config)
    primary = cfg["datasets"]["cicids2017_raw"]
    pcap_dir = Path(primary["pcap_dir"])
    csv_dir = Path(primary["corrected_csv_dir"])
    excluded = Path(primary["excluded_convenience_cache"])
    out_dir = ROOT / args.output_dir
    report_dir = ROOT / args.reports_dir

    pcap_paths = sorted([*pcap_dir.glob("*.pcap"), *pcap_dir.glob("*.pcapng")])
    csv_paths = sorted([*csv_dir.glob("*.csv"), *csv_dir.glob("*.CSV")])
    csv_by_day = {_day_from_name(path.name): path for path in csv_paths}

    csv_summaries = {path: _csv_summary(path) for path in csv_paths}
    rows: list[dict[str, Any]] = []
    for pcap in pcap_paths:
        day = _day_from_name(pcap.name)
        csv_path = csv_by_day.get(day)
        csv_summary = csv_summaries.get(csv_path, {}) if csv_path else {}
        rows.append(
            {
                "dataset": "CICIDS2017",
                "day": day,
                "pcap_path": str(pcap),
                "pcap_exists": pcap.exists(),
                "pcap_size_bytes": pcap.stat().st_size if pcap.exists() else "",
                "pcap_sha1_first_1mb": _sha1_prefix(pcap) if args.hash_prefix and pcap.exists() else "",
                "corrected_csv_path": str(csv_path) if csv_path else "",
                "corrected_csv_exists": bool(csv_path and csv_path.exists()),
                "corrected_csv_size_bytes": csv_path.stat().st_size if csv_path and csv_path.exists() else "",
                "csv_rows": csv_summary.get("csv_rows", ""),
                "csv_columns": csv_summary.get("csv_columns", ""),
                "first_timestamp": csv_summary.get("first_timestamp", ""),
                "last_timestamp": csv_summary.get("last_timestamp", ""),
                "label_counts_json": csv_summary.get("label_counts", {}),
                "protocol_counts_json": csv_summary.get("protocol_counts", {}),
                "source_policy": primary["source_policy"],
                "ready_ids2017_used": False,
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
            }
        )

    csv_unmatched = [path for path in csv_paths if _day_from_name(path.name) not in {_day_from_name(p.name) for p in pcap_paths}]
    missing_rows: list[dict[str, Any]] = []
    for day in sorted(set(DAY_KEYS.values())):
        pcap_exists = any(_day_from_name(path.name) == day for path in pcap_paths)
        csv_exists = day in csv_by_day
        if not pcap_exists or not csv_exists:
            missing_rows.append(
                {
                    "component": "raw_day_pair",
                    "day": day,
                    "status": "missing",
                    "pcap_exists": pcap_exists,
                    "corrected_csv_exists": csv_exists,
                    "reason": "raw PCAP and corrected CSV must both exist for full raw rerun",
                }
            )
    for path in csv_unmatched:
        missing_rows.append({"component": "corrected_csv", "path": str(path), "status": "unmatched_day"})

    artifact_status = _artifact_status(ROOT)
    for path in artifact_status["missing_expected_token_corpora"]:
        missing_rows.append({"component": "category_token_corpus", "path": path, "status": "missing"})
    if excluded.exists():
        missing_rows.append(
            {
                "component": "excluded_data_source",
                "path": str(excluded),
                "status": "excluded_by_policy",
                "reason": "per-class ready/ids2017 cache is not used for this reproducible raw-data pipeline",
            }
        )

    write_csv(out_dir / "pcap_manifest.csv", rows)
    write_csv(out_dir / "missing_data.csv", missing_rows)
    feature_rows = [
        {
            "artifact": "existing_cicids2017_interim_labeled_flows",
            "path": artifact_status["existing_flow_jsonl"],
            "exists": Path(artifact_status["existing_flow_jsonl"]).exists(),
            "role": "source flow cache for current category-token corpora",
        },
        {
            "artifact": "existing_cicids2017_zeek_aligned_labeled_flows",
            "path": artifact_status["existing_zeek_aligned_flow_jsonl"],
            "exists": Path(artifact_status["existing_zeek_aligned_flow_jsonl"]).exists(),
            "role": "large Zeek-aligned flow artifact",
        },
        {
            "artifact": "leave_one_unknown_splits",
            "path": str(ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown"),
            "exists": artifact_status["unknown_split_count"] > 0,
            "count": artifact_status["unknown_split_count"],
            "role": "leave-one unknown split definitions",
        },
        {
            "artifact": "category_token_corpora",
            "path": str(ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown" / "tokens_category"),
            "exists": artifact_status["category_token_corpus_count"] > 0,
            "count": artifact_status["category_token_corpus_count"],
            "role": "train-only profile/structural behavior token corpora",
        },
        {
            "artifact": "primitive_category_results",
            "path": str(ROOT / "results" / "primitive_categories"),
            "exists": artifact_status["primitive_category_result_csv_count"] > 0,
            "count": artifact_status["primitive_category_result_csv_count"],
            "role": "current profile/structural primitive category metrics",
        },
    ]
    write_csv(out_dir / "feature_manifest.csv", feature_rows)
    write_json(out_dir / "inventory_summary.json", {"command": command_record(sys.argv), "artifact_status": artifact_status, "raw_rows": rows})

    label_lines: list[str] = []
    for row in rows:
        label_lines.append(f"- {row['day']}: PCAP {row['pcap_size_bytes']} bytes, CSV rows {row['csv_rows']}, labels {row['label_counts_json']}")
    write_md(
        report_dir / "experiment_inventory.md",
        [
            "# FlowPrim Experiment Inventory",
            "",
            f"Generated at: {command_record(sys.argv)['created_at']}",
            "",
            "## Primary Data Policy",
            "",
            f"- Primary PCAP directory: `{pcap_dir}`",
            f"- Corrected CSV directory: `{csv_dir}`",
            f"- Excluded convenience cache: `{excluded}`",
            "- The excluded cache is not used by the pipeline or result summaries.",
            "- Raw IP, absolute timestamp, and complete five-tuple fields are restricted to joins, splits, deduplication, grouping, and audit metadata.",
            "",
            "## Raw Day Pairs",
            "",
            *label_lines,
            "",
            "## Existing Reproducible Artifacts",
            "",
            f"- Leave-one unknown split files: {artifact_status['unknown_split_count']}",
            f"- Category token corpora: {artifact_status['category_token_corpus_count']}",
            f"- Category vocab files: {artifact_status['category_vocab_count']}",
            f"- Primitive category result CSV files: {artifact_status['primitive_category_result_csv_count']}",
        ],
    )
    missing_lines = [
        f"- `{row.get('component')}` {row.get('day') or row.get('path') or ''}: {row.get('status')} {row.get('reason', '')}".rstrip()
        for row in missing_rows
    ]
    if not missing_lines:
        missing_lines = ["- No missing raw PCAP/corrected CSV day pairs were found. The per-class ready cache is still excluded by policy."]
    write_md(
        report_dir / "missing_data_report.md",
        [
            "# Missing / Skipped Data Report",
            "",
            "This report records unavailable or intentionally excluded inputs. Missing rows are not filled or interpolated.",
            "",
            "## Entries",
            "",
            *missing_lines,
        ],
    )
    print(out_dir / "pcap_manifest.csv")
    print(report_dir / "experiment_inventory.md")


if __name__ == "__main__":
    main()
