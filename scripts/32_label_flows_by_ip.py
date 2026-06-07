#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import html
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.data.label_policy import binary_label_for
from src.data.zeek_parser import aggregate_zeek_logs, iter_zeek_records, packets_from_zeek_row
from src.data.flow_aggregator import aggregate_packets
from src.utils.io import iter_jsonl, write_json, write_jsonl


IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


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


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", xml)
    return html.unescape(text)


def _text_for_ip_file(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return _docx_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def extract_ipv4_addresses(text: str) -> list[str]:
    ips: list[str] = []
    seen: set[str] = set()
    for match in IPV4_RE.finditer(text):
        ip = match.group(0)
        parts = ip.split(".")
        if any(int(part) > 255 for part in parts):
            continue
        if ip not in seen:
            ips.append(ip)
            seen.add(ip)
    return ips


def read_malicious_ips(paths: list[Path], explicit_ips: list[str] | None = None) -> list[str]:
    ips: list[str] = []
    seen: set[str] = set()
    for ip in explicit_ips or []:
        for parsed in extract_ipv4_addresses(ip):
            if parsed not in seen:
                ips.append(parsed)
                seen.add(parsed)
    for path in paths:
        for ip in extract_ipv4_addresses(_text_for_ip_file(path)):
            if ip not in seen:
                ips.append(ip)
                seen.add(ip)
    return ips


def label_flows_by_ip(
    flows: list[dict[str, Any]],
    malicious_ips: set[str],
    attack_label: str = "Botnet",
    benign_label: str = "BENIGN",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    matched_ip_counts: Counter[str] = Counter()
    endpoint_missing = 0
    for idx, flow in enumerate(flows):
        row = dict(flow)
        src_ip = str(row.get("src_ip", ""))
        dst_ip = str(row.get("dst_ip", ""))
        endpoints = [ip for ip in (src_ip, dst_ip) if ip]
        if not endpoints:
            endpoint_missing += 1
        matched = sorted(ip for ip in endpoints if ip in malicious_ips)
        if matched:
            label = attack_label
            for ip in matched:
                matched_ip_counts[ip] += 1
        else:
            label = benign_label
        row["label"] = label
        row["raw_label"] = label
        row["binary_label"] = binary_label_for(label)
        row["label_source"] = "malicious_ip_endpoint"
        row["matched_malicious_ips"] = matched
        row.setdefault("source_flow_id", row.get("flow_id", idx))
        labeled.append(row)

    label_counts = Counter(str(row.get("label", "UNKNOWN")) for row in labeled)
    binary_counts = Counter(str(row.get("binary_label", "UNKNOWN")) for row in labeled)
    stats = {
        "input_flows": len(flows),
        "labeled_flows": len(labeled),
        "label_counts": dict(sorted(label_counts.items())),
        "binary_counts": dict(sorted(binary_counts.items())),
        "malicious_ip_count": len(malicious_ips),
        "matched_malicious_ips": dict(sorted(matched_ip_counts.items())),
        "endpoint_missing": endpoint_missing,
    }
    return labeled, stats


def label_one_flow_by_ip(
    flow: dict[str, Any],
    malicious_ips: set[str],
    attack_label: str = "Botnet",
    benign_label: str = "BENIGN",
) -> dict[str, Any]:
    row = dict(flow)
    src_ip = str(row.get("src_ip", ""))
    dst_ip = str(row.get("dst_ip", ""))
    endpoints = [ip for ip in (src_ip, dst_ip) if ip]
    matched = sorted(ip for ip in endpoints if ip in malicious_ips)
    label = attack_label if matched else benign_label
    row["label"] = label
    row["raw_label"] = label
    row["binary_label"] = binary_label_for(label)
    row["label_source"] = "malicious_ip_endpoint"
    row["matched_malicious_ips"] = matched
    return row


def _flows_from_zeek_record(row: dict[str, Any]) -> list[dict[str, Any]]:
    packets = packets_from_zeek_row(row)
    if not packets:
        return []
    return aggregate_packets(packets)


def stream_label_zeek_flows_by_ip(
    paths: list[Path],
    malicious_ips: set[str],
    attack_label: str = "Botnet",
    benign_label: str = "BENIGN",
    max_flows_per_label: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    binary_counts: Counter[str] = Counter()
    matched_ip_counts: Counter[str] = Counter()
    input_rows = 0
    candidate_flows = 0
    capped_flows = 0
    empty_flow_rows = 0

    for path in paths:
        for record in iter_zeek_records(path):
            input_rows += 1
            flows = _flows_from_zeek_record(record)
            if not flows:
                empty_flow_rows += 1
                continue
            for flow in flows:
                candidate_flows += 1
                row = label_one_flow_by_ip(flow, malicious_ips, attack_label, benign_label)
                label = str(row.get("label", "UNKNOWN"))
                if max_flows_per_label is not None and label_counts[label] >= max_flows_per_label:
                    capped_flows += 1
                    continue
                row.setdefault("source_flow_id", row.get("flow_id", candidate_flows))
                selected.append(row)
                label_counts[label] += 1
                binary_counts[str(row.get("binary_label", "UNKNOWN"))] += 1
                for ip in row.get("matched_malicious_ips", []):
                    matched_ip_counts[str(ip)] += 1
            if max_flows_per_label is not None and label_counts[attack_label] >= max_flows_per_label and label_counts[benign_label] >= max_flows_per_label:
                break
        if max_flows_per_label is not None and label_counts[attack_label] >= max_flows_per_label and label_counts[benign_label] >= max_flows_per_label:
            break

    stats = {
        "input_rows": input_rows,
        "candidate_flows": candidate_flows,
        "labeled_flows": len(selected),
        "capped_flows": capped_flows,
        "empty_flow_rows": empty_flow_rows,
        "label_counts": dict(sorted(label_counts.items())),
        "binary_counts": dict(sorted(binary_counts.items())),
        "malicious_ip_count": len(malicious_ips),
        "matched_malicious_ips": dict(sorted(matched_ip_counts.items())),
        "max_flows_per_label": max_flows_per_label,
    }
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Label Zeek/flow records by whether either endpoint is a known malicious IP.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--zeek_logs", nargs="+", help="Zeek logs to aggregate before labeling.")
    input_group.add_argument("--flows", nargs="+", help="Existing JSONL flow files to label.")
    parser.add_argument("--malicious_ips_file", nargs="+", default=[], help="Text or docx files containing malicious IPs.")
    parser.add_argument("--malicious_ip", action="append", default=[], help="Explicit malicious IP. Can be repeated.")
    parser.add_argument("--attack_label", default="Botnet")
    parser.add_argument("--benign_label", default="BENIGN")
    parser.add_argument(
        "--max_flows_per_label",
        type=int,
        default=None,
        help="Stream-select at most this many flows per output label. Useful for large Zeek conn.log smoke runs.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats_out", default=None)
    args = parser.parse_args()

    ip_paths = _expand_inputs(args.malicious_ips_file, {".txt", ".csv", ".md", ".docx"})
    malicious_ips = set(read_malicious_ips(ip_paths, args.malicious_ip))
    if not malicious_ips:
        raise ValueError("no malicious IPs were provided or parsed")

    if args.zeek_logs:
        zeek_paths = _expand_inputs(args.zeek_logs, {".log", ".tsv", ".json", ".jsonl", ".ndjson"})
        if args.max_flows_per_label is not None:
            labeled, stats = stream_label_zeek_flows_by_ip(
                zeek_paths,
                malicious_ips,
                attack_label=args.attack_label,
                benign_label=args.benign_label,
                max_flows_per_label=args.max_flows_per_label,
            )
            flows = None
        else:
            flows = aggregate_zeek_logs(zeek_paths)
            labeled, stats = label_flows_by_ip(
                flows,
                malicious_ips,
                attack_label=args.attack_label,
                benign_label=args.benign_label,
            )
        input_paths = zeek_paths
        input_type = "zeek_logs"
    else:
        flow_paths = _expand_inputs(args.flows or [], {".jsonl"})
        flows = []
        for path in flow_paths:
            flows.extend(dict(row) for row in iter_jsonl(path))
        input_paths = flow_paths
        input_type = "flows"
        labeled, stats = label_flows_by_ip(
            flows,
            malicious_ips,
            attack_label=args.attack_label,
            benign_label=args.benign_label,
        )
    stats.update(
        {
            "input_type": input_type,
            "inputs": [str(path) for path in input_paths],
            "malicious_ips_file": [str(path) for path in ip_paths],
            "attack_label": args.attack_label,
            "benign_label": args.benign_label,
        }
    )
    write_jsonl(labeled, args.out)
    if args.stats_out:
        write_json(stats, args.stats_out)
    print(stats)


if __name__ == "__main__":
    main()
