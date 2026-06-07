from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from src.data.label_policy import binary_label_for, merged_cicids_label, normalize_label


CANONICAL_ATTACK_FAMILIES = {
    "BENIGN",
    "DoS",
    "DDoS",
    "Probe",
    "BruteForce",
    "Botnet",
    "WebAttack",
    "Infiltration",
    "Exfiltration",
    "Generic",
    "Exploit",
    "Fuzzers",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
    "OtherAttack",
    "Heartbleed",
}

KNOWN_DAYS = {
    "monday": "Monday",
    "tuesday": "Tuesday",
    "wednesday": "Wednesday",
    "thursday": "Thursday",
    "friday": "Friday",
    "saturday": "Saturday",
    "sunday": "Sunday",
}


def normalize_attack_family(label: object) -> str:
    normalized = normalize_label(label)
    if normalized in CANONICAL_ATTACK_FAMILIES:
        return normalized
    merged = merged_cicids_label(normalized)
    if merged == "PortScan":
        return "Probe"
    if merged == "Bot":
        return "Botnet"
    if merged in CANONICAL_ATTACK_FAMILIES:
        return merged
    if binary_label_for(merged) == "BENIGN":
        return "BENIGN"
    return "OtherAttack"


def infer_day(flow: dict[str, Any]) -> str:
    raw = str(flow.get("day") or "").strip()
    if raw:
        normalized = raw.lower()
        if normalized in KNOWN_DAYS:
            return KNOWN_DAYS[normalized]
        for day, canonical in KNOWN_DAYS.items():
            if normalized.startswith(day):
                return canonical
    source = str(flow.get("label_source_file") or flow.get("source_file") or flow.get("meta", {}).get("source_file", ""))
    lowered = source.lower()
    for day, canonical in KNOWN_DAYS.items():
        if day in lowered:
            return canonical
    return "unknown"


def normalize_flow_schema(flow: dict[str, Any], *, dataset: str = "CICIDS2017") -> dict[str, Any]:
    label = normalize_label(flow.get("label") or flow.get("attack_family") or "UNKNOWN")
    attack_family = normalize_attack_family(flow.get("attack_family") or label)
    proto = str(flow.get("proto") or flow.get("protocol") or "tcp").lower()
    lens = [int(float(item)) for item in flow.get("lens") or []]
    dirs = [1 if bool(item) else 0 for item in flow.get("dirs") or []]
    tss = [float(item) for item in flow.get("tss") or []]
    if flow.get("iats"):
        iats = [float(item) for item in flow.get("iats") or []]
    elif tss:
        iats = [0.0] + [max(0.0, cur - prev) for prev, cur in zip(tss[:-1], tss[1:])]
    else:
        iats = []
    meta = dict(flow.get("meta") or {})
    for key in ("label_source_file", "source_file", "raw_label", "uids"):
        if flow.get(key) is not None:
            meta[key] = flow.get(key)
    row = {
        "flow_id": str(flow.get("flow_id")),
        "dataset": str(flow.get("dataset") or dataset),
        "src_ip": str(flow.get("src_ip") or ""),
        "dst_ip": str(flow.get("dst_ip") or ""),
        "src_port": str(flow.get("src_port") or "0"),
        "dst_port": str(flow.get("dst_port") or "0"),
        "proto": proto,
        "protocol": proto.upper(),
        "lens": lens,
        "dirs": dirs,
        "tss": tss,
        "iats": iats,
        "label": label,
        "binary_label": binary_label_for(label),
        "attack_family": attack_family,
        "day": infer_day(flow),
        "split": flow.get("split"),
        "start_ts": flow.get("start_ts"),
        "end_ts": flow.get("end_ts"),
        "duration": flow.get("duration"),
        "packet_count": int(flow.get("packet_count") or len(lens)),
        "service_key": flow.get("service_key") or [str(flow.get("dst_ip") or ""), str(flow.get("dst_port") or "0"), proto.upper()],
        "meta": meta,
    }
    return row


def normalize_flows(flows: list[dict[str, Any]], *, dataset: str = "CICIDS2017") -> list[dict[str, Any]]:
    return [normalize_flow_schema(flow, dataset=dataset) for flow in flows]


def dataset_report(flows: list[dict[str, Any]], *, dataset: str = "CICIDS2017", source: str | None = None) -> dict[str, Any]:
    normalized = normalize_flows(flows, dataset=dataset)
    label_counts = Counter(str(flow.get("label")) for flow in normalized)
    family_counts = Counter(str(flow.get("attack_family")) for flow in normalized)
    binary_counts = Counter(str(flow.get("binary_label")) for flow in normalized)
    day_counts = Counter(str(flow.get("day")) for flow in normalized)
    proto_counts = Counter(str(flow.get("proto")) for flow in normalized)
    packet_lengths = [len(flow.get("lens") or []) for flow in normalized]
    missing = {
        "empty_lens": sum(1 for flow in normalized if not flow.get("lens")),
        "empty_dirs": sum(1 for flow in normalized if not flow.get("dirs")),
        "empty_tss": sum(1 for flow in normalized if not flow.get("tss")),
        "unknown_day": sum(1 for flow in normalized if str(flow.get("day")) == "unknown"),
    }
    return {
        "dataset": dataset,
        "source": source,
        "num_flows": len(normalized),
        "label_counts": dict(sorted(label_counts.items())),
        "attack_family_counts": dict(sorted(family_counts.items())),
        "binary_counts": dict(sorted(binary_counts.items())),
        "day_counts": dict(sorted(day_counts.items())),
        "proto_counts": dict(sorted(proto_counts.items())),
        "packet_count": {
            "avg": float(sum(packet_lengths) / len(packet_lengths)) if packet_lengths else 0.0,
            "p50": float(statistics.median(packet_lengths)) if packet_lengths else 0.0,
            "max": int(max(packet_lengths)) if packet_lengths else 0,
        },
        "missing_counts": missing,
        "schema_version": "flowprim_flow_schema_v2",
    }


def label_alignment_report(
    *,
    dataset: str,
    total_zeek_flows: int,
    matched_flows: int,
    unmatched_flows: int,
    ambiguous_matches: int = 0,
    time_deltas: list[float] | None = None,
    matched_flows_rows: list[dict[str, Any]] | None = None,
    dropped_reason_counts: dict[str, int] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    deltas = sorted(float(item) for item in (time_deltas or []))
    by_family = Counter()
    if matched_flows_rows:
        by_family.update(str(row.get("attack_family") or row.get("label") or "UNKNOWN") for row in matched_flows_rows)
    total = int(total_zeek_flows)
    ambiguous_rate = (float(ambiguous_matches) / float(matched_flows)) if matched_flows else 0.0
    p95_idx = int(round(0.95 * (len(deltas) - 1))) if deltas else 0
    return {
        "dataset": dataset,
        "total_zeek_flows": total,
        "matched_flows": int(matched_flows),
        "unmatched_flows": int(unmatched_flows),
        "ambiguous_matches": int(ambiguous_matches),
        "match_rate": (float(matched_flows) / float(total)) if total else 0.0,
        "ambiguous_rate": ambiguous_rate,
        "time_delta_mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        "time_delta_p95": float(deltas[p95_idx]) if deltas else 0.0,
        "match_by_attack_family": dict(sorted(by_family.items())),
        "dropped_reason_counts": dict(sorted((dropped_reason_counts or {}).items())),
        "notes": list(notes or []),
        "schema_version": "flowprim_label_alignment_report_v1",
    }


def split_coverage_report(flows: list[dict[str, Any]], splits: dict[str, list[str]]) -> dict[str, Any]:
    by_id = {str(flow.get("flow_id")): flow for flow in flows}
    all_labels = sorted({str(flow.get("attack_family") or flow.get("label") or "UNKNOWN") for flow in flows})
    report: dict[str, Any] = {"per_split": {}, "missing_by_split": {}}
    for split_name in ("train", "val", "test"):
        counts = Counter(
            str(by_id[flow_id].get("attack_family") or by_id[flow_id].get("label") or "UNKNOWN")
            for flow_id in splits.get(split_name, [])
            if flow_id in by_id
        )
        report["per_split"][split_name] = dict(sorted(counts.items()))
        report["missing_by_split"][split_name] = [label for label in all_labels if counts.get(label, 0) == 0]
    train_labels = set(report["per_split"].get("train", {}))
    test_labels = set(report["per_split"].get("test", {}))
    report["class_missing_in_train"] = sorted(test_labels - train_labels)
    report["class_missing_in_test"] = sorted(train_labels - test_labels)
    return report
