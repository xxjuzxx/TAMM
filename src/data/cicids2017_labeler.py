from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.label_policy import AttemptedPolicy, apply_attempted_policy, binary_label_for, normalize_label


def _norm_key(src_ip: str, src_port: str, dst_ip: str, dst_port: str, protocol: str) -> tuple[str, str, str, str, str]:
    return (str(src_ip), str(src_port), str(dst_ip), str(dst_port), str(protocol).upper())


def _reverse(key: tuple[str, str, str, str, str]) -> tuple[str, str, str, str, str]:
    src_ip, src_port, dst_ip, dst_port, protocol = key
    return (dst_ip, dst_port, src_ip, src_port, protocol)


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


def _timestamp_seconds(value: str, default: float = 0.0) -> float:
    parsed = _parse_timestamp_seconds(value)
    return default if parsed is None else parsed


def _parse_timestamp_seconds(value: str) -> float | None:
    numeric = _to_float(value, default=float("nan"))
    if numeric == numeric:
        return numeric
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S.%f %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %I:%M:%S.%f %p",
        "%d/%m/%Y %I:%M:%S %p",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def read_cicids_flow_csvs(paths: list[str | Path], attempted_policy: AttemptedPolicy = "keep") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_label = _row_value(row, "Label", "label", default=Path(path).stem)
                label = apply_attempted_policy(raw_label, attempted_policy)
                if label is None:
                    continue
                protocol = _row_value(row, "Protocol", "protocol", default="TCP")
                if protocol == "6":
                    protocol = "TCP"
                elif protocol == "17":
                    protocol = "UDP"
                timestamp_raw = _row_value(row, "Timestamp", "Flow Start", "start_ts", "frame.time_epoch", default="")
                parsed_start = _parse_timestamp_seconds(timestamp_raw)
                if parsed_start is None:
                    start = 0.0
                    timestamp_status = "missing" if not timestamp_raw else "invalid"
                else:
                    start = parsed_start
                    timestamp_status = "parsed"
                duration_raw = _to_float(_row_value(row, "Flow Duration", "duration", default="0"))
                duration = duration_raw / 1_000_000.0 if duration_raw > 10000 else duration_raw
                key = _norm_key(
                    _row_value(row, "Source IP", "src_ip", "srcip", "Src IP"),
                    _row_value(row, "Source Port", "src_port", "srcport", "Src Port"),
                    _row_value(row, "Destination IP", "dst_ip", "dstip", "Dst IP"),
                    _row_value(row, "Destination Port", "dst_port", "dstport", "Dst Port"),
                    protocol,
                )
                rows.append(
                    {
                        "key": key,
                        "reverse_key": _reverse(key),
                        "start_ts": start,
                        "end_ts": start + duration,
                        "timestamp_parse_status": timestamp_status,
                        "label": normalize_label(label),
                        "raw_label": normalize_label(raw_label),
                        "source_file": str(path),
                    }
                )
    return rows


def label_flows(
    flows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    tolerance_seconds: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    index: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in label_rows:
        index[row["key"]].append(row)
        index[row["reverse_key"]].append(row)
    labeled: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous_matches = 0
    time_deltas: list[float] = []
    for flow in flows:
        key = _norm_key(flow["src_ip"], flow["src_port"], flow["dst_ip"], flow["dst_port"], flow.get("protocol", "TCP"))
        start_ts = float(flow.get("start_ts") or 0.0)
        candidates = index.get(key, [])
        best = None
        best_delta = float("inf")
        valid_matches = 0
        for candidate in candidates:
            timestamp_status = candidate.get("timestamp_parse_status")
            if timestamp_status == "invalid":
                continue
            if timestamp_status == "missing" or (timestamp_status is None and candidate["start_ts"] <= 0):
                delta = 0.0
            elif candidate["start_ts"] - tolerance_seconds <= start_ts <= candidate["end_ts"] + tolerance_seconds:
                delta = abs(start_ts - candidate["start_ts"])
            else:
                continue
            valid_matches += 1
            if delta < best_delta:
                best = candidate
                best_delta = delta
        if best is None:
            unmatched.append(flow)
            continue
        if valid_matches > 1:
            ambiguous_matches += 1
        row = dict(flow)
        row["label"] = best["label"]
        row["raw_label"] = best.get("raw_label", best["label"])
        row["binary_label"] = binary_label_for(best["label"])
        row["label_source_file"] = best["source_file"]
        row["label_time_delta"] = float(best_delta if best_delta != float("inf") else 0.0)
        time_deltas.append(row["label_time_delta"])
        labeled.append(row)
    stats = {
        "num_flows": len(flows),
        "matched": len(labeled),
        "unmatched": len(unmatched),
        "ambiguous_matches": ambiguous_matches,
        "ambiguous_rate": (ambiguous_matches / len(labeled)) if labeled else 0.0,
        "match_rate": (len(labeled) / len(flows)) if flows else 0.0,
        "time_delta_mean": (sum(time_deltas) / len(time_deltas)) if time_deltas else 0.0,
        "time_delta_p95": sorted(time_deltas)[int(round(0.95 * (len(time_deltas) - 1)))] if time_deltas else 0.0,
        "label_timestamp_status_counts": dict(
            sorted(Counter(str(row.get("timestamp_parse_status", "unknown")) for row in label_rows).items())
        ),
        "label_counts": dict(sorted(Counter(row["label"] for row in labeled).items())),
    }
    return labeled, unmatched, stats
