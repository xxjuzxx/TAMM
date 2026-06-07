from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
from pathlib import Path
from typing import Any

from src.data.cicids2017_labeler import label_flows
from src.data.dataset_adapter import dataset_report, label_alignment_report, normalize_flows
from src.data.label_policy import binary_label_for
from src.data.label_policy import apply_attempted_policy


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


def _normalize_protocol(value: str) -> str:
    protocol = str(value or "TCP").strip().upper()
    if protocol == "6":
        return "TCP"
    if protocol == "17":
        return "UDP"
    if protocol == "0":
        return "0"
    return protocol


def _norm_key(src_ip: str, src_port: str, dst_ip: str, dst_port: str, protocol: str) -> tuple[str, str, str, str, str]:
    return (str(src_ip), str(src_port), str(dst_ip), str(dst_port), _normalize_protocol(protocol))


def _reverse(key: tuple[str, str, str, str, str]) -> tuple[str, str, str, str, str]:
    src_ip, src_port, dst_ip, dst_port, protocol = key
    return (dst_ip, dst_port, src_ip, src_port, protocol)


def _estimate_time_offset_seconds(
    flows: list[dict[str, Any]],
    retained_candidates: list[dict[str, Any]],
    *,
    max_pairs_per_key: int = 120,
    max_keys: int = 5000,
) -> tuple[float, dict[str, Any]]:
    """Estimate the Zeek-to-IDS2018 CSV time offset without using labels.

    IDS2018 CSV timestamps are wall-clock strings while PCAP/Zeek timestamps
    are epoch seconds. Some host-level PCAP shards contain repeated management
    flows with identical five-tuples, so a plain median over all pairwise
    deltas can be dominated by ambiguous repeated keys. This estimator uses
    five-tuple matches only for label alignment and chooses the most common
    hour-rounded offset across bounded per-key samples.
    """

    flow_by_key: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for flow in flows:
        key = _flow_key(flow)
        flow_by_key[key].append(flow)
        flow_by_key[_reverse(key)].append(flow)
    for values in flow_by_key.values():
        values.sort(key=lambda item: (float(item.get("start_ts") or 0.0), str(item.get("flow_id") or "")))

    candidate_by_key: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in retained_candidates:
        candidate_by_key[candidate["key"]].append(candidate)
        candidate_by_key[candidate["reverse_key"]].append(candidate)
    for values in candidate_by_key.values():
        values.sort(key=lambda item: float(item.get("start_ts") or 0.0))

    offset_votes: Counter[int] = Counter()
    minute_votes_by_hour: dict[int, Counter[int]] = defaultdict(Counter)
    sampled_keys = 0
    sampled_pairs = 0
    skipped_large_keys = 0
    for key in sorted(set(flow_by_key) & set(candidate_by_key)):
        if sampled_keys >= max_keys:
            break
        flows_for_key = [flow for flow in flow_by_key[key] if float(flow.get("start_ts") or 0.0) > 0]
        candidates_for_key = [
            candidate
            for candidate in candidate_by_key[key]
            if float(candidate.get("start_ts") or 0.0) > 0
        ]
        if not flows_for_key or not candidates_for_key:
            continue
        pair_count = len(flows_for_key) * len(candidates_for_key)
        if pair_count > max_pairs_per_key:
            skipped_large_keys += 1
            # Still sample deterministically from the head/tail to avoid losing
            # common attack keys that repeat heavily in a shard.
            flows_for_key = [*flows_for_key[:4], *flows_for_key[-4:]]
            candidates_for_key = [*candidates_for_key[:4], *candidates_for_key[-4:]]
        sampled_keys += 1
        for flow in flows_for_key:
            flow_start = float(flow.get("start_ts") or 0.0)
            for candidate in candidates_for_key:
                delta = flow_start - float(candidate.get("start_ts") or 0.0)
                if abs(delta) > 36 * 3600:
                    continue
                hour = int(round(delta / 3600.0) * 3600)
                minute = int(round(delta / 60.0) * 60)
                offset_votes[hour] += 1
                minute_votes_by_hour[hour][minute] += 1
                sampled_pairs += 1

    if not offset_votes:
        return 0.0, {
            "offset_estimator": "hour_mode",
            "offset_estimator_status": "no_votes",
            "offset_vote_counts_top": {},
            "offset_sampled_keys": sampled_keys,
            "offset_sampled_pairs": sampled_pairs,
            "offset_skipped_large_keys": skipped_large_keys,
            "offset_max_keys": max_keys,
            "offset_max_pairs_per_key": max_pairs_per_key,
        }

    best_hour, best_hour_votes = sorted(offset_votes.items(), key=lambda item: (-item[1], abs(item[0]), item[0]))[0]
    minute_counter = minute_votes_by_hour[best_hour]
    best_minute, best_minute_votes = sorted(minute_counter.items(), key=lambda item: (-item[1], abs(item[0] - best_hour), item[0]))[0]
    return float(best_minute), {
        "offset_estimator": "five_tuple_hour_mode_then_minute_mode",
        "offset_estimator_status": "ok",
        "offset_vote_counts_top": dict(offset_votes.most_common(10)),
        "offset_best_hour_seconds": int(best_hour),
        "offset_best_hour_votes": int(best_hour_votes),
        "offset_best_minute_seconds": int(best_minute),
        "offset_best_minute_votes": int(best_minute_votes),
        "offset_sampled_keys": sampled_keys,
        "offset_sampled_pairs": sampled_pairs,
        "offset_skipped_large_keys": skipped_large_keys,
        "offset_max_keys": max_keys,
        "offset_max_pairs_per_key": max_pairs_per_key,
    }


def _parse_ids2018_timestamp(value: str) -> tuple[float, str]:
    raw = str(value or "").strip()
    if not raw:
        return 0.0, "missing"
    numeric = _to_float(raw, default=float("nan"))
    if numeric == numeric:
        return numeric, "parsed_numeric"
    # IDS2018 public CSVs use day/month/year, e.g. 20/02/2018.
    for fmt in (
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %I:%M:%S.%f %p",
        "%d/%m/%Y %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).timestamp(), "parsed"
        except ValueError:
            pass
    return 0.0, "invalid"


def read_ids2018_flow_csvs(paths: list[str | Path], attempted_policy: str = "keep") -> list[dict[str, Any]]:
    """Read CSE-CIC-IDS2018 CICFlowMeter CSV labels.

    Most public IDS2018 CSV files omit source IP/port and destination IP, so
    they cannot support exact flow-level joins. Rows with complete five-tuples
    are retained for IDS2017-equivalent PCAP-derived alignment; incomplete rows
    are reported as unusable for exact joins by downstream stats.
    """

    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_label = _row_value(row, "Label", "label", default=Path(path).stem)
                label = apply_attempted_policy(raw_label, attempted_policy)  # type: ignore[arg-type]
                if label is None:
                    continue
                protocol = _normalize_protocol(_row_value(row, "Protocol", "protocol", default="TCP"))
                timestamp_raw = _row_value(row, "Timestamp", "Flow Start", "start_ts", default="")
                start, timestamp_status = _parse_ids2018_timestamp(timestamp_raw)
                duration_raw = _to_float(_row_value(row, "Flow Duration", "duration", default="0"))
                duration = duration_raw / 1_000_000.0 if duration_raw > 10000 else duration_raw
                src_ip = _row_value(row, "Src IP", "Source IP", "src_ip", "srcip")
                dst_ip = _row_value(row, "Dst IP", "Destination IP", "dst_ip", "dstip")
                src_port = _row_value(row, "Src Port", "Source Port", "src_port", "srcport")
                dst_port = _row_value(row, "Dst Port", "Destination Port", "dst_port", "dstport")
                key = _norm_key(src_ip, src_port, dst_ip, dst_port, protocol)
                complete_key = bool(src_ip and dst_ip and src_port and dst_port)
                rows.append(
                    {
                        "key": key,
                        "reverse_key": _reverse(key),
                        "start_ts": start,
                        "end_ts": start + duration,
                        "timestamp_parse_status": timestamp_status,
                        "label": _canonical_ids2018_family(label),
                        "raw_label": raw_label,
                        "source_file": str(path),
                        "ids2018_complete_five_tuple": complete_key,
                    }
                )
    return rows


def _canonical_ids2018_family(label: object) -> str:
    raw = str(label or "").strip()
    norm = raw.lower().replace("-", "").replace("_", "").replace(" ", "")
    if norm in {"benign", "normal"}:
        return "BENIGN"
    if "bot" in norm:
        return "Botnet"
    if "ddos" in norm:
        return "DDoS"
    if norm == "dos" or norm.startswith("dos"):
        return "DoS"
    if "bruteforce" in norm or "brute" in norm or "ftp" in norm or "ssh" in norm:
        return "BruteForce"
    if "web" in norm or "xss" in norm or "sql" in norm:
        return "WebAttack"
    if "infilteration" in norm or "infiltration" in norm:
        return "Infiltration"
    return raw or "UNKNOWN"


def _normalize_ids2018_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        family = _canonical_ids2018_family(item.get("label") or item.get("raw_label"))
        item["label"] = "BENIGN" if family == "BENIGN" else family
        item["raw_label"] = item.get("raw_label") or item["label"]
        out.append(item)
    return out


def _flow_key(flow: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return _norm_key(
        str(flow.get("src_ip") or ""),
        str(flow.get("src_port") or "0"),
        str(flow.get("dst_ip") or ""),
        str(flow.get("dst_port") or "0"),
        str(flow.get("protocol") or flow.get("proto") or "TCP"),
    )


def _label_flows_time_indexed(
    flows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    *,
    tolerance_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Join Zeek flows to label rows using five-tuples and indexed time windows."""

    buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(label_rows):
        item = dict(row)
        item["_candidate_id"] = idx
        keys = [item["key"]]
        if item["reverse_key"] != item["key"]:
            keys.append(item["reverse_key"])
        for key in keys:
            buckets[key].append(item)

    starts_by_key: dict[tuple[str, str, str, str, str], list[float]] = {}
    max_duration_by_key: dict[tuple[str, str, str, str, str], float] = {}
    for key, rows in buckets.items():
        rows.sort(key=lambda item: (float(item.get("start_ts") or 0.0), float(item.get("end_ts") or 0.0), int(item["_candidate_id"])))
        starts_by_key[key] = [float(item.get("start_ts") or 0.0) for item in rows]
        max_duration_by_key[key] = max(
            max(0.0, float(item.get("end_ts") or item.get("start_ts") or 0.0) - float(item.get("start_ts") or 0.0))
            for item in rows
        )

    labeled: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous_matches = 0
    time_deltas: list[float] = []
    window_sizes: list[int] = []
    valid_window_sizes: list[int] = []
    for flow in flows:
        key = _flow_key(flow)
        candidates = buckets.get(key)
        if not candidates:
            unmatched.append(flow)
            continue
        start_ts = float(flow.get("start_ts") or 0.0)
        starts = starts_by_key[key]
        max_duration = max_duration_by_key[key]
        lower = start_ts - max_duration - tolerance_seconds
        upper = start_ts + tolerance_seconds
        lo = bisect_left(starts, lower)
        hi = bisect_right(starts, upper)
        window = candidates[lo:hi]
        window_sizes.append(len(window))
        best = None
        best_delta = float("inf")
        valid_matches = 0
        seen_ids: set[int] = set()
        for candidate in window:
            candidate_id = int(candidate["_candidate_id"])
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            timestamp_status = candidate.get("timestamp_parse_status")
            if timestamp_status == "invalid":
                continue
            candidate_start = float(candidate.get("start_ts") or 0.0)
            candidate_end = float(candidate.get("end_ts") or candidate_start)
            if timestamp_status == "missing" or (timestamp_status is None and candidate_start <= 0):
                delta = 0.0
            elif candidate_start - tolerance_seconds <= start_ts <= candidate_end + tolerance_seconds:
                delta = abs(start_ts - candidate_start)
            else:
                continue
            valid_matches += 1
            if delta < best_delta:
                best = candidate
                best_delta = delta
        valid_window_sizes.append(valid_matches)
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
        "time_indexed_join": True,
        "candidate_window_mean": (sum(window_sizes) / len(window_sizes)) if window_sizes else 0.0,
        "valid_candidate_window_mean": (sum(valid_window_sizes) / len(valid_window_sizes)) if valid_window_sizes else 0.0,
        "candidate_window_max": max(window_sizes) if window_sizes else 0,
        "valid_candidate_window_max": max(valid_window_sizes) if valid_window_sizes else 0,
    }
    return labeled, unmatched, stats


def _iter_ids2018_label_rows(path: Path, attempted_policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_label = _row_value(row, "Label", "label", default=path.stem)
            label = apply_attempted_policy(raw_label, attempted_policy)  # type: ignore[arg-type]
            if label is None:
                continue
            src_ip = _row_value(row, "Src IP", "Source IP", "src_ip", "srcip")
            dst_ip = _row_value(row, "Dst IP", "Destination IP", "dst_ip", "dstip")
            src_port = _row_value(row, "Src Port", "Source Port", "src_port", "srcport")
            dst_port = _row_value(row, "Dst Port", "Destination Port", "dst_port", "dstport")
            if not (src_ip and dst_ip and src_port and dst_port):
                continue
            protocol = _normalize_protocol(_row_value(row, "Protocol", "protocol", default="TCP"))
            timestamp_raw = _row_value(row, "Timestamp", "Flow Start", "start_ts", default="")
            start, timestamp_status = _parse_ids2018_timestamp(timestamp_raw)
            duration_raw = _to_float(_row_value(row, "Flow Duration", "duration", default="0"))
            duration = duration_raw / 1_000_000.0 if duration_raw > 10000 else duration_raw
            key = _norm_key(src_ip, src_port, dst_ip, dst_port, protocol)
            rows.append(
                {
                    "key": key,
                    "reverse_key": _reverse(key),
                    "start_ts": start,
                    "end_ts": start + duration,
                    "timestamp_parse_status": timestamp_status,
                    "label": _canonical_ids2018_family(label),
                    "raw_label": raw_label,
                    "source_file": str(path),
                    "ids2018_complete_five_tuple": True,
                }
            )
    return rows


def align_ids2018_streaming_from_csvs(
    flows: list[dict[str, Any]],
    csv_paths: list[str | Path],
    *,
    tolerance_seconds: float = 2.0,
    attempted_policy: str = "keep",
    time_offset_seconds: float | None = None,
    auto_time_offset: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Memory-aware IDS2018 five-tuple/time label alignment.

    Only CSV rows whose forward or reverse five-tuple appears in the Zeek flow
    set are retained as candidates. This keeps the join exact while avoiding a
    full in-memory index over multi-million-row IDS2018 CSV files.
    """

    wanted: set[tuple[str, str, str, str, str]] = set()
    for flow in flows:
        key = _flow_key(flow)
        wanted.add(key)
        wanted.add(_reverse(key))

    retained_candidates: list[dict[str, Any]] = []
    csv_stats = {
        "csv_rows_scanned": 0,
        "csv_rows_missing_five_tuple": 0,
        "csv_candidate_rows_retained": 0,
        "csv_paths": [str(path) for path in csv_paths],
    }
    timestamp_status_counts: Counter[str] = Counter()
    raw_label_counts: Counter[str] = Counter()
    for raw_path in csv_paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                csv_stats["csv_rows_scanned"] += 1
                raw_label = _row_value(row, "Label", "label", default=path.stem)
                label = apply_attempted_policy(raw_label, attempted_policy)  # type: ignore[arg-type]
                if label is None:
                    continue
                raw_label_counts[str(raw_label)] += 1
                src_ip = _row_value(row, "Src IP", "Source IP", "src_ip", "srcip")
                dst_ip = _row_value(row, "Dst IP", "Destination IP", "dst_ip", "dstip")
                src_port = _row_value(row, "Src Port", "Source Port", "src_port", "srcport")
                dst_port = _row_value(row, "Dst Port", "Destination Port", "dst_port", "dstport")
                if not (src_ip and dst_ip and src_port and dst_port):
                    csv_stats["csv_rows_missing_five_tuple"] += 1
                    continue
                protocol = _normalize_protocol(_row_value(row, "Protocol", "protocol", default="TCP"))
                key = _norm_key(src_ip, src_port, dst_ip, dst_port, protocol)
                reverse_key = _reverse(key)
                if key not in wanted and reverse_key not in wanted:
                    continue
                timestamp_raw = _row_value(row, "Timestamp", "Flow Start", "start_ts", default="")
                start, timestamp_status = _parse_ids2018_timestamp(timestamp_raw)
                timestamp_status_counts[timestamp_status] += 1
                duration_raw = _to_float(_row_value(row, "Flow Duration", "duration", default="0"))
                duration = duration_raw / 1_000_000.0 if duration_raw > 10000 else duration_raw
                retained_candidates.append(
                    {
                    "key": key,
                    "reverse_key": reverse_key,
                    "start_ts": start,
                    "end_ts": start + duration,
                    "timestamp_parse_status": timestamp_status,
                    "label": _canonical_ids2018_family(label),
                    "raw_label": raw_label,
                    "source_file": str(path),
                    "ids2018_complete_five_tuple": True,
                    }
                )
                csv_stats["csv_candidate_rows_retained"] += 1

    offset_report: dict[str, Any] = {}
    estimated_offset = 0.0
    if time_offset_seconds is not None:
        estimated_offset = float(time_offset_seconds)
        offset_report = {
            "offset_estimator": "manual",
            "offset_estimator_status": "manual",
        }
    elif auto_time_offset and retained_candidates:
        estimated_offset, offset_report = _estimate_time_offset_seconds(flows, retained_candidates)
    else:
        offset_report = {
            "offset_estimator": "none",
            "offset_estimator_status": "disabled",
        }

    adjusted_label_rows: list[dict[str, Any]] = []
    for candidate in retained_candidates:
        adjusted = dict(candidate)
        adjusted["ids2018_time_offset_seconds"] = estimated_offset
        adjusted["start_ts"] = float(adjusted.get("start_ts") or 0.0) + estimated_offset
        adjusted["end_ts"] = float(adjusted.get("end_ts") or adjusted["start_ts"]) + estimated_offset
        adjusted_label_rows.append(adjusted)

    labeled, unmatched, stats = _label_flows_time_indexed(
        flows,
        adjusted_label_rows,
        tolerance_seconds=tolerance_seconds,
    )
    for row in labeled:
        row["attack_family"] = _canonical_ids2018_family(row.get("label"))
        row["label"] = "BENIGN" if row["attack_family"] == "BENIGN" else row["attack_family"]
        row["binary_label"] = "BENIGN" if row["attack_family"] == "BENIGN" else "ATTACK"
    normalized = normalize_flows(labeled, dataset="CSE-CIC-IDS2018")
    for row in normalized:
        row["attack_family"] = _canonical_ids2018_family(row.get("attack_family") or row.get("label"))
        row["label"] = "BENIGN" if row["attack_family"] == "BENIGN" else row["attack_family"]
        row["binary_label"] = "BENIGN" if row["attack_family"] == "BENIGN" else "ATTACK"
        row["dataset"] = "CSE-CIC-IDS2018"

    time_deltas = [
        float(row.get("label_time_delta", 0.0))
        for row in normalized
        if row.get("label_time_delta") is not None
    ]
    alignment = label_alignment_report(
        dataset="CSE-CIC-IDS2018",
        total_zeek_flows=len(flows),
        matched_flows=len(normalized),
        unmatched_flows=len(unmatched),
        ambiguous_matches=int(stats.get("ambiguous_matches", 0)),
        time_deltas=time_deltas,
        matched_flows_rows=normalized,
        dropped_reason_counts=stats.get("dropped_reason_counts", {}),
        notes=[
            "Alignment fields are for labeling/reporting only and must not enter model tokens.",
            f"tolerance_seconds={tolerance_seconds}",
            "CSV candidate filtering uses Zeek flow five-tuples only for label joining.",
        ],
    )
    alignment.update(csv_stats)
    alignment["raw_label_counts_scanned"] = dict(sorted(raw_label_counts.items()))
    alignment["candidate_timestamp_status_counts"] = dict(sorted(timestamp_status_counts.items()))
    alignment["ids2018_time_offset_seconds"] = estimated_offset
    alignment["ids2018_time_offset_mode"] = "manual" if time_offset_seconds is not None else ("auto_hour_mode" if auto_time_offset else "none")
    alignment.update(offset_report)
    return normalized, unmatched, alignment, dataset_report(normalized, dataset="CSE-CIC-IDS2018")


def align_and_adapt_ids2018(
    flows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    *,
    tolerance_seconds: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Align Zeek/tshark-derived flows to IDS2018 CSV labels and normalize schema.

    Five-tuples and timestamps are used only for label joining. They are retained
    as audit metadata by downstream code, but must not be used as behavior tokens
    or benign-memory grouping keys in FlowPrim's main path.
    """

    normalized_labels = _normalize_ids2018_labels(label_rows)
    labeled, unmatched, stats = label_flows(flows, normalized_labels, tolerance_seconds=tolerance_seconds)
    for row in labeled:
        row["attack_family"] = _canonical_ids2018_family(row.get("label"))
        row["label"] = "BENIGN" if row["attack_family"] == "BENIGN" else row["attack_family"]
        row["binary_label"] = "BENIGN" if row["attack_family"] == "BENIGN" else "ATTACK"
    normalized = normalize_flows(labeled, dataset="CSE-CIC-IDS2018")
    for row in normalized:
        row["attack_family"] = _canonical_ids2018_family(row.get("attack_family") or row.get("label"))
        row["label"] = "BENIGN" if row["attack_family"] == "BENIGN" else row["attack_family"]
        row["binary_label"] = "BENIGN" if row["attack_family"] == "BENIGN" else "ATTACK"
        row["dataset"] = "CSE-CIC-IDS2018"

    time_deltas = [
        float(row.get("label_time_delta", 0.0))
        for row in normalized
        if row.get("label_time_delta") is not None
    ]
    alignment = label_alignment_report(
        dataset="CSE-CIC-IDS2018",
        total_zeek_flows=len(flows),
        matched_flows=len(normalized),
        unmatched_flows=len(unmatched),
        ambiguous_matches=int(stats.get("ambiguous_matches", 0)),
        time_deltas=time_deltas,
        matched_flows_rows=normalized,
        dropped_reason_counts=stats.get("dropped_reason_counts", {}),
        notes=[
            "Alignment fields are for labeling/reporting only and must not enter model tokens.",
            f"tolerance_seconds={tolerance_seconds}",
            "IDS2018 labels are normalized to BENIGN, Botnet, DDoS, DoS, BruteForce, WebAttack, and Infiltration.",
        ],
    )
    report = dataset_report(normalized, dataset="CSE-CIC-IDS2018")
    report["ids2018_attack_family_counts"] = dict(
        sorted(Counter(str(row.get("attack_family")) for row in normalized).items())
    )
    return normalized, unmatched, alignment, report
