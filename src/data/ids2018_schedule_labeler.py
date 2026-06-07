from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import re
from typing import Any, Iterable

from src.data.dataset_adapter import dataset_report, normalize_flows
from src.data.label_policy import binary_label_for, merged_cicids_label
from src.utils.io import read_yaml


DAY_DIR_BY_ISO_DATE = {
    "2018-02-14": "Wednesday-14-02-2018",
    "2018-02-15": "Thursday-15-02-2018",
    "2018-02-16": "Friday-16-02-2018",
    "2018-02-20": "Tuesday-20-02-2018",
    "2018-02-21": "Wednesday-21-02-2018",
    "2018-02-22": "Thursday-22-02-2018",
    "2018-02-23": "Friday-23-02-2018",
    "2018-02-28": "Wednesday-28-02-2018",
    "2018-03-01": "Thursday-01-03-2018",
    "2018-03-02": "Friday-02-03-2018",
}


@dataclass(frozen=True)
class IDS2018AttackWindow:
    """Official IDS2018 schedule entry used only for label alignment."""

    window_id: str
    attack_name: str
    attack_family: str
    iso_date: str
    day_dir: str
    start_time: str
    finish_time: str
    start_ts: float
    end_ts: float
    attacker_ips: frozenset[str]
    victim_ips: frozenset[str]
    source_url: str
    confidence: str = "official_schedule"

    def overlaps(self, flow_start: float, flow_end: float, tolerance_seconds: float = 0.0) -> bool:
        end = flow_end if flow_end > 0 else flow_start
        return end >= self.start_ts - tolerance_seconds and flow_start <= self.end_ts + tolerance_seconds

    def matches_endpoints(self, src_ip: str, dst_ip: str) -> bool:
        src = str(src_ip)
        dst = str(dst_ip)
        return (src in self.attacker_ips and dst in self.victim_ips) or (
            dst in self.attacker_ips and src in self.victim_ips
        )


def extract_ipv4s(value: object) -> list[str]:
    """Extract syntactically valid IPv4 addresses from a source table cell."""

    ips: list[str] = []
    for candidate in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", str(value or "")):
        try:
            ips.append(str(ipaddress.ip_address(candidate)))
        except ValueError:
            continue
    return sorted(dict.fromkeys(ips))


def canonical_ids2018_family(label: object) -> str:
    """Map official IDS2018 attack names to FlowPrim family names."""

    merged = merged_cicids_label(label)
    if merged == "Bot":
        return "Botnet"
    if merged == "PortScan":
        return "Probe"
    return merged


def iso_date_from_official(value: str) -> str:
    """Convert official UNB table date strings to ISO dates."""

    raw = str(value).strip()
    date_part = raw.split("-", 1)[1] if raw[:3].lower() in {"wed", "thu", "fri", "tue"} else raw
    date_part = date_part.replace("Thurs-", "").replace("Thursday-", "")
    for fmt in ("%d-%m-%Y",):
        try:
            return datetime.strptime(date_part, fmt).date().isoformat()
        except ValueError:
            pass
    # Full strings such as Thursday-01-03-2018.
    parts = raw.split("-")
    if len(parts) >= 4:
        candidate = "-".join(parts[-3:])
        try:
            return datetime.strptime(candidate, "%d-%m-%Y").date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unsupported IDS2018 official date: {value!r}")


def schedule_time_to_epoch(iso_date: str, clock: str, offset_seconds: float) -> float:
    """Return Zeek-aligned epoch seconds for an official IDS2018 schedule time.

    The official schedule and processed CSV timestamps are wall-clock strings.
    The existing Tuesday exact-join audit aligns those timestamps to Zeek by
    adding 14,400 seconds. We apply the same offset to schedule labels.
    """

    hour, minute = (int(part) for part in str(clock).strip().split(":", 1))
    dt = datetime.fromisoformat(f"{iso_date}T{hour:02d}:{minute:02d}:00").replace(tzinfo=timezone.utc)
    return float(dt.timestamp() + float(offset_seconds))


def windows_from_manifest(manifest: dict[str, Any]) -> list[IDS2018AttackWindow]:
    """Build attack windows from `configs/ids2018_attack_schedule.yaml` data."""

    offset = float(manifest.get("schedule_to_zeek_offset_seconds", 14400.0))
    source_url = str(manifest.get("source_url") or "")
    seen: set[tuple[Any, ...]] = set()
    windows: list[IDS2018AttackWindow] = []
    for idx, item in enumerate(manifest.get("windows") or [], start=1):
        iso_date = str(item.get("iso_date") or iso_date_from_official(str(item.get("date") or "")))
        day_dir = str(item.get("day_dir") or DAY_DIR_BY_ISO_DATE.get(iso_date, iso_date))
        attack_name = str(item.get("attack_name") or "").strip()
        family = str(item.get("attack_family") or canonical_ids2018_family(attack_name))
        start_time = str(item.get("start_time") or item.get("attack_start_time") or "").strip()
        finish_time = str(item.get("finish_time") or item.get("attack_finish_time") or "").strip()
        attacker_ips = frozenset(item.get("attacker_ips") or extract_ipv4s(item.get("attacker") or ""))
        victim_ips = frozenset(item.get("victim_ips") or extract_ipv4s(item.get("victim") or ""))
        key = (iso_date, attack_name, start_time, finish_time, tuple(sorted(attacker_ips)), tuple(sorted(victim_ips)))
        if key in seen:
            continue
        seen.add(key)
        start_ts = schedule_time_to_epoch(iso_date, start_time, offset)
        end_ts = schedule_time_to_epoch(iso_date, finish_time, offset)
        if end_ts < start_ts:
            end_ts += 24 * 3600
        windows.append(
            IDS2018AttackWindow(
                window_id=str(item.get("window_id") or f"ids2018_schedule_{idx:02d}"),
                attack_name=attack_name,
                attack_family=family,
                iso_date=iso_date,
                day_dir=day_dir,
                start_time=start_time,
                finish_time=finish_time,
                start_ts=start_ts,
                end_ts=end_ts,
                attacker_ips=attacker_ips,
                victim_ips=victim_ips,
                source_url=str(item.get("source_url") or source_url),
                confidence=str(item.get("confidence") or "official_schedule"),
            )
        )
    windows.sort(key=lambda item: (item.start_ts, item.end_ts, item.attack_name, item.window_id))
    return windows


def load_attack_windows(path: str) -> tuple[list[IDS2018AttackWindow], dict[str, Any]]:
    """Read IDS2018 schedule manifest and return normalized windows."""

    manifest = read_yaml(path)
    return windows_from_manifest(manifest), manifest


def _flow_day_dir(flow: dict[str, Any]) -> str:
    raw = str(flow.get("ids2018_day") or flow.get("day_dir") or flow.get("day") or "").strip()
    return raw


def _label_one_flow(
    flow: dict[str, Any],
    windows: Iterable[IDS2018AttackWindow],
    *,
    day: str | None = None,
    tolerance_seconds: float = 0.0,
) -> tuple[dict[str, Any] | None, list[IDS2018AttackWindow]]:
    flow_start = float(flow.get("start_ts") or 0.0)
    flow_end = float(flow.get("end_ts") or flow_start)
    src_ip = str(flow.get("src_ip") or "")
    dst_ip = str(flow.get("dst_ip") or "")
    flow_day = _flow_day_dir(flow)
    matches: list[IDS2018AttackWindow] = []
    for window in windows:
        if day and window.day_dir != day:
            continue
        if flow_day and flow_day not in {"unknown", window.day_dir} and flow_day != window.day_dir:
            continue
        if not window.overlaps(flow_start, flow_end, tolerance_seconds=tolerance_seconds):
            continue
        if not window.matches_endpoints(src_ip, dst_ip):
            continue
        matches.append(window)
    if not matches:
        return None, []
    return flow, matches


def label_ids2018_flows_by_schedule(
    flows: list[dict[str, Any]],
    windows: list[IDS2018AttackWindow],
    *,
    day: str | None = None,
    tolerance_seconds: float = 0.0,
    ambiguous_policy: str = "quarantine",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Label PCAP-derived IDS2018 flows from official schedule/IP windows.

    The schedule fields are label-alignment metadata. They are retained in
    `meta` for audit and must not be emitted as behavior tokens or memory keys.
    """

    if ambiguous_policy not in {"quarantine", "first", "drop", "benign"}:
        raise ValueError(f"Unsupported ambiguous policy: {ambiguous_policy}")
    active_windows = [window for window in windows if day is None or window.day_dir == day]
    labeled: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    stats_counter: Counter[str] = Counter()
    window_hit_counter: Counter[str] = Counter()

    for flow in flows:
        _flow, matches = _label_one_flow(flow, active_windows, day=day, tolerance_seconds=tolerance_seconds)
        row = dict(flow)
        row["dataset"] = "CSE-CIC-IDS2018"
        if day and not row.get("day"):
            row["day"] = day
        if not matches:
            row["label"] = "BENIGN"
            row["attack_family"] = "BENIGN"
            row["binary_label"] = "BENIGN"
            meta = dict(row.get("meta") or {})
            meta["ids2018_label_source"] = "official_schedule_no_window_endpoint_match"
            meta["ids2018_label_protocol"] = "schedule_ip_time_window"
            row["meta"] = meta
            labeled.append(row)
            stats_counter["benign_no_schedule_match"] += 1
            continue

        families = sorted({match.attack_family for match in matches})
        if len(families) > 1:
            stats_counter["ambiguous_different_family"] += 1
            meta = dict(row.get("meta") or {})
            meta["ids2018_schedule_matches"] = [match.window_id for match in matches]
            meta["ids2018_ambiguous_families"] = families
            row["meta"] = meta
            if ambiguous_policy in {"quarantine", "drop"}:
                quarantine.append(row)
                continue
            if ambiguous_policy == "benign":
                row["label"] = "BENIGN"
                row["attack_family"] = "BENIGN"
                row["binary_label"] = "BENIGN"
                labeled.append(row)
                continue

        match = matches[0]
        family = match.attack_family
        meta = dict(row.get("meta") or {})
        meta.update(
            {
                "raw_label": match.attack_name,
                "ids2018_schedule_window_id": match.window_id,
                "ids2018_label_source": match.source_url,
                "ids2018_label_protocol": "schedule_ip_time_window",
                "ids2018_label_confidence": match.confidence,
                "ids2018_schedule_start_time": match.start_time,
                "ids2018_schedule_finish_time": match.finish_time,
                "ids2018_schedule_day": match.day_dir,
            }
        )
        row["meta"] = meta
        row["label"] = family
        row["attack_family"] = family
        row["binary_label"] = binary_label_for(family)
        labeled.append(row)
        stats_counter["attack_schedule_match"] += 1
        window_hit_counter[match.window_id] += 1

    normalized = normalize_flows(labeled, dataset="CSE-CIC-IDS2018")
    label_counts = Counter(str(row.get("attack_family") or row.get("label")) for row in normalized)
    alignment = {
        "schema_version": "flowprim_ids2018_schedule_alignment_v1",
        "dataset": "CSE-CIC-IDS2018",
        "label_protocol": "schedule_ip_time_window",
        "day_filter": day or "all",
        "total_zeek_flows": len(flows),
        "labeled_flows": len(normalized),
        "quarantine_flows": len(quarantine),
        "active_schedule_windows": len(active_windows),
        "label_counts": dict(sorted(label_counts.items())),
        "schedule_window_hit_counts": dict(sorted(window_hit_counter.items())),
        "reason_counts": dict(sorted(stats_counter.items())),
        "ambiguous_policy": ambiguous_policy,
        "tolerance_seconds": float(tolerance_seconds),
        "raw_ip_used_as_token": False,
        "absolute_time_used_as_token": False,
        "five_tuple_used_as_token": False,
        "protocol_service_used_as_memory_key": False,
        "notes": [
            "Official attacker/victim IPs and attack windows are used only for label alignment/audit.",
            "Raw IP addresses, absolute timestamps, complete five-tuples, protocol, and service are not behavior tokens or KNN memory grouping keys.",
            "Non-Tuesday IDS2018 schedule labels have weaker provenance than Tuesday exact five-tuple joins and CICIDS2017 corrected CSV joins.",
        ],
    }
    report = dataset_report(normalized, dataset="CSE-CIC-IDS2018", source="official_schedule_ip_time_window")
    report["ids2018_label_protocol"] = "schedule_ip_time_window"
    report["ids2018_schedule_windows"] = len(active_windows)
    return normalized, quarantine, alignment, report
