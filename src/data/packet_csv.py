from __future__ import annotations

import csv
import hashlib
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "frame.time_epoch",
    "frame.len",
    "ip.src",
    "ip.dst",
    "ipv6.src",
    "ipv6.dst",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
]


@dataclass
class FlowBuilder:
    flow_id: str
    src_ip: str
    src_port: str
    dst_ip: str
    dst_port: str
    protocol: str
    label: str
    dataset_file: str
    lens: list[int] = field(default_factory=list)
    dirs: list[bool] = field(default_factory=list)
    tss: list[float] = field(default_factory=list)

    def add_packet(self, ts: float, length: int, is_orig: bool) -> None:
        self.tss.append(ts)
        self.lens.append(length)
        self.dirs.append(is_orig)

    def to_record(self, max_packets: int | None = None) -> dict[str, Any]:
        order = sorted(range(len(self.tss)), key=self.tss.__getitem__)
        if max_packets is not None:
            order = order[:max_packets]
        lens = [self.lens[idx] for idx in order]
        dirs = [self.dirs[idx] for idx in order]
        abs_tss = [self.tss[idx] for idx in order]
        start_ts = abs_tss[0] if abs_tss else None
        end_ts = abs_tss[-1] if abs_tss else None
        tss = [max(0.0, ts - start_ts) for ts in abs_tss] if start_ts is not None else []
        iats = [0.0] + [max(0.0, cur - prev) for prev, cur in zip(abs_tss[:-1], abs_tss[1:])] if abs_tss else []
        duration = (end_ts - start_ts) if start_ts is not None and end_ts is not None and len(abs_tss) > 1 else 0.0
        path = Path(self.dataset_file)
        protocol = self.protocol.upper()
        return {
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "src_port": self.src_port,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "protocol": protocol,
            "proto": protocol.lower(),
            "service_key": [self.dst_ip, self.dst_port, self.protocol],
            "label": self.label,
            "attack_family": self.label,
            "binary_label": "BENIGN" if self.label.lower() == "benign" else "ATTACK",
            "dataset": "CICIDS2017",
            "day": path.stem,
            "dataset_file": self.dataset_file,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration": duration,
            "packet_count": len(tss),
            "lens": lens,
            "dirs": [1 if item else 0 for item in dirs],
            "tss": tss,
            "iats": iats,
            "split": None,
            "meta": {"source_file": self.dataset_file},
        }


def discover_packet_csvs(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    return sorted(root.glob("*/*.csv"))


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _packet_tuple(row: dict[str, str]) -> tuple[str, str, str, str, str] | None:
    src_ip = _first_non_empty(row.get("ip.src"), row.get("ipv6.src"))
    dst_ip = _first_non_empty(row.get("ip.dst"), row.get("ipv6.dst"))
    tcp_src = _first_non_empty(row.get("tcp.srcport"))
    tcp_dst = _first_non_empty(row.get("tcp.dstport"))
    udp_src = _first_non_empty(row.get("udp.srcport"))
    udp_dst = _first_non_empty(row.get("udp.dstport"))
    if tcp_src or tcp_dst:
        src_port = tcp_src
        dst_port = tcp_dst
        protocol = "TCP"
    elif udp_src or udp_dst:
        src_port = udp_src
        dst_port = udp_dst
        protocol = "UDP"
    else:
        src_port = "0"
        dst_port = "0"
        protocol = "OTHER"
    if not src_ip or not dst_ip:
        return None
    return src_ip, src_port or "0", dst_ip, dst_port or "0", protocol


def canonical_flow_key(packet_key: tuple[str, str, str, str, str]) -> tuple[tuple[str, str, str, str, str], bool]:
    src_ip, src_port, dst_ip, dst_port, protocol = packet_key
    forward = (src_ip, src_port, dst_ip, dst_port, protocol)
    reverse = (dst_ip, dst_port, src_ip, src_port, protocol)
    if forward <= reverse:
        return forward, True
    return reverse, False


def stable_flow_id(key: tuple[str, str, str, str, str], label: str, dataset_file: str) -> str:
    payload = "|".join([dataset_file, label, *key])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def build_flows_from_csvs(
    data_root: str | Path,
    max_flows_per_label: int | None = None,
    max_packets_per_flow: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flows: OrderedDict[tuple[str, str, str, str, str, str], FlowBuilder] = OrderedDict()
    packet_counts: Counter[str] = Counter()
    skipped = 0
    csv_paths = discover_packet_csvs(data_root)
    label_seen_counts: Counter[str] = Counter()
    for csv_path in csv_paths:
        label = csv_path.parent.name
        dataset_file = str(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if max_flows_per_label is not None and label_seen_counts[label] >= max_flows_per_label:
                    break
                packet_key = _packet_tuple(row)
                if packet_key is None:
                    skipped += 1
                    continue
                try:
                    ts = float(row["frame.time_epoch"])
                    length = int(float(row["frame.len"]))
                except (KeyError, TypeError, ValueError):
                    skipped += 1
                    continue
                canonical_key, is_orig = canonical_flow_key(packet_key)
                flow_map_key = (label, *canonical_key)
                if flow_map_key not in flows:
                    src_ip, src_port, dst_ip, dst_port, protocol = canonical_key
                    flows[flow_map_key] = FlowBuilder(
                        flow_id=stable_flow_id(canonical_key, label, dataset_file),
                        src_ip=src_ip,
                        src_port=src_port,
                        dst_ip=dst_ip,
                        dst_port=dst_port,
                        protocol=protocol,
                        label=label,
                        dataset_file=dataset_file,
                    )
                    label_seen_counts[label] += 1
                builder = flows[flow_map_key]
                if max_packets_per_flow is None or len(builder.lens) < max_packets_per_flow:
                    builder.add_packet(ts, length, is_orig)
                    packet_counts[label] += 1
    records = [builder.to_record(max_packets=max_packets_per_flow) for builder in flows.values()]
    records.sort(key=lambda item: (item["label"], item["start_ts"] or 0.0, item["flow_id"]))
    label_counts = Counter(record["label"] for record in records)
    binary_counts = Counter(record["binary_label"] for record in records)
    packet_per_label = defaultdict(int)
    for record in records:
        packet_per_label[record["label"]] += record["packet_count"]
    stats = {
        "data_root": str(data_root),
        "csv_files": [str(path) for path in csv_paths],
        "num_flows": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "binary_counts": dict(sorted(binary_counts.items())),
        "packet_counts_from_rows": dict(sorted(packet_counts.items())),
        "packet_counts_in_flows": dict(sorted(packet_per_label.items())),
        "skipped_packets": skipped,
        "max_flows_per_label": max_flows_per_label,
        "max_packets_per_flow": max_packets_per_flow,
    }
    return records, stats
