from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class PacketRecord:
    timestamp: float
    src_ip: str
    src_port: str
    dst_ip: str
    dst_port: str
    protocol: str
    length: int
    is_orig: bool | None = None
    appinfo: str | None = None
    uid: str | None = None


@dataclass
class AggregatedFlow:
    flow_id: str
    src_ip: str
    src_port: str
    dst_ip: str
    dst_port: str
    protocol: str
    appinfo: list[str] = field(default_factory=list)
    uids: list[str] = field(default_factory=list)
    lens: list[int] = field(default_factory=list)
    dirs: list[bool] = field(default_factory=list)
    tss: list[float] = field(default_factory=list)

    def add_packet(self, packet: PacketRecord, is_orig: bool) -> None:
        self.tss.append(float(packet.timestamp))
        self.lens.append(int(packet.length))
        self.dirs.append(bool(is_orig))
        if packet.appinfo:
            self.appinfo.append(packet.appinfo)
        if packet.uid:
            self.uids.append(packet.uid)

    def to_record(self) -> dict[str, Any]:
        order = sorted(range(len(self.tss)), key=self.tss.__getitem__)
        lens = [self.lens[idx] for idx in order]
        dirs = [self.dirs[idx] for idx in order]
        abs_tss = [self.tss[idx] for idx in order]
        start_ts = abs_tss[0] if abs_tss else None
        end_ts = abs_tss[-1] if abs_tss else None
        tss = [max(0.0, ts - start_ts) for ts in abs_tss] if start_ts is not None else []
        iats = [0.0] + [max(0.0, cur - prev) for prev, cur in zip(abs_tss[:-1], abs_tss[1:])] if abs_tss else []
        protocol = self.protocol.upper()
        return {
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "src_port": self.src_port,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "protocol": protocol,
            "proto": protocol.lower(),
            "service_key": [self.dst_ip, self.dst_port, protocol],
            "appinfo": self.appinfo,
            "uids": self.uids,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration": (end_ts - start_ts) if start_ts is not None and end_ts is not None and len(abs_tss) > 1 else 0.0,
            "packet_count": len(tss),
            "lens": lens,
            "dirs": [1 if item else 0 for item in dirs],
            "tss": tss,
            "iats": iats,
        }


def canonical_flow_key(packet: PacketRecord) -> tuple[tuple[str, str, str, str, str], bool]:
    forward = (packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port, packet.protocol)
    reverse = (packet.dst_ip, packet.dst_port, packet.src_ip, packet.src_port, packet.protocol)
    if forward <= reverse:
        return forward, True
    return reverse, False


def stable_flow_id(key: tuple[str, ...]) -> str:
    return hashlib.sha1("|".join(key).encode("utf-8")).hexdigest()[:20]


def aggregate_packets(packets: Iterable[PacketRecord]) -> list[dict[str, Any]]:
    flows: OrderedDict[tuple[str, ...], AggregatedFlow] = OrderedDict()
    for packet in packets:
        flow_key, canonical_orig = canonical_flow_key(packet)
        group_key = (*flow_key, f"uid:{packet.uid}") if packet.uid else flow_key
        if group_key not in flows:
            src_ip, src_port, dst_ip, dst_port, protocol = flow_key
            flows[group_key] = AggregatedFlow(
                flow_id=stable_flow_id(group_key),
                src_ip=src_ip,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                protocol=protocol,
            )
        is_orig = canonical_orig if packet.is_orig is None else bool(packet.is_orig)
        if not canonical_orig:
            is_orig = not is_orig if packet.is_orig is not None else False
        flows[group_key].add_packet(packet, is_orig=is_orig)
    records = [flow.to_record() for flow in flows.values()]
    records.sort(key=lambda item: (item["start_ts"] or 0.0, item["flow_id"]))
    return records
