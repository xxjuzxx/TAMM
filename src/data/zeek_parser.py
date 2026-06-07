from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.data.flow_aggregator import PacketRecord, aggregate_packets


def _field(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _is_packet_row(row: dict[str, Any]) -> bool:
    return any(key in row for key in ("applayerlength", "length", "frame_len", "frame.len"))


def _is_conn_row(row: dict[str, Any]) -> bool:
    return any(key in row for key in ("orig_pkts", "resp_pkts", "orig_bytes", "resp_bytes", "orig_ip_bytes", "resp_ip_bytes"))


def _normalize_protocol(value: Any) -> str:
    protocol = str(value or "TCP").upper()
    if protocol == "6":
        return "TCP"
    if protocol == "17":
        return "UDP"
    return protocol


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def packet_from_zeek_row(row: dict[str, Any]) -> PacketRecord:
    ts = float(_field(row, "timestamp", "ts", "time"))
    src_ip = str(_field(row, "srcip", "id.orig_h", "src_ip"))
    dst_ip = str(_field(row, "dstip", "id.resp_h", "dst_ip"))
    src_port = str(_field(row, "srcport", "id.orig_p", "src_port", default="0"))
    dst_port = str(_field(row, "dstport", "id.resp_p", "dst_port", default="0"))
    protocol = _normalize_protocol(_field(row, "proto", "protocol", default="TCP"))
    length = int(float(_field(row, "applayerlength", "length", "frame_len", "frame.len", default=0)))
    raw_is_orig = _field(row, "is_orig", "direction", default=True)
    if isinstance(raw_is_orig, str):
        is_orig = raw_is_orig.strip().lower() in {"true", "t", "1", "orig", "c2s"}
    else:
        is_orig = bool(raw_is_orig)
    return PacketRecord(
        timestamp=ts,
        src_ip=src_ip,
        src_port=src_port,
        dst_ip=dst_ip,
        dst_port=dst_port,
        protocol=protocol,
        length=length,
        is_orig=is_orig,
        appinfo=str(_field(row, "appinfo", "service", default="")),
        uid=str(_field(row, "uid", default="")),
    )


def _conn_row_to_packets(row: dict[str, Any]) -> list[PacketRecord]:
    ts = _to_float(_field(row, "timestamp", "ts", "time"))
    duration = max(_to_float(_field(row, "duration", "flow_duration", default=0.0)), 0.0)
    src_ip = str(_field(row, "srcip", "id.orig_h", "src_ip"))
    dst_ip = str(_field(row, "dstip", "id.resp_h", "dst_ip"))
    src_port = str(_field(row, "srcport", "id.orig_p", "src_port", default="0"))
    dst_port = str(_field(row, "dstport", "id.resp_p", "dst_port", default="0"))
    protocol = _normalize_protocol(_field(row, "proto", "protocol", default="TCP"))
    appinfo = str(_field(row, "appinfo", "service", default=""))
    uid = str(_field(row, "uid", default=""))
    orig_pkts = _to_int(_field(row, "orig_pkts", "fwd_pkts", "total_fwd_packets", default=0))
    resp_pkts = _to_int(_field(row, "resp_pkts", "bwd_pkts", "total_backward_packets", default=0))
    orig_bytes = _to_int(_field(row, "orig_bytes", "orig_ip_bytes", "fwd_bytes", default=0))
    resp_bytes = _to_int(_field(row, "resp_bytes", "resp_ip_bytes", "bwd_bytes", default=0))

    packets: list[PacketRecord] = []
    total_pkts = max(orig_pkts + resp_pkts, 1)
    step = duration / max(total_pkts - 1, 1)

    def add_many(count: int, total_bytes: int, is_orig: bool, offset: int) -> int:
        if count <= 0:
            return offset
        length = max(int(round(total_bytes / count)), 0)
        for _ in range(count):
            packets.append(
                PacketRecord(
                    timestamp=ts + step * offset,
                    src_ip=src_ip if is_orig else dst_ip,
                    src_port=src_port if is_orig else dst_port,
                    dst_ip=dst_ip if is_orig else src_ip,
                    dst_port=dst_port if is_orig else src_port,
                    protocol=protocol,
                    length=length,
                    is_orig=is_orig,
                    appinfo=appinfo,
                    uid=uid,
                )
            )
            offset += 1
        return offset

    offset = add_many(orig_pkts, orig_bytes, True, 0)
    add_many(resp_pkts, resp_bytes, False, offset)
    return packets


def packets_from_zeek_row(row: dict[str, Any]) -> list[PacketRecord]:
    if _is_packet_row(row):
        return [packet_from_zeek_row(row)]
    if _is_conn_row(row):
        return _conn_row_to_packets(row)
    return [packet_from_zeek_row(row)]


def _conn_uid_metadata(paths: Iterable[str | Path]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in iter_zeek_records(path):
            if not _is_conn_row(row):
                continue
            uid = str(_field(row, "uid", default=""))
            if not uid:
                continue
            metadata[uid] = {
                "proto": _normalize_protocol(_field(row, "proto", "protocol", default="TCP")),
                "service": str(_field(row, "service", "appinfo", default="")),
            }
    return metadata


def _paths_have_packet_rows(paths: Iterable[str | Path]) -> bool:
    for path in paths:
        for row in iter_zeek_records(path):
            if _is_packet_row(row):
                return True
    return False


def _apply_uid_metadata(row: dict[str, Any], uid_metadata: dict[str, dict[str, str]]) -> dict[str, Any]:
    uid = str(_field(row, "uid", default=""))
    if not uid:
        return row
    metadata = uid_metadata.get(uid)
    if not metadata:
        return row
    enriched = dict(row)
    if "proto" not in enriched and "protocol" not in enriched and metadata.get("proto"):
        enriched["proto"] = metadata["proto"]
    if "appinfo" not in enriched and "service" not in enriched and metadata.get("service"):
        enriched["service"] = metadata["service"]
    return enriched


def iter_zeek_json(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                yield json.loads(line)


def _decode_separator(raw: str) -> str:
    raw = raw.strip()
    if raw == r"\x09":
        return "\t"
    if raw == r"\x20":
        return " "
    try:
        return raw.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return "\t"


def _clean_zeek_value(value: str, empty_field: str, unset_field: str) -> str:
    if value in {empty_field, unset_field}:
        return ""
    return value


def iter_zeek_tsv(path: str | Path) -> Iterable[dict[str, Any]]:
    separator = "\t"
    empty_field = "(empty)"
    unset_field = "-"
    fields: list[str] | None = None
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#separator"):
                separator = _decode_separator(line[len("#separator") :])
                continue
            if line.startswith("#empty_field"):
                parts = line.split(separator, 1)
                empty_field = parts[1] if len(parts) > 1 else line.split(maxsplit=1)[-1]
                continue
            if line.startswith("#unset_field"):
                parts = line.split(separator, 1)
                unset_field = parts[1] if len(parts) > 1 else line.split(maxsplit=1)[-1]
                continue
            if line.startswith("#fields"):
                fields = line.split(separator)[1:]
                continue
            if line.startswith("#"):
                continue
            if fields is None:
                raise ValueError(f"Zeek TSV has data before #fields header: {path}")
            values = line.split(separator)
            row = {
                field: _clean_zeek_value(values[idx] if idx < len(values) else "", empty_field, unset_field)
                for idx, field in enumerate(fields)
            }
            yield row


def iter_zeek_records(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl", ".ndjson"} or _looks_like_json_lines(path):
        yield from iter_zeek_json(path)
        return
    yield from iter_zeek_tsv(path)


def _looks_like_json_lines(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.lstrip()
            if not stripped:
                continue
            return stripped.startswith("{")
    return False


def read_zeek_packets(paths: Iterable[str | Path]) -> list[PacketRecord]:
    paths = [Path(path) for path in paths]
    uid_metadata = _conn_uid_metadata(paths)
    has_packet_rows = _paths_have_packet_rows(paths)

    packets: list[PacketRecord] = []
    for path in paths:
        for row in iter_zeek_records(path):
            if has_packet_rows and _is_conn_row(row):
                continue
            packets.extend(packets_from_zeek_row(_apply_uid_metadata(row, uid_metadata)))
    return packets


def aggregate_zeek_logs(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return aggregate_packets(read_zeek_packets(paths))


def aggregate_zeek_json(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return aggregate_zeek_logs(paths)
