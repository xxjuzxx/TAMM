#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

import _bootstrap  # noqa: F401

from src.data.ids2018_schedule_labeler import IDS2018AttackWindow, load_attack_windows
from src.utils.io import write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZEEK_ROOT = ROOT / "outputs" / "ids2018_official_victim_external" / "zeek"
DEFAULT_OUT = ROOT / "data" / "interim" / "flows" / "ids2018" / "official_victim_external" / "raw_ids2018_official_victim_external_labeled_flows.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "interim" / "flows" / "ids2018" / "official_victim_external" / "merge_manifest.json"
DEFAULT_REPORT = ROOT / "reports" / "ids2018_official_victim_external_labeling.md"


def _json_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


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


def _normal_proto(value: Any) -> str:
    raw = str(value or "TCP").strip().upper()
    if raw == "6":
        return "TCP"
    if raw == "17":
        return "UDP"
    if raw == "1":
        return "ICMP"
    return raw or "TCP"


def _day_from_run_name(name: str) -> str:
    return name.split("__", 1)[0]


def _stable_flow_id(run_name: str, uid: str, src: str, sport: str, dst: str, dport: str, proto: str) -> str:
    key = "|".join([run_name, uid, src, sport, dst, dport, proto])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def _matches_window(flow: dict[str, Any], windows: list[IDS2018AttackWindow], day: str) -> list[IDS2018AttackWindow]:
    start_ts = float(flow.get("start_ts") or 0.0)
    end_ts = float(flow.get("end_ts") or start_ts)
    src_ip = str(flow.get("src_ip") or "")
    dst_ip = str(flow.get("dst_ip") or "")
    matches: list[IDS2018AttackWindow] = []
    for window in windows:
        if window.day_dir != day:
            continue
        if not window.overlaps(start_ts, end_ts):
            continue
        if not window.matches_endpoints(src_ip, dst_ip):
            continue
        matches.append(window)
    return matches


class FlowSketch:
    """Memory-bounded packet-order sketch for one Zeek UID."""

    def __init__(self, *, run_name: str, day: str, uid: str, max_packets: int) -> None:
        self.run_name = run_name
        self.day = day
        self.uid = uid
        self.max_packets = int(max_packets)
        self.src_ip = ""
        self.dst_ip = ""
        self.src_port = "0"
        self.dst_port = "0"
        self.protocol = "TCP"
        self.start_ts: float | None = None
        self.end_ts: float | None = None
        self.packet_count = 0
        self.byte_count = 0
        self._head: list[tuple[float, int, int]] = []
        self._tail: list[tuple[float, int, int]] = []
        self._tail_cap = max(0, self.max_packets // 3)
        self._head_cap = max(0, self.max_packets - self._tail_cap)

    def add(self, row: dict[str, Any]) -> None:
        ts = _to_float(row.get("timestamp") or row.get("ts"))
        length = max(0, _to_int(row.get("applayerlength") or row.get("length")))
        is_orig = row.get("is_orig", True)
        if isinstance(is_orig, str):
            direction = 1 if is_orig.strip().lower() in {"true", "t", "1", "orig", "c2s"} else 0
        else:
            direction = 1 if bool(is_orig) else 0
        if not self.src_ip:
            self.src_ip = str(row.get("srcip") or row.get("id.orig_h") or "")
            self.dst_ip = str(row.get("dstip") or row.get("id.resp_h") or "")
            self.src_port = str(row.get("srcport") or row.get("id.orig_p") or "0")
            self.dst_port = str(row.get("dstport") or row.get("id.resp_p") or "0")
            self.protocol = _normal_proto(row.get("proto") or row.get("protocol"))
        self.start_ts = ts if self.start_ts is None else min(self.start_ts, ts)
        self.end_ts = ts if self.end_ts is None else max(self.end_ts, ts)
        self.packet_count += 1
        self.byte_count += length
        item = (ts, direction, length)
        if len(self._head) < self._head_cap:
            self._head.append(item)
        elif self._tail_cap > 0:
            self._tail.append(item)
            if len(self._tail) > self._tail_cap:
                self._tail.pop(0)

    def to_flow(self, windows: list[IDS2018AttackWindow]) -> dict[str, Any]:
        packets = sorted(self._head + self._tail, key=lambda item: item[0])
        if packets:
            base_ts = packets[0][0]
            tss = [max(0.0, item[0] - base_ts) for item in packets]
            lens = [item[2] for item in packets]
            dirs = [item[1] for item in packets]
        else:
            tss, lens, dirs = [], [], []
        iats = [0.0] + [max(0.0, cur - prev) for prev, cur in zip(tss[:-1], tss[1:])] if tss else []
        flow_id = _stable_flow_id(self.run_name, self.uid, self.src_ip, self.src_port, self.dst_ip, self.dst_port, self.protocol)
        row: dict[str, Any] = {
            "flow_id": flow_id,
            "dataset": "CSE-CIC-IDS2018",
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "proto": self.protocol.lower(),
            "protocol": self.protocol,
            "service_key": [self.dst_ip, self.dst_port, self.protocol],
            "lens": lens,
            "dirs": dirs,
            "tss": tss,
            "iats": iats,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration": max(0.0, float(self.end_ts or 0.0) - float(self.start_ts or 0.0)),
            "packet_count": int(self.packet_count),
            "byte_count": int(self.byte_count),
            "day": self.day,
            "ids2018_day": self.day,
            "meta": {
                "source_run": self.run_name,
                "source_uid": self.uid,
                "ids2018_label_protocol": "schedule_ip_time_window",
                "packet_sketch_max_packets": self.max_packets,
                "packet_sketch_retained_packets": len(lens),
                "packet_sketch_policy": "head_tail_by_uid_order",
            },
        }
        matches = _matches_window(row, windows, self.day)
        meta = dict(row["meta"])
        if matches:
            match = matches[0]
            row["label"] = match.attack_family
            row["attack_family"] = match.attack_family
            row["binary_label"] = "ATTACK"
            meta.update(
                {
                    "raw_label": match.attack_name,
                    "ids2018_schedule_window_id": match.window_id,
                    "ids2018_schedule_day": match.day_dir,
                    "ids2018_label_source": match.source_url,
                    "ids2018_label_confidence": match.confidence,
                }
            )
        else:
            row["label"] = "BENIGN"
            row["attack_family"] = "BENIGN"
            row["binary_label"] = "BENIGN"
            meta["ids2018_label_source"] = "official_schedule_no_window_endpoint_match"
        row["meta"] = meta
        row["raw_ip_used_as_token"] = False
        row["absolute_time_used_as_token"] = False
        row["five_tuple_used_as_token"] = False
        row["protocol_service_used_as_memory_key"] = False
        return row


def _flush_oldest(flows: OrderedDict[str, FlowSketch], windows: list[IDS2018AttackWindow], handle: Any, counters: Counter[str]) -> None:
    _uid, sketch = flows.popitem(last=False)
    row = sketch.to_flow(windows)
    handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    counters["flows_written"] += 1
    counters[f"label:{row['attack_family']}"] += 1
    counters[f"day:{row['day']}"] += 1


def _run_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "Features.log").exists())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_labeled_flows(args: argparse.Namespace) -> dict[str, Any]:
    windows, manifest = load_attack_windows(args.schedule)
    zeek_root = Path(args.zeek_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    total = Counter()
    with out_path.open("w", encoding="utf-8") as handle:
        for run_dir in _run_dirs(zeek_root):
            run_name = run_dir.name
            day = _day_from_run_name(run_name)
            features = run_dir / "Features.log"
            active: OrderedDict[str, FlowSketch] = OrderedDict()
            counters: Counter[str] = Counter()
            for row in _json_rows(features):
                uid = str(row.get("uid") or "")
                if not uid:
                    counters["rows_missing_uid"] += 1
                    continue
                sketch = active.get(uid)
                if sketch is None:
                    if len(active) >= args.max_open_flows:
                        _flush_oldest(active, windows, handle, counters)
                    sketch = FlowSketch(run_name=run_name, day=day, uid=uid, max_packets=args.max_packets_per_flow)
                    active[uid] = sketch
                else:
                    active.move_to_end(uid)
                sketch.add(row)
                counters["feature_rows"] += 1
            while active:
                _flush_oldest(active, windows, handle, counters)
            total.update(counters)
            run_rows.append(
                {
                    "run_name": run_name,
                    "day": day,
                    "features_log": str(features),
                    "features_bytes": features.stat().st_size,
                    **dict(sorted(counters.items())),
                }
            )
            print(json.dumps(run_rows[-1], sort_keys=True))
    label_counts = {key.removeprefix("label:"): value for key, value in total.items() if key.startswith("label:")}
    summary = {
        "schema_version": "flowprim_ids2018_official_victim_streaming_label_v1",
        "zeek_root": str(zeek_root),
        "output": str(out_path),
        "schedule": str(args.schedule),
        "schedule_source_url": manifest.get("source_url"),
        "schedule_to_zeek_offset_seconds": manifest.get("schedule_to_zeek_offset_seconds"),
        "run_count": len(run_rows),
        "max_packets_per_flow": int(args.max_packets_per_flow),
        "max_open_flows": int(args.max_open_flows),
        "total_counters": dict(sorted(total.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "token_safety": {
            "raw_ip_used_as_token": False,
            "absolute_time_used_as_token": False,
            "five_tuple_used_as_token": False,
            "protocol_service_used_as_memory_key": False,
            "protocol_service_used_as_behavior_token": False,
        },
        "notes": [
            "Official schedule/IP/time windows are used only for label alignment and audit metadata.",
            "The emitted packet evidence uses direction, relative IAT, and length sketches; raw IPs and absolute timestamps are not behavior tokens.",
            "Each flow retains a bounded head/tail packet sketch to make IDS2018 full official-victim extraction tractable.",
        ],
        "runs": run_rows,
    }
    write_json(summary, args.manifest)
    _write_csv(Path(args.run_manifest_csv), run_rows)
    _write_report(Path(args.report), summary)
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# IDS2018 Official-Victim Streaming Labeling",
        "",
        f"- Zeek root: `{summary['zeek_root']}`",
        f"- Output JSONL: `{summary['output']}`",
        f"- Run count: {summary['run_count']}",
        f"- Max packet sketch per flow: {summary['max_packets_per_flow']}",
        "",
        "## Label Counts",
        "",
    ]
    for label, count in summary.get("label_counts", {}).items():
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Leakage Controls",
            "",
            "- Raw IP addresses are used only for schedule label alignment and audit metadata.",
            "- Absolute timestamps are used only for schedule label alignment, ordering, and split metadata.",
            "- Complete five-tuples are not emitted as behavior tokens or memory grouping keys.",
            "- Protocol/service are not used as behavior tokens, threshold features, or KNN memory keys in this external validation path.",
            "",
            "## Run Manifest",
            "",
            "| run | day | feature rows | flows written | bytes |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("runs", []):
        lines.append(
            f"| {row.get('run_name')} | {row.get('day')} | {row.get('feature_rows', 0)} | "
            f"{row.get('flows_written', 0)} | {row.get('features_bytes', 0)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-label IDS2018 official-victim AllFeas logs into FlowPrim raw flow JSONL.")
    parser.add_argument("--zeek-root", default=str(DEFAULT_ZEEK_ROOT))
    parser.add_argument("--schedule", default=str(ROOT / "configs" / "ids2018_attack_schedule.yaml"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--run-manifest-csv", default=str(DEFAULT_MANIFEST).replace(".json", "_runs.csv"))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--max-packets-per-flow", type=int, default=256)
    parser.add_argument("--max-open-flows", type=int, default=250000)
    args = parser.parse_args()
    summary = build_labeled_flows(args)
    print(json.dumps({k: summary[k] for k in ("output", "run_count", "label_counts")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
