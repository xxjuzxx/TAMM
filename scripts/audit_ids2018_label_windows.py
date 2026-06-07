#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import _bootstrap  # noqa: F401
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def _parse_time(day: str, hm: str, offset_seconds: int) -> float:
    hour, minute = [int(part) for part in hm.split(":", 1)]
    dt = datetime.fromisoformat(day).replace(hour=hour, minute=minute, second=0, microsecond=0, tzinfo=timezone.utc)
    return dt.timestamp() + int(offset_seconds)


def _windows(schedule_path: Path, attacks: set[str]) -> list[dict[str, Any]]:
    payload = yaml.safe_load(schedule_path.read_text(encoding="utf-8"))
    offset = int(payload.get("schedule_to_zeek_offset_seconds") or 0)
    rows: list[dict[str, Any]] = []
    for item in payload.get("windows", []):
        family = str(item.get("attack_family") or "")
        if family not in attacks:
            continue
        start = _parse_time(str(item["iso_date"]), str(item["start_time"]), offset)
        finish = _parse_time(str(item["iso_date"]), str(item["finish_time"]), offset)
        rows.append(
            {
                "window_id": item.get("window_id"),
                "attack_family": family,
                "attack_name": item.get("attack_name"),
                "day_dir": item.get("day_dir"),
                "start_ts": start,
                "finish_ts": finish,
                "victim_ips": set(str(ip) for ip in item.get("victim_ips", [])),
                "attacker_ips": set(str(ip) for ip in item.get("attacker_ips", [])),
            }
        )
    return rows


def _period_for_flow(row: dict[str, Any], windows: list[dict[str, Any]], pad_seconds: int) -> tuple[str, str]:
    day = str(row.get("day") or row.get("ids2018_day") or "")
    ts = float(row.get("start_ts") or 0.0)
    src = str(row.get("src_ip") or "")
    dst = str(row.get("dst_ip") or "")
    for win in windows:
        if day != win["day_dir"]:
            continue
        endpoint_match = src in win["victim_ips"] or dst in win["victim_ips"] or src in win["attacker_ips"] or dst in win["attacker_ips"]
        if not endpoint_match:
            continue
        if win["start_ts"] <= ts <= win["finish_ts"]:
            return "attack_window_endpoint", str(win["window_id"])
        if win["start_ts"] - pad_seconds <= ts < win["start_ts"]:
            return "pre_window_endpoint", str(win["window_id"])
        if win["finish_ts"] < ts <= win["finish_ts"] + pad_seconds:
            return "post_window_endpoint", str(win["window_id"])
    return "outside_or_nonendpoint", ""


def _flow_shape(row: dict[str, Any]) -> str:
    lens = row.get("lens") or []
    dirs = row.get("dirs") or []
    lens_sig = ",".join(str(x) for x in lens[:12])
    dirs_sig = ",".join(str(x) for x in dirs[:12])
    return f"p{row.get('packet_count')}|d{round(float(row.get('duration') or 0.0), 3)}|l[{lens_sig}]|r[{dirs_sig}]"


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "flows": 0,
            "packet_count_mean": "",
            "packet_count_p50": "",
            "duration_mean": "",
            "duration_p50": "",
            "byte_count_mean": "",
            "unique_shapes": 0,
            "top_shape_share": "",
        }
    pkt = sorted(float(row.get("packet_count") or 0) for row in rows)
    dur = sorted(float(row.get("duration") or 0) for row in rows)
    byt = [float(row.get("byte_count") or sum(row.get("lens") or []) or 0) for row in rows]
    shape_counts = Counter(_flow_shape(row) for row in rows)
    return {
        "flows": len(rows),
        "packet_count_mean": sum(pkt) / len(pkt),
        "packet_count_p50": pkt[len(pkt) // 2],
        "duration_mean": sum(dur) / len(dur),
        "duration_p50": dur[len(dur) // 2],
        "byte_count_mean": sum(byt) / len(byt),
        "unique_shapes": len(shape_counts),
        "top_shape_share": shape_counts.most_common(1)[0][1] / len(rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _load_low_scores(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row.get("flow_id")) for row in csv.DictReader(handle) if row.get("flow_id")}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    attacks = set(args.attacks)
    windows = _windows(Path(args.schedule), attacks)
    low_score_ids = _load_low_scores(Path(args.low_score_flows))
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    low_score_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []

    for row in _jsonl(Path(args.flow_jsonl)):
        family = str(row.get("attack_family") or row.get("label") or "")
        if family not in attacks and family != "BENIGN":
            continue
        period, window_id = _period_for_flow(row, windows, args.adjacent_seconds)
        if period == "outside_or_nonendpoint" and family not in attacks:
            continue
        key = (family, period, window_id, str(row.get("day") or ""))
        grouped[key].append(row)
        if str(row.get("flow_id")) in low_score_ids:
            low_score_rows.append(
                {
                    "flow_id": row.get("flow_id"),
                    "attack_family": family,
                    "period": period,
                    "window_id": window_id,
                    "day": row.get("day"),
                    "start_ts": row.get("start_ts"),
                    "packet_count": row.get("packet_count"),
                    "duration": row.get("duration"),
                    "byte_count": row.get("byte_count"),
                    "shape": _flow_shape(row),
                    "src_ip": row.get("src_ip"),
                    "dst_ip": row.get("dst_ip"),
                    "src_port": row.get("src_port"),
                    "dst_port": row.get("dst_port"),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for (family, period, window_id, day), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "family": family,
                "period": period,
                "window_id": window_id,
                "day": day,
                **_stats(rows),
            }
        )
        for shape, count in Counter(_flow_shape(row) for row in rows).most_common(args.top_shapes):
            shape_rows.append(
                {
                    "family": family,
                    "period": period,
                    "window_id": window_id,
                    "day": day,
                    "shape_count": count,
                    "shape_share": count / max(len(rows), 1),
                    "shape": shape,
                }
            )

    _write_csv(out_dir / "window_period_summary.csv", summary_rows)
    _write_csv(out_dir / "top_flow_shapes_by_period.csv", shape_rows)
    _write_csv(out_dir / "low_score_attack_flow_alignment.csv", low_score_rows)
    _write_report(out_dir, summary_rows, low_score_rows, args)
    return {"output": str(out_dir), "summary_rows": len(summary_rows), "low_score_rows": len(low_score_rows)}


def _fmt(value: Any) -> str:
    if value == "" or value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _write_report(out_dir: Path, rows: list[dict[str, Any]], low_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# IDS2018 Label-window Audit",
        "",
        f"Adjacent window size: {args.adjacent_seconds} seconds.",
        "",
        "## Period Summary",
        "",
        "| Family | Period | Window | Day | Flows | Pkt mean | Pkt p50 | Duration mean | Unique shapes | Top shape share |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["family"] == "BENIGN" and row["period"] == "outside_or_nonendpoint":
            continue
        lines.append(
            "| {family} | {period} | {window} | {day} | {flows} | {pkt_mean} | {pkt_p50} | {dur} | {shapes} | {share} |".format(
                family=row["family"],
                period=row["period"],
                window=row["window_id"] or "-",
                day=row["day"],
                flows=row["flows"],
                pkt_mean=_fmt(row["packet_count_mean"]),
                pkt_p50=_fmt(row["packet_count_p50"]),
                dur=_fmt(row["duration_mean"]),
                shapes=row["unique_shapes"],
                share=_fmt(row["top_shape_share"]),
            )
        )
    period_counts = Counter(row["period"] for row in low_rows)
    lines.extend(
        [
            "",
            "## Low-score Attack-flow Alignment",
            "",
            f"Low-score flow ids matched: {len(low_rows)}.",
            "",
            "| Period | Count |",
            "|---|---:|",
        ]
    )
    for period, count in sorted(period_counts.items()):
        lines.append(f"| {period} | {count} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `window_period_summary.csv`",
            "- `top_flow_shapes_by_period.csv`",
            "- `low_score_attack_flow_alignment.csv`",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit IDS2018 schedule-label windows.")
    parser.add_argument("--flow-jsonl", required=True)
    parser.add_argument("--schedule", default=str(ROOT / "configs" / "ids2018_attack_schedule.yaml"))
    parser.add_argument("--low-score-flows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attacks", nargs="+", default=["Botnet", "BruteForce"])
    parser.add_argument("--adjacent-seconds", type=int, default=3600)
    parser.add_argument("--top-shapes", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
