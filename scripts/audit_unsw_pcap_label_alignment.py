#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import dpkt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNSW_ROOT = Path("data/raw/UNSW-NB15")
DEFAULT_PCAP_DIR = UNSW_ROOT / "pcap files" / "pcaps 17-2-2015"
DEFAULT_CSV_GLOB = str(UNSW_ROOT / "CSV Files" / "UNSW-NB15_*.csv")
DEFAULT_GT = UNSW_ROOT / "CSV Files" / "NUSW-NB15_GT.csv"
DEFAULT_OUT = ROOT / "results" / "unsw_nb15_pcap_alignment_pilot"

UNSW_COLUMNS = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "Sload",
    "Dload",
    "Spkts",
    "Dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smeansz",
    "dmeansz",
    "trans_depth",
    "res_bdy_len",
    "Sjit",
    "Djit",
    "Stime",
    "Ltime",
    "Sintpkt",
    "Dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "Label",
]


def _ip_text(raw: bytes) -> str:
    return ".".join(str(b) for b in raw)


def _canon_proto(p: int) -> str:
    return {6: "tcp", 17: "udp", 1: "icmp"}.get(int(p), str(p))


def _flow_key(src: str, sport: int, dst: str, dport: int, proto: str) -> tuple[str, int, str, int, str]:
    left = (src, int(sport))
    right = (dst, int(dport))
    if left <= right:
        return (src, int(sport), dst, int(dport), proto)
    return (dst, int(dport), src, int(sport), proto)


def _family(value: object, label: object = None) -> str:
    try:
        if label is not None and int(float(label)) == 0:
            return "BENIGN"
    except Exception:
        pass
    raw = str(value).strip()
    if raw.lower() in {"", "nan", "none", "-", "normal"}:
        return "BENIGN"
    return raw


def _read_pcap_flows(path: Path, max_packets: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flows: dict[tuple[str, int, str, int, str], dict[str, Any]] = {}
    packets = 0
    skipped = Counter()
    started = time.perf_counter()
    with path.open("rb") as handle:
        reader = dpkt.pcap.Reader(handle)
        linktype = reader.datalink()
        for ts, buf in reader:
            packets += 1
            if max_packets and packets > max_packets:
                break
            try:
                if linktype == dpkt.pcap.DLT_LINUX_SLL:
                    ip = dpkt.sll.SLL(buf).data
                elif hasattr(dpkt.pcap, "DLT_LINUX_SLL2") and linktype == dpkt.pcap.DLT_LINUX_SLL2:
                    ip = dpkt.sll2.SLL2(buf).data
                else:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                if not isinstance(ip, dpkt.ip.IP):
                    skipped["non_ipv4"] += 1
                    continue
                proto = _canon_proto(ip.p)
                if proto == "tcp" and isinstance(ip.data, dpkt.tcp.TCP):
                    sport, dport = int(ip.data.sport), int(ip.data.dport)
                elif proto == "udp" and isinstance(ip.data, dpkt.udp.UDP):
                    sport, dport = int(ip.data.sport), int(ip.data.dport)
                elif proto == "icmp":
                    sport, dport = 0, 0
                else:
                    skipped["unsupported_transport"] += 1
                    continue
                src = _ip_text(ip.src)
                dst = _ip_text(ip.dst)
                key = _flow_key(src, sport, dst, dport, proto)
                flow = flows.get(key)
                if flow is None:
                    flow = {
                        "srcip": key[0],
                        "sport": key[1],
                        "dstip": key[2],
                        "dsport": key[3],
                        "proto": proto,
                        "start_ts": float(ts),
                        "end_ts": float(ts),
                        "packet_count": 0,
                        "byte_count": 0,
                        "orig_packets": 0,
                        "resp_packets": 0,
                        "orig_bytes": 0,
                        "resp_bytes": 0,
                    }
                    flows[key] = flow
                direction_orig = src == flow["srcip"] and sport == flow["sport"]
                flow["end_ts"] = float(ts)
                flow["packet_count"] += 1
                flow["byte_count"] += int(len(buf))
                if direction_orig:
                    flow["orig_packets"] += 1
                    flow["orig_bytes"] += int(len(buf))
                else:
                    flow["resp_packets"] += 1
                    flow["resp_bytes"] += int(len(buf))
            except Exception:
                skipped["parse_error"] += 1
    rows = list(flows.values())
    for idx, row in enumerate(rows):
        row["flow_id"] = f"{path.stem}:{idx}"
        row["duration"] = float(row["end_ts"] - row["start_ts"])
    return rows, {"pcap": str(path), "linktype": int(linktype), "packets_seen": int(min(packets, max_packets) if max_packets else packets), "flows": len(rows), "skipped": dict(skipped), "elapsed_seconds": time.perf_counter() - started}


def _csv_paths(pattern: str) -> list[Path]:
    return sorted(Path("/").glob(pattern.lstrip("/"))) if pattern.startswith("/") else sorted(Path().glob(pattern))


def _build_csv_index(paths: list[Path], *, chunksize: int) -> tuple[dict[tuple[str, int, str, int, str], list[dict[str, Any]]], dict[str, Any]]:
    index: dict[tuple[str, int, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    total = 0
    for path in paths:
        for chunk in pd.read_csv(path, header=None, names=UNSW_COLUMNS, chunksize=chunksize, low_memory=False):
            total += len(chunk)
            for row in chunk[["srcip", "sport", "dstip", "dsport", "proto", "Stime", "Ltime", "attack_cat", "Label"]].itertuples(index=False):
                try:
                    key = _flow_key(str(row.srcip), int(float(row.sport)), str(row.dstip), int(float(row.dsport)), str(row.proto).lower())
                    family = _family(row.attack_cat, row.Label)
                    index[key].append({"stime": float(row.Stime), "ltime": float(row.Ltime), "family": family, "label": int(float(row.Label))})
                    counts[family] += 1
                except Exception:
                    counts["csv_parse_error"] += 1
    return index, {"csv_rows": total, "csv_family_counts": dict(sorted(counts.items()))}


def _match_flows(flows: list[dict[str, Any]], index: dict[tuple[str, int, str, int, str], list[dict[str, Any]]], tolerance: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    status = Counter()
    family_counts = Counter()
    for flow in flows:
        key = (flow["srcip"], int(flow["sport"]), flow["dstip"], int(flow["dsport"]), flow["proto"])
        candidates = index.get(key, [])
        in_time = [
            cand
            for cand in candidates
            if float(cand["stime"]) - tolerance <= float(flow["end_ts"]) and float(cand["ltime"]) + tolerance >= float(flow["start_ts"])
        ]
        row = dict(flow)
        if not candidates:
            row.update({"alignment_status": "unmatched_fivetuple", "attack_family": "", "binary_label": ""})
            status["unmatched_fivetuple"] += 1
        elif not in_time:
            row.update({"alignment_status": "unmatched_time", "attack_family": "", "binary_label": "", "candidate_count": len(candidates)})
            status["unmatched_time"] += 1
        else:
            families = sorted({str(cand["family"]) for cand in in_time})
            labels = sorted({int(cand["label"]) for cand in in_time})
            if len(families) == 1 and len(labels) == 1:
                family = families[0]
                row.update({"alignment_status": "matched", "attack_family": family, "binary_label": "BENIGN" if labels[0] == 0 else "ATTACK", "candidate_count": len(in_time)})
                status["matched"] += 1
                family_counts[family] += 1
            else:
                row.update({"alignment_status": "ambiguous", "attack_family": "|".join(families), "binary_label": "", "candidate_count": len(in_time)})
                status["ambiguous"] += 1
        out.append(row)
    return out, {"alignment_status_counts": dict(status), "matched_family_counts": dict(sorted(family_counts.items()))}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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


def _capinfos(path: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(["capinfos", "-c", "-a", "-e", "-u", str(path)], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UNSW-NB15 PCAP flow to full-CSV label alignment on existing local PCAP subset.")
    parser.add_argument("--pcap-dir", default=str(DEFAULT_PCAP_DIR))
    parser.add_argument("--csv", nargs="+", default=[DEFAULT_CSV_GLOB])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pcaps", nargs="+", default=["1.pcap"])
    parser.add_argument("--max-packets-per-pcap", type=int, default=200000)
    parser.add_argument("--tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcap_dir = Path(args.pcap_dir)
    pcaps = [pcap_dir / item for item in args.pcaps]
    csv_paths: list[Path] = []
    for item in args.csv:
        csv_paths.extend(_csv_paths(item))
    csv_paths = [path for path in csv_paths if path.exists()]
    if not csv_paths:
        raise FileNotFoundError("No UNSW CSV paths found for label index.")
    index, csv_meta = _build_csv_index(csv_paths, chunksize=args.chunksize)
    all_rows: list[dict[str, Any]] = []
    pcap_rows: list[dict[str, Any]] = []
    for pcap in pcaps:
        capinfo = _capinfos(pcap)
        if not pcap.exists():
            pcap_rows.append({"pcap": str(pcap), "status": "missing"})
            continue
        flows, extract_meta = _read_pcap_flows(pcap, args.max_packets_per_pcap)
        aligned, align_meta = _match_flows(flows, index, args.tolerance_seconds)
        all_rows.extend(aligned)
        pcap_rows.append(
            {
                "pcap": str(pcap),
                "status": "ok",
                "size_bytes": pcap.stat().st_size,
                "linktype": extract_meta.get("linktype", ""),
                "packets_seen": extract_meta["packets_seen"],
                "flows_extracted": extract_meta["flows"],
                "alignment_status_counts": json.dumps(align_meta["alignment_status_counts"], sort_keys=True),
                "matched_family_counts": json.dumps(align_meta["matched_family_counts"], sort_keys=True),
                "capinfos_returncode": capinfo["returncode"],
                "capinfos_stderr": capinfo["stderr"].strip()[:500],
            }
        )
    status_counts = Counter(str(row.get("alignment_status")) for row in all_rows)
    matched_counts = Counter(str(row.get("attack_family")) for row in all_rows if row.get("alignment_status") == "matched")
    _write_jsonl(out_dir / "unsw_pcap_alignment_pilot_flows.jsonl", all_rows)
    _write_csv(out_dir / "unsw_pcap_alignment_pilot_pcaps.csv", pcap_rows)
    manifest = {
        "pcap_dir": str(pcap_dir),
        "pcaps": [str(path) for path in pcaps],
        "csv_paths": [str(path) for path in csv_paths],
        "max_packets_per_pcap": int(args.max_packets_per_pcap),
        "tolerance_seconds": float(args.tolerance_seconds),
        "csv_meta": csv_meta,
        "pcap_rows": pcap_rows,
        "overall_alignment_status_counts": dict(status_counts),
        "overall_matched_family_counts": dict(sorted(matched_counts.items())),
        "boundary": {
            "five_tuple_and_absolute_time_used_for_label_alignment_only": True,
            "raw_ip_used_as_behavior_token": False,
            "absolute_timestamp_used_as_behavior_token": False,
            "complete_five_tuple_used_as_behavior_token": False,
        },
    }
    (out_dir / "unsw_pcap_alignment_pilot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# UNSW-NB15 PCAP Label-alignment Pilot",
        "",
        f"PCAPs: {', '.join(args.pcaps)}",
        f"Max packets per PCAP: {args.max_packets_per_pcap}",
        f"Tolerance seconds: {args.tolerance_seconds}",
        "",
        "## Overall Alignment",
        "",
        f"- Status counts: `{dict(status_counts)}`",
        f"- Matched family counts: `{dict(sorted(matched_counts.items()))}`",
        "",
        "Five-tuple and absolute timestamps are used only for label alignment and audit metadata, not behavior tokens or memory keys.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "status_counts": dict(status_counts), "matched_family_counts": dict(sorted(matched_counts.items()))}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
