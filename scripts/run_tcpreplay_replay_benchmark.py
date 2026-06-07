#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "tcpreplay_replay"
DEFAULT_PCAP = Path("data/raw/CIC-IDS2017/pcaps/Monday-WorkingHours.pcap_ISCX.pcap")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
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


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rss_kib(pid: int) -> int | None:
    path = Path("/proc") / str(pid) / "status"
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) if len(parts) >= 2 else None
    except FileNotFoundError:
        return None
    return None


def _children(pid: int) -> list[int]:
    task = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        text = task.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return []
    return [int(item) for item in text.split()] if text else []


def _tree_rss_kib(pid: int) -> int:
    total = 0
    stack = [pid]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        rss = _rss_kib(cur)
        if rss is not None:
            total += rss
        stack.extend(_children(cur))
    return total


def _monitor_command(
    cmd: list[str],
    *,
    sample_interval: float,
    timeout: float | None = None,
    stdout_file: Path | None = None,
    stderr_file: Path | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    out_handle = stdout_file.open("w", encoding="utf-8") if stdout_file is not None else subprocess.PIPE
    err_handle = stderr_file.open("w", encoding="utf-8") if stderr_file is not None else subprocess.PIPE
    proc = subprocess.Popen(cmd, stdout=out_handle, stderr=err_handle, text=True)
    samples: list[tuple[float, int]] = []
    timed_out = False
    stdout = ""
    stderr = ""
    try:
        while proc.poll() is None:
            now = time.perf_counter() - start
            samples.append((now, _tree_rss_kib(proc.pid)))
            if timeout is not None and now > timeout:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(max(0.01, sample_interval))
        stdout_pipe, stderr_pipe = proc.communicate()
        stdout = stdout_pipe or ""
        stderr = stderr_pipe or ""
    finally:
        if stdout_file is not None and hasattr(out_handle, "close"):
            out_handle.close()
        if stderr_file is not None and hasattr(err_handle, "close"):
            err_handle.close()
    elapsed = time.perf_counter() - start
    if not samples:
        samples.append((elapsed, 0))
    rss_values = [rss for _, rss in samples]
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "rss_peak_mib": max(rss_values) / 1024.0,
        "rss_avg_mib": (sum(rss_values) / max(len(rss_values), 1)) / 1024.0,
        "rss_samples": len(samples),
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


def _count_packet_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def _tcpreplay_attempt(pcap: Path, iface: str, out_dir: Path, sample_interval: float, limit: int | None) -> dict[str, Any]:
    tcpreplay = shutil.which("tcpreplay")
    if tcpreplay is None:
        return {
            "experiment": "tcpreplay",
            "status": "blocked",
            "reason": "tcpreplay_not_installed",
            "pcap": str(pcap),
            "interface": iface,
            "tcpreplay_path": "",
        }
    cmd = [tcpreplay, "--topspeed", "--intf1", iface]
    if limit is not None and int(limit) > 0:
        cmd.extend(["--limit", str(int(limit))])
    cmd.append(str(pcap))
    result = _monitor_command(cmd, sample_interval=sample_interval)
    status = "ok" if result["returncode"] == 0 else "failed"
    stderr_lower = str(result.get("stderr", "")).lower()
    if "operation not permitted" in stderr_lower or "permission" in stderr_lower:
        reason = "permission_denied_raw_socket"
    else:
        reason = "" if status == "ok" else "tcpreplay_returned_nonzero"
    stdout = str(result.get("stdout", ""))
    actual = re.search(r"Actual:\s+(\d+)\s+packets\s+\((\d+)\s+bytes\)\s+sent\s+in\s+([0-9.]+)\s+seconds", stdout)
    rated = re.search(r"Rated:\s+([0-9.]+)\s+Bps,\s+([0-9.]+)\s+Mbps,\s+([0-9.]+)\s+pps", stdout)
    flows = re.search(r"Flows:\s+(\d+)\s+flows,\s+([0-9.]+)\s+fps,\s+(\d+)\s+flow packets,\s+(\d+)\s+non-flow", stdout)
    successful = re.search(r"Successful packets:\s+(\d+)", stdout)
    failed = re.search(r"Failed packets:\s+(\d+)", stdout)
    return {
        "experiment": "tcpreplay",
        "status": status,
        "reason": reason,
        "pcap": str(pcap),
        "pcap_size_bytes": pcap.stat().st_size if pcap.exists() else "",
        "limit_packets": int(limit) if limit is not None and int(limit) > 0 else "",
        "interface": iface,
        "tcpreplay_path": tcpreplay,
        "elapsed_seconds": result["elapsed_seconds"],
        "send_elapsed_seconds": float(actual.group(3)) if actual else "",
        "packets_observed": int(actual.group(1)) if actual else "",
        "bytes_sent": int(actual.group(2)) if actual else "",
        "bytes_per_second": float(rated.group(1)) if rated else "",
        "mib_per_second": float(rated.group(1)) / (1024.0 * 1024.0) if rated else "",
        "mbps": float(rated.group(2)) if rated else "",
        "packets_per_second": float(rated.group(3)) if rated else "",
        "tcpreplay_flows": int(flows.group(1)) if flows else "",
        "tcpreplay_flows_per_second": float(flows.group(2)) if flows else "",
        "tcpreplay_flow_packets": int(flows.group(3)) if flows else "",
        "tcpreplay_non_flow_packets": int(flows.group(4)) if flows else "",
        "successful_packets": int(successful.group(1)) if successful else "",
        "failed_packets": int(failed.group(1)) if failed else "",
        "rss_peak_mib": result["rss_peak_mib"],
        "rss_avg_mib": result["rss_avg_mib"],
        "rss_samples": result["rss_samples"],
        "returncode": result["returncode"],
        "stdout_path": str(out_dir / "tcpreplay_stdout.txt"),
        "stderr_path": str(out_dir / "tcpreplay_stderr.txt"),
        "cmd": " ".join(cmd),
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def _tcpdump_fallback(pcap: Path, max_packets: int, out_dir: Path, sample_interval: float) -> dict[str, Any]:
    tcpdump = shutil.which("tcpdump")
    if tcpdump is None:
        return {
            "experiment": "tcpdump_pcap_scan_fallback",
            "status": "blocked",
            "reason": "tcpdump_not_installed",
            "pcap": str(pcap),
        }
    cmd = [
        tcpdump,
        "-r",
        str(pcap),
        "-c",
        str(max_packets),
        "-w",
        "/dev/null",
    ]
    stdout_path = out_dir / "tcpdump_stdout.txt"
    stderr_path = out_dir / "tcpdump_stderr.txt"
    result = _monitor_command(cmd, sample_interval=sample_interval, stdout_file=stdout_path, stderr_file=stderr_path)
    packets = int(max_packets) if result["returncode"] == 0 else 0
    pcap_size = pcap.stat().st_size if pcap.exists() else 0
    return {
        "experiment": "tcpdump_pcap_scan_fallback",
        "status": "ok" if result["returncode"] == 0 else "failed",
        "reason": "fallback_not_tcpreplay",
        "pcap": str(pcap),
        "pcap_size_bytes": pcap_size,
        "max_packets": int(max_packets),
        "packets_observed": int(packets),
        "elapsed_seconds": result["elapsed_seconds"],
        "packets_per_second": float(packets / max(result["elapsed_seconds"], 1e-12)),
        "input_mib_per_second_upper_bound": float((pcap_size / (1024.0 * 1024.0)) / max(result["elapsed_seconds"], 1e-12)),
        "rss_peak_mib": result["rss_peak_mib"],
        "rss_avg_mib": result["rss_avg_mib"],
        "rss_samples": result["rss_samples"],
        "returncode": result["returncode"],
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "cmd": " ".join(cmd),
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def _load_knn_replay_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summarize_knn(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("memory_policy")), str(row.get("threshold_mode"))), []).append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "false_positive_rate",
        "false_alerts_per_10k_benign",
        "attack_recall_online",
        "query_ms_p50",
        "query_ms_p95",
        "query_ms_p99",
        "throughput_flows_per_second",
        "memory_mib_estimate",
    ]
    for (memory_policy, threshold_mode), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "experiment": "flowprim_knn_online_replay",
            "status": "ok",
            "memory_policy": memory_policy,
            "threshold_mode": threshold_mode,
            "runs": len(items),
            "reason": "algorithm_replay_not_tcpreplay",
        }
        for metric in metrics:
            vals: list[float] = []
            for item in items:
                value = item.get(metric)
                try:
                    vals.append(float(value))
                except (TypeError, ValueError):
                    pass
            if vals:
                row[f"{metric}_mean"] = sum(vals) / len(vals)
                row[f"{metric}_max"] = max(vals)
        out.append(row)
    return out


def _write_report(rows: list[dict[str, Any]], out_dir: Path) -> None:
    tcpreplay_row = next((row for row in rows if row.get("experiment") == "tcpreplay"), {})
    tcpreplay_reason = str(tcpreplay_row.get("reason", ""))
    if tcpreplay_reason == "tcpreplay_not_installed":
        tcpreplay_note = "In this environment, `tcpreplay` is not installed."
    elif tcpreplay_reason == "permission_denied_raw_socket":
        tcpreplay_note = "In this environment, `tcpreplay` is installed, but the current user lacks raw packet transmit permission on the selected interface."
    else:
        tcpreplay_note = f"`tcpreplay` status: {tcpreplay_row.get('status', 'unknown')} ({tcpreplay_reason or 'no error'})."
    lines = [
        "# tcpreplay Replay Benchmark Report",
        "",
        "This report separates true tcpreplay status from fallback PCAP scanning and FlowPrim KNN online replay metrics. Fallback rows are not tcpreplay results.",
        "",
        "| experiment | status | reason | peak MiB | avg MiB | speed | speed unit | notes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        speed = row.get("packets_per_second") or row.get("throughput_flows_per_second_mean") or ""
        speed_unit = "pps" if row.get("packets_per_second") not in {None, ""} else "flows/s" if row.get("throughput_flows_per_second_mean") not in {None, ""} else ""
        peak = row.get("rss_peak_mib") or row.get("memory_mib_estimate_mean") or ""
        avg = row.get("rss_avg_mib") or row.get("memory_mib_estimate_mean") or ""
        notes = row.get("cmd") or f"{row.get('memory_policy', '')}/{row.get('threshold_mode', '')}"
        lines.append(
            f"| {row.get('experiment', '')} | {row.get('status', '')} | {row.get('reason', '')} | "
            f"{float(peak):.2f} | {float(avg):.2f} | {float(speed):.2f} | {speed_unit} | `{notes}` |"
            if peak != "" and avg != "" and speed != ""
            else f"| {row.get('experiment', '')} | {row.get('status', '')} | {row.get('reason', '')} |  |  |  | {speed_unit} | `{notes}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A true tcpreplay experiment requires the `tcpreplay` binary and raw packet transmit permissions on the selected interface.",
            f"- {tcpreplay_note}",
            "- The `tcpdump_pcap_scan_fallback` row measures offline PCAP read/parse throughput and process RSS.",
            "- The `flowprim_knn_online_replay` rows measure the current algorithm's replay-time scoring speed and memory footprint over verified token artifacts.",
            "- If the interface is `lo`, tcpreplay throughput is a local injection benchmark, not an external physical-link deployment measurement.",
        ]
    )
    (out_dir / "tcpreplay_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attempt tcpreplay benchmark and record fallback replay-like metrics.")
    parser.add_argument("--pcap", default=str(DEFAULT_PCAP))
    parser.add_argument("--iface", default="lo")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--max-packets", type=int, default=100000)
    parser.add_argument("--tcpreplay-limit", type=int, default=None)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--knn-summary", default=str(ROOT / "results" / "online_replay_knn" / "online_replay_summary.csv"))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcap = Path(args.pcap)
    rows: list[dict[str, Any]] = []

    tcpreplay_row = _tcpreplay_attempt(pcap, args.iface, out_dir, args.sample_interval, args.tcpreplay_limit)
    (out_dir / "tcpreplay_stdout.txt").write_text(str(tcpreplay_row.pop("stdout", "")), encoding="utf-8")
    (out_dir / "tcpreplay_stderr.txt").write_text(str(tcpreplay_row.pop("stderr", "")), encoding="utf-8")
    rows.append(tcpreplay_row)

    fallback_row = _tcpdump_fallback(pcap, args.max_packets, out_dir, args.sample_interval)
    fallback_row.pop("stdout", "")
    fallback_row.pop("stderr", "")
    rows.append(fallback_row)

    rows.extend(_summarize_knn(_load_knn_replay_summary(Path(args.knn_summary))))
    _write_csv(rows, out_dir / "tcpreplay_replay_metrics.csv")
    _write_json(
        {
            "pcap": str(pcap),
            "iface": args.iface,
            "max_packets": int(args.max_packets),
            "tcpreplay_available": shutil.which("tcpreplay") is not None,
            "note": "Fallback rows are not true tcpreplay results.",
        },
        out_dir / "tcpreplay_replay_manifest.json",
    )
    _write_report(rows, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows), "tcpreplay_status": tcpreplay_row.get("status")}, sort_keys=True))


if __name__ == "__main__":
    main()
