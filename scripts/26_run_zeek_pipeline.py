#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.data.label_policy import ATTEMPTED_POLICIES
from src.utils.io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}", flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _default_prefix(input_path: str) -> str:
    path = Path(input_path)
    name = path.stem if path.suffix else path.name
    return f"zeek_{name.lower().replace('-', '_')}_pcap"


def _resolve_existing_path(path: str, *, base: Path = PROJECT_ROOT) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"path does not exist: {candidate}")
    return str(candidate)


def collect_zeek_logs(zeek_out_dir: str | Path) -> list[Path]:
    root = Path(zeek_out_dir)
    feature_logs = sorted(root.rglob("Features.log"))
    conn_logs = sorted(root.rglob("conn.log"))
    if not feature_logs and not conn_logs:
        raise FileNotFoundError(f"no Features.log or conn.log found under {root}")

    selected: list[Path] = []
    seen: set[Path] = set()
    if feature_logs:
        for feature_log in feature_logs:
            for log_path in (feature_log, feature_log.with_name("conn.log")):
                if log_path.exists() and log_path not in seen:
                    selected.append(log_path)
                    seen.add(log_path)
        for conn_log in conn_logs:
            if conn_log not in seen:
                selected.append(conn_log)
                seen.add(conn_log)
    else:
        selected.extend(conn_logs)
    return selected


def _output_paths(prefix: str, processed_dir: str, token_dir: str) -> dict[str, Path]:
    processed = Path(processed_dir)
    tokens = Path(token_dir)
    return {
        "labeled_flows": processed / f"{prefix}_labeled_flows.jsonl",
        "unmatched_flows": processed / f"{prefix}_unmatched_flows.jsonl",
        "label_stats": processed / f"{prefix}_label_stats.json",
        "profile_primitives": processed / f"{prefix}_profile_primitives.jsonl",
        "profile_stats": processed / f"{prefix}_profile_primitives_stats.json",
        "tokens": tokens / f"{prefix}_tokens_packet_profile.pt",
        "vocab": tokens / f"{prefix}_vocab_packet_profile.json",
        "manifest": processed / f"{prefix}_manifest.json",
    }


def _manifest(args: argparse.Namespace, zeek_logs: list[Path], outputs: dict[str, Path]) -> dict[str, Any]:
    return {
        "input": args.input,
        "prefix": args.prefix,
        "zeek_out_dir": args.zeek_out_dir,
        "zeek_script": args.zeek_script,
        "ignore_checksums": bool(args.ignore_checksums),
        "label_csv": args.label_csv,
        "attempted_policy": args.attempted_policy,
        "tolerance_seconds": float(args.tolerance_seconds),
        "config": args.config,
        "profile_mode": args.profile_mode,
        "max_len": args.max_len,
        "base_vocab": args.base_vocab,
        "use_service_context": bool(args.use_service_context),
        "record_service_context": bool(args.record_service_context),
        "use_service_tokens": bool(args.use_service_tokens),
        "service_context_window_seconds": args.service_context_window_seconds,
        "command": shlex.join(sys.argv),
        "zeek_logs": [str(path) for path in zeek_logs],
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Zeek-first PCAP -> labeled flows -> profile primitives -> token dataset pipeline.")
    parser.add_argument("--input", required=True, help="PCAP file or directory.")
    parser.add_argument("--label_csv", nargs="+", required=True, help="Corrected/official CICIDS flow label CSV files.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to input-derived prefix.")
    parser.add_argument("--zeek_out_dir", default=None)
    parser.add_argument("--processed_dir", default="outputs/processed")
    parser.add_argument("--token_dir", default="outputs/tokens")
    parser.add_argument("--zeek_script", default="../third_party/BSTS-Net/AllFeas.zeek")
    parser.add_argument("--ignore_checksums", action="store_true")
    parser.add_argument("--attempted_policy", choices=ATTEMPTED_POLICIES, default="drop")
    parser.add_argument("--tolerance_seconds", type=float, default=2.0)
    parser.add_argument("--config", default="configs/cicids2017.yaml")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--profile_mode", dest="profile_mode", choices=["none", "packet", "summary", "full"], default="packet")
    parser.add_argument("--base_vocab", default=None)
    parser.add_argument("--use_service_context", action="store_true")
    parser.add_argument("--record_service_context", action="store_true")
    parser.add_argument("--use_service_tokens", action="store_true")
    parser.add_argument("--service_context_window_seconds", type=float, default=None)
    parser.add_argument("--skip_zeek", action="store_true", help="Reuse an existing --zeek_out_dir.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.prefix = args.prefix or _default_prefix(args.input)
    args.zeek_out_dir = args.zeek_out_dir or f"outputs/{args.prefix}_allfeas"
    outputs = _output_paths(args.prefix, args.processed_dir, args.token_dir)

    input_path = _resolve_existing_path(args.input)
    label_csvs = [_resolve_existing_path(path) for path in args.label_csv]
    config_path = _resolve_existing_path(args.config)
    zeek_script = _resolve_existing_path(args.zeek_script) if args.zeek_script else ""

    if not args.skip_zeek:
        zeek_cmd = [
            "scripts/00_run_zeek.sh",
            "--input",
            input_path,
            "--out_dir",
            args.zeek_out_dir,
        ]
        if zeek_script:
            zeek_cmd.extend(["--zeek_script", zeek_script])
        if args.ignore_checksums:
            zeek_cmd.append("--ignore_checksums")
        _run(zeek_cmd, dry_run=args.dry_run)

    zeek_logs = [] if args.dry_run else collect_zeek_logs(args.zeek_out_dir)
    if args.dry_run:
        zeek_logs = [Path(args.zeek_out_dir) / "Features.log", Path(args.zeek_out_dir) / "conn.log"]
    print("Zeek logs:")
    for log_path in zeek_logs:
        print(f"  {log_path}")

    label_cmd = [
        "python",
        "scripts/01_label_flows.py",
        "--zeek_logs",
        *[str(path) for path in zeek_logs],
        "--label_csv",
        *label_csvs,
        "--attempted_policy",
        args.attempted_policy,
        "--tolerance_seconds",
        str(args.tolerance_seconds),
        "--out",
        str(outputs["labeled_flows"]),
        "--unmatched_out",
        str(outputs["unmatched_flows"]),
        "--stats_out",
        str(outputs["label_stats"]),
    ]
    _run(label_cmd, dry_run=args.dry_run)

    profile_cmd = [
        "python",
        "scripts/02_extract_profile_primitives.py",
        "--flows",
        str(outputs["labeled_flows"]),
        "--out",
        str(outputs["profile_primitives"]),
        "--stats_out",
        str(outputs["profile_stats"]),
        "--config",
        config_path,
    ]
    _run(profile_cmd, dry_run=args.dry_run)

    token_cmd = [
        "python",
        "scripts/03_build_tokens.py",
        "--flows",
        str(outputs["labeled_flows"]),
        "--profile_primitives",
        str(outputs["profile_primitives"]),
        "--out",
        str(outputs["tokens"]),
        "--vocab",
        str(outputs["vocab"]),
        "--config",
        config_path,
        "--max_len",
        str(args.max_len),
        "--profile_mode",
        args.profile_mode,
    ]
    if args.base_vocab:
        token_cmd.extend(["--base_vocab", _resolve_existing_path(args.base_vocab)])
    if args.use_service_context:
        token_cmd.append("--use_service_context")
    if args.record_service_context:
        token_cmd.append("--record_service_context")
    if args.use_service_tokens:
        token_cmd.append("--use_service_tokens")
    if args.service_context_window_seconds is not None:
        token_cmd.extend(["--service_context_window_seconds", str(args.service_context_window_seconds)])
    _run(token_cmd, dry_run=args.dry_run)

    if not args.dry_run:
        write_json(_manifest(args, zeek_logs, outputs), outputs["manifest"])
        print(f"Manifest: {outputs['manifest']}")


if __name__ == "__main__":
    main()
