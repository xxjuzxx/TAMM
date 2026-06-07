#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import write_json


PANEL: list[tuple[str, str]] = [
    ("clean", "outputs/tokens/cicids2017_tokens_2k_behavior.pt"),
    ("low_rate_c2_070", "outputs/tokens/cicids2017_tokens_2k_behavior_low_rate_c2_070.pt"),
    ("packet_delete_010", "outputs/tokens/cicids2017_tokens_2k_behavior_packet_delete_010.pt"),
    ("packet_insert_010", "outputs/tokens/cicids2017_tokens_2k_behavior_packet_insert_010.pt"),
    ("direction_flip_010", "outputs/tokens/cicids2017_tokens_2k_behavior_direction_flip_010.pt"),
    ("length_align_050", "outputs/tokens/cicids2017_tokens_2k_behavior_length_align_050.pt"),
    ("length_padding_050", "outputs/tokens/cicids2017_tokens_2k_behavior_length_padding_050.pt"),
    ("iat_jitter_030", "outputs/tokens/cicids2017_tokens_2k_behavior_iat_jitter_030.pt"),
    ("delay_stretch_050", "outputs/tokens/cicids2017_tokens_2k_behavior_delay_stretch_050.pt"),
    ("burst_split_050", "outputs/tokens/cicids2017_tokens_2k_behavior_burst_split_050.pt"),
    ("burst_merge_050", "outputs/tokens/cicids2017_tokens_2k_behavior_burst_merge_050.pt"),
    ("benign_flow_insert_010", "outputs/tokens/cicids2017_tokens_2k_behavior_benign_flow_insert_010.pt"),
    ("short_flow_delete_050", "outputs/tokens/cicids2017_tokens_2k_behavior_short_flow_delete_050.pt"),
]


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_row(name: str, path: str | Path) -> dict[str, Any]:
    payload = _load_json(path)
    summary = payload.get("summary", {})
    stateful = payload.get("stateful_service") or {}
    alert_guard = payload.get("alert_guard") or {}
    row = {
        "name": name,
        "summary_path": str(path),
        "tokens": payload.get("tokens"),
        "metadata_tokens": payload.get("metadata_tokens"),
        "threshold": payload.get("threshold"),
        "threshold_source": payload.get("threshold_source"),
        "num_eval_flows": int(payload.get("num_eval_flows") or summary.get("num_flows") or 0),
        "macro_f1": float(summary.get("macro_f1") or 0.0),
        "weighted_f1": float(summary.get("weighted_f1") or 0.0),
        "false_positive_rate": float(summary.get("false_positive_rate") or 0.0),
        "attack_recall_online": float(summary.get("attack_recall_online") or 0.0),
        "false_positives": int(summary.get("false_positives") or 0),
        "true_positives": int(summary.get("true_positives") or 0),
        "confusion_matrix": summary.get("confusion_matrix"),
        "stateful_enabled": bool(stateful.get("enabled", False)),
        "stateful_only_alerts": int(stateful.get("stateful_only_alerts") or 0),
        "stateful_missing_key_flows": int(stateful.get("stateful_missing_key_flows") or 0),
        "stateful_guard_blocked_flows": int(stateful.get("stateful_guard_blocked_flows") or 0),
        "stateful_updated_flows": int(stateful.get("stateful_updated_flows") or 0),
        "alert_guard_enabled": bool(alert_guard.get("enabled", False)),
        "alert_guard_blocked_predictions": int(alert_guard.get("alert_guard_blocked_predictions") or 0),
        "alert_guard_blocked_false_positives": int(alert_guard.get("alert_guard_blocked_false_positives") or 0),
        "alert_guard_blocked_true_positives": int(alert_guard.get("alert_guard_blocked_true_positives") or 0),
    }
    if stateful.get("model_only_summary"):
        model_only = stateful["model_only_summary"]
        row["model_only_macro_f1"] = float(model_only.get("macro_f1") or 0.0)
        row["model_only_false_positive_rate"] = float(model_only.get("false_positive_rate") or 0.0)
        row["model_only_attack_recall_online"] = float(model_only.get("attack_recall_online") or 0.0)
        row["model_only_confusion_matrix"] = model_only.get("confusion_matrix")
    return row


def _worst(rows: list[dict[str, Any]], key: str, prefer_low: bool = False) -> dict[str, Any] | None:
    if not rows:
        return None
    selector = min if prefer_low else max
    return selector(rows, key=lambda row: float(row[key]))


def summarize_panel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_panels": 0,
            "worst_macro_f1": None,
            "worst_attack_recall": None,
            "worst_false_positive_rate": None,
        }
    worst_macro = _worst(rows, "macro_f1", prefer_low=True)
    worst_recall = _worst(rows, "attack_recall_online", prefer_low=True)
    worst_fpr = _worst(rows, "false_positive_rate", prefer_low=False)
    return {
        "num_panels": len(rows),
        "mean_macro_f1": float(sum(float(row["macro_f1"]) for row in rows) / len(rows)),
        "worst_macro_f1": {"name": worst_macro["name"], "value": worst_macro["macro_f1"]},
        "worst_attack_recall": {"name": worst_recall["name"], "value": worst_recall["attack_recall_online"]},
        "worst_false_positive_rate": {"name": worst_fpr["name"], "value": worst_fpr["false_positive_rate"]},
    }


def _replay_command(args: argparse.Namespace, name: str, token_path: str, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/16_online_replay.py",
        "--tokens",
        token_path,
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
        "--split",
        args.split,
        "--scope",
        args.scope,
        "--threshold",
        str(args.threshold),
        "--consecutive_alerts",
        str(args.consecutive_alerts),
        "--fpr_windows",
        *[str(window) for window in args.fpr_windows],
        "--out",
        str(out_dir),
    ]
    if args.enable_stateful_service:
        cmd.extend(
            [
                "--enable_stateful_service",
                "--stateful_window_seconds",
                str(args.stateful_window_seconds),
                "--stateful_min_count",
                str(args.stateful_min_count),
                "--stateful_max_packets",
                str(args.stateful_max_packets),
                "--stateful_min_short_ratio",
                str(args.stateful_min_short_ratio),
                "--stateful_min_model_score",
                str(args.stateful_min_model_score),
                "--stateful_merge",
                args.stateful_merge,
            ]
        )
        if args.stateful_allowed_protocols is not None:
            cmd.extend(["--stateful_allowed_protocols", *args.stateful_allowed_protocols])
        if args.stateful_allowed_ports is not None:
            cmd.extend(["--stateful_allowed_ports", *args.stateful_allowed_ports])
        if args.stateful_excluded_ports is not None:
            cmd.extend(["--stateful_excluded_ports", *args.stateful_excluded_ports])
    if args.alert_allowed_protocols is not None:
        cmd.extend(["--alert_allowed_protocols", *args.alert_allowed_protocols])
    if args.alert_allowed_ports is not None:
        cmd.extend(["--alert_allowed_ports", *args.alert_allowed_ports])
    if args.alert_excluded_ports is not None:
        cmd.extend(["--alert_excluded_ports", *args.alert_excluded_ports])
    if name == "clean" and args.clean_metadata_tokens:
        cmd.extend(["--metadata_tokens", args.clean_metadata_tokens])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/model_behavior_composer.yaml")
    parser.add_argument("--split", choices=["stratified", "chronological", "temporal_stratified"], default="temporal_stratified")
    parser.add_argument("--scope", choices=["train", "val", "test", "val_test", "all"], default="test")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--consecutive_alerts", type=int, default=1)
    parser.add_argument("--fpr_windows", nargs="*", type=float, default=[60.0, 300.0, 3600.0])
    parser.add_argument("--enable_stateful_service", action="store_true")
    parser.add_argument("--stateful_window_seconds", type=float, default=2.0)
    parser.add_argument("--stateful_min_count", type=int, default=40)
    parser.add_argument("--stateful_max_packets", type=int, default=4)
    parser.add_argument("--stateful_min_short_ratio", type=float, default=0.75)
    parser.add_argument("--stateful_min_model_score", type=float, default=0.01)
    parser.add_argument("--stateful_merge", choices=["or", "model_only"], default="or")
    parser.add_argument("--stateful_allowed_protocols", nargs="*", default=None)
    parser.add_argument("--stateful_allowed_ports", nargs="*", default=None)
    parser.add_argument("--stateful_excluded_ports", nargs="*", default=None)
    parser.add_argument("--alert_allowed_protocols", nargs="*", default=None)
    parser.add_argument("--alert_allowed_ports", nargs="*", default=None)
    parser.add_argument("--alert_excluded_ports", nargs="*", default=None)
    parser.add_argument("--clean_metadata_tokens", default=None)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    commands: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name, token_path in PANEL:
        out_dir = out_root / name
        summary_path = out_dir / "online_replay_summary.json"
        cmd = _replay_command(args, name, token_path, out_dir)
        commands.append({"name": name, "cmd": cmd, "summary_path": str(summary_path)})
        if not args.dry_run and not (args.skip_existing and summary_path.exists()):
            subprocess.run(cmd, check=True)
        if summary_path.exists():
            rows.append(_summary_row(name, summary_path))

    payload = {
        "checkpoint": args.checkpoint,
        "threshold": float(args.threshold),
        "split": args.split,
        "scope": args.scope,
        "stateful_service": {
            "enabled": bool(args.enable_stateful_service),
            "window_seconds": float(args.stateful_window_seconds),
            "min_count": int(args.stateful_min_count),
            "max_packets": int(args.stateful_max_packets),
            "min_short_ratio": float(args.stateful_min_short_ratio),
            "min_model_score": float(args.stateful_min_model_score),
            "merge": args.stateful_merge,
            "allowed_protocols": args.stateful_allowed_protocols,
            "allowed_ports": args.stateful_allowed_ports,
            "excluded_ports": args.stateful_excluded_ports,
        },
        "alert_guard": {
            "enabled": bool(args.alert_allowed_protocols is not None or args.alert_allowed_ports is not None or args.alert_excluded_ports is not None),
            "allowed_protocols": args.alert_allowed_protocols,
            "allowed_ports": args.alert_allowed_ports,
            "excluded_ports": args.alert_excluded_ports,
        },
        "panel_summary": summarize_panel(rows),
        "results": rows,
        "commands": commands,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(payload, out_root / "online_replay_panel_summary.json")
    print(payload["panel_summary"])


if __name__ == "__main__":
    main()
