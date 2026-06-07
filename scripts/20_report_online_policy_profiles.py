#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import read_yaml, write_json


def _read_json_optional(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _missing_validation_paths(validation: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in ("primary_summary", "panel_summary", "all_scope_summary", "audit"):
        path = validation.get(key)
        if path and not Path(path).exists():
            missing.append(path)
    return missing


def _summary_metrics(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    summary = payload.get("summary", payload.get("panel_summary", {}))
    out = {
        "macro_f1": summary.get("macro_f1"),
        "mean_macro_f1": summary.get("mean_macro_f1"),
        "false_positive_rate": summary.get("false_positive_rate"),
        "attack_recall_online": summary.get("attack_recall_online"),
        "confusion_matrix": summary.get("confusion_matrix"),
        "num_flows": summary.get("num_flows"),
        "num_panels": summary.get("num_panels"),
        "worst_macro_f1": summary.get("worst_macro_f1"),
        "worst_attack_recall": summary.get("worst_attack_recall"),
        "worst_false_positive_rate": summary.get("worst_false_positive_rate"),
        "detected_labels": summary.get("detected_labels"),
        "missed_labels": summary.get("missed_labels"),
    }
    return {key: value for key, value in out.items() if value is not None}


def _audit_metrics(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    keys = [
        "blocked_true_positives",
        "blocked_false_positives",
        "passed_true_positives",
        "missed_attacks",
        "benign_alerts_after_guard",
        "blocked_true_positive_labels",
        "blocked_true_positive_services",
        "blocked_false_positive_services",
    ]
    return {key: payload.get(key) for key in keys if key in payload}


def _extend_values(cmd: list[str], option: str, values: Any) -> None:
    if values is None:
        return
    if isinstance(values, (list, tuple)):
        if not values:
            return
        cmd.extend([option, *[str(value) for value in values]])
    else:
        cmd.extend([option, str(values)])


def _stateful_args(profile: dict[str, Any]) -> list[str]:
    stateful = profile.get("stateful_service") or {}
    if not stateful.get("enabled", False):
        return []
    cmd = [
        "--enable_stateful_service",
        "--stateful_window_seconds",
        str(stateful.get("window_seconds", 2.0)),
        "--stateful_min_count",
        str(stateful.get("min_count", 40)),
        "--stateful_max_packets",
        str(stateful.get("max_packets", 4)),
        "--stateful_min_short_ratio",
        str(stateful.get("min_short_ratio", 0.75)),
        "--stateful_min_model_score",
        str(stateful.get("min_model_score", 0.0)),
    ]
    if stateful.get("merge"):
        cmd.extend(["--stateful_merge", str(stateful["merge"])])
    _extend_values(cmd, "--stateful_allowed_protocols", stateful.get("allowed_protocols"))
    _extend_values(cmd, "--stateful_allowed_ports", stateful.get("allowed_ports"))
    _extend_values(cmd, "--stateful_excluded_ports", stateful.get("excluded_ports"))
    return cmd


def _alert_guard_args(profile: dict[str, Any]) -> list[str]:
    guard = profile.get("alert_guard") or {}
    if not guard.get("enabled", False):
        return []
    cmd: list[str] = []
    _extend_values(cmd, "--alert_allowed_protocols", guard.get("allowed_protocols"))
    _extend_values(cmd, "--alert_allowed_ports", guard.get("allowed_ports"))
    _extend_values(cmd, "--alert_excluded_ports", guard.get("excluded_ports"))
    return cmd


def _summary_kind(path: str | None, payload: dict[str, Any] | None) -> str | None:
    if not path:
        return None
    if payload and "panel_summary" in payload:
        return "panel"
    if payload and "summary" in payload:
        return "online_replay"
    if payload and ("blocked_true_positives" in payload or "blocked_false_positives" in payload):
        return "audit"
    name = Path(path).name
    if name == "online_replay_panel_summary.json":
        return "panel"
    if name == "online_replay_summary.json":
        return "online_replay"
    if name == "alert_guard_audit.json":
        return "audit"
    return None


def _scope_for_replay(validation_key: str, payload: dict[str, Any] | None) -> str:
    if payload and payload.get("scope"):
        return str(payload["scope"])
    if validation_key == "all_scope_summary":
        return "all"
    return "test"


def _online_replay_command(
    profile: dict[str, Any],
    out_path: str,
    validation_key: str,
    payload: dict[str, Any] | None,
) -> list[str]:
    cmd = [
        "python3",
        "scripts/16_online_replay.py",
        "--tokens",
        str(profile.get("tokens")),
    ]
    if profile.get("metadata_tokens"):
        cmd.extend(["--metadata_tokens", str(profile["metadata_tokens"])])
    cmd.extend(
        [
            "--checkpoint",
            str(profile.get("checkpoint")),
            "--config",
            str(profile.get("config", "configs/model_behavior_composer.yaml")),
            "--split",
            str(profile.get("split", "temporal_stratified")),
            "--scope",
            _scope_for_replay(validation_key, payload),
            "--threshold",
            str(profile.get("threshold")),
            "--consecutive_alerts",
            "1",
            "--fpr_windows",
            "60",
            "300",
            "3600",
        ]
    )
    cmd.extend(_stateful_args(profile))
    cmd.extend(_alert_guard_args(profile))
    cmd.extend(["--out", str(Path(out_path).parent)])
    return cmd


def _panel_command(profile: dict[str, Any], out_path: str, payload: dict[str, Any] | None) -> list[str]:
    cmd = [
        "python3",
        "scripts/18_run_online_replay_panel.py",
        "--checkpoint",
        str(profile.get("checkpoint")),
        "--config",
        str(profile.get("config", "configs/model_behavior_composer.yaml")),
        "--split",
        str(profile.get("split", "temporal_stratified")),
        "--scope",
        str(payload.get("scope", "test") if payload else "test"),
        "--threshold",
        str(profile.get("threshold")),
    ]
    cmd.extend(_stateful_args(profile))
    cmd.extend(_alert_guard_args(profile))
    if profile.get("metadata_tokens"):
        cmd.extend(["--clean_metadata_tokens", str(profile["metadata_tokens"])])
    cmd.extend(["--out_root", str(Path(out_path).parent)])
    return cmd


def _audit_command(out_path: str) -> list[str]:
    out = Path(out_path)
    return [
        "python3",
        "scripts/19_audit_alert_guard.py",
        "--scores",
        str(out.parent / "online_replay_scores.jsonl"),
        "--out",
        str(out),
    ]


def _reproduce_commands(profile: dict[str, Any], validation: dict[str, Any], payloads: dict[str, dict[str, Any] | None]) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for key in ("primary_summary", "panel_summary", "all_scope_summary", "audit"):
        path = validation.get(key)
        if not path:
            continue
        payload = payloads.get(key)
        kind = _summary_kind(path, payload)
        if kind == "online_replay":
            commands[key] = _online_replay_command(profile, path, key, payload)
        elif kind == "panel":
            commands[key] = _panel_command(profile, path, payload)
        elif kind == "audit":
            commands[key] = _audit_command(path)
    return commands


def build_report(config_path: str | Path) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    profiles = cfg.get("profiles", {})
    rows: list[dict[str, Any]] = []
    for name, profile in profiles.items():
        validation = profile.get("validation", {})
        primary = _read_json_optional(validation.get("primary_summary"))
        panel = _read_json_optional(validation.get("panel_summary"))
        all_scope = _read_json_optional(validation.get("all_scope_summary"))
        audit = _read_json_optional(validation.get("audit"))
        missing_paths = _missing_validation_paths(validation)
        payloads = {
            "primary_summary": primary,
            "panel_summary": panel,
            "all_scope_summary": all_scope,
            "audit": audit,
        }
        rows.append(
            {
                "name": name,
                "description": profile.get("description"),
                "threshold": profile.get("threshold"),
                "stateful_service": profile.get("stateful_service", {}),
                "alert_guard": profile.get("alert_guard", {}),
                "validation_notes": validation.get("notes"),
                "primary_metrics": _summary_metrics(primary),
                "panel_metrics": _summary_metrics(panel),
                "all_scope_metrics": _summary_metrics(all_scope),
                "audit_metrics": _audit_metrics(audit),
                "missing_validation_paths": missing_paths,
                "reproduce_commands": _reproduce_commands(profile, validation, payloads),
                "paths": {
                    "primary_summary": validation.get("primary_summary"),
                    "panel_summary": validation.get("panel_summary"),
                    "all_scope_summary": validation.get("all_scope_summary"),
                    "audit": validation.get("audit"),
                },
            }
        )
    return {"config": str(config_path), "profiles": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="configs/online_policy_profiles.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--strict", action="store_true", help="Fail if a configured validation artifact is missing.")
    args = parser.parse_args()

    report = build_report(args.profiles)
    missing = [
        {"name": row["name"], "paths": row["missing_validation_paths"]}
        for row in report["profiles"]
        if row["missing_validation_paths"]
    ]
    if args.strict and missing:
        raise SystemExit(f"missing validation artifacts: {missing}")
    write_json(report, args.out)
    print({"profiles": [row["name"] for row in report["profiles"]], "missing": missing})


if __name__ == "__main__":
    main()
