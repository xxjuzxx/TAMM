from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("online_policy_profiles", scripts_dir / "20_report_online_policy_profiles.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_policy_profile_report_collects_metrics(tmp_path: Path) -> None:
    module = _load_report_module()
    primary = tmp_path / "summary.json"
    panel = tmp_path / "panel.json"
    audit = tmp_path / "audit.json"
    primary.write_text(
        json.dumps({"summary": {"macro_f1": 0.99, "false_positive_rate": 0.01, "attack_recall_online": 1.0}}),
        encoding="utf-8",
    )
    panel.write_text(
        json.dumps({"panel_summary": {"mean_macro_f1": 0.98, "worst_macro_f1": {"name": "x", "value": 0.9}}}),
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps({"blocked_true_positives": 5, "blocked_false_positives": 1}),
        encoding="utf-8",
    )
    config = tmp_path / "profiles.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "p": {
                        "description": "profile",
                        "threshold": 0.16,
                        "stateful_service": {"enabled": False},
                        "alert_guard": {"enabled": True},
                        "validation": {
                            "primary_summary": str(primary),
                            "panel_summary": str(panel),
                            "audit": str(audit),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = module.build_report(config)
    row = report["profiles"][0]
    assert row["name"] == "p"
    assert row["primary_metrics"]["macro_f1"] == 0.99
    assert row["panel_metrics"]["mean_macro_f1"] == 0.98
    assert row["audit_metrics"]["blocked_true_positives"] == 5
    assert row["missing_validation_paths"] == []
    primary_cmd = row["reproduce_commands"]["primary_summary"]
    panel_cmd = row["reproduce_commands"]["panel_summary"]
    audit_cmd = row["reproduce_commands"]["audit"]
    assert primary_cmd[:2] == ["python3", "scripts/16_online_replay.py"]
    assert "--scope" in primary_cmd
    assert primary_cmd[primary_cmd.index("--scope") + 1] == "test"
    assert panel_cmd[:2] == ["python3", "scripts/18_run_online_replay_panel.py"]
    assert audit_cmd[:2] == ["python3", "scripts/19_audit_alert_guard.py"]


def test_online_policy_profile_report_records_missing_validation_paths(tmp_path: Path) -> None:
    module = _load_report_module()
    missing = tmp_path / "missing_summary.json"
    config = tmp_path / "profiles.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "p": {
                        "description": "profile",
                        "validation": {
                            "primary_summary": str(missing),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = module.build_report(config)
    row = report["profiles"][0]
    assert row["primary_metrics"] is None
    assert row["missing_validation_paths"] == [str(missing)]


def test_online_policy_profile_report_builds_stateful_guard_commands(tmp_path: Path) -> None:
    module = _load_report_module()
    all_scope = tmp_path / "online_replay_summary.json"
    panel = tmp_path / "online_replay_panel_summary.json"
    all_scope.write_text(json.dumps({"scope": "all", "summary": {"macro_f1": 0.9}}), encoding="utf-8")
    panel.write_text(json.dumps({"scope": "test", "panel_summary": {"mean_macro_f1": 0.9}}), encoding="utf-8")
    config = tmp_path / "profiles.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "p": {
                        "checkpoint": "model.pt",
                        "tokens": "tokens.pt",
                        "metadata_tokens": "metadata.pt",
                        "split": "temporal_stratified",
                        "threshold": 0.16,
                        "stateful_service": {
                            "enabled": True,
                            "window_seconds": 2.0,
                            "min_count": 40,
                            "max_packets": 4,
                            "min_short_ratio": 0.75,
                            "min_model_score": 0.01,
                            "allowed_protocols": ["TCP"],
                            "allowed_ports": ["80"],
                        },
                        "alert_guard": {
                            "enabled": True,
                            "allowed_protocols": ["TCP"],
                            "allowed_ports": ["80"],
                        },
                        "validation": {
                            "all_scope_summary": str(all_scope),
                            "panel_summary": str(panel),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    row = module.build_report(config)["profiles"][0]
    all_scope_cmd = row["reproduce_commands"]["all_scope_summary"]
    panel_cmd = row["reproduce_commands"]["panel_summary"]
    assert all_scope_cmd[all_scope_cmd.index("--scope") + 1] == "all"
    assert "--metadata_tokens" in all_scope_cmd
    assert "--enable_stateful_service" in all_scope_cmd
    assert "--stateful_allowed_protocols" in all_scope_cmd
    assert "--stateful_allowed_ports" in all_scope_cmd
    assert "--alert_allowed_protocols" in all_scope_cmd
    assert "--alert_allowed_ports" in all_scope_cmd
    assert "--clean_metadata_tokens" in panel_cmd
    assert "metadata.pt" in panel_cmd
