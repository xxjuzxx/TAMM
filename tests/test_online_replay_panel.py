from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_panel_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("online_replay_panel", scripts_dir / "18_run_online_replay_panel.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_replay_panel_summary_reports_worst_metrics() -> None:
    module = _load_panel_module()
    rows = [
        {"name": "clean", "macro_f1": 0.99, "attack_recall_online": 1.0, "false_positive_rate": 0.01},
        {"name": "packet_insert", "macro_f1": 0.95, "attack_recall_online": 0.98, "false_positive_rate": 0.02},
        {"name": "low_rate", "macro_f1": 0.97, "attack_recall_online": 0.99, "false_positive_rate": 0.00},
    ]
    summary = module.summarize_panel(rows)
    assert summary["num_panels"] == 3
    assert summary["worst_macro_f1"] == {"name": "packet_insert", "value": 0.95}
    assert summary["worst_attack_recall"] == {"name": "packet_insert", "value": 0.98}
    assert summary["worst_false_positive_rate"] == {"name": "packet_insert", "value": 0.02}


def test_online_replay_panel_command_adds_clean_metadata_only_for_clean() -> None:
    module = _load_panel_module()
    args = SimpleNamespace(
        checkpoint="model.pt",
        config="configs/model_behavior_composer.yaml",
        split="temporal_stratified",
        scope="test",
        threshold=0.16,
        consecutive_alerts=1,
        fpr_windows=[60.0, 300.0],
        enable_stateful_service=True,
        stateful_window_seconds=2.0,
        stateful_min_count=40,
        stateful_max_packets=4,
        stateful_min_short_ratio=0.75,
        stateful_min_model_score=0.01,
        stateful_merge="or",
        stateful_allowed_protocols=["TCP"],
        stateful_allowed_ports=["80"],
        stateful_excluded_ports=["53"],
        alert_allowed_protocols=["TCP"],
        alert_allowed_ports=["80"],
        alert_excluded_ports=["53"],
        clean_metadata_tokens="svcctx.pt",
    )
    clean_cmd = module._replay_command(args, "clean", "clean.pt", Path("out") / "clean")
    low_rate_cmd = module._replay_command(args, "low_rate_c2_070", "low.pt", Path("out") / "low")
    assert "--metadata_tokens" in clean_cmd
    assert "svcctx.pt" in clean_cmd
    assert "--metadata_tokens" not in low_rate_cmd
    assert "--enable_stateful_service" in clean_cmd
    assert "--stateful_allowed_protocols" in clean_cmd
    assert "TCP" in clean_cmd
    assert "--stateful_allowed_ports" in clean_cmd
    assert "80" in clean_cmd
    assert "--stateful_excluded_ports" in clean_cmd
    assert "53" in clean_cmd
    assert "--alert_allowed_protocols" in clean_cmd
    assert "--alert_allowed_ports" in clean_cmd
    assert "--alert_excluded_ports" in clean_cmd
