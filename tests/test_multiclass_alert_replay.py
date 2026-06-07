from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "multiclass_alert_replay",
        scripts_dir / "36_eval_zeek_multiclass_alert_replay.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binary_metrics_reports_fpr_and_attack_recall() -> None:
    module = _load_module()

    metrics = module._binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.9, 0.8, 0.2]),
        threshold=0.5,
    )

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["attack_recall"] == 0.5
    assert metrics["num_benign"] == 2
    assert metrics["num_attack"] == 2


def test_maximin_threshold_uses_worst_condition() -> None:
    module = _load_module()
    calibration_items = [
        (np.array([0, 1]), np.array([0.4, 0.6])),
        (np.array([0, 1]), np.array([0.2, 0.8])),
    ]

    threshold, value, mean_value = module._best_pooled_threshold(
        calibration_items,
        metric="macro_f1",
        policy="maximin",
    )

    assert 0.4 < threshold <= 0.6
    assert value == 1.0
    assert mean_value == 1.0


def test_online_replay_summary_orders_by_time_and_reports_delay() -> None:
    module = _load_module()
    token_data = {
        "meta": [
            {"flow_id": "attack-2", "label": "DDoS", "start_ts": 20.0},
            {"flow_id": "benign", "label": "BENIGN", "start_ts": 0.0},
            {"flow_id": "attack-1", "label": "DDoS", "start_ts": 10.0},
        ]
    }
    indices = np.array([0, 1, 2])
    y_binary = np.array([1, 0, 1])
    alert_scores = np.array([0.8, 0.9, 0.2])

    out = module._online_replay_summary(
        token_data,
        indices,
        y_binary,
        alert_scores,
        threshold=0.5,
        consecutive_alerts=1,
        fpr_windows=[60.0],
    )
    summary = out["summary"]

    assert summary["false_positives"] == 1
    assert summary["true_positives"] == 1
    assert summary["false_positive_rate"] == 1.0
    assert summary["attack_recall_online"] == 0.5
    assert summary["time_to_first_false_positive_seconds"] == 0.0
    assert summary["detected_labels"] == 1
    assert summary["median_delay_seconds"] == 10.0
    assert summary["median_delay_flows"] == 1.0
