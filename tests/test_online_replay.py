from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load_online_replay_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("online_replay", scripts_dir / "16_online_replay.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_summary_reports_delay_and_false_positive_timing() -> None:
    module = _load_online_replay_module()
    rows = [
        {
            "stream_pos": 0,
            "binary_label_id": 0,
            "label": "Benign",
            "prediction": 0,
            "attack_probability": 0.01,
            "is_false_positive": False,
            "is_true_positive": False,
            "start_ts": 0.0,
        },
        {
            "stream_pos": 1,
            "binary_label_id": 0,
            "label": "Benign",
            "prediction": 1,
            "attack_probability": 0.90,
            "is_false_positive": True,
            "is_true_positive": False,
            "start_ts": 10.0,
        },
        {
            "stream_pos": 2,
            "binary_label_id": 1,
            "label": "DDoS",
            "prediction": 0,
            "attack_probability": 0.20,
            "is_false_positive": False,
            "is_true_positive": False,
            "start_ts": 20.0,
        },
        {
            "stream_pos": 3,
            "binary_label_id": 1,
            "label": "DDoS",
            "prediction": 1,
            "attack_probability": 0.80,
            "is_false_positive": False,
            "is_true_positive": True,
            "start_ts": 25.0,
        },
        {
            "stream_pos": 4,
            "binary_label_id": 1,
            "label": "Bot",
            "prediction": 0,
            "attack_probability": 0.20,
            "is_false_positive": False,
            "is_true_positive": False,
            "start_ts": 30.0,
        },
    ]

    out = module.summarize_replay(rows, consecutive_alerts=1, fpr_windows=[15.0])
    summary = out["summary"]
    assert summary["false_positives"] == 1
    assert summary["false_positive_rate"] == 0.5
    assert summary["time_to_first_false_positive_seconds"] == 10.0
    assert summary["max_sliding_false_positive_rate"]["15.0"] == 0.5
    assert summary["attack_labels"] == 2
    assert summary["detected_labels"] == 1
    assert summary["missed_labels"] == 1
    assert summary["mean_delay_seconds"] == 5.0
    assert summary["mean_delay_flows"] == 1.0


def test_best_threshold_handles_single_class_warmup() -> None:
    module = _load_online_replay_module()
    threshold, value = module._best_threshold([0, 0, 0], [0.01, 0.02, 0.03], "macro_f1")
    assert threshold > 0.03
    assert value == 1.0
    assert module._label_counts([0, 0, 0]) == {"benign": 3, "attack": 0, "num_classes": 1}


def test_calibration_requires_configured_class_count() -> None:
    module = _load_online_replay_module()
    labels = np.array([0, 0, 0])
    items = [(0, [0.99, 0.01]), (1, [0.98, 0.02]), (2, [0.97, 0.03])]
    try:
        module._calibrate_threshold(labels, items, metric="macro_f1", min_classes=2)
    except ValueError as exc:
        assert "min_calibration_classes" in str(exc)
    else:
        raise AssertionError("expected ValueError for single-class calibration")
    threshold, value, counts = module._calibrate_threshold(labels, items, metric="macro_f1", min_classes=1)
    assert threshold > 0.03
    assert value == 1.0
    assert counts == {"benign": 3, "attack": 0, "num_classes": 1}


def test_stateful_service_monitor_uses_prior_rows_and_service_keys() -> None:
    module = _load_online_replay_module()
    token_data = {
        "binary_labels": torch.tensor([0, 0, 0, 0]),
        "meta": [
            {"flow_id": "svc-a-1", "start_ts": 0.0, "service_key": ["dst", "80", "TCP"], "packet_count": 2},
            {"flow_id": "svc-b-1", "start_ts": 0.5, "service_key": ["dst", "443", "TCP"], "packet_count": 2},
            {"flow_id": "svc-a-2", "start_ts": 1.0, "service_key": ["dst", "80", "TCP"], "packet_count": 2},
            {"flow_id": "svc-a-3", "start_ts": 1.5, "service_key": ["dst", "80", "TCP"], "packet_count": 2},
        ],
    }
    rows = module._replay_rows(
        token_data,
        np.array([0, 1, 2, 3]),
        np.array([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01]]),
        threshold=0.9,
        warmup_count=0,
        enable_stateful_service=True,
        stateful_window_seconds=2.0,
        stateful_min_count=3,
        stateful_max_packets=4,
        stateful_min_short_ratio=1.0,
        stateful_min_model_score=0.0,
    )

    by_id = {row["flow_id"]: row for row in rows}
    assert by_id["svc-a-1"]["stateful_recent_count"] == 0
    assert by_id["svc-b-1"]["stateful_recent_count"] == 0
    assert by_id["svc-a-2"]["stateful_recent_count"] == 1
    assert by_id["svc-a-3"]["stateful_recent_count"] == 2
    assert by_id["svc-a-3"]["model_prediction"] == 0
    assert by_id["svc-a-3"]["stateful_prediction"] == 1
    assert by_id["svc-a-3"]["prediction"] == 1


def test_stateful_service_monitor_respects_window_and_score_gate() -> None:
    module = _load_online_replay_module()
    token_data = {
        "binary_labels": torch.tensor([0, 0, 0]),
        "meta": [
            {"flow_id": "old", "start_ts": 0.0, "service_key": ["dst", "80", "TCP"], "packet_count": 2},
            {"flow_id": "current-low-score", "start_ts": 3.0, "service_key": ["dst", "80", "TCP"], "packet_count": 2},
            {"flow_id": "current-high-score", "start_ts": 3.5, "service_key": ["dst", "80", "TCP"], "packet_count": 2},
        ],
    }
    rows = module._replay_rows(
        token_data,
        np.array([0, 1, 2]),
        np.array([[0.99, 0.01], [0.999, 0.001], [0.90, 0.10]]),
        threshold=0.9,
        warmup_count=0,
        enable_stateful_service=True,
        stateful_window_seconds=1.0,
        stateful_min_count=2,
        stateful_max_packets=4,
        stateful_min_short_ratio=1.0,
        stateful_min_model_score=0.01,
    )

    by_id = {row["flow_id"]: row for row in rows}
    assert by_id["current-low-score"]["stateful_recent_count"] == 0
    assert by_id["current-low-score"]["stateful_prediction"] == 0
    assert by_id["current-high-score"]["stateful_recent_count"] == 1
    assert by_id["current-high-score"]["stateful_prediction"] == 1


def test_stateful_service_monitor_skips_rows_without_service_key() -> None:
    module = _load_online_replay_module()
    token_data = {
        "binary_labels": torch.tensor([0, 0, 0]),
        "meta": [
            {"flow_id": "a", "start_ts": 0.0, "packet_count": 2, "dataset_file": "benign.csv"},
            {"flow_id": "b", "start_ts": 0.1, "packet_count": 2, "dataset_file": "benign.csv"},
            {"flow_id": "c", "start_ts": 0.2, "packet_count": 2, "dataset_file": "benign.csv"},
        ],
    }
    rows = module._replay_rows(
        token_data,
        np.array([0, 1, 2]),
        np.array([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01]]),
        threshold=0.9,
        warmup_count=0,
        enable_stateful_service=True,
        stateful_window_seconds=2.0,
        stateful_min_count=2,
        stateful_max_packets=4,
        stateful_min_short_ratio=1.0,
        stateful_min_model_score=0.0,
    )

    assert [row["stateful_missing_key"] for row in rows] == [True, True, True]
    assert [row["stateful_prediction"] for row in rows] == [0, 0, 0]
    assert [row["prediction"] for row in rows] == [0, 0, 0]


def test_stateful_service_monitor_respects_protocol_and_port_guards() -> None:
    module = _load_online_replay_module()
    token_data = {
        "binary_labels": torch.tensor([0, 0, 0, 0]),
        "meta": [
            {"flow_id": "dns-1", "start_ts": 0.0, "service_key": ["dst", "53", "UDP"], "packet_count": 2},
            {"flow_id": "dns-2", "start_ts": 0.1, "service_key": ["dst", "53", "UDP"], "packet_count": 2},
            {"flow_id": "web-1", "start_ts": 1.0, "service_key": ["dst", "80", "TCP"], "packet_count": 2},
            {"flow_id": "web-2", "start_ts": 1.1, "service_key": ["dst", "80", "TCP"], "packet_count": 2},
        ],
    }
    rows = module._replay_rows(
        token_data,
        np.array([0, 1, 2, 3]),
        np.array([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01]]),
        threshold=0.9,
        warmup_count=0,
        enable_stateful_service=True,
        stateful_window_seconds=2.0,
        stateful_min_count=2,
        stateful_max_packets=4,
        stateful_min_short_ratio=1.0,
        stateful_min_model_score=0.0,
        stateful_allowed_protocols={"TCP"},
        stateful_allowed_ports={"80"},
        stateful_excluded_ports={"53"},
    )

    by_id = {row["flow_id"]: row for row in rows}
    assert by_id["dns-2"]["stateful_guard_allowed"] is False
    assert by_id["dns-2"]["stateful_prediction"] == 0
    assert by_id["web-2"]["stateful_guard_allowed"] is True
    assert by_id["web-2"]["stateful_prediction"] == 1
    assert by_id["web-2"]["stateful_service_port"] == "80"
    assert by_id["web-2"]["stateful_service_protocol"] == "TCP"


def test_alert_guard_blocks_final_predictions_by_service() -> None:
    module = _load_online_replay_module()
    token_data = {
        "binary_labels": torch.tensor([0, 1]),
        "meta": [
            {"flow_id": "dns", "start_ts": 0.0, "service_key": ["dst", "53", "UDP"], "packet_count": 5},
            {"flow_id": "web", "start_ts": 1.0, "service_key": ["dst", "80", "TCP"], "packet_count": 5},
        ],
    }
    rows = module._replay_rows(
        token_data,
        np.array([0, 1]),
        np.array([[0.01, 0.99], [0.01, 0.99]]),
        threshold=0.9,
        warmup_count=0,
        alert_allowed_protocols={"TCP"},
        alert_allowed_ports={"80"},
    )

    by_id = {row["flow_id"]: row for row in rows}
    assert by_id["dns"]["pre_guard_prediction"] == 1
    assert by_id["dns"]["alert_guard_allowed"] is False
    assert by_id["dns"]["alert_guard_blocked_prediction"] is True
    assert by_id["dns"]["prediction"] == 0
    assert by_id["web"]["alert_guard_allowed"] is True
    assert by_id["web"]["prediction"] == 1


def test_metadata_tokens_merge_service_fields_by_matching_flow_id() -> None:
    module = _load_online_replay_module()
    token_data = {
        "meta": [
            {"flow_id": "a", "packet_count": 2},
            {"flow_id": "b", "packet_count": 3},
        ]
    }
    metadata_data = {
        "meta": [
            {"flow_id": "a", "service_key": ["dst", "80", "TCP"]},
            {"flow_id": "b", "service_key": ["dst", "443", "TCP"], "service_context": {"recent_count": 1}},
        ]
    }
    merged = module._merged_metadata_rows(token_data, metadata_data)
    assert merged[0]["packet_count"] == 2
    assert merged[0]["service_key"] == ["dst", "80", "TCP"]
    assert merged[1]["service_context"] == {"recent_count": 1}

    try:
        module._merged_metadata_rows(token_data, {"meta": [{"flow_id": "x"}, {"flow_id": "b"}]})
    except ValueError as exc:
        assert "row order does not match" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched metadata flow_id")
