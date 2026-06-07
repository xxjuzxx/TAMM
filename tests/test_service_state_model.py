from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "38_train_service_state_model.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("service_state_model", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_service_state_features_use_prior_same_service_rows_only() -> None:
    module = _module()
    meta_rows = [
        {"flow_id": "a1", "start_ts": 0.0, "service_key": ["dst", "80", "TCP"], "packet_count": 2, "token_count": 10},
        {"flow_id": "b1", "start_ts": 0.5, "service_key": ["dst", "443", "TCP"], "packet_count": 2, "token_count": 10},
        {"flow_id": "a2", "start_ts": 1.0, "service_key": ["dst", "80", "TCP"], "packet_count": 3, "token_count": 11},
        {"flow_id": "a3", "start_ts": 3.5, "service_key": ["dst", "80", "TCP"], "packet_count": 10, "token_count": 12},
    ]

    matrix, names = module.build_service_state_matrix(meta_rows, windows=[2.0])
    by_name = {name: idx for idx, name in enumerate(names)}

    assert matrix[0, by_name["w2_log_recent_count"]] == 0.0
    assert matrix[1, by_name["w2_log_recent_count"]] == 0.0
    assert matrix[2, by_name["w2_log_recent_count"]] > 0.0
    assert matrix[2, by_name["w2_episode_short4_count_log"]] > matrix[0, by_name["w2_episode_short4_count_log"]]
    assert matrix[3, by_name["w2_log_recent_count"]] == 0.0
    assert matrix[3, by_name["prior_total_log_count"]] > matrix[2, by_name["prior_total_log_count"]]


def test_metadata_merge_copies_service_context_without_reordering() -> None:
    module = _module()
    primary = {
        "meta": [
            {"flow_id": "f1", "packet_count": 1},
            {"flow_id": "f2", "packet_count": 2, "service_key": ["base", "80", "TCP"]},
        ]
    }
    supplemental = {
        "meta": [
            {"flow_id": "f1", "service_key": ["dst", "53", "UDP"], "service_context": {"recent_count": 1}},
            {"flow_id": "f2", "service_key": ["dst", "443", "TCP"], "service_context": {"recent_count": 2}},
        ]
    }

    rows = module._merge_metadata_rows(primary, supplemental)

    assert rows[0]["service_key"] == ["dst", "53", "UDP"]
    assert rows[0]["service_context"] == {"recent_count": 1}
    assert rows[1]["service_key"] == ["base", "80", "TCP"]
    assert rows[1]["service_context"] == {"recent_count": 2}


def test_metadata_merge_rejects_mismatched_flow_order() -> None:
    module = _module()
    primary = {"meta": [{"flow_id": "f1"}]}
    supplemental = {"meta": [{"flow_id": "f2", "service_key": ["dst", "80", "TCP"]}]}

    try:
        module._merge_metadata_rows(primary, supplemental)
    except ValueError as exc:
        assert "row order does not match" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched metadata order")


def test_feature_set_selection_splits_current_history_and_identity() -> None:
    module = _module()
    names = [
        "current_log_packet_count",
        "missing_service_key",
        "prior_total_log_count",
        "w2_log_recent_count",
        "proto_TCP",
        "port_80",
    ]

    assert module._feature_indices(names, "all") == [0, 1, 2, 3, 4, 5]
    assert module._feature_indices(names, "history") == [2, 3]
    assert module._feature_indices(names, "current") == [0, 1, 4, 5]
