from __future__ import annotations

import pytest

from src.data.leakage_check import LeakageError, assert_no_leakage, build_leakage_report


def test_leakage_check_rejects_overlapping_flow_ids() -> None:
    report = build_leakage_report(
        split_payload={"splits": {"train": ["a"], "val": ["a"], "test": []}},
        profile_manifest={"train_only": True, "splits": "splits.json"},
        token_manifest={"train_only": True, "splits": "splits.json", "threshold_tuning_split": "val"},
        vocab={"[PAD]": 0, "[CLS]": 1},
    )
    assert report["passed"] is False
    with pytest.raises(LeakageError):
        assert_no_leakage(report)


def test_leakage_check_rejects_raw_identifier_vocab() -> None:
    report = build_leakage_report(
        split_payload={"splits": {"train": ["a"], "val": ["b"], "test": ["c"]}},
        profile_manifest={"train_only": True, "splits": "splits.json"},
        token_manifest={"train_only": True, "splits": "splits.json", "threshold_tuning_split": "val"},
        vocab={"[PAD]": 0, "10.0.0.1": 1, "DPORT_443": 2},
    )
    assert report["passed"] is False
    assert report["checks"]["L4_vocab_no_raw_ip"]["ok"] is False
    assert report["checks"]["L6_vocab_no_raw_port"]["ok"] is False


def test_leakage_check_accepts_clean_train_only_artifacts() -> None:
    report = build_leakage_report(
        split_payload={"splits": {"train": ["a"], "val": ["b"], "test": ["c"]}},
        profile_manifest={"train_only": True, "splits": "splits.json"},
        token_manifest={"train_only": True, "splits": "splits.json", "threshold_tuning_split": "val"},
        vocab={"[PAD]": 0, "PKT_LEN_1": 1, "PKT_IAT_2": 2},
        token_data={"meta": [{"flow_id": "a", "has_ip_token": False, "has_abs_time_token": False, "has_port_token": False}]},
    )
    assert report["passed"] is True
