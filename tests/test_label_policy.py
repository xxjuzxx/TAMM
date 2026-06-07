from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cicids2017_labeler import read_cicids_flow_csvs
from src.data.label_policy import apply_attempted_policy, binary_label_for, merged_cicids_label


def _baseline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "21_train_cicflowmeter_baseline.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("cicflowmeter_baseline", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_attempted_policy_variants() -> None:
    assert apply_attempted_policy("Attempted-DDoS", "keep") == "Attempted-DDoS"
    assert apply_attempted_policy("Attempted-DDoS", "drop") is None
    assert apply_attempted_policy("Attempted-DDoS", "attack") == "DDoS"
    assert apply_attempted_policy("Attempted-DDoS", "benign") == "BENIGN"
    assert binary_label_for("Normal") == "BENIGN"
    assert binary_label_for("Attempted-DDoS") == "ATTACK"
    assert merged_cicids_label("Web Attack - Sql Injection") == "WebAttack"


def test_read_cicids_flow_csvs_parses_timestamp_and_drops_attempted(tmp_path) -> None:
    path = tmp_path / "flows.csv"
    path.write_text(
        "\n".join(
            [
                "Source IP,Source Port,Destination IP,Destination Port,Protocol,Timestamp,Flow Duration,Label",
                "10.0.0.1,1234,10.0.0.2,80,6,2017-07-05 10:00:00,2000000,Attempted-DDoS",
                "10.0.0.3,1235,10.0.0.4,443,6,2017-07-05 10:00:01,1,Benign",
            ]
        ),
        encoding="utf-8",
    )
    rows = read_cicids_flow_csvs([path], attempted_policy="drop")
    assert len(rows) == 1
    assert rows[0]["label"] == "Benign"
    assert rows[0]["start_ts"] > 1_400_000_000
    assert rows[0]["end_ts"] == rows[0]["start_ts"] + 1


def test_read_cicids_flow_csvs_parses_fractional_timestamps_as_utc(tmp_path) -> None:
    path = tmp_path / "flows.csv"
    path.write_text(
        "\n".join(
            [
                "Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Flow Duration,Label",
                "172.16.0.1,36198,192.168.10.50,80,6,2017-07-06 13:40:13.192475,5070496,Web Attack - SQL Injection",
            ]
        ),
        encoding="utf-8",
    )
    rows = read_cicids_flow_csvs([path], attempted_policy="drop")
    expected = datetime(2017, 7, 6, 13, 40, 13, 192475, tzinfo=timezone.utc).timestamp()
    assert len(rows) == 1
    assert rows[0]["label"] == "Web Attack - SQL Injection"
    assert rows[0]["start_ts"] == expected
    assert rows[0]["end_ts"] == expected + 5.070496


def test_invalid_label_timestamp_does_not_fallback_to_tuple_only_match(tmp_path) -> None:
    from src.data.cicids2017_labeler import label_flows

    path = tmp_path / "flows.csv"
    path.write_text(
        "\n".join(
            [
                "Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Flow Duration,Label",
                "10.0.0.1,1234,10.0.0.2,80,6,not-a-timestamp,1000000,DDoS",
            ]
        ),
        encoding="utf-8",
    )
    rows = read_cicids_flow_csvs([path], attempted_policy="drop")
    flow = {
        "src_ip": "10.0.0.1",
        "src_port": "1234",
        "dst_ip": "10.0.0.2",
        "dst_port": "80",
        "protocol": "TCP",
        "start_ts": 100.0,
    }
    labeled, unmatched, stats = label_flows([flow], rows, tolerance_seconds=2.0)
    assert not labeled
    assert unmatched == [flow]
    assert stats["label_timestamp_status_counts"] == {"invalid": 1}


def test_cicflowmeter_baseline_label_policy_drop_and_attack() -> None:
    baseline = _baseline_module()
    frame = pd.DataFrame(
        {
            "Label": ["Benign", "Attempted-DDoS", "DDoS"],
            "Flow Duration": [1, 2, 3],
        }
    )
    filtered, labels, stats = baseline._apply_label_policy(frame, "Label", "drop")
    assert filtered.shape[0] == 2
    assert labels.tolist() == ["Benign", "DDoS"]
    assert stats["dropped_rows"] == 1

    filtered, labels, stats = baseline._apply_label_policy(frame, "Label", "attack")
    assert filtered.shape[0] == 3
    assert labels.tolist() == ["Benign", "DDoS", "DDoS"]
    assert stats["resolved_label_counts"]["DDoS"] == 2


def test_cicflowmeter_raw_label_temporal_split_keeps_webattack_subtypes() -> None:
    baseline = _baseline_module()
    frame = pd.DataFrame(
        {
            "Label": (
                ["Benign"] * 5
                + ["Web Attack - Brute Force"] * 5
                + ["Web Attack - SQL Injection"] * 5
                + ["Web Attack - XSS"] * 5
            ),
            "Flow Duration": list(range(20)),
        }
    )

    filtered, labels, _ = baseline._apply_label_policy(frame, "Label", "drop")
    y, target_names = baseline._labels(labels, "multiclass_merged")
    groups = baseline._raw_label_group_ids(labels)
    train_idx, val_idx, test_idx = baseline._temporal_stratified_by_group_indices(
        y,
        groups,
        val_ratio=0.2,
        test_ratio=0.2,
        order_values=np.arange(len(filtered)),
    )

    assert target_names == ["BENIGN", "WebAttack"]
    assert len(set(y[groups == groups[5]].tolist())) == 1
    assert len(set(groups[5:15].tolist())) == 2
    assert labels.iloc[train_idx].value_counts().to_dict() == {
        "Benign": 3,
        "Web Attack - Brute Force": 3,
        "Web Attack - SQL Injection": 3,
        "Web Attack - XSS": 3,
    }
    assert labels.iloc[val_idx].value_counts().to_dict() == {
        "Benign": 1,
        "Web Attack - Brute Force": 1,
        "Web Attack - SQL Injection": 1,
        "Web Attack - XSS": 1,
    }
    assert labels.iloc[test_idx].value_counts().to_dict() == {
        "Benign": 1,
        "Web Attack - Brute Force": 1,
        "Web Attack - SQL Injection": 1,
        "Web Attack - XSS": 1,
    }
