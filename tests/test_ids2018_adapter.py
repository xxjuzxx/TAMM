from __future__ import annotations

from src.data.ids2018_adapter import _estimate_time_offset_seconds, _label_flows_time_indexed


def test_ids2018_time_offset_estimator_uses_hour_mode() -> None:
    flows = [
        {
            "flow_id": "f1",
            "src_ip": "10.0.0.1",
            "src_port": "1111",
            "dst_ip": "10.0.0.2",
            "dst_port": "80",
            "protocol": "TCP",
            "start_ts": 114400.0,
        },
        {
            "flow_id": "f2",
            "src_ip": "10.0.0.3",
            "src_port": "2222",
            "dst_ip": "10.0.0.2",
            "dst_port": "80",
            "protocol": "TCP",
            "start_ts": 114410.0,
        },
    ]
    candidates = [
        {
            "key": ("10.0.0.1", "1111", "10.0.0.2", "80", "TCP"),
            "reverse_key": ("10.0.0.2", "80", "10.0.0.1", "1111", "TCP"),
            "start_ts": 100000.0,
        },
        {
            "key": ("10.0.0.3", "2222", "10.0.0.2", "80", "TCP"),
            "reverse_key": ("10.0.0.2", "80", "10.0.0.3", "2222", "TCP"),
            "start_ts": 100010.0,
        },
    ]
    offset, report = _estimate_time_offset_seconds(flows, candidates)
    assert offset == 14400.0
    assert report["offset_estimator_status"] == "ok"


def test_ids2018_time_indexed_join_selects_nearest_candidate() -> None:
    flows = [
        {
            "flow_id": "f1",
            "src_ip": "10.0.0.1",
            "src_port": "1111",
            "dst_ip": "10.0.0.2",
            "dst_port": "80",
            "protocol": "TCP",
            "start_ts": 102.0,
        }
    ]
    labels = [
        {
            "key": ("10.0.0.1", "1111", "10.0.0.2", "80", "TCP"),
            "reverse_key": ("10.0.0.2", "80", "10.0.0.1", "1111", "TCP"),
            "start_ts": 100.0,
            "end_ts": 101.0,
            "timestamp_parse_status": "parsed",
            "label": "BENIGN",
            "raw_label": "Benign",
            "source_file": "labels.csv",
        },
        {
            "key": ("10.0.0.1", "1111", "10.0.0.2", "80", "TCP"),
            "reverse_key": ("10.0.0.2", "80", "10.0.0.1", "1111", "TCP"),
            "start_ts": 102.1,
            "end_ts": 103.0,
            "timestamp_parse_status": "parsed",
            "label": "DDoS",
            "raw_label": "DDoS attacks-LOIC-HTTP",
            "source_file": "labels.csv",
        },
    ]
    labeled, unmatched, stats = _label_flows_time_indexed(flows, labels, tolerance_seconds=2.0)
    assert not unmatched
    assert len(labeled) == 1
    assert labeled[0]["label"] == "DDoS"
    assert stats["time_indexed_join"] is True
