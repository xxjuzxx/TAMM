from __future__ import annotations

from pathlib import Path

from src.utils.io import write_jsonl
from scripts.eval_crossnet_rolling_aggregation import _group_key, _merge_flow_metadata


def test_rolling_metadata_flows_fill_src_dst_fields(tmp_path: Path) -> None:
    predictions = [
        {"flow_id": "f1", "scores": {"a": 0.8, "b": 0.2}},
        {"flow_id": "f2", "scores": {"a": 0.2, "b": 0.8}},
    ]
    metadata_path = tmp_path / "flows.jsonl"
    write_jsonl(
        [
            {"flow_id": "f1", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "protocol": "TCP"},
            {"flow_id": "f2", "src_ip": "10.0.0.3", "dst_ip": "10.0.0.4", "protocol": "UDP"},
        ],
        metadata_path,
    )

    merged, summary = _merge_flow_metadata(predictions, metadata_path)

    assert summary["matched_metadata_rows"] == 2
    assert summary["missing_metadata_rows"] == 0
    assert _group_key(merged[0], "src_dst_proto") == ("src_dst_proto", "10.0.0.1", "10.0.0.2", "TCP")
    assert _group_key(merged[1], "src_dst_pair") == ("src_dst_pair", "10.0.0.3", "10.0.0.4")
