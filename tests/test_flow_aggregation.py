from __future__ import annotations

from src.data.flow_aggregator import PacketRecord, aggregate_packets


def test_flow_aggregation_outputs_relative_packet_times() -> None:
    flows = aggregate_packets(
        [
            PacketRecord(100.0, "10.0.0.1", "1111", "10.0.0.2", "80", "TCP", 60, True),
            PacketRecord(100.3, "10.0.0.2", "80", "10.0.0.1", "1111", "TCP", 120, False),
        ]
    )
    assert len(flows) == 1
    assert flows[0]["packet_count"] == 2
    assert flows[0]["duration"] == 0.29999999999999716
    assert flows[0]["lens"] == [60, 120]
