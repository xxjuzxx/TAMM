from __future__ import annotations

from src.features.profile_primitives import extract_all_profile_primitives


def flow(flow_id: str, lens: list[int], dirs: list[bool], service: str = "svc") -> dict:
    return {
        "flow_id": flow_id,
        "service_key": ["10.0.0.2", service, "TCP"],
        "label": "Benign",
        "lens": lens,
        "dirs": dirs,
        "tss": [float(idx) for idx in range(len(lens))],
    }


def test_all_profile_primitive_types_trigger() -> None:
    flows = [
        flow("short", [10, 20, 10], [True, False, True]),
        flow("same", [42, 42, 42, 42, 42, 42], [True, True, True, True, True, True]),
        flow("repeat_dup_a", [7, 7, 9, 9, 7, 7, 9, 9], [True, True, False, False, True, True, False, False]),
        flow("repeat_dup_b", [7, 7, 9, 9, 7, 7, 9, 9], [True, True, False, False, True, True, False, False]),
        flow("local_pkt_a", [11, 12, 30, 31, 11, 12], [True, True, False, False, True, True], service="local-a"),
        flow("local_pkt_b", [11, 12, 30, 31, 11, 12], [True, True, False, False, True, True], service="local-b"),
    ]
    rows, stats = extract_all_profile_primitives(flows, {"short_flow_packet_threshold": 6})
    by_id = {row["flow_id"]: row["profile"] for row in rows}
    assert by_id["short"]["short"] is not None
    assert by_id["same"]["same"] is not None
    assert by_id["repeat_dup_a"]["repeat"]
    assert by_id["repeat_dup_a"]["duplicate"]
    assert by_id["local_pkt_a"]["packet"]
    assert by_id["local_pkt_a"]["local"]
    assert stats["peer_grouping"] == "global_train_ordered_chunks"
    assert set(stats["profile_primitive_flow_counts"]).issuperset({"short", "same", "packet", "local", "repeat", "duplicate"})
