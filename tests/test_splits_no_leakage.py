from __future__ import annotations

from src.data.splits import build_split, split_lookup


def _flow(idx: int, label: str) -> dict:
    return {"flow_id": f"f{idx}", "label": label, "attack_family": label, "start_ts": float(idx)}


def test_stratified_random_split_has_no_flow_overlap() -> None:
    flows = [_flow(idx, "BENIGN" if idx < 10 else "DDoS") for idx in range(20)]
    payload = build_split(flows, "stratified_random", val_ratio=0.1, test_ratio=0.2, seed=7)
    lookup = split_lookup(payload)
    assert len(lookup) == len(flows)
    assert payload["counts"]["train"] + payload["counts"]["val"] + payload["counts"]["test"] == len(flows)
    assert set(payload["splits"]["train"]).isdisjoint(payload["splits"]["val"])
    assert set(payload["splits"]["train"]).isdisjoint(payload["splits"]["test"])
    assert set(payload["splits"]["val"]).isdisjoint(payload["splits"]["test"])


def test_temporal_split_preserves_time_order() -> None:
    flows = [_flow(idx, "BENIGN") for idx in range(10)]
    payload = build_split(flows, "temporal", val_ratio=0.2, test_ratio=0.2, seed=1)
    assert payload["splits"]["train"] == [f"f{idx}" for idx in range(6)]
    assert payload["splits"]["val"] == ["f6", "f7"]
    assert payload["splits"]["test"] == ["f8", "f9"]


def test_temporal_chronological_alias_is_stratified() -> None:
    flows = [_flow(idx, "BENIGN" if idx < 10 else "DDoS") for idx in range(20)]
    payload = build_split(flows, "temporal_chronological", val_ratio=0.1, test_ratio=0.2, seed=1)
    assert payload["canonical_split"] == "temporal_stratified"
    assert payload["label_counts"]["test"]["BENIGN"] == 2
    assert payload["label_counts"]["test"]["DDoS"] == 2


def test_day_wise_split_reports_missing_classes() -> None:
    flows = [
        {**_flow(0, "BENIGN"), "day": "Monday"},
        {**_flow(1, "DDoS"), "day": "Tuesday"},
        {**_flow(2, "BENIGN"), "day": "Wednesday"},
        {**_flow(3, "PortScan"), "day": "Thursday"},
    ]
    payload = build_split(flows, "day_wise", train_days=["Monday", "Tuesday"], val_days=["Wednesday"], test_days=["Thursday"])
    assert payload["counts"] == {"train": 2, "val": 1, "test": 1}
    assert payload["coverage"]["class_missing_in_train"] == ["PortScan"]


def test_leave_one_attack_out_test_contains_benign_and_held_out_attack() -> None:
    flows = [_flow(idx, "BENIGN") for idx in range(10)]
    flows.extend(_flow(10 + idx, "DDoS") for idx in range(5))
    flows.extend(_flow(20 + idx, "Bot") for idx in range(5))
    payload = build_split(
        flows,
        "leave_one_attack_out",
        val_ratio=0.1,
        test_ratio=0.2,
        seed=1,
        leave_label="Bot",
        leave_one_mode="classification",
    )
    assert payload["leave_label"] == "Bot"
    assert payload["label_counts"]["test"]["BENIGN"] > 0
    assert payload["label_counts"]["test"]["Bot"] == 5
    assert "DDoS" in payload["label_counts"]["train"]


def test_few_label_split_supports_train_fraction() -> None:
    flows = [_flow(idx, "BENIGN" if idx < 100 else "DDoS") for idx in range(200)]
    payload = build_split(flows, "few_label", val_ratio=0.1, test_ratio=0.2, seed=1, train_fraction=0.1)
    assert payload["train_fraction"] == 0.1
    assert payload["counts"]["train"] == 14
    assert payload["label_counts"]["train"] == {"BENIGN": 7, "DDoS": 7}
