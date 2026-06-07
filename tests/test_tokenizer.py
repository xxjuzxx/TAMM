from __future__ import annotations

from src.data.splits import build_split
from src.features.behavior_tokens import build_behavior_token_dataset


def _flow(idx: int, label: str = "BENIGN") -> dict:
    return {
        "flow_id": f"f{idx}",
        "src_ip": f"10.0.0.{idx}",
        "dst_ip": "172.16.0.1",
        "src_port": str(1000 + idx),
        "dst_port": "443",
        "proto": "tcp",
        "protocol": "TCP",
        "lens": [60, 120, 60],
        "dirs": [1, 0, 1],
        "tss": [0.0, 0.01, 0.02],
        "iats": [0.0, 0.01, 0.01],
        "label": label,
        "attack_family": label,
        "binary_label": "BENIGN" if label == "BENIGN" else "ATTACK",
        "dataset": "CICIDS2017",
        "day": "Monday",
        "start_ts": float(idx),
        "duration": 0.02,
        "packet_count": 3,
    }


def _profile_row(flow_id: str) -> dict:
    return {
        "flow_id": flow_id,
        "profile": {"short": None, "same": None, "packet": [], "local": [], "repeat": [], "duplicate": []},
    }


def test_behavior_tokenizer_uses_train_only_vocab_and_clean_meta_flags() -> None:
    flows = [_flow(idx, "BENIGN" if idx < 5 else "DDoS") for idx in range(10)]
    split_payload = build_split(flows, "temporal", val_ratio=0.2, test_ratio=0.2, seed=1)
    token_data, stats = build_behavior_token_dataset(
        flows,
        [_profile_row(flow["flow_id"]) for flow in flows],
        split_payload,
        {"max_len": 64, "max_packets": 16, "use_service_tokens": False},
    )
    assert token_data["train_only"] is True
    assert stats["has_ip_token"] is False
    assert stats["has_port_token"] is False
    assert all(meta["has_ip_token"] is False for meta in token_data["meta"])
    assert all(meta["has_abs_time_token"] is False for meta in token_data["meta"])
    assert all(meta["has_port_token"] is False for meta in token_data["meta"])
    assert not any("10.0.0." in token for token in token_data["vocab"])
    assert not any("443" in token for token in token_data["vocab"])
    assert "avg_len" in stats
    assert "p95_len" in stats
    assert "truncation_rate" in stats
    assert "unk_token_rate" in stats
    assert "profile_token_coverage" in stats
    assert "rhythm_token_coverage" in stats
    assert set(stats["per_split_stats"]) == {"train", "val", "test"}


def test_behavior_tokenizer_flags_port_tokens_when_service_tokens_enabled() -> None:
    flows = [_flow(idx, "BENIGN" if idx < 5 else "DDoS") for idx in range(10)]
    split_payload = build_split(flows, "temporal", val_ratio=0.2, test_ratio=0.2, seed=1)
    token_data, stats = build_behavior_token_dataset(
        flows,
        [_profile_row(flow["flow_id"]) for flow in flows],
        split_payload,
        {"max_len": 64, "max_packets": 16, "use_service_tokens": True},
    )
    assert stats["has_port_token"] is True
    assert any(meta["has_port_token"] is True for meta in token_data["meta"])
    assert any(token.startswith("DPORT_") or token.startswith("PORT_") for token in token_data["vocab"])
