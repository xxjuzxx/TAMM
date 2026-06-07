from __future__ import annotations

from src.features.tokenizer import PrimitiveTrafficTokenizer, build_token_dataset


def _flow(flow_id: str, start_ts: float, packet_count: int, service: str = "svc") -> dict:
    return {
        "flow_id": flow_id,
        "service_key": ["10.0.0.2", service, "TCP"],
        "label": "Benign",
        "binary_label": "BENIGN",
        "lens": [10] * packet_count,
        "dirs": [True] * packet_count,
        "tss": [start_ts + idx * 0.001 for idx in range(packet_count)],
        "packet_count": packet_count,
        "start_ts": start_ts,
        "duration": max(0.0, (packet_count - 1) * 0.001),
    }


def _profile_row(flow_id: str) -> dict:
    return {
        "flow_id": flow_id,
        "profile": {
            "short": None,
            "same": None,
            "packet": [],
            "local": [],
            "repeat": [],
            "duplicate": [],
        },
    }


def test_service_context_uses_only_prior_flows() -> None:
    flows = [
        _flow("first", start_ts=10.0, packet_count=3),
        _flow("second", start_ts=11.0, packet_count=4),
        _flow("third", start_ts=15.0, packet_count=8),
    ]
    tokenizer = PrimitiveTrafficTokenizer(
        profile_mode="none",
        use_service_context=True,
        service_context_window_seconds=2.0,
        service_context_short_packet_threshold=6,
    )
    token_data, _ = build_token_dataset(flows, [_profile_row(flow["flow_id"]) for flow in flows], tokenizer)
    contexts = {row["flow_id"]: row["service_context"] for row in token_data["meta"]}
    assert contexts["first"]["recent_count"] == 0
    assert contexts["second"]["recent_count"] == 1
    assert contexts["second"]["recent_short"] == 1
    assert contexts["third"]["recent_count"] == 0


def test_service_context_does_not_mix_services() -> None:
    flows = [
        _flow("svc_a", start_ts=10.0, packet_count=3, service="a"),
        _flow("svc_b", start_ts=10.5, packet_count=3, service="b"),
    ]
    tokenizer = PrimitiveTrafficTokenizer(profile_mode="none", use_service_context=True, service_context_window_seconds=2.0)
    token_data, _ = build_token_dataset(flows, [_profile_row(flow["flow_id"]) for flow in flows], tokenizer)
    contexts = {row["flow_id"]: row["service_context"] for row in token_data["meta"]}
    assert contexts["svc_a"]["recent_count"] == 0
    assert contexts["svc_b"]["recent_count"] == 0


def test_service_tokens_can_be_enabled() -> None:
    flow = _flow("svc_tokens", start_ts=10.0, packet_count=4)
    flow["protocol"] = "6"
    flow["dst_port"] = "443"
    flow["appinfo"] = ["ssl", "tls"]
    flow["conn_state"] = "SF"
    tokenizer = PrimitiveTrafficTokenizer(profile_mode="none", use_service_tokens=True)

    tokens = tokenizer.flow_tokens(flow, _profile_row(flow["flow_id"]))
    token_data, stats = build_token_dataset([flow], [_profile_row(flow["flow_id"])], tokenizer)

    assert "PROTO_TCP" in tokens
    assert "PORT_SYSTEM" in tokens
    assert "APP_TLS" in tokens
    assert "STATE_SF" in tokens
    assert token_data["use_service_tokens"] is True
    assert stats["use_service_tokens"] is True


def test_tokenizer_preserves_predefined_split_in_meta() -> None:
    flow = _flow("split_row", start_ts=10.0, packet_count=4)
    flow["split"] = "train"
    tokenizer = PrimitiveTrafficTokenizer(profile_mode="none")

    token_data, _ = build_token_dataset([flow], [_profile_row(flow["flow_id"])], tokenizer)

    assert token_data["meta"][0]["split"] == "train"
