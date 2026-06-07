from __future__ import annotations

from src.features.tokenizer import PrimitiveTrafficTokenizer, build_token_dataset


def _flow(flow_id: str = "flow1") -> dict:
    return {
        "flow_id": flow_id,
        "service_key": ["10.0.0.2", "443", "TCP"],
        "src_ip": "10.1.1.1",
        "dst_ip": "10.1.1.2",
        "src_port": 55555,
        "dst_port": 443,
        "dataset_file": "crossnet_app_label_capture.jsonl",
        "label": "secret_app",
        "binary_label": "ATTACK",
        "lens": [60, 80, 1200, 1000, 72, 76],
        "dirs": [True, True, False, False, True, True],
        "tss": [0.0, 0.01, 0.4, 0.42, 1.1, 1.12],
        "packet_count": 6,
        "start_ts": 10.0,
        "duration": 1.12,
    }


def _profile_row(flow_id: str = "flow1") -> dict:
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


def test_new_tokens_are_disabled_by_default() -> None:
    flow = _flow()
    profile = _profile_row()
    default_tokens = PrimitiveTrafficTokenizer(profile_mode="none").flow_tokens(flow, profile)
    explicit_tokens = PrimitiveTrafficTokenizer(
        profile_mode="none",
        use_burst_shape_tokens=False,
        use_first_k_signature=False,
        first_k=5,
    ).flow_tokens(flow, profile)

    assert default_tokens == explicit_tokens
    assert not any(token.startswith("BURST_PKTN_") for token in default_tokens)
    assert not any(token.startswith("FIRST_") for token in default_tokens)


def test_burst_shape_and_first_k_tokens_are_emitted_and_saved() -> None:
    flow = _flow()
    profile = _profile_row()
    tokenizer = PrimitiveTrafficTokenizer(
        profile_mode="none",
        use_burst_shape_tokens=True,
        use_first_k_signature=True,
        first_k=3,
    )

    tokens = tokenizer.flow_tokens(flow, profile)
    token_data, stats = build_token_dataset([flow], [profile], tokenizer)

    assert any(token.startswith("BURST_PKTN_") for token in tokens)
    assert any(token.startswith("BURST_BYTES_C2S_") for token in tokens)
    assert "FIRST_DIR_PATTERN_CCS" in tokens
    assert any(token.startswith("FIRST_LEN_BIN_1_") for token in tokens)
    assert any(token.startswith("FIRST_IAT_PATTERN_") for token in tokens)
    assert token_data["use_burst_shape_tokens"] is True
    assert token_data["use_first_k_signature"] is True
    assert token_data["first_k"] == 3
    assert token_data["meta"][0]["src_ip"] == flow["src_ip"]
    assert token_data["meta"][0]["dst_ip"] == flow["dst_ip"]
    assert token_data["meta"][0]["src_port"] == flow["src_port"]
    assert token_data["meta"][0]["dst_port"] == flow["dst_port"]
    assert stats["use_burst_shape_tokens"] is True
    assert stats["use_first_k_signature"] is True
    assert stats["first_k"] == 3
    assert stats["avg_raw_token_length"] >= stats["avg_token_length"]
    assert "FIRST_DIR_PATTERN_CCS" in token_data["vocab"]


def test_new_tokens_do_not_copy_forbidden_metadata_values() -> None:
    flow = _flow()
    profile = _profile_row()
    tokenizer = PrimitiveTrafficTokenizer(
        profile_mode="none",
        use_burst_shape_tokens=True,
        use_first_k_signature=True,
        first_k=5,
        use_service_tokens=False,
    )
    tokens = tokenizer.flow_tokens(flow, profile)
    joined = " ".join(tokens)

    forbidden_fragments = [
        "10.1.1.1",
        "10.1.1.2",
        "55555",
        "443",
        "secret_app",
        "flow1",
        "crossnet_app_label_capture",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in joined
