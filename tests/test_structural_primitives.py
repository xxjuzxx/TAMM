from __future__ import annotations

from src.features.structural_primitives import (
    StructuralPrimitiveConfig,
    build_train_only_structural_primitive_vocabulary,
    extract_structural_primitive_candidates,
    family_from_token,
    filter_triggers,
)


def test_structural_extracts_structural_tokens_not_single_fields() -> None:
    tokens = [
        "[CLS]",
        "FLOW_PKTN_5",
        "PKT_DIR_C2S",
        "PKT_LEN_3",
        "PKT_IAT_1",
        "BURST_START",
        "PKT_DIR_C2S",
        "PKT_LEN_3",
        "PKT_IAT_1",
        "BURST_MID",
        "PKT_DIR_C2S",
        "PKT_LEN_3",
        "PKT_IAT_1",
        "BURST_END",
        "PKT_DIR_S2C",
        "PKT_LEN_10",
        "PKT_IAT_12",
        "BURST_SINGLE",
        "BURST_PKTN_1",
        "BURST_DUR_0",
        "BURST_BYTES_C2S_7",
        "BURST_BYTES_S2C_0",
        "BURST_DIR_C2S",
        "BURST_IAT_MED_1",
        "BURST_GAP_NEXT_1",
        "BURST_LEN_MEAN_3",
        "BURST_LEN_STD_0",
        "BURST_PKTN_1",
        "BURST_DUR_0",
        "BURST_BYTES_C2S_0",
        "BURST_BYTES_S2C_7",
        "BURST_DIR_S2C",
        "BURST_IAT_MED_1",
        "BURST_GAP_NEXT_1",
        "BURST_LEN_MEAN_10",
        "BURST_LEN_STD_0",
        "[SEP]",
    ]
    cfg = StructuralPrimitiveConfig(
        enabled=True,
        enable_packet_shape_primitives=True,
        enable_burst_shape_primitives=True,
        enable_timing_rhythm_primitives=True,
        enable_direction_transition_primitives=True,
        enable_composite_primitives=True,
        min_support=1,
    )

    names = {trigger.name for trigger in extract_structural_primitive_candidates(tokens, cfg)}

    assert "PRIM_STRUCT_PKT_C2S_SMALL_RUN_3" in names
    assert "PRIM_STRUCT_PKT_SAME_LEN_RUN_C2S_3_L3" in names
    assert "PRIM_STRUCT_PKT_LEN_SPIKE" in names
    assert "PRIM_STRUCT_BURST_REQ_RESP_PAIR" in names
    assert "PRIM_STRUCT_DIR_TWO_PHASE_C2S_THEN_S2C" in names
    assert not any(name in {"PRIM_STRUCT_LEN_SMALL", "PRIM_STRUCT_DIR_C2S"} for name in names)


def test_structural_vocab_is_train_only_and_filters_rare_tokens() -> None:
    cfg = StructuralPrimitiveConfig(
        enabled=True,
        enable_packet_shape_primitives=True,
        min_support=2,
    )
    base = ["PKT_DIR_C2S", "PKT_LEN_3", "PKT_IAT_1", "PKT_DIR_C2S", "PKT_LEN_3", "PKT_IAT_1", "PKT_DIR_C2S", "PKT_LEN_3", "PKT_IAT_1"]
    rows = [
        extract_structural_primitive_candidates(base, cfg),
        extract_structural_primitive_candidates(base, cfg),
        extract_structural_primitive_candidates(["PKT_DIR_S2C", "PKT_LEN_3", "PKT_IAT_1", "PKT_DIR_S2C", "PKT_LEN_3", "PKT_IAT_1"], cfg),
    ]

    vocab = build_train_only_structural_primitive_vocabulary(rows, [0, 1], min_support=2)
    filtered_test = filter_triggers(rows[2], vocab)

    assert "PRIM_STRUCT_PKT_C2S_SMALL_RUN_3" in vocab
    assert all(trigger.name in vocab for trigger in filtered_test)
    assert "PRIM_STRUCT_PKT_S2C_SMALL_RUN_2" not in vocab


def test_burst_interval_periodic_is_timing_family() -> None:
    assert family_from_token("PRIM_STRUCT_BURST_INTERVAL_PERIODIC") == "timing_rhythm"
    assert family_from_token("PRIM_STRUCT_POS_PREFIX_C2S_SMALL_RUN") == "position_aware"
    assert family_from_token("PRIM_STRUCT_PARAM_IAT_CV_LOW") == "parameterized"


def test_structural_position_and_parameter_switches_are_optional() -> None:
    tokens = [
        "PKT_DIR_C2S",
        "PKT_LEN_3",
        "PKT_IAT_1",
        "PKT_DIR_C2S",
        "PKT_LEN_3",
        "PKT_IAT_1",
        "PKT_DIR_C2S",
        "PKT_LEN_3",
        "PKT_IAT_12",
        "PKT_DIR_S2C",
        "PKT_LEN_10",
        "PKT_IAT_12",
        "BURST_PKTN_1",
        "BURST_BYTES_C2S_7",
        "BURST_BYTES_S2C_0",
        "BURST_DIR_C2S",
        "BURST_GAP_NEXT_1",
        "BURST_PKTN_1",
        "BURST_BYTES_C2S_0",
        "BURST_BYTES_S2C_7",
        "BURST_DIR_S2C",
        "BURST_GAP_NEXT_1",
    ]
    base_cfg = StructuralPrimitiveConfig(
        enabled=True,
        enable_packet_shape_primitives=True,
        enable_burst_shape_primitives=True,
        enable_timing_rhythm_primitives=True,
        min_support=1,
    )
    structural1_cfg = StructuralPrimitiveConfig(
        enabled=True,
        enable_packet_shape_primitives=True,
        enable_burst_shape_primitives=True,
        enable_timing_rhythm_primitives=True,
        enable_position_aware_primitives=True,
        enable_parameterized_primitives=True,
        min_support=1,
    )

    base_names = {trigger.name for trigger in extract_structural_primitive_candidates(tokens, base_cfg)}
    structural1_names = {trigger.name for trigger in extract_structural_primitive_candidates(tokens, structural1_cfg)}

    assert not any(name.startswith(("PRIM_STRUCT_POS_", "PRIM_STRUCT_PARAM_")) for name in base_names)
    assert any(name.startswith("PRIM_STRUCT_POS_") for name in structural1_names)
    assert any(name.startswith("PRIM_STRUCT_PARAM_") for name in structural1_names)
