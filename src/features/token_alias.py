from __future__ import annotations

from collections.abc import Iterable


SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}

PROFILE_PRIMITIVE_PREFIX = "PRIM_PROFILE_"
STRUCTURAL_PRIMITIVE_PREFIX = "PRIM_STRUCT_"
RAW_PROFILE_PRIMITIVE_PREFIX = "RAW_PRIM_PROFILE_"

PROFILE_PRIMITIVE_TYPE_TO_TOKEN = {
    "SHORT": "PRIM_PROFILE_SHORT_FLOW",
    "SAME": "PRIM_PROFILE_SAME_LEN",
    "PKT": "PRIM_PROFILE_PKT_DIST",
    "LOCAL": "PRIM_PROFILE_LOCAL_SIM",
    "REPEAT": "PRIM_PROFILE_REPEAT_SEG",
    "DUP": "PRIM_PROFILE_DUP_SEG",
    "NONE": "PRIM_PROFILE_NONE",
}

PROFILE_PRIMITIVE_TOKEN_TO_TYPE = {value: key for key, value in PROFILE_PRIMITIVE_TYPE_TO_TOKEN.items()}


def canonical_token(token: str) -> str:
    """Return a canonical FlowPrim token spelling for current artifacts."""

    tok = str(token)
    if tok in SPECIAL_TOKENS:
        return tok
    if tok.startswith(("PRIM_PROFILE_", "PRIM_STRUCT_", "RAW_PRIM_PROFILE_")):
        return tok
    return tok


def canonical_tokens(tokens: Iterable[str]) -> list[str]:
    return [canonical_token(token) for token in tokens]


def profile_primitive_token(kind: str) -> str:
    return PROFILE_PRIMITIVE_TYPE_TO_TOKEN.get(str(kind).upper(), f"PRIM_PROFILE_{str(kind).upper()}")


def profile_primitive_type(token: str) -> str | None:
    return PROFILE_PRIMITIVE_TOKEN_TO_TYPE.get(canonical_token(token))


def is_profile_primitive_token(token: str, *, include_none: bool = False) -> bool:
    tok = canonical_token(token)
    return tok.startswith(PROFILE_PRIMITIVE_PREFIX) and (include_none or tok != "PRIM_PROFILE_NONE")


def is_profile_token(token: str, *, include_none: bool = False) -> bool:
    return is_profile_primitive_token(token, include_none=include_none)


def is_structural_primitive_token(token: str) -> bool:
    return canonical_token(token).startswith(STRUCTURAL_PRIMITIVE_PREFIX)


def is_structural_token(token: str) -> bool:
    return is_structural_primitive_token(token)


def is_packet_token(token: str) -> bool:
    tok = canonical_token(token)
    return tok.startswith(("PKT_DIR_", "PKT_LEN_", "PKT_IAT_", "PKT_LEN_TRANS_"))


def is_flow_summary_token(token: str) -> bool:
    tok = canonical_token(token)
    return tok.startswith(("FLOW_PKTN_", "FLOW_DUR_", "FLOW_BURSTN_", "FLOW_DIR_RATIO_", "FLOW_COUNT_"))


def is_burst_token(token: str) -> bool:
    tok = canonical_token(token)
    return tok.startswith(("BURST_", "BURST_TRANS_", "BURST_RUN_", "BURST_TURN_"))


def is_packet_burst_token(token: str) -> bool:
    return is_packet_token(token) or is_burst_token(token) or is_flow_summary_token(token)


def is_rhythm_token(token: str) -> bool:
    return canonical_token(token).startswith("RHY_")


def is_match_token(token: str) -> bool:
    return canonical_token(token).startswith("MATCH_")
