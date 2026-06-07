from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import torch

from .bins import count_bin, iat_bin, log_bin, ratio_bin
from .burst import burst_count, burst_positions
from .token_alias import RAW_PROFILE_PRIMITIVE_PREFIX, profile_primitive_token


SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
PROFILE_PRIMITIVE_ORDER = ["SHORT", "SAME", "PKT", "LOCAL", "REPEAT", "DUP", "NONE"]


@dataclass
class Vocabulary:
    token_to_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for token in SPECIAL_TOKENS:
            self.add(token)

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
        return self.token_to_id[token]

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token_to_id["[UNK]"]
        return [self.token_to_id.get(token, unk) for token in tokens]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["[PAD]"]

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)


KNOWN_APP_TOKENS = {
    "dhcp": "APP_DHCP",
    "dns": "APP_DNS",
    "ftp": "APP_FTP",
    "gssapi": "APP_GSSAPI",
    "http": "APP_HTTP",
    "irc": "APP_IRC",
    "ntlm": "APP_NTLM",
    "ntp": "APP_NTP",
    "rdp": "APP_RDP",
    "sip": "APP_SIP",
    "smb": "APP_SMB",
    "smtp": "APP_SMTP",
    "snmp": "APP_SNMP",
    "ssh": "APP_SSH",
    "ssl": "APP_TLS",
    "tls": "APP_TLS",
}

KNOWN_CONN_STATES = {"S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "OTH"}
BURST_SHAPE_BIN_PREFIXES = [
    "BURST_PKTN",
    "BURST_DUR",
    "BURST_BYTES_C2S",
    "BURST_BYTES_S2C",
    "BURST_IAT_MED",
    "BURST_GAP_NEXT",
    "BURST_LEN_MEAN",
    "BURST_LEN_STD",
]
FIRST_IAT_PATTERN_CODES = ("Z", "S", "M", "L")
MAX_STATIC_FIRST_K_VOCAB = 5
CONTEXT_PROFILE_BIN_PREFIXES_BASE = [
    "CTX_RECENT",
    "CTX_PKT_RATE",
    "CTX_BYTE_RATE",
    "CTX_GAP_MED",
    "CTX_GAP_STD",
    "CTX_BURST_RECENT",
]
CONTEXT_PROFILE_BIN_PREFIXES = CONTEXT_PROFILE_BIN_PREFIXES_BASE
TRANSITION_BIN_PREFIXES = [
    "BURST_TRANS_GAP",
    "BURST_TRANS_BYTES_RATIO",
    "BURST_TRANS_PKTN_RATIO",
    "BURST_RUN_C2S",
    "BURST_RUN_S2C",
    "BURST_TURN_COUNT",
]
LEN_TRANS_CODES = ("UP", "DOWN", "STABLE")


def default_vocab(max_bin: int = 15) -> Vocabulary:
    vocab = Vocabulary()
    for prefix in ["PKT_LEN", "PKT_IAT", "FLOW_COUNT", "FLOW_PKTN", "FLOW_DUR", "FLOW_BURSTN", "FLOW_DIR_RATIO", "SVCREC", "SVCSHORT", "SVCGAP", "SVCPKT"]:
        for idx in range(max_bin + 1):
            vocab.add(f"{prefix}_{idx}")
    for token in ["PKT_DIR_C2S", "PKT_DIR_S2C", "BURST_START", "BURST_MID", "BURST_END", "BURST_SINGLE"]:
        vocab.add(token)
    for token in ["SVC_COLD", "SVC_WARM", "SVC_HOT", "SVC_SHORT_NONE", "SVC_SHORT_LOW", "SVC_SHORT_MED", "SVC_SHORT_HIGH"]:
        vocab.add(token)
    for token in ["PROTO_TCP", "PROTO_UDP", "PROTO_ICMP", "PROTO_OTHER"]:
        vocab.add(token)
    for token in ["APP_NONE", "APP_OTHER", *sorted(set(KNOWN_APP_TOKENS.values()))]:
        vocab.add(token)
    for token in ["STATE_NONE", "STATE_OTHER", *[f"STATE_{state}" for state in sorted(KNOWN_CONN_STATES)]]:
        vocab.add(token)
    for prim in PROFILE_PRIMITIVE_ORDER:
        vocab.add(profile_primitive_token(prim))
    for rhythm in ["RHY_SHORT", "RHY_REQ_LONGRESP", "RHY_CLIENT_BURST", "RHY_PERIODIC", "RHY_SCAN_LIKE", "RHY_INTERACTIVE", "RHY_GENERIC"]:
        vocab.add(rhythm)
    for match in ["MATCH_EXACT", "MATCH_SUM", "MATCH_FUZZY", "MATCH_SHIFT2", "MATCH_SINGLE"]:
        vocab.add(match)
    return vocab


def _extend_service_token_vocab(vocab: Vocabulary, max_bin: int) -> None:
    for idx in range(max_bin + 1):
        vocab.add(f"DPORT_{idx}")
    for token in ["PORT_ZERO", "PORT_SYSTEM", "PORT_REGISTERED", "PORT_DYNAMIC", "PORT_OTHER"]:
        vocab.add(token)


def _extend_optional_token_vocab(
    vocab: Vocabulary,
    max_bin: int,
    *,
    use_burst_shape_tokens: bool,
    use_first_k_signature: bool,
    first_k: int,
    use_context_profile_tokens: bool,
    use_transition_profile_tokens: bool,
) -> None:
    if use_burst_shape_tokens:
        for prefix in BURST_SHAPE_BIN_PREFIXES:
            for idx in range(max_bin + 1):
                vocab.add(f"{prefix}_{idx}")
        for token in ["BURST_DIR_C2S", "BURST_DIR_S2C"]:
            vocab.add(token)
    if use_first_k_signature:
        static_k = min(max(1, int(first_k)), MAX_STATIC_FIRST_K_VOCAB)
        vocab.add("FIRST_DIR_PATTERN_EMPTY")
        for size in range(1, static_k + 1):
            for pattern in product(("C", "S"), repeat=size):
                vocab.add(f"FIRST_DIR_PATTERN_{''.join(pattern)}")
        for pos in range(1, static_k + 1):
            for idx in range(max_bin + 1):
                vocab.add(f"FIRST_LEN_BIN_{pos}_{idx}")
        vocab.add("FIRST_IAT_PATTERN_EMPTY")
        for size in range(1, static_k):
            for pattern in product(FIRST_IAT_PATTERN_CODES, repeat=size):
                vocab.add(f"FIRST_IAT_PATTERN_{''.join(pattern)}")
    if use_context_profile_tokens:
        for prefix in CONTEXT_PROFILE_BIN_PREFIXES:
            for idx in range(max_bin + 1):
                vocab.add(f"{prefix}_{idx}")
        for token in ["CTX_COLD", "CTX_WARM", "CTX_HOT", "CTX_BURSTY_LOW", "CTX_BURSTY_MED", "CTX_BURSTY_HIGH"]:
            vocab.add(token)
    if use_transition_profile_tokens:
        for prefix in TRANSITION_BIN_PREFIXES:
            for idx in range(max_bin + 1):
                vocab.add(f"{prefix}_{idx}")
        for src in ("C2S", "S2C"):
            for dst in ("C2S", "S2C"):
                vocab.add(f"BURST_TRANS_DIR_{src}_TO_{dst}")
        for code in LEN_TRANS_CODES:
            vocab.add(f"PKT_LEN_TRANS_{code}")


def _flow_rhythm(flow: dict[str, Any], profile: dict[str, Any]) -> str:
    pkt_n = len(flow["lens"])
    dirs = [bool(item) for item in flow["dirs"]]
    c2s = sum(1 for item in dirs if item)
    s2c = pkt_n - c2s
    if profile.get("short"):
        return "RHY_SHORT"
    if profile.get("repeat") or profile.get("duplicate"):
        return "RHY_PERIODIC"
    if pkt_n > 0 and c2s / pkt_n > 0.85:
        return "RHY_CLIENT_BURST"
    if s2c > c2s * 3 and c2s <= 3:
        return "RHY_REQ_LONGRESP"
    if pkt_n <= 8 and c2s >= s2c:
        return "RHY_SCAN_LIKE"
    if pkt_n > 3 and abs(c2s - s2c) <= max(2, pkt_n * 0.2):
        return "RHY_INTERACTIVE"
    return "RHY_GENERIC"


def _service_key(flow: dict[str, Any]) -> tuple[str, ...]:
    key = flow.get("service_key")
    if isinstance(key, (list, tuple)):
        return tuple(str(item) for item in key)
    return (str(key),)


def _service_context_rows(
    flows: list[dict[str, Any]],
    window_seconds: float,
    short_packet_threshold: int,
    count_bin_max: int,
) -> dict[str, dict[str, Any]]:
    order = sorted(
        range(len(flows)),
        key=lambda idx: (float(flows[idx].get("start_ts") or idx), str(flows[idx].get("flow_id", idx))),
    )
    history: dict[tuple[str, ...], deque[tuple[float, int]]] = defaultdict(deque)
    packet_history: dict[tuple[str, ...], deque[tuple[float, int, int, int]]] = defaultdict(deque)
    context: dict[str, dict[str, Any]] = {}
    for idx in order:
        flow = flows[idx]
        start_ts = float(flow.get("start_ts") or idx)
        key = _service_key(flow)
        service_history = history[key]
        service_packets = packet_history[key]
        while service_history and start_ts - service_history[0][0] > window_seconds:
            service_history.popleft()
        while service_packets and start_ts - service_packets[0][0] > window_seconds:
            service_packets.popleft()
        recent_count = len(service_history)
        recent_short = sum(flag for _, flag in service_history)
        recent_packets = sum(count for _, count, _bytes, _bursts in service_packets)
        recent_bytes = sum(byte_count for _ts, _count, byte_count, _bursts in service_packets)
        recent_bursts = sum(bursts for _ts, _count, _bytes, bursts in service_packets)
        last_gap = start_ts - service_history[-1][0] if service_history else None
        gap_values: list[float] = []
        prev_ts: float | None = None
        for ts, _flag in service_history:
            if prev_ts is not None:
                gap_values.append(max(0.0, ts - prev_ts))
            prev_ts = ts
        short_ratio = recent_short / recent_count if recent_count else 0.0
        context[flow["flow_id"]] = {
            "recent_count": recent_count,
            "recent_short": recent_short,
            "recent_packets": recent_packets,
            "recent_bytes": recent_bytes,
            "recent_bursts": recent_bursts,
            "gap_values": gap_values,
            "short_ratio": short_ratio,
            "last_gap": last_gap,
            "tokens": _service_context_tokens(
                recent_count,
                recent_short,
                recent_packets,
                short_ratio,
                last_gap,
                count_bin_max=count_bin_max,
            ),
        }
        pkt_n = int(flow.get("packet_count") or len(flow.get("lens", [])))
        bursts = burst_count(flow, threshold_seconds=0.1)
        service_history.append((start_ts, 1 if pkt_n < short_packet_threshold else 0))
        service_packets.append((start_ts, pkt_n, _flow_byte_count(flow), bursts))
    return context


def _flow_byte_count(flow: dict[str, Any]) -> int:
    if flow.get("byte_count") is not None:
        try:
            return int(float(flow.get("byte_count")))
        except (TypeError, ValueError):
            pass
    return int(sum(max(0, int(item)) for item in flow.get("lens", [])))


def _context_profile_tokens(service_context: dict[str, Any] | None, count_bin_max: int) -> list[str]:
    ctx = service_context or {}
    recent_count = int(ctx.get("recent_count") or 0)
    recent_packets = int(ctx.get("recent_packets") or 0)
    recent_bytes = int(ctx.get("recent_bytes") or 0)
    recent_bursts = int(ctx.get("recent_bursts") or 0)
    last_gap = ctx.get("last_gap")
    gap_values = [float(item) for item in ctx.get("gap_values") or []]
    if recent_count == 0:
        heat = "CTX_COLD"
    elif recent_count < 4:
        heat = "CTX_WARM"
    else:
        heat = "CTX_HOT"
    burst_ratio = recent_bursts / recent_count if recent_count else 0.0
    if burst_ratio < 1.0:
        burst_level = "CTX_BURSTY_LOW"
    elif burst_ratio < 3.0:
        burst_level = "CTX_BURSTY_MED"
    else:
        burst_level = "CTX_BURSTY_HIGH"
    gap_med = _median(gap_values) if gap_values else (0.0 if last_gap is None else max(0.0, float(last_gap)))
    gap_std = _std(gap_values, sum(gap_values) / len(gap_values)) if gap_values else 0.0
    duration = max(1e-6, sum(gap_values) if gap_values else (0.0 if last_gap is None else max(0.0, float(last_gap))))
    pkt_rate = recent_packets / duration if duration > 0 else 0.0
    byte_rate = recent_bytes / duration if duration > 0 else 0.0
    return [
        heat,
        burst_level,
        f"CTX_RECENT_{count_bin(recent_count, count_bin_max)}",
        f"CTX_PKT_RATE_{log_bin(pkt_rate, count_bin_max)}",
        f"CTX_BYTE_RATE_{log_bin(byte_rate, count_bin_max)}",
        f"CTX_GAP_MED_{iat_bin(gap_med, count_bin_max)}",
        f"CTX_GAP_STD_{iat_bin(gap_std, count_bin_max)}",
        f"CTX_BURST_RECENT_{count_bin(recent_bursts, count_bin_max)}",
    ]


def _service_context_tokens(
    recent_count: int,
    recent_short: int,
    recent_packets: int,
    short_ratio: float,
    last_gap: float | None,
    count_bin_max: int,
) -> list[str]:
    if recent_count == 0:
        heat = "SVC_COLD"
    elif recent_count < 4:
        heat = "SVC_WARM"
    else:
        heat = "SVC_HOT"
    if recent_short == 0:
        short_level = "SVC_SHORT_NONE"
    elif short_ratio < 0.25:
        short_level = "SVC_SHORT_LOW"
    elif short_ratio < 0.75:
        short_level = "SVC_SHORT_MED"
    else:
        short_level = "SVC_SHORT_HIGH"
    gap_value = 0.0 if last_gap is None else max(0.0, float(last_gap))
    return [
        heat,
        short_level,
        f"SVCREC_{count_bin(recent_count, count_bin_max)}",
        f"SVCSHORT_{count_bin(recent_short, count_bin_max)}",
        f"SVCPKT_{count_bin(recent_packets, count_bin_max)}",
        f"SVCGAP_{iat_bin(gap_value, count_bin_max)}",
    ]


def _normal_protocol(value: Any) -> str:
    protocol = str(value or "").strip().upper()
    if protocol == "6":
        protocol = "TCP"
    elif protocol == "17":
        protocol = "UDP"
    elif protocol == "1":
        protocol = "ICMP"
    if protocol in {"TCP", "UDP", "ICMP"}:
        return protocol
    return "OTHER"


def _destination_port(flow: dict[str, Any]) -> int | None:
    raw_port = flow.get("dst_port")
    if raw_port in (None, ""):
        service_key = flow.get("service_key")
        if isinstance(service_key, (list, tuple)) and len(service_key) > 1:
            raw_port = service_key[1]
    try:
        return int(float(raw_port))
    except (TypeError, ValueError):
        return None


def _port_class(port: int | None) -> str:
    if port is None:
        return "PORT_OTHER"
    if port == 0:
        return "PORT_ZERO"
    if 0 < port < 1024:
        return "PORT_SYSTEM"
    if port < 49152:
        return "PORT_REGISTERED"
    if port <= 65535:
        return "PORT_DYNAMIC"
    return "PORT_OTHER"


def _app_token(flow: dict[str, Any]) -> str:
    appinfo = flow.get("appinfo")
    if isinstance(appinfo, list):
        values = [str(item).strip().lower() for item in appinfo if str(item).strip()]
    elif appinfo:
        values = [str(appinfo).strip().lower()]
    else:
        values = []
    if not values:
        return "APP_NONE"
    service = Counter(values).most_common(1)[0][0]
    return KNOWN_APP_TOKENS.get(service, "APP_OTHER")


def _state_token(flow: dict[str, Any]) -> str:
    state = str(flow.get("conn_state") or flow.get("state") or "").strip().upper()
    if not state:
        return "STATE_NONE"
    if state in KNOWN_CONN_STATES:
        return f"STATE_{state}"
    return "STATE_OTHER"


def _service_feature_tokens(flow: dict[str, Any], count_bin_max: int) -> list[str]:
    protocol = _normal_protocol(flow.get("protocol"))
    port = _destination_port(flow)
    port_for_bin = 0 if port is None else max(0, port)
    return [
        f"PROTO_{protocol}",
        _port_class(port),
        f"DPORT_{log_bin(port_for_bin, count_bin_max)}",
        _app_token(flow),
        _state_token(flow),
    ]


def _packet_iats(flow: dict[str, Any]) -> list[float]:
    tss = [float(item) for item in flow.get("tss", [])]
    return [0.0] + [max(0.0, cur - prev) for prev, cur in zip(tss[:-1], tss[1:])]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _burst_spans(flow: dict[str, Any], threshold_seconds: float) -> list[tuple[int, int]]:
    lens = flow.get("lens", [])
    dirs = flow.get("dirs", [])
    tss = [float(item) for item in flow.get("tss", [])]
    if not lens:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(lens)):
        same_dir = bool(dirs[idx]) == bool(dirs[idx - 1])
        close = (tss[idx] - tss[idx - 1]) <= threshold_seconds
        if not (same_dir and close):
            spans.append((start, idx))
            start = idx
    spans.append((start, len(lens)))
    return spans


def _select_burst_spans(spans: list[tuple[int, int]], max_bursts: int = 4) -> list[tuple[int, int]]:
    if len(spans) <= max_bursts:
        return spans
    head_count = max_bursts - 1
    return [*spans[:head_count], spans[-1]]


def _std(values: list[float], mean: float) -> float:
    if len(values) <= 1:
        return 0.0
    return (sum((float(item) - mean) ** 2 for item in values) / len(values)) ** 0.5


def _burst_shape_tokens(
    flow: dict[str, Any],
    *,
    threshold_seconds: float,
    count_bin_max: int,
    length_bin_max: int,
    iat_bin_max: int,
) -> list[str]:
    lens = [max(0, int(item)) for item in flow.get("lens", [])]
    dirs = [bool(item) for item in flow.get("dirs", [])]
    tss = [float(item) for item in flow.get("tss", [])]
    if not lens:
        return []
    iats = _packet_iats(flow)
    tokens: list[str] = []
    for start, end in _select_burst_spans(_burst_spans(flow, threshold_seconds), max_bursts=4):
        burst_lens = lens[start:end]
        burst_dirs = dirs[start:end]
        if not burst_lens:
            continue
        pkt_n = len(burst_lens)
        duration = max(0.0, tss[end - 1] - tss[start]) if end - 1 < len(tss) else 0.0
        c2s_bytes = sum(length for length, direction in zip(burst_lens, burst_dirs) if direction)
        s2c_bytes = sum(length for length, direction in zip(burst_lens, burst_dirs) if not direction)
        within_iats = iats[start + 1 : end]
        gap_next = max(0.0, tss[end] - tss[end - 1]) if end < len(tss) else 0.0
        mean_len = sum(burst_lens) / pkt_n
        std_len = _std([float(item) for item in burst_lens], mean_len)
        tokens.extend(
            [
                f"BURST_PKTN_{count_bin(pkt_n, count_bin_max)}",
                f"BURST_DUR_{log_bin(duration * 1000.0, count_bin_max)}",
                f"BURST_BYTES_C2S_{log_bin(c2s_bytes, count_bin_max)}",
                f"BURST_BYTES_S2C_{log_bin(s2c_bytes, count_bin_max)}",
                "BURST_DIR_C2S" if c2s_bytes >= s2c_bytes else "BURST_DIR_S2C",
                f"BURST_IAT_MED_{iat_bin(_median(within_iats), iat_bin_max)}",
                f"BURST_GAP_NEXT_{iat_bin(gap_next, iat_bin_max)}",
                f"BURST_LEN_MEAN_{log_bin(mean_len, length_bin_max)}",
                f"BURST_LEN_STD_{log_bin(std_len, length_bin_max)}",
            ]
        )
    return tokens


def _first_iat_code(seconds: float, max_bin: int) -> str:
    bin_id = iat_bin(seconds, max_bin)
    if bin_id == 0:
        return "Z"
    if bin_id <= 4:
        return "S"
    if bin_id <= 8:
        return "M"
    return "L"


def _first_k_signature_tokens(flow: dict[str, Any], *, first_k: int, length_bin_max: int, iat_bin_max: int) -> list[str]:
    lens = [max(0, int(item)) for item in flow.get("lens", [])]
    dirs = [bool(item) for item in flow.get("dirs", [])]
    if not lens:
        return ["FIRST_DIR_PATTERN_EMPTY", "FIRST_IAT_PATTERN_EMPTY"]
    k = min(max(1, int(first_k)), len(lens))
    direction_pattern = "".join("C" if item else "S" for item in dirs[:k])
    tokens = [f"FIRST_DIR_PATTERN_{direction_pattern}"]
    tokens.extend(f"FIRST_LEN_BIN_{idx + 1}_{log_bin(lens[idx], length_bin_max)}" for idx in range(k))
    iats = _packet_iats(flow)
    if k <= 1:
        tokens.append("FIRST_IAT_PATTERN_EMPTY")
    else:
        tokens.append(f"FIRST_IAT_PATTERN_{''.join(_first_iat_code(iats[idx], iat_bin_max) for idx in range(1, k))}")
    return tokens


def _transition_profile_tokens(
    flow: dict[str, Any],
    *,
    threshold_seconds: float,
    count_bin_max: int,
    length_bin_max: int,
    iat_bin_max: int,
    max_transitions: int = 4,
) -> list[str]:
    lens = [max(0, int(item)) for item in flow.get("lens", [])]
    dirs = [bool(item) for item in flow.get("dirs", [])]
    tss = [float(item) for item in flow.get("tss", [])]
    if not lens:
        return []
    tokens: list[str] = []
    spans = _burst_spans(flow, threshold_seconds)
    selected_spans = _select_burst_spans(spans, max_bursts=max_transitions + 1)
    for (left_start, left_end), (right_start, right_end) in zip(selected_spans[:-1], selected_spans[1:]):
        left_lens = lens[left_start:left_end]
        right_lens = lens[right_start:right_end]
        if not left_lens or not right_lens:
            continue
        left_dir = "C2S" if sum(1 for item in dirs[left_start:left_end] if item) >= len(left_lens) / 2 else "S2C"
        right_dir = "C2S" if sum(1 for item in dirs[right_start:right_end] if item) >= len(right_lens) / 2 else "S2C"
        left_bytes = sum(left_lens)
        right_bytes = sum(right_lens)
        gap = max(0.0, tss[right_start] - tss[left_end - 1]) if right_start < len(tss) and left_end - 1 < len(tss) else 0.0
        byte_ratio = left_bytes / max(1.0, float(right_bytes))
        pkt_ratio = len(left_lens) / max(1.0, float(len(right_lens)))
        tokens.extend(
            [
                f"BURST_TRANS_DIR_{left_dir}_TO_{right_dir}",
                f"BURST_TRANS_GAP_{iat_bin(gap, iat_bin_max)}",
                f"BURST_TRANS_BYTES_RATIO_{ratio_bin(byte_ratio / (1.0 + byte_ratio), count_bin_max)}",
                f"BURST_TRANS_PKTN_RATIO_{ratio_bin(pkt_ratio / (1.0 + pkt_ratio), count_bin_max)}",
            ]
        )
    c2s_runs: list[int] = []
    s2c_runs: list[int] = []
    turns = 0
    if dirs:
        cur_dir = dirs[0]
        run_len = 0
        for direction in dirs:
            if direction == cur_dir:
                run_len += 1
            else:
                (c2s_runs if cur_dir else s2c_runs).append(run_len)
                turns += 1
                cur_dir = direction
                run_len = 1
        (c2s_runs if cur_dir else s2c_runs).append(run_len)
    tokens.extend(
        [
            f"BURST_RUN_C2S_{count_bin(max(c2s_runs, default=0), count_bin_max)}",
            f"BURST_RUN_S2C_{count_bin(max(s2c_runs, default=0), count_bin_max)}",
            f"BURST_TURN_COUNT_{count_bin(turns, count_bin_max)}",
        ]
    )
    for left, right in zip(lens[:8], lens[1:9]):
        if right > left * 1.25:
            code = "UP"
        elif left > right * 1.25:
            code = "DOWN"
        else:
            code = "STABLE"
        tokens.append(f"PKT_LEN_TRANS_{code}")
    return tokens


def _profile_position_tags(flow: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    tags = [profile_primitive_token("NONE")] * len(flow["lens"])
    if profile.get("short"):
        return [profile_primitive_token("SHORT")] * len(flow["lens"])
    if profile.get("same"):
        tags = [profile_primitive_token("SAME")] * len(flow["lens"])
    packet_lens = {item["length"] for item in profile.get("packet", []) if "length" in item}
    repeat_lens = {item["length"] for item in profile.get("repeat", []) if "length" in item}
    for idx, length in enumerate(flow["lens"]):
        if length in repeat_lens:
            tags[idx] = profile_primitive_token("REPEAT")
        elif length in packet_lens and tags[idx] == profile_primitive_token("NONE"):
            tags[idx] = profile_primitive_token("PKT")
    if profile.get("local"):
        # Local segment matches do not carry exact positions in this compressed representation.
        for idx, tag in enumerate(tags):
            if tag == profile_primitive_token("NONE"):
                tags[idx] = profile_primitive_token("LOCAL")
                break
    if profile.get("duplicate"):
        for idx in range(len(tags) - 1, -1, -1):
            if tags[idx] == profile_primitive_token("NONE"):
                tags[idx] = profile_primitive_token("DUP")
                break
    return tags


def _raw_profile_id_tokens(profile: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key in ("short", "same"):
        item = profile.get(key)
        if item:
            tokens.append(f"{RAW_PROFILE_PRIMITIVE_PREFIX}{item.get('type', key).upper()}_{item.get('len_bin', 'NA')}_{item.get('count_bin', 'NA')}")
    for key in ("packet", "local", "repeat", "duplicate"):
        for idx, item in enumerate(profile.get(key, [])[:16]):
            parts = [f"{RAW_PROFILE_PRIMITIVE_PREFIX}{key.upper()}", str(idx)]
            for field in ("type", "dir", "len_bin", "count_bin", "match"):
                if field in item:
                    parts.append(str(item[field]))
            tokens.append("_".join(parts))
    return tokens


class PrimitiveTrafficTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        max_packets: int = 96,
        length_bin_max: int = 15,
        iat_bin_max: int = 15,
        count_bin_max: int = 15,
        burst_iat_threshold_ms: int = 100,
        use_profile_tokens: bool | None = None,
        profile_mode: str | None = "full",
        use_service_context: bool = False,
        record_service_context: bool = False,
        use_service_tokens: bool = False,
        service_context_window_seconds: float = 2.0,
        service_context_short_packet_threshold: int = 6,
        use_burst_shape_tokens: bool = False,
        use_first_k_signature: bool = False,
        first_k: int = 3,
        use_context_profile_tokens: bool | None = None,
        use_transition_profile_tokens: bool | None = None,
        use_raw_profile_id_tokens: bool | None = None,
        vocab: Vocabulary | None = None,
    ) -> None:
        self.max_len = max_len
        self.max_packets = max_packets
        self.length_bin_max = length_bin_max
        self.iat_bin_max = iat_bin_max
        self.count_bin_max = count_bin_max
        self.burst_threshold_seconds = burst_iat_threshold_ms / 1000.0
        use_profile = True if use_profile_tokens is None else bool(use_profile_tokens)
        resolved_profile_mode = str(profile_mode or "full")
        if use_profile_tokens is not None:
            use_profile = bool(use_profile_tokens)
        resolved_context_profile = bool(use_context_profile_tokens) if use_context_profile_tokens is not None else False
        resolved_transition_profile = bool(use_transition_profile_tokens) if use_transition_profile_tokens is not None else False
        resolved_raw_profile = bool(use_raw_profile_id_tokens) if use_raw_profile_id_tokens is not None else False
        if not use_profile:
            resolved_profile_mode = "none"
        if resolved_profile_mode not in {"none", "packet", "summary", "full"}:
            raise ValueError(f"Unsupported profile_mode: {resolved_profile_mode}")
        self.profile_mode = resolved_profile_mode
        self.use_service_context = use_service_context
        self.record_service_context = record_service_context or use_service_context
        self.use_service_tokens = use_service_tokens
        self.service_context_window_seconds = float(service_context_window_seconds)
        self.service_context_short_packet_threshold = int(service_context_short_packet_threshold)
        self.use_burst_shape_tokens = bool(use_burst_shape_tokens)
        self.use_first_k_signature = bool(use_first_k_signature)
        self.first_k = max(1, int(first_k))
        self.use_context_profile_tokens = resolved_context_profile
        self.use_transition_profile_tokens = resolved_transition_profile
        self.use_raw_profile_id_tokens = resolved_raw_profile
        self.vocab = vocab or default_vocab(max(length_bin_max, iat_bin_max, count_bin_max))
        if self.use_service_tokens:
            _extend_service_token_vocab(self.vocab, self.count_bin_max)
        _extend_optional_token_vocab(
            self.vocab,
            max(length_bin_max, iat_bin_max, count_bin_max),
            use_burst_shape_tokens=self.use_burst_shape_tokens,
            use_first_k_signature=self.use_first_k_signature,
            first_k=self.first_k,
            use_context_profile_tokens=self.use_context_profile_tokens,
            use_transition_profile_tokens=self.use_transition_profile_tokens,
        )

    def flow_tokens(self, flow: dict[str, Any], profile_row: dict[str, Any], service_context: dict[str, Any] | None = None) -> list[str]:
        return self.raw_flow_tokens(flow, profile_row, service_context=service_context)[: self.max_len]

    def raw_flow_tokens(self, flow: dict[str, Any], profile_row: dict[str, Any], service_context: dict[str, Any] | None = None) -> list[str]:
        profile = profile_row.get("profile", {})
        pkt_n = len(flow["lens"])
        duration = float(flow.get("duration") or 0.0)
        dirs = [bool(item) for item in flow["dirs"]]
        c2s_ratio = (sum(1 for item in dirs if item) / pkt_n) if pkt_n else 0.0
        rhythm_profile = profile if self.profile_mode != "none" else {}
        tokens = [
            "[CLS]",
            f"FLOW_PKTN_{log_bin(pkt_n, self.count_bin_max)}",
            f"FLOW_DUR_{log_bin(duration * 1000.0, self.count_bin_max)}",
            f"FLOW_BURSTN_{log_bin(burst_count(flow, self.burst_threshold_seconds), self.count_bin_max)}",
            f"FLOW_DIR_RATIO_{ratio_bin(c2s_ratio, self.count_bin_max)}",
            _flow_rhythm(flow, rhythm_profile),
        ]
        if self.use_service_context:
            tokens.extend((service_context or {}).get("tokens", _service_context_tokens(0, 0, 0, 0.0, None, self.count_bin_max)))
        if self.use_context_profile_tokens:
            tokens.extend(_context_profile_tokens(service_context, self.count_bin_max))
        if self.use_service_tokens:
            tokens.extend(_service_feature_tokens(flow, self.count_bin_max))
        if self.use_burst_shape_tokens:
            tokens.extend(
                _burst_shape_tokens(
                    flow,
                    threshold_seconds=self.burst_threshold_seconds,
                    count_bin_max=self.count_bin_max,
                    length_bin_max=self.length_bin_max,
                    iat_bin_max=self.iat_bin_max,
                )
            )
        if self.use_first_k_signature:
            tokens.extend(
                _first_k_signature_tokens(
                    flow,
                    first_k=self.first_k,
                    length_bin_max=self.length_bin_max,
                    iat_bin_max=self.iat_bin_max,
                )
            )
        if self.use_transition_profile_tokens:
            tokens.extend(
                _transition_profile_tokens(
                    flow,
                    threshold_seconds=self.burst_threshold_seconds,
                    count_bin_max=self.count_bin_max,
                    length_bin_max=self.length_bin_max,
                    iat_bin_max=self.iat_bin_max,
                )
            )
        if self.use_raw_profile_id_tokens:
            tokens.extend(_raw_profile_id_tokens(profile))

        if self.profile_mode in {"summary", "full"}:
            for key in ["short", "same"]:
                if profile.get(key):
                    tokens.append(profile_primitive_token(profile[key]["type"]))
                    if "len_bin" in profile[key]:
                        tokens.append(f"PKT_LEN_{profile[key]['len_bin']}")
                    if "count_bin" in profile[key]:
                        tokens.append(f"FLOW_COUNT_{profile[key]['count_bin']}")
            for item in profile.get("packet", [])[:8]:
                tokens.extend([profile_primitive_token(item["type"]), f"PKT_DIR_{item['dir']}", f"PKT_LEN_{item['len_bin']}", f"FLOW_COUNT_{item['count_bin']}"])
            for item in profile.get("local", [])[:4]:
                tokens.append(profile_primitive_token("LOCAL"))
                tokens.append(f"PKT_DIR_{item['dir']}")
                tokens.append(f"MATCH_{item['match']}")
                tokens.extend(f"PKT_LEN_{bin_id}" for bin_id in item["len_bins"][:4])
            for item in profile.get("repeat", [])[:4]:
                tokens.extend([profile_primitive_token("REPEAT"), f"PKT_LEN_{item['len_bin']}", f"FLOW_COUNT_{item['count_bin']}"])
            for item in profile.get("duplicate", [])[:4]:
                tokens.extend([profile_primitive_token("DUP"), f"FLOW_COUNT_{item['count_bin']}"])
                tokens.extend(f"PKT_LEN_{bin_id}" for bin_id in item["left_bins"][:2] + item["right_bins"][:2])

        iats = _packet_iats(flow)
        bursts = burst_positions(flow, threshold_seconds=self.burst_threshold_seconds)
        profile_tags = _profile_position_tags(flow, profile) if self.profile_mode in {"packet", "full"} else [profile_primitive_token("NONE")] * len(flow["lens"])
        packet_indices = self._select_packet_indices(flow, profile_tags)
        for idx in packet_indices:
            direction = "C2S" if bool(flow["dirs"][idx]) else "S2C"
            tokens.extend(
                [
                    f"PKT_DIR_{direction}",
                    f"PKT_LEN_{log_bin(int(flow['lens'][idx]), self.length_bin_max)}",
                    f"PKT_IAT_{iat_bin(iats[idx], self.iat_bin_max)}",
                    bursts[idx],
                ]
            )
            if self.profile_mode in {"packet", "full"}:
                tokens.append(profile_tags[idx])
        tokens.append("[SEP]")
        return tokens

    def _select_packet_indices(self, flow: dict[str, Any], profile_tags: list[str]) -> list[int]:
        pkt_n = len(flow["lens"])
        if pkt_n <= self.max_packets:
            return list(range(pkt_n))
        head_n = self.max_packets // 2
        tail_n = self.max_packets // 4
        selected = set(range(head_n))
        selected.update(range(max(head_n, pkt_n - tail_n), pkt_n))
        for idx, tag in enumerate(profile_tags):
            if tag != profile_primitive_token("NONE"):
                selected.add(idx)
                if len(selected) >= self.max_packets:
                    break
        return sorted(selected)[: self.max_packets]

    def encode_row(self, flow: dict[str, Any], profile_row: dict[str, Any], service_context: dict[str, Any] | None = None) -> dict[str, Any]:
        raw_tokens = self.raw_flow_tokens(flow, profile_row, service_context=service_context)
        tokens = raw_tokens[: self.max_len]
        for token in tokens:
            self.vocab.add(token)
        input_ids = self.vocab.encode(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.vocab.pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            token_type_ids.extend([0] * pad_len)
        return {
            "input_ids": input_ids[: self.max_len],
            "attention_mask": attention_mask[: self.max_len],
            "token_type_ids": token_type_ids[: self.max_len],
            "length": min(len(tokens), self.max_len),
            "meta": {
                "flow_id": flow["flow_id"],
                "split": flow.get("split"),
                "label": flow["label"],
                "binary_label": flow["binary_label"],
                "packet_count": flow["packet_count"],
                "token_count": min(len(tokens), self.max_len),
                "raw_token_count": len(raw_tokens),
                "truncated": len(raw_tokens) > self.max_len,
                "start_ts": flow.get("start_ts"),
                "end_ts": flow.get("end_ts"),
                "duration": flow.get("duration"),
                "dataset_file": flow.get("dataset_file"),
                "src_ip": flow.get("src_ip"),
                "dst_ip": flow.get("dst_ip"),
                "src_port": flow.get("src_port"),
                "dst_port": flow.get("dst_port"),
                "protocol": flow.get("protocol"),
                "service_key": flow.get("service_key"),
                "service_context": service_context,
            },
        }


def build_token_dataset(
    flows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    tokenizer: PrimitiveTrafficTokenizer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_by_id = {row["flow_id"]: row for row in profile_rows}
    service_context_by_id = (
        _service_context_rows(
            flows,
            window_seconds=tokenizer.service_context_window_seconds,
            short_packet_threshold=tokenizer.service_context_short_packet_threshold,
            count_bin_max=tokenizer.count_bin_max,
        )
        if tokenizer.record_service_context
        else {}
    )
    labels = sorted({flow["label"] for flow in flows})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    binary_to_id = {"BENIGN": 0, "ATTACK": 1}
    encoded = []
    for flow in flows:
        if flow["flow_id"] not in profile_by_id:
            continue
        row = tokenizer.encode_row(flow, profile_by_id[flow["flow_id"]], service_context=service_context_by_id.get(flow["flow_id"]))
        row["label"] = label_to_id[flow["label"]]
        row["binary_label"] = binary_to_id[flow["binary_label"]]
        encoded.append(row)
    tensors = {
        "input_ids": torch.tensor([row["input_ids"] for row in encoded], dtype=torch.long),
        "attention_mask": torch.tensor([row["attention_mask"] for row in encoded], dtype=torch.long),
        "token_type_ids": torch.tensor([row["token_type_ids"] for row in encoded], dtype=torch.long),
        "labels": torch.tensor([row["label"] for row in encoded], dtype=torch.long),
        "binary_labels": torch.tensor([row["binary_label"] for row in encoded], dtype=torch.long),
        "meta": [row["meta"] for row in encoded],
        "label_to_id": label_to_id,
        "binary_label_to_id": binary_to_id,
        "vocab": tokenizer.vocab.to_dict(),
        "max_len": tokenizer.max_len,
        "profile_mode": tokenizer.profile_mode,
        "use_service_context": tokenizer.use_service_context,
        "record_service_context": tokenizer.record_service_context,
        "use_service_tokens": tokenizer.use_service_tokens,
        "use_burst_shape_tokens": tokenizer.use_burst_shape_tokens,
        "use_first_k_signature": tokenizer.use_first_k_signature,
        "first_k": tokenizer.first_k,
        "use_context_profile_tokens": tokenizer.use_context_profile_tokens,
        "use_transition_profile_tokens": tokenizer.use_transition_profile_tokens,
    }
    raw_lengths = [int(row["meta"].get("raw_token_count", row["length"])) for row in encoded]
    truncated_count = sum(1 for length in raw_lengths if length > tokenizer.max_len)
    stats = {
        "num_flows": len(encoded),
        "labels": label_to_id,
        "binary_labels": binary_to_id,
        "vocab_size": len(tokenizer.vocab.token_to_id),
        "token_count_hist": dict(Counter(row["length"] for row in encoded)),
        "raw_token_count_hist": dict(Counter(raw_lengths)),
        "avg_token_length": float(sum(row["length"] for row in encoded) / len(encoded)) if encoded else 0.0,
        "avg_raw_token_length": float(sum(raw_lengths) / len(raw_lengths)) if raw_lengths else 0.0,
        "truncated_count": int(truncated_count),
        "truncation_ratio": float(truncated_count / len(encoded)) if encoded else 0.0,
        "use_service_context": tokenizer.use_service_context,
        "record_service_context": tokenizer.record_service_context,
        "use_service_tokens": tokenizer.use_service_tokens,
        "use_burst_shape_tokens": tokenizer.use_burst_shape_tokens,
        "use_first_k_signature": tokenizer.use_first_k_signature,
        "first_k": tokenizer.first_k,
        "use_context_profile_tokens": tokenizer.use_context_profile_tokens,
        "use_transition_profile_tokens": tokenizer.use_transition_profile_tokens,
    }
    return tensors, stats
