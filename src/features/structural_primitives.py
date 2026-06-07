from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from .token_alias import STRUCTURAL_PRIMITIVE_PREFIX, canonical_tokens


PRIM_STRUCT_PREFIX = STRUCTURAL_PRIMITIVE_PREFIX


@dataclass(frozen=True)
class StructuralPrimitiveConfig:
    """Configuration for deriving structural primitive tokens from behavior tokens."""

    enabled: bool = False
    enable_packet_shape_primitives: bool = False
    enable_burst_shape_primitives: bool = False
    enable_timing_rhythm_primitives: bool = False
    enable_direction_transition_primitives: bool = False
    enable_composite_primitives: bool = False
    enable_position_aware_primitives: bool = False
    enable_parameterized_primitives: bool = False
    min_support: int = 5
    ngram_sizes: tuple[int, ...] = (2, 3)
    small_len_bin_max: int = 6
    low_iat_bin_max: int = 2
    high_iat_bin_min: int = 10
    len_spike_delta: int = 4
    max_structural_primitives_per_family: int = 24

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "StructuralPrimitiveConfig":
        """Build a config from a mapping while preserving defaults for missing keys."""

        if not data:
            return cls()
        values = dict(data)
        if "ngram_sizes" in values:
            values["ngram_sizes"] = tuple(int(item) for item in values["ngram_sizes"])
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})

    def is_enabled(self) -> bool:
        return bool(self.enabled) or any(
            [
                self.enable_packet_shape_primitives,
                self.enable_burst_shape_primitives,
                self.enable_timing_rhythm_primitives,
                self.enable_direction_transition_primitives,
                self.enable_composite_primitives,
                self.enable_position_aware_primitives,
                self.enable_parameterized_primitives,
            ]
        )


@dataclass
class StructuralPrimitiveTrigger:
    """A single structural primitive trigger with provenance for audit and aggregation."""

    name: str
    family: str
    count: int = 1
    positions: list[int] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "count": int(self.count),
            "positions": list(self.positions),
            "source": self.source,
        }


def _struct_token(token: str) -> str:
    return canonical_tokens([token])[0]


def _struct_name(suffix: str) -> str:
    return f"{PRIM_STRUCT_PREFIX}{suffix}"


def family_from_token(token: str) -> str:
    """Return the family for a canonical structural primitive token."""

    token = _struct_token(token)
    if token.startswith("PRIM_STRUCT_BURST_INTERVAL_"):
        return "timing_rhythm"
    if token.startswith("PRIM_STRUCT_PKT_"):
        return "packet_shape"
    if token.startswith("PRIM_STRUCT_BURST_"):
        return "burst_shape"
    if token.startswith("PRIM_STRUCT_IAT_") or token.startswith("PRIM_STRUCT_FLOW_"):
        return "timing_rhythm"
    if token.startswith("PRIM_STRUCT_DIR_"):
        return "direction_transition"
    if token.startswith("PRIM_STRUCT_COMP_"):
        return "composite"
    if token.startswith("PRIM_STRUCT_POS_"):
        return "position_aware"
    if token.startswith("PRIM_STRUCT_PARAM_"):
        return "parameterized"
    return "unknown"


def family_slug(family: str) -> str:
    return {
        "packet_shape": "packet",
        "burst_shape": "burst",
        "timing_rhythm": "timing",
        "direction_transition": "direction",
        "composite": "composite",
        "position_aware": "position",
        "parameterized": "parameter",
    }.get(family, family)


def _int_suffix(token: str, prefix: str, default: int = 0) -> int:
    if not token.startswith(prefix):
        return default
    try:
        return int(token.removeprefix(prefix))
    except ValueError:
        return default


def _bucket_run(length: int) -> int:
    if length <= 2:
        return 2
    if length == 3:
        return 3
    if length <= 5:
        return 4
    if length <= 8:
        return 8
    return 16


def _sanitize(parts: Iterable[str]) -> str:
    return "_".join(str(part).replace("-", "").replace(":", "").replace("/", "") for part in parts)


def _position_bucket(position: int | None, max_position: int) -> str:
    if position is None or max_position <= 0:
        return "UNKNOWN"
    ratio = float(position) / max(float(max_position), 1.0)
    if ratio < 1.0 / 3.0:
        return "PREFIX"
    if ratio < 2.0 / 3.0:
        return "MIDDLE"
    return "TAIL"


def _base_position_name(name: str) -> str:
    prefix_map = {
        "PRIM_STRUCT_PKT_C2S_SMALL_RUN": "C2S_SMALL_RUN",
        "PRIM_STRUCT_PKT_S2C_SMALL_RUN": "S2C_SMALL_RUN",
        "PRIM_STRUCT_PKT_SAME_LEN_RUN_C2S": "C2S_SAME_LEN_RUN",
        "PRIM_STRUCT_PKT_SAME_LEN_RUN_S2C": "S2C_SAME_LEN_RUN",
        "PRIM_STRUCT_PKT_LEN_SPIKE": "LEN_SPIKE",
        "PRIM_STRUCT_PKT_LEN_RAMP_UP": "LEN_RAMP_UP",
        "PRIM_STRUCT_PKT_LEN_RAMP_DOWN": "LEN_RAMP_DOWN",
        "PRIM_STRUCT_BURST_TEMPLATE_DUP": "DUP_BURST",
        "PRIM_STRUCT_BURST_REQ_RESP_REPEAT": "REQ_RESP_REPEAT",
        "PRIM_STRUCT_BURST_REQ_RESP_PAIR": "REQ_RESP_PAIR",
        "PRIM_STRUCT_IAT_BURSTY_THEN_IDLE": "BURSTY_THEN_IDLE",
        "PRIM_STRUCT_COMP_LONG_IDLE_THEN_DUP": "LONG_IDLE_THEN_DUP",
        "PRIM_STRUCT_COMP_REQ_RESP_TEMPLATE_REPEAT": "REQ_RESP_TEMPLATE_REPEAT",
        "PRIM_STRUCT_COMP_LOW_RATE_PERIODIC_SMALL_PKT": "LOW_RATE_PERIODIC_SMALL_PKT",
    }
    for prefix, label in prefix_map.items():
        if name.startswith(prefix):
            return label
    if name.startswith("PRIM_STRUCT_PKT_DIR_LEN_NGRAM_"):
        return "DIR_LEN_NGRAM"
    return name.removeprefix(PRIM_STRUCT_PREFIX)


def _bin_ge(value: int, *, small: int, medium: int) -> str:
    if value <= small:
        return f"LE_{small}"
    if value <= medium:
        return f"{small + 1}_{medium}"
    return f"GE_{medium + 1}"


def parse_packet_events(tokens: list[str]) -> list[dict[str, Any]]:
    """Parse direction/length/IAT packet events from FlowPrim behavior-token order."""

    tokens = canonical_tokens(tokens)
    events: list[dict[str, Any]] = []
    for idx in range(0, max(0, len(tokens) - 2)):
        direction = tokens[idx]
        if direction not in {"PKT_DIR_C2S", "PKT_DIR_S2C"}:
            continue
        if not tokens[idx + 1].startswith("PKT_LEN_") or not tokens[idx + 2].startswith("PKT_IAT_"):
            continue
        burst = tokens[idx + 3] if idx + 3 < len(tokens) and tokens[idx + 3] in {"BURST_START", "BURST_MID", "BURST_END", "BURST_SINGLE"} else ""
        events.append(
            {
                "pos": idx,
                "dir": direction.removeprefix("PKT_DIR_"),
                "len": _int_suffix(tokens[idx + 1], "PKT_LEN_"),
                "iat": _int_suffix(tokens[idx + 2], "PKT_IAT_"),
                "burst_tag": burst,
            }
        )
    return events


def parse_burst_events(tokens: list[str]) -> list[dict[str, Any]]:
    """Parse compact burst-shape records from canonical BURST_* token blocks."""

    tokens = canonical_tokens(tokens)
    bursts: list[dict[str, Any]] = []
    idx = 0
    while idx < len(tokens):
        if not tokens[idx].startswith("BURST_PKTN_"):
            idx += 1
            continue
        cur: dict[str, Any] = {"pos": idx, "pktn": _int_suffix(tokens[idx], "BURST_PKTN_")}
        j = idx + 1
        while j < len(tokens) and tokens[j].startswith("BURST_") and not tokens[j].startswith("BURST_PKTN_"):
            tok = tokens[j]
            if tok.startswith("BURST_DUR_"):
                cur["dur"] = _int_suffix(tok, "BURST_DUR_")
            elif tok.startswith("BURST_BYTES_C2S_"):
                cur["bytes_c2s"] = _int_suffix(tok, "BURST_BYTES_C2S_")
            elif tok.startswith("BURST_BYTES_S2C_"):
                cur["bytes_s2c"] = _int_suffix(tok, "BURST_BYTES_S2C_")
            elif tok in {"BURST_DIR_C2S", "BURST_DIR_S2C", "BURST_DIR_MIXED"}:
                cur["dir"] = tok.removeprefix("BURST_DIR_")
            elif tok.startswith("BURST_IAT_MED_"):
                cur["iat_med"] = _int_suffix(tok, "BURST_IAT_MED_")
            elif tok.startswith("BURST_GAP_NEXT_"):
                cur["gap_next"] = _int_suffix(tok, "BURST_GAP_NEXT_")
            elif tok.startswith("BURST_LEN_MEAN_"):
                cur["len_mean"] = _int_suffix(tok, "BURST_LEN_MEAN_")
            elif tok.startswith("BURST_LEN_STD_"):
                cur["len_std"] = _int_suffix(tok, "BURST_LEN_STD_")
            j += 1
        bursts.append(cur)
        idx = j
    return bursts


def _add(counter: dict[str, StructuralPrimitiveTrigger], name: str, family: str, *, position: int | None = None, source: str = "") -> None:
    item = counter.get(name)
    if item is None:
        counter[name] = StructuralPrimitiveTrigger(name=name, family=family, count=1, positions=[] if position is None else [int(position)], source=source)
    else:
        item.count += 1
        if position is not None:
            item.positions.append(int(position))


def _cap_family(triggers: list[StructuralPrimitiveTrigger], max_per_family: int) -> list[StructuralPrimitiveTrigger]:
    by_family: dict[str, list[StructuralPrimitiveTrigger]] = defaultdict(list)
    for trigger in triggers:
        by_family[trigger.family].append(trigger)
    out: list[StructuralPrimitiveTrigger] = []
    for family in sorted(by_family):
        selected = sorted(by_family[family], key=lambda item: (-item.count, item.name))[: max(1, int(max_per_family))]
        out.extend(selected)
    return sorted(out, key=lambda item: (item.family, item.name))


def extract_structural_primitive_candidates(tokens: list[str], config: StructuralPrimitiveConfig | None = None) -> list[StructuralPrimitiveTrigger]:
    """Extract raw structural primitive candidates from one tokenized flow.

    The extractor only uses direction, length-bin, IAT-bin, burst-shape, and
    transition tokens. It does not inspect raw IP addresses, absolute time, or
    complete five-tuples.
    """

    cfg = config or StructuralPrimitiveConfig()
    packets = parse_packet_events(tokens)
    bursts = parse_burst_events(tokens)
    found: dict[str, StructuralPrimitiveTrigger] = {}
    names_seen: set[str] = set()

    def emit(name: str, family: str, *, position: int | None = None, source: str = "") -> None:
        names_seen.add(name)
        _add(found, name, family, position=position, source=source)

    dirs = [str(item["dir"]) for item in packets]
    lens = [int(item["len"]) for item in packets]
    iats = [int(item["iat"]) for item in packets]

    if cfg.enable_packet_shape_primitives and packets:
        for n in cfg.ngram_sizes:
            if n <= 1 or len(packets) < n:
                continue
            for start in range(0, len(packets) - n + 1):
                pat: list[str] = []
                for item in packets[start : start + n]:
                    pat.extend([str(item["dir"]), f"L{int(item['len'])}"])
                emit(f"PRIM_STRUCT_PKT_DIR_LEN_NGRAM_{_sanitize(pat)}", "packet_shape", position=int(packets[start]["pos"]), source=f"packet dir+len {n}-gram")

        run_start = 0
        for idx in range(1, len(packets) + 1):
            end_run = idx == len(packets) or packets[idx]["dir"] != packets[run_start]["dir"] or packets[idx]["len"] > cfg.small_len_bin_max
            if packets[run_start]["len"] <= cfg.small_len_bin_max and end_run:
                run_len = idx - run_start
                if run_len >= 2:
                    direction = packets[run_start]["dir"]
                    emit(
                        f"PRIM_STRUCT_PKT_{direction}_SMALL_RUN_{_bucket_run(run_len)}",
                        "packet_shape",
                        position=int(packets[run_start]["pos"]),
                        source="same-direction small-packet run",
                    )
                run_start = idx

        run_start = 0
        for idx in range(1, len(packets) + 1):
            end_run = idx == len(packets) or packets[idx]["dir"] != packets[run_start]["dir"] or packets[idx]["len"] != packets[run_start]["len"]
            if end_run:
                run_len = idx - run_start
                if run_len >= 2:
                    emit(
                        f"PRIM_STRUCT_PKT_SAME_LEN_RUN_{packets[run_start]['dir']}_{_bucket_run(run_len)}_L{packets[run_start]['len']}",
                        "packet_shape",
                        position=int(packets[run_start]["pos"]),
                        source="same-direction same-length run",
                    )
                run_start = idx

        for idx in range(1, len(lens)):
            delta = lens[idx] - lens[idx - 1]
            if abs(delta) >= cfg.len_spike_delta:
                emit("PRIM_STRUCT_PKT_LEN_SPIKE", "packet_shape", position=int(packets[idx]["pos"]), source="adjacent length-bin jump")
        for idx in range(0, len(lens) - 2):
            if lens[idx] < lens[idx + 1] < lens[idx + 2]:
                emit("PRIM_STRUCT_PKT_LEN_RAMP_UP", "packet_shape", position=int(packets[idx]["pos"]), source="three-packet length ramp up")
            if lens[idx] > lens[idx + 1] > lens[idx + 2]:
                emit("PRIM_STRUCT_PKT_LEN_RAMP_DOWN", "packet_shape", position=int(packets[idx]["pos"]), source="three-packet length ramp down")

    if cfg.enable_burst_shape_primitives:
        burst_dirs = [str(item.get("dir", "")) for item in bursts if item.get("dir")]
        if burst_dirs:
            counts = Counter(burst_dirs)
            direction, count = counts.most_common(1)[0]
            if direction in {"C2S", "S2C"} and count / max(len(burst_dirs), 1) >= 0.8:
                emit(f"PRIM_STRUCT_BURST_SINGLE_DIR_DOMINANT_{direction}", "burst_shape", source="dominant burst direction")
            req_resp = sum(1 for left, right in zip(burst_dirs, burst_dirs[1:]) if left == "C2S" and right == "S2C")
            if req_resp >= 1:
                emit("PRIM_STRUCT_BURST_REQ_RESP_PAIR", "burst_shape", source="C2S burst followed by S2C burst")
            if req_resp >= 2:
                emit(f"PRIM_STRUCT_BURST_REQ_RESP_REPEAT_{_bucket_run(req_resp)}", "burst_shape", source="repeated request-response burst pairs")
            turns = sum(1 for left, right in zip(burst_dirs, burst_dirs[1:]) if left != right)
            low_gap = sum(1 for item in bursts if int(item.get("gap_next", 99)) <= cfg.low_iat_bin_max)
            if len(burst_dirs) >= 4 and turns >= len(burst_dirs) - 2 and low_gap >= max(1, len(burst_dirs) // 2):
                emit("PRIM_STRUCT_BURST_ALT_FAST", "burst_shape", source="fast alternating burst directions")
            templates = [(item.get("dir"), item.get("pktn"), item.get("len_mean")) for item in bursts]
            if any(cur == prev for prev, cur in zip(templates, templates[1:])):
                emit("PRIM_STRUCT_BURST_TEMPLATE_DUP", "burst_shape", source="adjacent duplicate burst template")
        if bursts:
            c2s_bytes = sum(int(item.get("bytes_c2s", 0)) for item in bursts)
            s2c_bytes = sum(int(item.get("bytes_s2c", 0)) for item in bursts)
            if c2s_bytes >= s2c_bytes + 4:
                emit("PRIM_STRUCT_BURST_ASYM_BYTES_C2S", "burst_shape", source="C2S byte-bin dominance across bursts")
            if s2c_bytes >= c2s_bytes + 4:
                emit("PRIM_STRUCT_BURST_ASYM_BYTES_S2C", "burst_shape", source="S2C byte-bin dominance across bursts")
            short_bursts = sum(1 for item in bursts if int(item.get("pktn", 0)) <= 1)
            if short_bursts >= 4:
                emit("PRIM_STRUCT_BURST_SHORT_MANY", "burst_shape", source="many one-packet bursts")

    if cfg.enable_timing_rhythm_primitives and iats:
        iat_counts = Counter(iats)
        common_iat, common_count = iat_counts.most_common(1)[0]
        if len(iats) >= 4 and common_count / max(len(iats), 1) >= 0.7:
            emit("PRIM_STRUCT_IAT_LOW_VARIANCE", "timing_rhythm", source="dominant IAT bin")
            if common_iat > 0:
                emit("PRIM_STRUCT_IAT_REGULAR_BEACON", "timing_rhythm", source="regular nonzero IAT bin")
        if len(iats) >= 4 and min(iats) <= cfg.low_iat_bin_max and max(iats) >= cfg.high_iat_bin_min:
            emit("PRIM_STRUCT_IAT_HEAVY_TAIL", "timing_rhythm", source="low median and high-tail IAT bins")
        for idx in range(0, len(iats) - 3):
            if max(iats[idx : idx + 3]) <= cfg.low_iat_bin_max and iats[idx + 3] >= cfg.high_iat_bin_min:
                emit("PRIM_STRUCT_IAT_BURSTY_THEN_IDLE", "timing_rhythm", position=int(packets[idx]["pos"]), source="low-IAT run followed by high-IAT gap")
        if len(iats) <= 8 and sum(1 for item in iats if item >= cfg.high_iat_bin_min) >= 2:
            emit("PRIM_STRUCT_FLOW_SLOW_DRIP", "timing_rhythm", source="few packets with repeated high IAT bins")
        gaps = [int(item.get("gap_next", 0)) for item in bursts if "gap_next" in item]
        if len(gaps) >= 3 and Counter(gaps).most_common(1)[0][1] >= max(2, len(gaps) - 1):
            emit("PRIM_STRUCT_BURST_INTERVAL_PERIODIC", "timing_rhythm", source="repeated burst gap bins")

    if cfg.enable_direction_transition_primitives and dirs:
        if all(item == "C2S" for item in dirs):
            emit("PRIM_STRUCT_DIR_MONO_C2S", "direction_transition", source="all parsed packet directions are C2S")
            emit("PRIM_STRUCT_DIR_SERVER_RESPONSE_ABSENT", "direction_transition", source="no parsed S2C packet response")
        if all(item == "S2C" for item in dirs):
            emit("PRIM_STRUCT_DIR_MONO_S2C", "direction_transition", source="all parsed packet directions are S2C")
        turns = [idx for idx, (left, right) in enumerate(zip(dirs, dirs[1:]), start=1) if left != right]
        if len(dirs) >= 4 and len(turns) >= len(dirs) - 2:
            emit("PRIM_STRUCT_DIR_ALTERNATING", "direction_transition", source="packet direction alternates")
        if len(turns) == 1:
            left, right = dirs[0], dirs[-1]
            if left == "C2S" and right == "S2C":
                emit("PRIM_STRUCT_DIR_TWO_PHASE_C2S_THEN_S2C", "direction_transition", position=turns[0], source="single direction transition")
            if left == "S2C" and right == "C2S":
                emit("PRIM_STRUCT_DIR_TWO_PHASE_S2C_THEN_C2S", "direction_transition", position=turns[0], source="single direction transition")

    if cfg.enable_composite_primitives:
        has = names_seen.__contains__
        if any(name.startswith("PRIM_STRUCT_PKT_C2S_SMALL_RUN") for name in names_seen) and (has("PRIM_STRUCT_IAT_LOW_VARIANCE") or has("PRIM_STRUCT_IAT_REGULAR_BEACON")):
            emit("PRIM_STRUCT_COMP_FAST_C2S_SMALL_REPEAT", "composite", source="C2S small-packet run plus low-variance timing")
        if has("PRIM_STRUCT_BURST_TEMPLATE_DUP") and (has("PRIM_STRUCT_IAT_LOW_VARIANCE") or has("PRIM_STRUCT_BURST_INTERVAL_PERIODIC")):
            emit("PRIM_STRUCT_COMP_DUP_BURST_LOW_IAT", "composite", source="duplicate burst template plus regular/low IAT")
        if has("PRIM_STRUCT_DIR_MONO_C2S") and any(name.startswith("PRIM_STRUCT_PKT_SAME_LEN_RUN_C2S") for name in names_seen) and len(packets) <= 6:
            emit("PRIM_STRUCT_COMP_SHORT_FLOW_SAME_LEN_C2S", "composite", source="short mono-C2S same-length packet run")
        if has("PRIM_STRUCT_BURST_REQ_RESP_PAIR") and (has("PRIM_STRUCT_BURST_TEMPLATE_DUP") or any(name.startswith("PRIM_STRUCT_BURST_REQ_RESP_REPEAT") for name in names_seen)):
            emit("PRIM_STRUCT_COMP_REQ_RESP_TEMPLATE_REPEAT", "composite", source="request-response burst pattern repeat")
        if has("PRIM_STRUCT_IAT_BURSTY_THEN_IDLE") and has("PRIM_STRUCT_BURST_TEMPLATE_DUP"):
            emit("PRIM_STRUCT_COMP_LONG_IDLE_THEN_DUP", "composite", source="idle gap plus duplicate burst template")
        if has("PRIM_STRUCT_IAT_REGULAR_BEACON") and any(name.endswith("SMALL_RUN_2") or "_SMALL_RUN_" in name for name in names_seen):
            emit("PRIM_STRUCT_COMP_LOW_RATE_PERIODIC_SMALL_PKT", "composite", source="periodic timing plus small-packet repetition")

    if cfg.enable_position_aware_primitives and found:
        max_pos = max([int(item.get("pos", 0)) for item in packets] + [int(item.get("pos", 0)) for item in bursts] + [1])
        for trigger in list(found.values()):
            if not trigger.name.startswith(("PRIM_STRUCT_PKT_", "PRIM_STRUCT_BURST_", "PRIM_STRUCT_IAT_", "PRIM_STRUCT_FLOW_", "PRIM_STRUCT_COMP_")):
                continue
            pos = trigger.positions[0] if trigger.positions else None
            bucket = _position_bucket(pos, max_pos)
            if bucket == "UNKNOWN":
                continue
            emit(
                f"PRIM_STRUCT_POS_{bucket}_{_base_position_name(trigger.name)}",
                "position_aware",
                position=pos,
                source=f"position bucket for {trigger.name}",
            )

    if cfg.enable_parameterized_primitives:
        for direction in ("C2S", "S2C"):
            max_run = 0
            run_start = 0
            for idx in range(1, len(packets) + 1):
                end_run = idx == len(packets) or packets[idx]["dir"] != packets[run_start]["dir"] or packets[idx]["len"] > cfg.small_len_bin_max
                if packets and packets[run_start]["dir"] == direction and packets[run_start]["len"] <= cfg.small_len_bin_max and end_run:
                    max_run = max(max_run, idx - run_start)
                if end_run:
                    run_start = idx
            if max_run >= 2:
                emit(
                    f"PRIM_STRUCT_PARAM_{direction}_SMALL_RUN_LEN_{_bin_ge(max_run, small=3, medium=5)}",
                    "parameterized",
                    source="maximum same-direction small-packet run length bin",
                )
        if bursts:
            c2s_bytes = sum(int(item.get("bytes_c2s", 0)) for item in bursts)
            s2c_bytes = sum(int(item.get("bytes_s2c", 0)) for item in bursts)
            low = max(min(c2s_bytes, s2c_bytes), 1)
            high = max(c2s_bytes, s2c_bytes)
            ratio = int(high / low)
            if high >= low + 4:
                emit(
                    f"PRIM_STRUCT_PARAM_BURST_ASYM_RATIO_{_bin_ge(ratio, small=3, medium=7)}",
                    "parameterized",
                    source="burst byte-bin asymmetry ratio bin",
                )
            req_resp = sum(1 for left, right in zip([str(item.get("dir", "")) for item in bursts], [str(item.get("dir", "")) for item in bursts][1:]) if left == "C2S" and right == "S2C")
            if req_resp >= 2:
                emit(
                    f"PRIM_STRUCT_PARAM_REQ_RESP_REPEAT_{_bin_ge(req_resp, small=2, medium=3)}",
                    "parameterized",
                    source="request-response repeat count bin",
                )
            emit(
                f"PRIM_STRUCT_PARAM_BURST_COUNT_{_bin_ge(len(bursts), small=3, medium=8)}",
                "parameterized",
                source="burst count bin",
            )
        if len(iats) >= 4:
            mean_iat = sum(iats) / max(len(iats), 1)
            variance = sum((value - mean_iat) ** 2 for value in iats) / max(len(iats), 1)
            cv = 0.0 if mean_iat <= 0 else (variance ** 0.5) / mean_iat
            if cv < 0.25:
                cv_bin = "LOW"
            elif cv < 1.0:
                cv_bin = "MID"
            else:
                cv_bin = "HIGH"
            emit(f"PRIM_STRUCT_PARAM_IAT_CV_{cv_bin}", "parameterized", source="IAT coefficient-of-variation bin")
            long_idle = max(iats)
            if long_idle >= cfg.high_iat_bin_min:
                emit(
                    f"PRIM_STRUCT_PARAM_LONG_IDLE_BIN_{_bin_ge(long_idle, small=10, medium=13)}",
                    "parameterized",
                    source="maximum IAT long-idle bin",
                )

    return _cap_family(list(found.values()), cfg.max_structural_primitives_per_family)


def build_train_only_structural_primitive_vocabulary(
    row_triggers: list[list[StructuralPrimitiveTrigger]],
    train_indices: Iterable[int],
    *,
    min_support: int = 5,
) -> dict[str, int]:
    """Create a deterministic train-only structural primitive vocabulary."""

    support: Counter[str] = Counter()
    for idx in train_indices:
        support.update({trigger.name for trigger in row_triggers[int(idx)]})
    selected = sorted(name for name, count in support.items() if count >= int(min_support))
    return {name: support[name] for name in selected}


def filter_triggers(
    triggers: list[StructuralPrimitiveTrigger],
    vocabulary: dict[str, int],
) -> list[StructuralPrimitiveTrigger]:
    """Keep only triggers admitted by the train-only structural primitive vocabulary."""

    allowed = set(vocabulary)
    return [trigger for trigger in triggers if trigger.name in allowed]
