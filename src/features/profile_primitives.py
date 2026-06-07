from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .bins import count_bin, log_bin


def extract_short_profile_primitive(flow: dict[str, Any], threshold: int = 6) -> dict[str, Any] | None:
    pkt_n = len(flow["lens"])
    if pkt_n < threshold:
        return {"type": "SHORT", "len": pkt_n, "count_bin": count_bin(pkt_n)}
    return None


def extract_same_len_profile_primitive(flow: dict[str, Any], len_bin_max: int = 15, count_bin_max: int = 15) -> dict[str, Any] | None:
    lens = flow["lens"]
    if lens and len(set(lens)) == 1:
        return {
            "type": "SAME",
            "length": lens[0],
            "len_bin": log_bin(lens[0], len_bin_max),
            "count": len(lens),
            "count_bin": count_bin(len(lens), count_bin_max),
        }
    return None


def split_segments_by_direction(flow: dict[str, Any]) -> list[dict[str, Any]]:
    lens = flow["lens"]
    dirs = flow["dirs"]
    if not lens:
        return []
    segments: list[dict[str, Any]] = []
    start = 0
    cur_dir = bool(dirs[0])
    for idx, direction in enumerate(dirs):
        direction = bool(direction)
        if direction != cur_dir:
            segments.append({"dir": cur_dir, "start": start, "end": idx, "lens": lens[start:idx]})
            start = idx
            cur_dir = direction
    segments.append({"dir": cur_dir, "start": start, "end": len(lens), "lens": lens[start:]})
    return segments


def _direction_counts(flow: dict[str, Any]) -> dict[bool, Counter[int]]:
    res = {True: Counter(), False: Counter()}
    for length, direction in zip(flow["lens"], flow["dirs"]):
        res[bool(direction)][int(length)] += 1
    return res


def _best_common_counts(index: int, counters: list[Counter[int]]) -> Counter[int]:
    if len(counters) == 1:
        return Counter(counters[0])
    cur = counters[index]
    best_score = 0
    best = Counter()
    for other_idx, other in enumerate(counters):
        if other_idx == index:
            continue
        common = Counter({length: min(count, other[length]) for length, count in cur.items() if length in other})
        score = sum(common.values())
        if score > best_score:
            best_score = score
            best = common
    return best


def extract_bidir_packet_profile_primitives(
    flows_in_peer_chunk: list[dict[str, Any]],
    len_bin_max: int = 15,
    count_bin_max: int = 15,
    max_items: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    all_counts = [_direction_counts(flow) for flow in flows_in_peer_chunk]
    true_counts = [item[True] for item in all_counts]
    false_counts = [item[False] for item in all_counts]
    out: dict[str, list[dict[str, Any]]] = {}
    for idx, flow in enumerate(flows_in_peer_chunk):
        prims: list[dict[str, Any]] = []
        for direction, counters in [(True, true_counts), (False, false_counts)]:
            common = _best_common_counts(idx, counters)
            if not common and len(counters) == 1:
                common = counters[0]
            for length, cnt in common.most_common(max_items):
                prims.append(
                    {
                        "type": "PKT",
                        "dir": "C2S" if direction else "S2C",
                        "length": int(length),
                        "len_bin": log_bin(length, len_bin_max),
                        "count": int(cnt),
                        "count_bin": count_bin(int(cnt), count_bin_max),
                    }
                )
        out[flow["flow_id"]] = prims[:max_items]
    return out


def _segment_match(candidate: list[int], peer: list[int], peer_plus_2: list[int] | None = None) -> tuple[int, list[int], str]:
    if candidate == peer:
        return len(candidate), list(candidate), "EXACT"
    if abs(sum(candidate) - sum(peer)) <= 1:
        return len(candidate), list(candidate), "SUM"
    score = 0
    matched: list[int] = []
    peer_set = set(peer)
    peer2_set = set(peer_plus_2 or [])
    mode = "FUZZY"
    for item in candidate:
        if item in peer_set:
            score += 1
            matched.append(item)
        elif item in peer2_set:
            score += 1
            matched.append(item)
            mode = "SHIFT2"
        else:
            matched.append(-1)
    return score, matched, mode


def _best_local_segments(index: int, segment_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if len(segment_lists) == 1:
        return [
            {"idx": idx, "dir": "C2S" if seg["dir"] else "S2C", "pattern": list(seg["lens"]), "match": "SINGLE"}
            for idx, seg in enumerate(segment_lists[0])
        ]
    cur = segment_lists[index]
    best_score = -1
    best_segments: list[dict[str, Any]] = []
    for other_idx, peer_segments in enumerate(segment_lists):
        if other_idx == index:
            continue
        score = 0
        matched_segments: list[dict[str, Any]] = []
        for seg_idx, seg in enumerate(cur):
            if seg_idx >= len(peer_segments):
                break
            peer = peer_segments[seg_idx]["lens"]
            peer_plus_2 = peer_segments[seg_idx + 2]["lens"] if seg_idx + 2 < len(peer_segments) else None
            seg_score, pattern, mode = _segment_match(seg["lens"], peer, peer_plus_2)
            score += seg_score
            matched_segments.append(
                {
                    "idx": seg_idx,
                    "dir": "C2S" if seg["dir"] else "S2C",
                    "pattern": pattern,
                    "match": mode,
                }
            )
        if score > best_score:
            best_score = score
            best_segments = matched_segments
    return best_segments


def extract_local_segment_profile_primitives(
    flows_in_peer_chunk: list[dict[str, Any]],
    len_bin_max: int = 15,
    max_items: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    segment_lists = [split_segments_by_direction(flow) for flow in flows_in_peer_chunk]
    out: dict[str, list[dict[str, Any]]] = {}
    for idx, flow in enumerate(flows_in_peer_chunk):
        local = _best_local_segments(idx, segment_lists)
        packed: list[dict[str, Any]] = []
        for item in local:
            pattern = [length for length in item["pattern"] if length >= 0]
            if not pattern:
                continue
            packed.append(
                {
                    "type": "LOCAL",
                    "idx": item["idx"],
                    "dir": item["dir"],
                    "len_bins": [log_bin(length, len_bin_max) for length in pattern[:6]],
                    "match": item["match"],
                }
            )
        out[flow["flow_id"]] = packed[:max_items]
    return out


def extract_repeat_profile_primitives(
    flow: dict[str, Any],
    zero_len_min_repeat: int = 6,
    normal_min_repeat: int = 2,
    len_bin_max: int = 15,
    count_bin_max: int = 15,
) -> list[dict[str, Any]]:
    repeats: Counter[int] = Counter()
    for segment in split_segments_by_direction(flow):
        lens = segment["lens"]
        if len(lens) > 1 and len(set(lens)) == 1:
            repeats[int(lens[0])] += len(lens)
    out: list[dict[str, Any]] = []
    for length, cnt in repeats.most_common():
        threshold = zero_len_min_repeat if length == 0 else normal_min_repeat
        if cnt < threshold:
            continue
        out.append(
            {
                "type": "REPEAT",
                "length": length,
                "len_bin": log_bin(length, len_bin_max),
                "count": int(cnt),
                "count_bin": count_bin(int(cnt), count_bin_max),
            }
        )
    return out


def extract_duplicate_profile_primitives(
    flow: dict[str, Any],
    duplicate_min_repeat: int = 2,
    len_bin_max: int = 15,
    count_bin_max: int = 15,
) -> list[dict[str, Any]]:
    segments = split_segments_by_direction(flow)
    counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    for cur, prev in zip(segments[1:], segments[:-1]):
        key = (tuple(sorted(set(cur["lens"]))), tuple(sorted(set(prev["lens"]))))
        counts[key] += 1
    out: list[dict[str, Any]] = []
    for pair, cnt in counts.most_common():
        if pair == ((0,), (0,)) or cnt < duplicate_min_repeat:
            continue
        left, right = pair
        out.append(
            {
                "type": "DUP",
                "left_bins": [log_bin(length, len_bin_max) for length in left[:4]],
                "right_bins": [log_bin(length, len_bin_max) for length in right[:4]],
                "count": int(cnt),
                "count_bin": count_bin(int(cnt), count_bin_max),
            }
        )
    return out


def extract_all_profile_primitives(
    flows: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config or {}
    len_bin_max = int(cfg.get("length_bin_max", 15))
    count_bin_max = int(cfg.get("count_bin_max", 15))
    packet_by_flow: dict[str, list[dict[str, Any]]] = {}
    local_by_flow: dict[str, list[dict[str, Any]]] = {}
    max_peer_flows = int(cfg.get("max_peer_flows", 128))
    ordered_flows = sorted(flows, key=lambda item: (item.get("start_ts") or 0.0, item["flow_id"]))
    for start in range(0, len(ordered_flows), max_peer_flows):
        chunk = ordered_flows[start : start + max_peer_flows]
        packet_by_flow.update(
            extract_bidir_packet_profile_primitives(
                chunk,
                len_bin_max=len_bin_max,
                count_bin_max=count_bin_max,
                max_items=int(cfg.get("max_packet_profile_primitives", 12)),
            )
        )
        local_by_flow.update(
            extract_local_segment_profile_primitives(
                chunk,
                len_bin_max=len_bin_max,
                max_items=int(cfg.get("max_local_profile_primitives", 8)),
            )
        )

    rows: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    for flow in flows:
        profile = {
            "short": extract_short_profile_primitive(flow, threshold=int(cfg.get("short_flow_packet_threshold", 6))),
            "same": extract_same_len_profile_primitive(flow, len_bin_max=len_bin_max, count_bin_max=count_bin_max),
            "packet": packet_by_flow.get(flow["flow_id"], []),
            "local": local_by_flow.get(flow["flow_id"], []),
            "repeat": extract_repeat_profile_primitives(
                flow,
                zero_len_min_repeat=int(cfg.get("zero_len_min_repeat", 6)),
                normal_min_repeat=int(cfg.get("normal_min_repeat", 2)),
                len_bin_max=len_bin_max,
                count_bin_max=count_bin_max,
            ),
            "duplicate": extract_duplicate_profile_primitives(
                flow,
                duplicate_min_repeat=int(cfg.get("duplicate_min_repeat", 2)),
                len_bin_max=len_bin_max,
                count_bin_max=count_bin_max,
            ),
        }
        for key, value in profile.items():
            if isinstance(value, list):
                if value:
                    type_counts[key] += 1
            elif value:
                type_counts[key] += 1
        rows.append({"flow_id": flow["flow_id"], "label": flow["label"], "profile": profile})
    stats = {
        "num_flows": len(flows),
        "num_peer_chunks": (len(ordered_flows) + max_peer_flows - 1) // max_peer_flows if max_peer_flows > 0 else 0,
        "peer_grouping": "global_train_ordered_chunks",
        "profile_primitive_flow_counts": dict(sorted(type_counts.items())),
    }
    return rows, stats
