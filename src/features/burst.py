from __future__ import annotations

from typing import Any


def burst_positions(flow: dict[str, Any], threshold_seconds: float = 0.1) -> list[str]:
    lens = flow["lens"]
    dirs = flow["dirs"]
    tss = flow["tss"]
    if not lens:
        return []
    groups: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(lens)):
        same_dir = bool(dirs[idx]) == bool(dirs[idx - 1])
        close = (float(tss[idx]) - float(tss[idx - 1])) <= threshold_seconds
        if not (same_dir and close):
            groups.append((start, idx))
            start = idx
    groups.append((start, len(lens)))
    positions = ["BURST_SINGLE"] * len(lens)
    for start, end in groups:
        size = end - start
        if size == 1:
            positions[start] = "BURST_SINGLE"
            continue
        positions[start] = "BURST_START"
        for idx in range(start + 1, end - 1):
            positions[idx] = "BURST_MID"
        positions[end - 1] = "BURST_END"
    return positions


def burst_count(flow: dict[str, Any], threshold_seconds: float = 0.1) -> int:
    positions = burst_positions(flow, threshold_seconds=threshold_seconds)
    return sum(1 for pos in positions if pos in {"BURST_START", "BURST_SINGLE"})
