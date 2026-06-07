from __future__ import annotations

import copy
from typing import Any

import numpy as np


PER_FLOW_PERTURBATION_MODES = {
    "packet_delete",
    "packet_insert",
    "direction_flip",
    "length_padding",
    "length_align",
    "iat_jitter",
    "delay_stretch",
    "burst_split",
    "burst_merge",
    "low_rate_c2",
}

FLOW_LEVEL_PERTURBATION_MODES = {
    "short_flow_delete",
    "benign_flow_insert",
}

PERTURBATION_MODES = PER_FLOW_PERTURBATION_MODES | FLOW_LEVEL_PERTURBATION_MODES


def _refresh_flow_metadata(flow: dict[str, Any], lens: list[int], dirs: list[Any], tss: list[float]) -> dict[str, Any]:
    flow["lens"] = lens
    flow["dirs"] = dirs
    flow["tss"] = tss
    flow["packet_count"] = len(lens)
    flow["start_ts"] = tss[0] if tss else None
    flow["end_ts"] = tss[-1] if tss else None
    flow["duration"] = (tss[-1] - tss[0]) if len(tss) > 1 else 0.0
    return flow


def _positive_iats(tss: list[float]) -> np.ndarray:
    if len(tss) <= 1:
        return np.array([], dtype=np.float64)
    return np.maximum(np.diff(np.array(tss, dtype=np.float64)), 1e-6)


def perturb_flow(flow: dict[str, Any], mode: str, strength: float, rng: np.random.Generator) -> dict[str, Any]:
    out = copy.deepcopy(flow)
    lens = list(out.get("lens", []))
    dirs = list(out.get("dirs", []))
    tss = [float(item) for item in out.get("tss", [])]
    if not lens:
        return out

    if mode == "packet_delete":
        keep = rng.random(len(lens)) >= strength
        if not np.any(keep):
            keep[int(rng.integers(0, len(lens)))] = True
        lens = [item for item, flag in zip(lens, keep.tolist()) if flag]
        dirs = [item for item, flag in zip(dirs, keep.tolist()) if flag]
        tss = [item for item, flag in zip(tss, keep.tolist()) if flag]
    elif mode == "packet_insert":
        insert_mask = rng.random(len(lens)) < strength
        if not np.any(insert_mask):
            insert_mask[int(rng.integers(0, len(lens)))] = True
        new_lens: list[int] = []
        new_dirs: list[Any] = []
        new_tss: list[float] = []
        iats = _positive_iats(tss)
        median_iat = float(np.median(iats)) if iats.size else 1e-3
        for idx, (length, direction, ts) in enumerate(zip(lens, dirs, tss)):
            new_lens.append(int(length))
            new_dirs.append(direction)
            new_tss.append(float(ts))
            if insert_mask[idx]:
                scale = float(rng.uniform(0.5, 1.5))
                new_lens.append(max(0, int(round(length * scale))))
                new_dirs.append(direction if rng.random() >= 0.25 else (not bool(direction)))
                new_tss.append(float(ts + median_iat * 0.5))
        order = np.argsort(np.array(new_tss, dtype=np.float64), kind="stable")
        lens = [new_lens[int(idx)] for idx in order]
        dirs = [new_dirs[int(idx)] for idx in order]
        tss = [new_tss[int(idx)] for idx in order]
    elif mode == "direction_flip":
        flip = rng.random(len(dirs)) < strength
        if not np.any(flip):
            flip[int(rng.integers(0, len(dirs)))] = True
        dirs = [not bool(direction) if flag else direction for direction, flag in zip(dirs, flip.tolist())]
    elif mode == "length_padding":
        max_pad = max(1, int(round(64 * strength)))
        pads = rng.integers(1, max_pad + 1, size=len(lens))
        lens = [int(length + pad) for length, pad in zip(lens, pads.tolist())]
    elif mode == "length_align":
        alignments = [64, 128, 256]
        idx = min(len(alignments) - 1, max(0, int(np.ceil(strength * len(alignments))) - 1))
        boundary = alignments[idx]
        lens = [int(((int(length) + boundary - 1) // boundary) * boundary) for length in lens]
    elif mode == "iat_jitter":
        if len(tss) > 1:
            base = tss[0]
            iats = np.diff(np.array(tss, dtype=np.float64))
            jitter = rng.normal(loc=0.0, scale=np.maximum(np.abs(iats) * strength, 1e-6))
            iats = np.maximum(iats + jitter, 0.0)
            tss = [float(base)]
            for delta in iats.tolist():
                tss.append(float(tss[-1] + delta))
    elif mode == "delay_stretch":
        if len(tss) > 1:
            base = tss[0]
            iats = _positive_iats(tss)
            stretch = 1.0 + max(0.0, strength) * rng.uniform(0.5, 1.5, size=len(iats))
            iats = iats * stretch
            tss = [float(base)]
            for delta in iats.tolist():
                tss.append(float(tss[-1] + delta))
    elif mode == "burst_split":
        if len(tss) > 1:
            base = tss[0]
            iats = _positive_iats(tss)
            threshold = float(np.quantile(iats, 0.25)) if len(iats) > 1 else float(iats[0])
            split = iats <= threshold
            if not np.any(split):
                split[int(rng.integers(0, len(iats)))] = True
            iats = iats.copy()
            iats[split] = iats[split] * (1.0 + 10.0 * max(strength, 0.01))
            tss = [float(base)]
            for delta in iats.tolist():
                tss.append(float(tss[-1] + delta))
    elif mode == "burst_merge":
        if len(tss) > 1:
            base = tss[0]
            iats = _positive_iats(tss)
            threshold = float(np.quantile(iats, 0.75)) if len(iats) > 1 else float(iats[0])
            merge = iats >= threshold
            if not np.any(merge):
                merge[int(rng.integers(0, len(iats)))] = True
            iats = iats.copy()
            iats[merge] = np.maximum(iats[merge] * max(0.0, 1.0 - strength), 1e-6)
            tss = [float(base)]
            for delta in iats.tolist():
                tss.append(float(tss[-1] + delta))
    elif mode == "low_rate_c2":
        binary_label = str(out.get("binary_label", "")).upper()
        raw_label = str(out.get("label", "")).lower()
        if binary_label == "ATTACK" or raw_label not in {"benign", "normal"}:
            keep_n = max(2, int(round(len(lens) * max(0.05, 1.0 - strength))))
            if keep_n < len(lens):
                keep_idx = np.linspace(0, len(lens) - 1, num=keep_n, dtype=int).tolist()
                lens = [int(lens[idx]) for idx in keep_idx]
                dirs = [dirs[idx] for idx in keep_idx]
                base = tss[keep_idx[0]] if tss else 0.0
                iats = _positive_iats(tss)
                median_iat = float(np.median(iats)) if iats.size else 1.0
                gap = max(median_iat, 1e-3) * (1.0 + 20.0 * max(strength, 0.01))
                tss = [float(base + idx * gap) for idx in range(len(lens))]
    else:
        raise ValueError(f"Unsupported perturbation mode: {mode}")

    return _refresh_flow_metadata(out, lens, dirs, tss)


def _perturb_short_flow_delete(flows: list[dict[str, Any]], strength: float, rng: np.random.Generator) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for flow in flows:
        pkt_n = len(flow.get("lens", []))
        if pkt_n < 6 and rng.random() < strength:
            continue
        out.append(copy.deepcopy(flow))
    return out or [copy.deepcopy(flows[int(rng.integers(0, len(flows)))])]


def _shift_flow_time(flow: dict[str, Any], new_start: float) -> dict[str, Any]:
    old_tss = [float(item) for item in flow.get("tss", [])]
    if not old_tss:
        return flow
    offset = new_start - old_tss[0]
    tss = [float(ts + offset) for ts in old_tss]
    return _refresh_flow_metadata(flow, list(flow.get("lens", [])), list(flow.get("dirs", [])), tss)


def _perturb_benign_flow_insert(flows: list[dict[str, Any]], strength: float, rng: np.random.Generator) -> list[dict[str, Any]]:
    out = [copy.deepcopy(flow) for flow in flows]
    benign = [
        flow
        for flow in flows
        if str(flow.get("binary_label", "")).upper() == "BENIGN" or str(flow.get("label", "")).lower() in {"benign", "normal"}
    ]
    if not benign:
        return out
    n_insert = max(1, int(round(len(flows) * strength)))
    anchors = [float(flow.get("start_ts") or 0.0) for flow in flows if flow.get("start_ts") is not None]
    min_ts = min(anchors) if anchors else 0.0
    max_ts = max(anchors) if anchors else float(n_insert)
    for idx in range(n_insert):
        src = copy.deepcopy(benign[int(rng.integers(0, len(benign)))])
        src["flow_id"] = f"{src.get('flow_id', 'flow')}_benign_insert_{idx:06d}"
        src["dataset_file"] = f"{src.get('dataset_file', '')}#benign_insert"
        src["label"] = "Benign"
        src["binary_label"] = "BENIGN"
        new_start = float(rng.uniform(min_ts, max_ts + 1e-6))
        out.append(_shift_flow_time(src, new_start))
    out.sort(key=lambda item: (float(item.get("start_ts") or 0.0), str(item.get("flow_id", ""))))
    return out


def perturb_flows(flows: list[dict[str, Any]], mode: str, strength: float, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if mode == "short_flow_delete":
        return _perturb_short_flow_delete(flows, strength, rng)
    if mode == "benign_flow_insert":
        return _perturb_benign_flow_insert(flows, strength, rng)
    return [perturb_flow(flow, mode, strength, rng) for flow in flows]
