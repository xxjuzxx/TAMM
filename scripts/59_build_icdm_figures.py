#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "middle", weight: str = "400", color: str = "#243042") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{_escape(text)}</text>'


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * max(0.0, min(1.0, t))))


def _color(value: float) -> str:
    # Low values: light gray-blue; high values: saturated teal. Kept print-friendly.
    r = _lerp(238, 25, value)
    g = _lerp(242, 132, value)
    b = _lerp(246, 136, value)
    return f"#{r:02x}{g:02x}{b:02x}"


def _box(x: float, y: float, w: float, h: float, title: str, lines: list[str], *, fill: str = "#ffffff", stroke: str = "#2563eb") -> list[str]:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
        _svg_text(x + w / 2, y + 23, title, size=13, weight="700", color="#111827"),
    ]
    for idx, line in enumerate(lines):
        parts.append(_svg_text(x + w / 2, y + 47 + idx * 17, line, size=10, color="#475467"))
    return parts


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" '
        'stroke="#334155" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>'
    )


def build_framework_overview(out_dir: Path) -> Path:
    width = 1040
    height = 430
    top = 78
    box_w = 148
    box_h = 112
    gap = 18
    x0 = 26
    xs = [x0 + i * (box_w + gap) for i in range(6)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 28, "FlowPrim Behavior-Only Framework", size=19, weight="700"),
        _svg_text(width / 2, 51, "Traffic behavior primitive mining for low-FPR malicious traffic diagnosis", size=12, color="#5b6678"),
    ]
    boxes = [
        ("Traffic sources", ["PCAP / Zeek", "packet CSV", "flow CSV fallback"], "#eff6ff", "#2563eb"),
        ("Flow construction", ["bidirectional flows", "label alignment", "audit metadata"], "#ecfeff", "#0891b2"),
        ("Primitive mining", ["profile primitives", "structural motifs", "trigger provenance"], "#f0fdf4", "#16a34a"),
        ("Behavior tokens", ["global stats", "packet/burst tokens", "primitive tokens"], "#fff7ed", "#ea580c"),
        ("Behavior composer", ["Transformer", "supervised heads", "optional embedding"], "#f5f3ff", "#7c3aed"),
        ("Diagnosis heads", ["binary detection", "KNN benign memory", "P99 threshold"], "#fdf2f8", "#db2777"),
    ]
    for x, (title, lines, fill, stroke) in zip(xs, boxes):
        parts.extend(_box(x, top, box_w, box_h, title, lines, fill=fill, stroke=stroke))
    for i in range(5):
        parts.append(_arrow(xs[i] + box_w + 2, top + box_h / 2, xs[i + 1] - 5, top + box_h / 2))

    y2 = 242
    parts.extend(
        _box(
            86,
            y2,
            238,
            86,
            "Shortcut control",
            ["No IP/time/5-tuple", "protocol/service tokens", "metadata only"],
            fill="#f8fafc",
            stroke="#64748b",
        )
    )
    parts.extend(
        _box(
            370,
            y2,
            238,
            86,
            "Fit-only-on-training rule",
            ["train-only vocab", "benign memory", "validation threshold"],
            fill="#f8fafc",
            stroke="#64748b",
        )
    )
    parts.extend(
        _box(
            654,
            y2,
            238,
            86,
            "Analyst-facing record",
            ["score + threshold", "primitives + tokens", "nearest evidence"],
            fill="#f8fafc",
            stroke="#64748b",
        )
    )
    parts.append(_arrow(xs[1] + box_w / 2, top + box_h + 8, 205, y2 - 5))
    parts.append(_arrow(xs[3] + box_w / 2, top + box_h + 8, 489, y2 - 5))
    parts.append(_arrow(xs[5] + box_w / 2, top + box_h + 8, 773, y2 - 5))
    parts.append(_svg_text(width / 2, 374, "Reported low-FPR diagnosis uses behavior evidence: direction, length, timing, burst structure, and primitive triggers.", size=12, color="#475467"))
    parts.append(_svg_text(width / 2, 395, "Protocol, service, host identifiers, absolute time, and five-tuples are retained only for compatibility and audit metadata.", size=12, color="#475467"))
    parts.append("</svg>")
    out = out_dir / "fig_flowprim_framework.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def build_primitive_token_example(out_dir: Path) -> Path:
    width = 620
    height = 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 27, "Primitive Mining and Tokenization Example", size=18, weight="700"),
        _svg_text(width / 2, 50, "Illustrative flow: packet evidence is converted into composable behavior tokens", size=13, color="#5b6678"),
    ]
    x0 = 62
    y0 = 84
    pkt_w = 64
    pkt_h = 46
    packets = [
        ("C->S", "60B", "0ms", "#dbeafe"),
        ("C->S", "60B", "2ms", "#dbeafe"),
        ("C->S", "60B", "3ms", "#dbeafe"),
        ("S->C", "66B", "5ms", "#ecfdf5"),
        ("S->C", "66B", "6ms", "#ecfdf5"),
        ("S->C", "66B", "8ms", "#ecfdf5"),
    ]
    parts.append(_svg_text(24, y0 + 26, "Packets", size=13, anchor="start", weight="700"))
    for i, (direction, length, iat, fill) in enumerate(packets):
        x = x0 + i * (pkt_w + 10)
        parts.append(f'<rect x="{x}" y="{y0}" width="{pkt_w}" height="{pkt_h}" rx="6" fill="{fill}" stroke="#2563eb" stroke-width="1.1"/>')
        parts.append(_svg_text(x + pkt_w / 2, y0 + 17, direction, size=11, weight="700"))
        parts.append(_svg_text(x + pkt_w / 2, y0 + 32, f"{length}, {iat}", size=10, color="#475467"))
    parts.append(
        '<path d="M72,151 C150,174 224,174 302,151" stroke="#16a34a" stroke-width="2" fill="none"/>'
    )
    parts.append(
        '<path d="M406,151 C456,174 506,174 556,151" stroke="#16a34a" stroke-width="2" fill="none"/>'
    )
    parts.append(_svg_text(188, 188, "REPEAT: same-direction equal-length run", size=12, weight="700", color="#166534"))
    parts.append(_svg_text(480, 188, "DUP: adjacent repeated segment", size=12, weight="700", color="#166534"))

    tag_y = 220
    tags = [
        ("Global", ["PKTN=6", "DUR_BIN=short", "DIR_RATIO=1.0"], "#eff6ff", "#2563eb"),
        ("Primitive", ["PRIM_PROFILE_REPEAT", "PRIM_PROFILE_DUP", "PRIM_STRUCT_REQ_RESP"], "#f0fdf4", "#16a34a"),
        ("Packet/Burst", ["D=C2S", "LEN=60", "IAT=small", "BURST=0"], "#fff7ed", "#ea580c"),
    ]
    for i, (title, lines, fill, stroke) in enumerate(tags):
        x = 34 + i * 146
        parts.extend(_box(x, tag_y, 130, 98, title, lines, fill=fill, stroke=stroke))
    parts.append(_arrow(300, 202, 300, tag_y - 9).replace("url(#arrow)", "url(#arrow2)"))
    parts.append(_svg_text(width / 2, 350, "x_f = [CLS, global, primitive, packet/burst tokens]", size=14, color="#111827"))
    parts.append(_svg_text(width / 2, 374, "Audit metadata such as protocol, service, IPs, absolute time, and five-tuples is not emitted as behavior tokens.", size=12, color="#475467"))
    parts.append("</svg>")
    out = out_dir / "fig_primitive_token_example.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def _find_calibration_row() -> dict[str, str]:
    rows = _read_csv(Path("paper_icdm_applied_2026/experiments/unknown/botnet_low_fpr_sweep.csv"))
    for row in rows:
        if (
            row.get("feature_filter") == "packet_burst"
            and row.get("transform") == "binary_l2"
            and row.get("scorer") == "knn_euclidean"
            and row.get("k") == "3"
            and row.get("group_mode") == "protocol"
        ):
            return row
    raise FileNotFoundError("Selected Botnet calibration row not found")


def build_low_fpr_calibration(out_dir: Path) -> Path:
    row = _find_calibration_row()
    width = 620
    height = 330
    left = 84
    right = 562
    top = 84
    bottom = 246
    keys = [
        "benign_score_min",
        "benign_score_p50",
        "benign_score_p90",
        "benign_score_p95",
        "benign_score_p99",
        "benign_score_max",
        "attack_score_min",
        "attack_score_p50",
        "attack_score_p90",
        "attack_score_p95",
        "attack_score_p99",
        "attack_score_max",
        "val_p99_0_threshold",
        "threshold_at_1pct_fpr",
        "threshold_at_0_1pct_fpr",
    ]
    values = [_num(row.get(key)) for key in keys]
    lo = min(values)
    hi = max(values)
    pad = (hi - lo) * 0.08 if hi > lo else 0.1
    lo -= pad
    hi += pad

    def sx(value: float) -> float:
        return left + (value - lo) / (hi - lo) * (right - left)

    def quantile_band(prefix: str, y: float, color: str, label: str) -> list[str]:
        qmin = _num(row[f"{prefix}_score_min"])
        q50 = _num(row[f"{prefix}_score_p50"])
        q90 = _num(row[f"{prefix}_score_p90"])
        q95 = _num(row[f"{prefix}_score_p95"])
        q99 = _num(row[f"{prefix}_score_p99"])
        qmax = _num(row[f"{prefix}_score_max"])
        return [
            f'<line x1="{sx(qmin):.1f}" y1="{y:.1f}" x2="{sx(qmax):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="5" stroke-linecap="round" opacity="0.35"/>',
            f'<rect x="{sx(q90):.1f}" y="{y - 16:.1f}" width="{max(2, sx(q99) - sx(q90)):.1f}" height="32" rx="4" fill="{color}" opacity="0.30"/>',
            f'<line x1="{sx(q50):.1f}" y1="{y - 22:.1f}" x2="{sx(q50):.1f}" y2="{y + 22:.1f}" stroke="{color}" stroke-width="3"/>',
            _svg_text(left - 12, y + 5, label, size=14, anchor="end", weight="700"),
            _svg_text(sx(q50), y - 29, f"p50={q50:.2f}", size=12, weight="700", color=color),
            _svg_text(sx(q99), y + 37, f"p99={q99:.2f}", size=12, weight="700", color=color),
        ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 27, "Low-FPR Calibration View", size=18, weight="700"),
        _svg_text(width / 2, 50, "Representative Botnet leave-one memory setting, seed 42 score quantiles", size=13, color="#5b6678"),
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#334155" stroke-width="1.2"/>',
    ]
    for tick in [lo, (lo + hi) / 2, hi]:
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 5}" stroke="#334155" stroke-width="1"/>')
        parts.append(_svg_text(x, bottom + 22, f"{tick:.2f}", size=12, color="#475467"))
    parts.extend(quantile_band("benign", top + 46, "#2563eb", "Benign"))
    parts.extend(quantile_band("attack", top + 124, "#dc2626", "Attack"))
    thresholds = [
        ("P99 val", _num(row["val_p99_0_threshold"]), "#7c3aed", 0),
        ("1%FPR", _num(row["threshold_at_1pct_fpr"]), "#0f766e", 1),
        ("0.1%FPR", _num(row["threshold_at_0_1pct_fpr"]), "#b45309", 2),
    ]
    for label, value, color, offset in thresholds:
        x = sx(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top + 12}" x2="{x:.1f}" y2="{bottom}" stroke="{color}" stroke-width="1.8" stroke-dasharray="5 4"/>')
        parts.append(_svg_text(x + 3, top + 16 + offset * 16, f"{label}={value:.2f}", size=12, anchor="start", weight="700", color=color))
    parts.append(_svg_text(width / 2, 304, "High global ranking can still leave an unstable extreme benign tail, motivating validation calibration.", size=12, color="#475467"))
    parts.append("</svg>")
    out = out_dir / "fig_low_fpr_calibration.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def build_primitive_heatmap(out_dir: Path) -> Path:
    src = Path("experiments/behavior_profile_ccfa_temporal_posthoc_20260525.csv")
    rows = [
        row
        for row in _read_csv(src)
        if row.get("domain") == "CICIDS2017_temporal_interim_posthoc"
        and row.get("group_field") == "attack_family"
        and row.get("group_value") not in {"DoS", "Infiltration"}
    ]
    primitives = [
        ("short", "short_trigger_rate"),
        ("same", "same_trigger_rate"),
        ("repeat", "repeat_trigger_rate"),
        ("duplicate", "duplicate_trigger_rate"),
    ]
    labels = [str(row["group_value"]) for row in rows]
    cell_w = 78
    cell_h = 38
    left = 118
    top = 82
    width = left + cell_w * len(primitives) + 28
    height = top + cell_h * len(rows) + 54
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 27, "Class x Primitive Trigger Rate", size=18, weight="700"),
        _svg_text(width / 2, 51, "CICIDS2017 temporal post-hoc profile", size=13, color="#5b6678"),
    ]
    for j, (label, _key) in enumerate(primitives):
        parts.append(_svg_text(left + j * cell_w + cell_w / 2, top - 15, label, size=14, weight="700"))
    for i, row in enumerate(rows):
        y = top + i * cell_h
        parts.append(_svg_text(left - 9, y + 25, labels[i], size=13, anchor="end", weight="700"))
        for j, (_label, key) in enumerate(primitives):
            value = _num(row.get(key))
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" rx="2" fill="{_color(value)}"/>')
            parts.append(_svg_text(x + cell_w / 2, y + 25, f"{value:.2f}", size=15, color="#0f172a" if value < 0.55 else "#ffffff", weight="700"))
    parts.append("</svg>")
    out = out_dir / "fig_class_primitive_heatmap.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def build_token_ablation_bar(out_dir: Path) -> Path:
    rows = _read_csv(Path("paper_icdm_applied_2026/experiments/tables/table_token_ablation.csv"))
    width = 520
    height = 330
    left = 62
    bottom = 260
    plot_h = 178
    bar_w = 60
    gap = 38
    min_y = 0.96
    max_y = 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 27, "Token Composition Ablation", size=18, weight="700"),
        _svg_text(width / 2, 50, "3-seed Zeek 6K temporal split, Macro-F1", size=13, color="#5b6678"),
        f'<line x1="{left}" y1="{bottom}" x2="{width - 50}" y2="{bottom}" stroke="#344054" stroke-width="1"/>',
        f'<line x1="{left}" y1="{bottom - plot_h}" x2="{left}" y2="{bottom}" stroke="#344054" stroke-width="1"/>',
    ]
    for tick in [0.96, 0.97, 0.98, 0.99, 1.0]:
        y = bottom - ((tick - min_y) / (max_y - min_y)) * plot_h
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{width - 50}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(_svg_text(left - 9, y + 5, f"{tick:.2f}", size=13, anchor="end", color="#5b6678"))
    palette = ["#3b82f6", "#14b8a6", "#10b981", "#64748b"]
    label_map = {
        "packet_only": "packet",
        "packet_burst": "+burst",
        "packet_burst_profile": "+profile",
        "full_rhythm": "full",
    }
    for i, row in enumerate(rows):
        value = _num(row.get("macro_f1"))
        std = _num(row.get("macro_f1_std"))
        x = left + 24 + i * (bar_w + gap)
        h = ((value - min_y) / (max_y - min_y)) * plot_h
        y = bottom - h
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="3" fill="{palette[i % len(palette)]}"/>')
        err = (std / (max_y - min_y)) * plot_h
        cx = x + bar_w / 2
        parts.append(f'<line x1="{cx:.1f}" y1="{max(bottom - h - err, bottom - plot_h):.1f}" x2="{cx:.1f}" y2="{min(bottom - h + err, bottom):.1f}" stroke="#111827" stroke-width="1.2"/>')
        label = label_map.get(str(row.get("variant", "")), str(row.get("variant", "")).replace("_", " "))
        parts.append(_svg_text(cx, bottom + 22, label, size=14))
        parts.append(_svg_text(cx, y - 9, f"{value:.4f}", size=15, weight="700"))
    parts.append("</svg>")
    out = out_dir / "fig_token_ablation_macro_f1.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def build_unknown_bar(out_dir: Path) -> Path:
    src_3seed = Path("paper_icdm_applied_2026/experiments/tables/table_unknown_low_fpr_3seed.csv")
    if src_3seed.exists():
        rows = _read_csv(src_3seed)
        r01_key = "recall_at_0_1pct_fpr_mean"
        r1_key = "recall_at_1pct_fpr_mean"
        title_suffix = "3-seed mean"
    else:
        rows = _read_csv(Path("paper_icdm_applied_2026/experiments/tables/table_unknown_low_fpr.csv"))
        r01_key = "recall_at_0_1pct_fpr"
        r1_key = "recall_at_1pct_fpr"
        title_suffix = "seed-42 full sweep"
    width = 560
    height = 360
    left = 54
    bottom = 268
    plot_h = 185
    bar_w = 26
    group_gap = 39
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 27, "Leave-One Unknown Attack Recall", size=18, weight="700"),
        _svg_text(width / 2, 50, f"Low-FPR test-oracle setting ({title_suffix})", size=13, color="#5b6678"),
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom - tick * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 50}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(_svg_text(left - 8, y + 5, f"{tick:.2f}", size=13, anchor="end", color="#5b6678"))
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{width - 50}" y2="{bottom}" stroke="#344054" stroke-width="1"/>')
    palette = {"r01": "#ef4444", "r1": "#0ea5e9"}
    for i, row in enumerate(rows):
        x0 = left + 28 + i * (bar_w * 2 + group_gap)
        values = [("r01", _num(row.get(r01_key))), ("r1", _num(row.get(r1_key)))]
        for j, (name, value) in enumerate(values):
            h = value * plot_h
            x = x0 + j * (bar_w + 4)
            parts.append(f'<rect x="{x}" y="{bottom - h:.1f}" width="{bar_w}" height="{h:.1f}" rx="3" fill="{palette[name]}"/>')
            parts.append(_svg_text(x + bar_w / 2, bottom - h - 8, f"{value:.2f}", size=15, weight="700"))
        parts.append(_svg_text(x0 + bar_w + 2, bottom + 22, str(row.get("unknown_attack")), size=13))
    legend_y = 318
    legend_x0 = 176
    legend_x1 = 310
    parts.append(f'<rect x="{legend_x0}" y="{legend_y}" width="14" height="14" fill="{palette["r01"]}"/>')
    parts.append(_svg_text(legend_x0 + 20, legend_y + 12, "R@0.1%FPR", size=13, anchor="start"))
    parts.append(f'<rect x="{legend_x1}" y="{legend_y}" width="14" height="14" fill="{palette["r1"]}"/>')
    parts.append(_svg_text(legend_x1 + 20, legend_y + 12, "R@1%FPR", size=13, anchor="start"))
    parts.append("</svg>")
    out = out_dir / "fig_unknown_low_fpr_recall.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def build_robustness_bar(out_dir: Path) -> Path:
    rows = _read_csv(Path("paper_icdm_applied_2026/experiments/tables/table_robustness.csv"))
    width = 620
    height = 330
    left = 62
    bottom = 258
    plot_h = 178
    bar_w = 40
    gap = 27
    max_drop = max(_num(row.get("drop_from_clean")) for row in rows) or 0.1
    max_drop = max(0.08, max_drop * 1.2)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(width / 2, 27, "Robustness Drop From Clean", size=18, weight="700"),
        _svg_text(width / 2, 50, "3-seed Macro-F1 drop under perturbations", size=13, color="#5b6678"),
    ]
    for tick in [0.0, max_drop / 2, max_drop]:
        y = bottom - (tick / max_drop) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 50}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(_svg_text(left - 9, y + 5, f"{tick:.2f}", size=13, anchor="end", color="#5b6678"))
    label_map = {
        "clean": "clean",
        "packet_delete_010": "delete",
        "packet_insert_010": "insert",
        "direction_flip_010": "flip",
        "length_align_050": "align",
        "length_padding_050": "pad",
        "low_rate_c2_070": "low-rate",
    }
    for i, row in enumerate(rows):
        value = _num(row.get("drop_from_clean"))
        x = left + 24 + i * (bar_w + gap)
        h = (value / max_drop) * plot_h
        parts.append(f'<rect x="{x}" y="{bottom - h:.1f}" width="{bar_w}" height="{h:.1f}" rx="3" fill="#f59e0b"/>')
        label = label_map.get(str(row.get("condition", "")), str(row.get("condition", "")).replace("_", " "))
        parts.append(_svg_text(x + bar_w / 2, bottom + 22, label, size=13))
        parts.append(_svg_text(x + bar_w / 2, bottom - h - 8, f"{value:.3f}", size=15, weight="700"))
    parts.append("</svg>")
    out = out_dir / "fig_robustness_macro_f1_drop.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SVG figures for the ICDM Applied Track package.")
    parser.add_argument("--out_dir", default="paper_icdm_applied_2026/figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_framework_overview(out_dir),
        build_primitive_token_example(out_dir),
        build_primitive_heatmap(out_dir),
        build_low_fpr_calibration(out_dir),
        build_token_ablation_bar(out_dir),
        build_unknown_bar(out_dir),
        build_robustness_bar(out_dir),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
