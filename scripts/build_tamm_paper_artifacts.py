#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import subprocess
from pathlib import Path
from typing import Iterable

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper" / "flowprim_motif_memory_icdm"
TABLE_DIR = PAPER / "tables"
FIG_DIR = PAPER / "figures"


def _fmt(value: object, digits: int = 4) -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "--"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value).replace("_", r"\_")


def _latex_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [r"\begin{tabular}{" + "l" * len(headers) + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _write_pdf_from_svg(svg_path: Path) -> None:
    pdf_path = svg_path.with_suffix(".pdf")
    try:
        subprocess.run(["cairosvg", str(svg_path), "-o", str(pdf_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        try:
            import cairosvg

            cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        except Exception as exc:
            print(f"warning: could not convert {svg_path} to PDF: {exc}")


def _svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "middle", weight: str = "400") -> str:
    parts = html.escape(str(text)).split("\n")
    out = [f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" font-family="Arial, Helvetica, sans-serif" fill="#23313d">']
    for i, part in enumerate(parts):
        dy = 0 if i == 0 else size * 1.15
        out.append(f'<tspan x="{x}" dy="{dy}">{part}</tspan>')
    out.append("</text>")
    return "".join(out)


def _write_svg(name: str, width: int, height: int, body: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>
'''
    path = FIG_DIR / f"{name}.svg"
    path.write_text(svg, encoding="utf-8")
    _write_pdf_from_svg(path)


def _rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="#34424f" stroke-width="1.2"/>'


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'''<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L7,3 z" fill="#34424f"/></marker></defs>
<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#34424f" stroke-width="1.3" marker-end="url(#arrow)"/>'''


def _line(x1: float, y1: float, x2: float, y2: float, color: str = "#34424f", width: float = 1.2, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _short_label(text: str, max_len: int = 18) -> str:
    s = str(text).replace("_", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "."


def build_memory_governance_table(results_dir: Path) -> None:
    path = results_dir / "memory_optimization" / "summary_by_setting.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    runtime_path = results_dir / "runtime_unified" / "memory_governance_runtime_unified_summary.csv"
    runtime_df = pd.read_csv(runtime_path) if runtime_path.exists() else pd.DataFrame()
    runtime_by_strategy = {}
    if not runtime_df.empty:
        runtime_by_strategy = {str(r["strategy"]): r for _, r in runtime_df.iterrows()}
    wanted = [
        ("baseline", "uniform_exact", "uniform_exact", "Exact full memory"),
        ("indexed_retrieval", "coreset_tail_preserving_0.5", "coreset_tail_preserving_0.5", "Tail coreset 50\\%"),
        ("indexed_retrieval", "coreset_random_0.5", "coreset_random_0.5", "Random coreset 50\\%"),
        ("token_weighting", "tfidf_train_only", "tfidf_train_only", "Train-only TF-IDF"),
        ("calibration", "evt_tail_p99", "evt_tail_p99", "EVT tail P99"),
        ("memory_governance", "tail_aware_update", "tail_aware_update", "Tail-aware update"),
        ("memory_governance", "random_benign_update", "random_benign_update", "Random benign update"),
        ("memory_governance", "oracle_pollution_5pct_attack_diagnostic", "oracle_pollution_5pct_attack_diagnostic", "5\\% attack pollution"),
    ]
    rows = []
    for group, setting, runtime_key, label in wanted:
        sub = df[(df["experiment_group"] == group) & (df["setting"] == setting)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        runtime = runtime_by_strategy.get(runtime_key)
        ms_value = runtime.get("batch_ms_per_flow_mean_mean") if runtime is not None else r.get("query_ms_per_flow_mean")
        rows.append(
            [
                label,
                _fmt(r.get("auroc_mean")),
                _fmt(r.get("recall_at_0_1pct_fpr_mean")),
                _fmt(r.get("recall_at_1pct_fpr_mean")),
                _fmt(r.get("p99_realized_fpr_mean")),
                _fmt(r.get("false_alerts_per_10k_benign_mean"), 1),
                _fmt(ms_value, 3),
                _fmt(r.get("memory_size_mean"), 0),
            ]
        )
    ms_header = "Batch ms/flow" if runtime_by_strategy else "ms/flow"
    (TABLE_DIR / "table_memory_governance_optimization.tex").write_text(
        _latex_table(["Setting", "AUROC", "R@0.1\\%", "R@1\\%", "P99 FPR", "Alerts/10k", ms_header, "Memory"], rows),
        encoding="utf-8",
    )


def build_dm_motif_table(results_dir: Path) -> None:
    path = results_dir / "motif_selection_dm" / "motif_selection_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.sort_values(["feature_view"]).iterrows():
        rows.append(
            [
                str(r["feature_view"]).replace("_", r"\_"),
                _fmt(r.get("auroc")),
                _fmt(r.get("recall_at_0_1pct_fpr")),
                _fmt(r.get("recall_at_1pct_fpr")),
                _fmt(r.get("val_p99_realized_fpr")),
                _fmt(r.get("false_alerts_per_10k_benign"), 1),
                _fmt(r.get("selected_motif_count"), 1),
                _fmt(r.get("vocab_size"), 1),
            ]
        )
    (TABLE_DIR / "table_dm_motif_selection.tex").write_text(
        _latex_table(["View", "AUROC", "R@0.1\\%", "R@1\\%", "P99 FPR", "Alerts/10k", "Motifs", "Vocab"], rows),
        encoding="utf-8",
    )


def build_candidate_source_table(results_dir: Path) -> None:
    path = results_dir / "motif_source_comparison" / "motif_selection_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    label = {
        "expert_only": "Expert only",
        "data_mined_only": "Train-mined only",
        "expert_plus_data_mined": "Expert + train-mined",
    }
    view = {
        "selected_motifs_only": "Motifs only",
        "packet_burst_selected_motifs": "Packet/burst + motifs",
    }
    rows = []
    for _, r in df.sort_values(["candidate_source_mode", "feature_view"]).iterrows():
        rows.append(
            [
                label.get(str(r["candidate_source_mode"]), str(r["candidate_source_mode"]).replace("_", " ")),
                view.get(str(r["feature_view"]), str(r["feature_view"]).replace("_", " ")),
                _fmt(r.get("auroc")),
                _fmt(r.get("recall_at_0_1pct_fpr")),
                _fmt(r.get("recall_at_1pct_fpr")),
                _fmt(r.get("val_p99_realized_fpr")),
                _fmt(r.get("false_alerts_per_10k_benign"), 1),
                _fmt(r.get("selected_motif_count"), 1),
            ]
        )
    (TABLE_DIR / "table_motif_candidate_source_comparison.tex").write_text(
        _latex_table(["Candidate source", "View", "AUROC", "R@0.1\\%", "R@1\\%", "P99 FPR", "Alerts/10k", "Motifs"], rows),
        encoding="utf-8",
    )


def build_selection_stage_table(results_dir: Path) -> None:
    root = results_dir / "motif_selection" / "dictionaries"
    if not root.exists():
        return
    rows_raw = []
    for report in root.glob("*/motif_selection_report.csv"):
        if "_full_utility_" not in str(report.parent.name):
            continue
        try:
            df = pd.read_csv(report)
        except Exception:
            continue
        if "selection_strategy" in df and not (df["selection_strategy"] == "full_utility").any():
            continue
        total = len(df)
        after_support = int((~df["filtered_by_support"].astype(bool)).sum()) if "filtered_by_support" in df else total
        selected = int(df["selected"].astype(bool).sum()) if "selected" in df else 0
        rows_raw.append((total, after_support, after_support, after_support, selected))
    if not rows_raw:
        return
    arr = np.asarray(rows_raw, dtype=float)
    means = arr.mean(axis=0)
    rows = [
        ["Initial candidates", _fmt(means[0], 1), "Expert/profile/structural candidate tokens observed in train-fitted artifacts"],
        ["After support filter", _fmt(means[1], 1), "Train-only min/max support removes rare or ubiquitous motifs"],
        ["After stability/tail filters", _fmt(means[2], 1), "Permissive in current artifact; candidate set is mostly unchanged"],
        ["Before redundancy pruning", _fmt(means[3], 1), "Utility-ranked valid candidates"],
        ["Final selected", _fmt(means[4], 1), "Full-utility greedy selection with redundancy pruning"],
    ]
    (TABLE_DIR / "table_motif_selection_stage_counts.tex").write_text(
        _latex_table(["Stage", "Mean motifs", "Interpretation"], rows),
        encoding="utf-8",
    )


def build_explanation_table(results_dir: Path) -> None:
    path = results_dir / "explanation_contrast" / "explanation_contrast_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.sort_values(["feature_view"]).iterrows():
        rows.append(
            [
                str(r["feature_view"]).replace("_", r"\_"),
                _fmt(r.get("runs"), 0),
                _fmt(r.get("alerts"), 1),
                _fmt(r.get("contrast_coverage")),
                _fmt(r.get("mean_alert_only_motif_count"), 2),
                _fmt(r.get("mean_score_drop_top3"), 4),
            ]
        )
    (TABLE_DIR / "table_explanation_contrast.tex").write_text(
        _latex_table(["View", "Runs", "Alerts", "Coverage", "Alert-only motifs", "$\\Delta$score top-3"], rows),
        encoding="utf-8",
    )


def draw_framework() -> None:
    xs = [20, 160, 300, 440, 580]
    labels = ["Traffic evidence\nPCAP / Zeek / CSV", "Canonical\nbehavior sequence", "Expert-guided\ncandidate motifs", "Train-only\nmotif dictionary", "Benign motif-memory\nscore + diagnosis"]
    fills = ["#d9e8f5", "#e8f1df", "#f8e5c7", "#eadff1", "#dfeee9"]
    body = []
    for x, lab, fill in zip(xs, labels, fills):
        body.append(_rect(x, 64, 112, 60, fill))
        body.append(_svg_text(x + 56, 87, lab, size=11))
    for x in [132, 272, 412, 552]:
        body.append(_arrow(x + 6, 94, x + 24, 94))
    body.append(_svg_text(350, 165, "Behavior-only path: no raw IP, absolute time, five-tuple, protocol, or service as tokens or memory keys", size=11))
    _write_svg("fig_tamm_framework", 720, 190, "\n".join(body))


def draw_algorithm() -> None:
    steps = [
        "Generate interpretable candidates",
        "Fit train-only support and stability",
        "Estimate benign-tail sensitivity",
        "Score utility; penalize redundancy",
        "Freeze dictionary; emit transactions",
        "Calibrate benign-only threshold",
    ]
    body = []
    for i, step in enumerate(steps):
        y = 22 + i * 42
        body.append(_rect(28, y, 300, 28, "#f3f5f7"))
        body.append(_svg_text(178, y + 18, f"{i+1}. {step}", size=11))
        if i < len(steps) - 1:
            body.append(_arrow(178, y + 30, 178, y + 39))
    _write_svg("fig_behavior_primitive_algorithm", 360, 286, "\n".join(body))


def draw_token_example() -> None:
    xs = [48, 92, 136, 180, 224, 268, 312]
    dirs = ["S", "C", "S", "C", "C", "S", "S"]
    lens = ["L6", "L6", "L6", "L6", "L8", "L6", "L6"]
    body = []
    for x, d, l in zip(xs, dirs, lens):
        body.append(f'<circle cx="{x}" cy="54" r="14" fill="#d9e8f5" stroke="#34424f"/>')
        body.append(_svg_text(x, 59, d, size=11, weight="700"))
        body.append(_svg_text(x, 88, l, size=9))
    for x, label in [(92, "DM_SEQ_DIRLEN"), (188, "PRIM_STRUCT_REPEAT"), (272, "DM_TRANS")]:
        body.append(_rect(x - 55, 118, 110, 26, "#eadff1"))
        body.append(_svg_text(x, 135, label, size=8))
    body.append(_svg_text(180, 178, "Motif transaction = packet/burst items + selected motif items + evidence spans", size=10))
    _write_svg("fig_primitive_token_example", 360, 205, "\n".join(body))


def draw_bar_chart(name: str, labels: list[str], series: list[tuple[str, list[float], str]], *, ylabel: str = "", ymax: float | None = None) -> None:
    width, height = 360, 250
    left, right, top, bottom = 46, 16, 18, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    max_val = ymax or max(max(vals) for _, vals, _ in series) * 1.1
    body = [f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#34424f"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#34424f"/>']
    n = len(labels)
    group_w = plot_w / max(n, 1)
    bar_w = group_w / (len(series) + 1)
    for si, (sname, vals, color) in enumerate(series):
        for i, val in enumerate(vals):
            h = plot_h * float(val) / max_val
            x = left + i * group_w + (si + 0.4) * bar_w
            y = top + plot_h - h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{h:.1f}" fill="{color}"/>')
    for i, lab in enumerate(labels):
        x = left + i * group_w + group_w * 0.45
        body.append(_svg_text(x, height - 36, lab.replace("_", "\n"), size=8))
    body.append(_svg_text(13, top + plot_h / 2, ylabel, size=9, anchor="middle"))
    lx = left + 6
    for j, (sname, _, color) in enumerate(series):
        body.append(f'<rect x="{lx + j*96}" y="8" width="10" height="10" fill="{color}"/>')
        body.append(_svg_text(lx + 16 + j*96, 17, sname, size=8, anchor="start"))
    _write_svg(name, width, height, "\n".join(body))


def draw_applied_workflow() -> None:
    body: list[str] = []
    lanes = [
        ("Traffic\nartifacts", "PCAP / Zeek\npacket CSV\nflow summary*", "#d9e8f5"),
        ("Behavioral\nevents", "direction\nlength bin\nIAT bin\nburst\nlocal repeat", "#e8f1df"),
        ("Candidate\nmotifs", "expert-guided\ntrain-mined\nsequential", "#f8e5c7"),
        ("View-specific\nvocabulary", "fixed behavior-only\nutility-selected\ncompact view", "#eadff1"),
        ("View-specific\ntransaction\nvector", "packet/burst\n+ motif items", "#f7e4dc"),
        ("Benign\nmemory", "KNN deviation\nnearest benign\nprovenance", "#dfeee9"),
        ("Benign-only\ncalibration", "P99 threshold\nEVT / empirical\np-value diagnostics\nalert budget", "#e9edf5"),
        ("Diagnosis\nrecord", "score threshold\ndecision motifs\nnearest evidence", "#e0f0ef"),
    ]
    x0, y0, w, h, gap = 18, 42, 82, 88, 7
    for i, (title, detail, fill) in enumerate(lanes):
        x = x0 + i * (w + gap)
        body.append(_rect(x, y0, w, h, fill))
        body.append(_svg_text(x + w / 2, y0 + 21, title, size=10, weight="700"))
        body.append(_svg_text(x + w / 2, y0 + 49, detail, size=8))
        if i < len(lanes) - 1:
            body.append(_arrow(x + w + 1, y0 + h / 2, x + w + gap - 2, y0 + h / 2))
    body.append(_rect(34, 150, 324, 38, "#fff4d8"))
    body.append(_svg_text(196, 165, "No payload, raw IP, absolute time, five-tuple,", size=9, weight="700"))
    body.append(_svg_text(196, 179, "protocol, or service as behavior tokens or memory keys", size=9))
    body.append(_svg_text(72, 202, "* flow summaries are domain-shift diagnostics, not full motif evidence", size=8, anchor="start"))
    body.append(_rect(386, 154, 146, 30, "#e8f1df"))
    body.append(_svg_text(459, 173, "Fixed behavior-only:\nmain diagnosis result", size=9, weight="700"))
    body.append(_rect(552, 154, 138, 30, "#eadff1"))
    body.append(_svg_text(621, 173, "Utility-selected:\ncompact audit view", size=9))
    _write_svg("fig1_tamm_applied_workflow", 740, 218, "\n".join(body))


def draw_data_capability() -> None:
    body: list[str] = []
    body.append(_rect(22, 18, 316, 48, "#d9e8f5"))
    body.append(_svg_text(180, 38, "Full behavior evidence", size=12, weight="700"))
    body.append(_svg_text(180, 56, "CICIDS2017 PCAP/Zeek  |  extra benign PCAP  |  Botnet2014 PCAP/Zeek", size=8))
    body.append(_rect(22, 84, 316, 66, "#e8f1df"))
    body.append(_svg_text(180, 105, "Supports packet-order motif mining", size=11, weight="700"))
    body.append(_svg_text(180, 124, "direction, length bin, IAT, burst span,\ntransition, repetition, structural motifs", size=9))
    body.append(_rect(22, 174, 316, 48, "#f8e5c7"))
    body.append(_svg_text(180, 194, "Tabular domain-shift diagnostics", size=12, weight="700"))
    body.append(_svg_text(180, 212, "IDS2018 flow summary  |  UNSW-NB15 flow summary", size=8))
    body.append(_rect(22, 240, 316, 50, "#fff4d8"))
    body.append(_svg_text(180, 260, "Leakage control", size=11, weight="700"))
    body.append(_svg_text(180, 278, "Payload is unused; join fields, labels, protocol/service,\nhosts, and time are audit metadata only", size=8))
    body.append(_arrow(180, 66, 180, 84))
    body.append(_arrow(180, 150, 180, 174))
    body.append(_svg_text(180, 318, "Full packet/burst validation is separated from tabular diagnostics.", size=9, weight="700"))
    _write_svg("fig2_data_capability_leakage", 360, 334, "\n".join(body))


def draw_candidate_to_dictionary() -> None:
    body: list[str] = []
    body.append(_rect(22, 22, 138, 58, "#f8e5c7"))
    body.append(_svg_text(91, 44, "Expert-guided", size=11, weight="700"))
    body.append(_svg_text(91, 63, "PROFILE / STRUCT\nSHORT / REPEAT", size=8))
    body.append(_rect(200, 22, 138, 58, "#d9e8f5"))
    body.append(_svg_text(269, 44, "Train-mined", size=11, weight="700"))
    body.append(_svg_text(269, 63, "DIRLEN / BURST\nIAT / TRANS", size=8))
    body.append(_arrow(160, 51, 198, 51))
    body.append(_rect(112, 104, 136, 38, "#f3f5f7"))
    body.append(_svg_text(180, 128, "candidate pool", size=10, weight="700"))
    body.append(_arrow(91, 80, 145, 104))
    body.append(_arrow(269, 80, 215, 104))

    stages = [
        ("train-only\nsupport", "#e8f1df"),
        ("stability\ncoverage", "#e8f1df"),
        ("benign-tail\nsensitivity", "#fff4d8"),
        ("redundancy\npruning", "#eadff1"),
    ]
    sx, sy, sw, sh, gap = 8, 166, 72, 42, 8
    body.append(_arrow(180, 142, 56, 166))
    for i, (lab, fill) in enumerate(stages):
        x = sx + i * (sw + gap)
        body.append(_rect(x, sy, sw, sh, fill))
        body.append(_svg_text(x + sw / 2, sy + 18, lab, size=8))
        if i < len(stages) - 1:
            body.append(_arrow(x + sw + 1, sy + sh / 2, x + sw + gap - 2, sy + sh / 2))
    body.append(_arrow(sx + 4 * (sw + gap) - gap + 2, sy + sh / 2, 292, 224))
    body.append(_rect(218, 224, 118, 42, "#dfeee9"))
    body.append(_svg_text(277, 242, "selected motif", size=9, weight="700"))
    body.append(_svg_text(277, 257, "dictionary", size=9, weight="700"))
    body.append(_rect(20, 226, 184, 38, "#fffdf6"))
    body.append(_svg_text(112, 243, "Expert-guided + train-mined candidates;", size=8, weight="700"))
    body.append(_svg_text(112, 256, "fitting split + benign validation instantiate the dictionary.", size=7.5))
    _write_svg("fig3_candidate_to_dictionary", 360, 286, "\n".join(body))


def draw_motif_transaction_evidence() -> None:
    body: list[str] = []
    xs = [36, 70, 104, 138, 172, 206, 240, 274, 308]
    dirs = ["C", "C", "S", "S", "C", "C", "C", "S", "S"]
    fills = ["#d9e8f5" if d == "C" else "#e8f1df" for d in dirs]
    for x, d, fill in zip(xs, dirs, fills):
        body.append(f'<circle cx="{x}" cy="38" r="12" fill="{fill}" stroke="#34424f" stroke-width="1"/>')
        body.append(_svg_text(x, 42, d, size=10, weight="700"))
    body.append(_svg_text(172, 70, "test flow packet direction / length / timing / burst evidence", size=9))
    body.append(_arrow(172, 78, 172, 104))
    items = [
        ("packet/burst\n+ motif", "#d9e8f5"),
        ("transaction\nitems", "#eadff1"),
        ("transaction\nhistogram", "#f7e4dc"),
    ]
    for i, (lab, fill) in enumerate(items):
        x = 42 + i * 104
        body.append(_rect(x, 108, 88, 32, fill))
        body.append(_svg_text(x + 44, 124 if i == 2 else 128, lab, size=8 if i == 2 else 9, weight="700"))
        if i < len(items) - 1:
            body.append(_arrow(x + 88, 124, x + 102, 124))
    body.append(_arrow(172, 144, 172, 170))
    body.append(_rect(28, 172, 140, 42, "#dfeee9"))
    body.append(_svg_text(98, 193, "example nearest benign\nmemory neighbor", size=9))
    body.append(_rect(194, 172, 138, 42, "#fff4d8"))
    body.append(_svg_text(263, 193, "motif contrast\nalert-only / missing", size=9))
    body.append(_arrow(168, 193, 194, 193))
    body.append(_arrow(263, 214, 263, 238))
    body.append(_rect(58, 240, 250, 42, "#e9edf5"))
    body.append(_svg_text(183, 225, "C/S = client-to-server / server-to-client", size=8, weight="700"))
    body.append(_svg_text(183, 258, "diagnosis record: score, threshold, decision,", size=9, weight="700"))
    body.append(_svg_text(183, 273, "active motifs, nearest evidence, distance provenance", size=8))
    _write_svg("fig4_motif_transaction_evidence", 360, 300, "\n".join(body))


def draw_low_fpr_dashboard(results_dir: Path) -> None:
    p = results_dir / "main_detection" / "unknown_best_settings_3seed_runs.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    attack_col = next((c for c in ["heldout_attack", "unknown_attack", "attack"] if c in df.columns), None)
    if attack_col is None:
        return
    g = df.groupby(attack_col)[["recall_at_0_1pct_fpr", "recall_at_1pct_fpr"]].agg(["mean", "std"]).sort_index()
    labels = list(g.index)
    vals01 = g[("recall_at_0_1pct_fpr", "mean")].to_numpy(dtype=float).tolist()
    vals1 = g[("recall_at_1pct_fpr", "mean")].to_numpy(dtype=float).tolist()
    vals01.append(float(np.mean(vals01)))
    vals1.append(float(np.mean(vals1)))
    labels.append("Aggregate")
    draw_bar_chart(
        "fig5_low_fpr_unknown_recall",
        labels,
        [("R@0.1%FPR", vals01, "#7aa6c2"), ("R@1%FPR", vals1, "#d09253")],
        ylabel="Recall",
        ymax=1.0,
    )
    print({"fig5_source": str(p), "R01_mean": _fmt(np.mean(vals01[:-1])), "R1_mean": _fmt(np.mean(vals1[:-1]))})


def draw_calibration_alert_budget(results_dir: Path) -> None:
    fixed_p = results_dir / "primitive_categories" / "primitive_category_unknown_metrics.csv"
    memory_p = results_dir / "memory_optimization" / "summary_by_setting.csv"
    conformal_p = results_dir / "motif_selection" / "motif_selection_conformal_calibration.csv"
    if not fixed_p.exists():
        return
    fixed = pd.read_csv(fixed_p)
    fixed = fixed[(fixed["feature_view"] == "packet_burst_plus_profile_structural") & (fixed["memory_scope"] == "global")]
    if fixed.empty:
        return
    oracle_fpr = float(fixed["actual_fpr_at_1pct_fpr"].mean())
    oracle_recall = float(fixed["recall_at_1pct_fpr"].mean())
    p99_fpr = float(fixed["val_p99_realized_fpr"].mean())
    p99_recall = float(fixed["val_p99_attack_recall"].mean())
    evt_fpr = None
    if memory_p.exists():
        memory = pd.read_csv(memory_p)
        evt = memory[(memory["experiment_group"] == "calibration") & (memory["setting"] == "evt_tail_p99")]
        if not evt.empty:
            evt_fpr = float(evt.iloc[0]["p99_realized_fpr_mean"])
    empirical_fpr = None
    if conformal_p.exists():
        conformal = pd.read_csv(conformal_p)
        empirical = conformal[
            (conformal["motif_selection_strategy"] == "full_utility")
            & (conformal["feature_view"] == "packet_burst_selected_motifs")
            & (conformal["target_alpha"].astype(float) == 0.01)
        ]
        if not empirical.empty:
            empirical_fpr = float(empirical["realized_fpr"].mean())

    fpr_rows = [
        ("Oracle\n1%", oracle_fpr, "#7aa6c2"),
        ("Fixed\nP99", p99_fpr, "#d09253"),
    ]
    if evt_fpr is not None:
        fpr_rows.append(("EVT\nP99", evt_fpr, "#8fbf8f"))
    if empirical_fpr is not None:
        fpr_rows.append(("Emp.\n.01", empirical_fpr, "#b85c5c"))

    prevalences = [0.001, 0.01, 0.05]
    budget_rows = []
    for prevalence in prevalences:
        benign = 10000.0 * (1.0 - prevalence)
        attack = 10000.0 * prevalence
        budget_rows.append(
            (
                prevalence,
                benign * p99_fpr + attack * p99_recall,
                benign * oracle_fpr + attack * oracle_recall,
            )
        )

    width, height = 360, 342
    body: list[str] = []
    # Panel A: mainline P99 and labeled calibration diagnostics.
    x0, y0, pw, ph = 46, 34, 278, 104
    body.append(_svg_text(x0 + pw / 2, 18, "A. Realized FPR by candidate", size=11, weight="700"))
    body.append(_line(x0, y0 + ph, x0 + pw, y0 + ph))
    body.append(_line(x0, y0, x0, y0 + ph))
    ymax = max(0.04, max(v for _, v, _ in fpr_rows) * 1.25)
    bar_w = pw / max(len(fpr_rows), 1) * 0.48
    for i, (label, val, color) in enumerate(fpr_rows):
        h = ph * val / ymax
        x = x0 + 18 + i * (pw / max(len(fpr_rows), 1))
        body.append(f'<rect x="{x:.1f}" y="{y0+ph-h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        body.append(_svg_text(x + bar_w / 2, y0 + ph + 16, label, size=7))
        body.append(_svg_text(x + bar_w / 2, y0 + ph - h - 4, f"{val*100:.1f}%", size=8))
    for ref, lab in [(0.01, "1%"), (0.005, "0.5%")]:
        yy = y0 + ph - ph * ref / ymax
        body.append(_line(x0, yy, x0 + pw, yy, "#7f8b95", 0.8, "3 3"))
        body.append(_svg_text(x0 + pw + 8, yy + 3, lab, size=7, anchor="start"))
    body.append(_svg_text(14, y0 + ph / 2, "FPR", size=9))
    # Panel B: prevalence-normalized alert budget.
    x1, y1, pw2, ph2 = 46, 206, 278, 92
    body.append(_svg_text(x1 + pw2 / 2, 188, "B. Alert budget under prevalence shift", size=11, weight="700"))
    body.append(_line(x1, y1 + ph2, x1 + pw2, y1 + ph2))
    body.append(_line(x1, y1, x1, y1 + ph2))
    maxv = max(1.0, max(max(row[1], row[2]) for row in budget_rows) * 1.1)
    gw = pw2 / max(len(budget_rows), 1)
    bw = gw / 3.0
    for i, (prevalence, p99_alerts, oracle_alerts) in enumerate(budget_rows):
        for j, (val, color) in enumerate([(p99_alerts, "#d09253"), (oracle_alerts, "#7aa6c2")]):
            h = ph2 * val / maxv
            x = x1 + i * gw + 20 + j * bw
            body.append(f'<rect x="{x:.1f}" y="{y1+ph2-h:.1f}" width="{bw*0.82:.1f}" height="{h:.1f}" fill="{color}"/>')
        body.append(_svg_text(x1 + i * gw + 36, y1 + ph2 + 17, f"{prevalence*100:g}%", size=8))
    body.append(f'<rect x="{x1+12}" y="{height-42}" width="10" height="10" fill="#d09253"/>')
    body.append(_svg_text(x1 + 27, height - 33, "fixed P99", size=8, anchor="start"))
    body.append(f'<rect x="{x1+132}" y="{height-42}" width="10" height="10" fill="#7aa6c2"/>')
    body.append(_svg_text(x1 + 147, height - 33, "oracle 1% FPR", size=8, anchor="start"))
    body.append(_svg_text(15, y1 + ph2 / 2, "alerts/10k", size=8))
    body.append(_svg_text(180, 166, "Fixed-view P99: 2.58% FPR, 258.3 false alerts/10k benign.", size=9, weight="700"))
    _write_svg("fig6_calibration_alert_budget", width, height, "\n".join(body))

    width, height = 720, 154
    body = [
        "<style>",
        "  text { font-family: Arial, Helvetica, sans-serif; fill: #23313d; }",
        "  .title { font-size: 12px; font-weight: 700; }",
        "  .tick { font-size: 9px; }",
        "  .tiny { font-size: 8px; }",
        "  .legend { font-size: 9px; }",
        "</style>",
        "",
        "<!-- A. Realized FPR by candidate -->",
        '<text x="182" y="16" text-anchor="middle" class="title">A. Realized FPR by candidate</text>',
        _line(42, 100, 322, 100, "#34424f", 1.1),
        _line(42, 28, 42, 100, "#34424f", 1.1),
    ]
    x0, y0, pw, ph = 42, 28, 280, 72
    ymax = max(0.04, max(v for _, v, _ in fpr_rows) * 1.25)
    bar_w = pw / max(len(fpr_rows), 1) * 0.44
    for i, (label, val, color) in enumerate(fpr_rows):
        h = ph * val / ymax
        x = x0 + 22 + i * (pw / max(len(fpr_rows), 1))
        body.append(f'<rect x="{x:.1f}" y="{y0+ph-h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        body.append(f'<text x="{x + bar_w / 2:.1f}" y="115" text-anchor="middle" class="tick">{html.escape(label).replace(chr(10), " ")}</text>')
        body.append(f'<text x="{x + bar_w / 2:.1f}" y="{y0+ph-h-3:.1f}" text-anchor="middle" class="tick">{val*100:.1f}%</text>')
    for ref, lab in [(0.01, "1%"), (0.005, "0.5%")]:
        yy = y0 + ph - ph * ref / ymax
        body.append(_line(x0, yy, x0 + pw, yy, "#7f8b95", 0.7, "3 3"))
        body.append(f'<text x="326" y="{yy+2.6:.1f}" text-anchor="start" class="tiny">{lab}</text>')
    body.append('<text x="15" y="65" text-anchor="middle" class="tiny">FPR</text>')
    body.extend(
        [
            "",
            "<!-- B. Alert load by prevalence -->",
            '<text x="542" y="16" text-anchor="middle" class="title">B. Alert load by prevalence</text>',
            _line(402, 100, 682, 100, "#34424f", 1.1),
            _line(402, 28, 402, 100, "#34424f", 1.1),
        ]
    )
    x1, y1, pw2, ph2 = 402, 28, 280, 72
    maxv = max(1.0, max(max(row[1], row[2]) for row in budget_rows) * 1.1)
    gw = pw2 / max(len(budget_rows), 1)
    bw = gw / 3.0
    for i, (prevalence, p99_alerts, oracle_alerts) in enumerate(budget_rows):
        for j, (val, color) in enumerate([(p99_alerts, "#d09253"), (oracle_alerts, "#7aa6c2")]):
            h = ph2 * val / maxv
            x = x1 + i * gw + 26 + j * bw
            body.append(f'<rect x="{x:.1f}" y="{y1+ph2-h:.1f}" width="{bw*0.72:.1f}" height="{h:.1f}" fill="{color}"/>')
        body.append(f'<text x="{x1 + i * gw + 46.5:.1f}" y="115" text-anchor="middle" class="tick">{prevalence*100:g}%</text>')
    body.extend(
        [
            '<text x="372" y="66" text-anchor="middle" class="tiny">alerts/10k</text>',
            "",
            "<!-- Shared legend -->",
            '<rect x="62" y="132" width="9" height="9" fill="#d09253"/>',
            '<text x="76" y="140" text-anchor="start" class="legend">fixed P99</text>',
            '<rect x="176" y="132" width="9" height="9" fill="#7aa6c2"/>',
            '<text x="190" y="140" text-anchor="start" class="legend">oracle 1% FPR</text>',
            '<rect x="330" y="132" width="9" height="9" fill="#8fbf8f"/>',
            '<text x="344" y="140" text-anchor="start" class="legend">EVT diagnostic</text>',
            '<rect x="482" y="132" width="9" height="9" fill="#b85c5c"/>',
            '<text x="496" y="140" text-anchor="start" class="legend">empirical diagnostic</text>',
            '<text x="690" y="140" text-anchor="end" font-size="9" font-weight="700">mainline fixed view</text>',
        ]
    )
    _write_svg("fig6_calibration_alert_budget_compact", width, height, "\n".join(body))
    print({"fig6_sources": [str(fixed_p), str(memory_p), str(conformal_p)]})


def draw_memory_governance() -> None:
    p = ROOT / "results" / "memory_optimization" / "summary_by_setting.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    wanted = [
        ("baseline", "uniform_exact", "Exact"),
        ("indexed_retrieval", "coreset_tail_preserving_0.5", "Tail\ncore50"),
        ("indexed_retrieval", "coreset_random_0.5", "Random\ncore50"),
        ("token_weighting", "tfidf_train_only", "TF-IDF"),
        ("memory_governance", "tail_aware_update", "Tail\nupdate"),
        ("memory_governance", "random_benign_update", "Random\nupdate"),
        ("memory_governance", "oracle_pollution_5pct_attack_diagnostic", "5% attack\npollution"),
    ]
    rows = []
    for group, setting, label in wanted:
        sub = df[(df["experiment_group"] == group) & (df["setting"] == setting)]
        if not sub.empty:
            r = sub.iloc[0]
            rows.append((label, _safe_float(r.get("auroc_mean")), _safe_float(r.get("recall_at_1pct_fpr_mean")), _safe_float(r.get("p99_realized_fpr_mean")), _safe_float(r.get("query_ms_per_flow_mean")), _safe_float(r.get("memory_size_mean"))))
    if not rows:
        return
    width, height = 720, 320
    body: list[str] = []
    # Panel A grouped bars.
    x0, y0, pw, ph = 48, 42, 350, 180
    body.append(_svg_text(x0 + pw / 2, 20, "A. Memory strategy versus diagnosis quality", size=11, weight="700"))
    body.append(_line(x0, y0 + ph, x0 + pw, y0 + ph))
    body.append(_line(x0, y0, x0, y0 + ph))
    gw = pw / len(rows)
    bw = gw / 3.1
    for i, (lab, auroc, recall, p99, _ms, _mem) in enumerate(rows):
        for j, (val, color) in enumerate([(auroc, "#7aa6c2"), (recall, "#d09253"), (p99, "#b85c5c")]):
            h = ph * min(val, 1.0)
            x = x0 + i * gw + 4 + j * bw
            body.append(f'<rect x="{x:.1f}" y="{y0+ph-h:.1f}" width="{bw*0.85:.1f}" height="{h:.1f}" fill="{color}"/>')
        body.append(_svg_text(x0 + i * gw + gw / 2, y0 + ph + 18, lab, size=8))
    body.append(f'<rect x="{x0+8}" y="{height-50}" width="10" height="10" fill="#7aa6c2"/>')
    body.append(_svg_text(x0 + 23, height - 41, "AUROC", size=9, anchor="start"))
    body.append(f'<rect x="{x0+74}" y="{height-50}" width="10" height="10" fill="#d09253"/>')
    body.append(_svg_text(x0 + 89, height - 41, "R@1%FPR", size=9, anchor="start"))
    body.append(f'<rect x="{x0+176}" y="{height-50}" width="10" height="10" fill="#b85c5c"/>')
    body.append(_svg_text(x0 + 191, height - 41, "P99 FPR", size=9, anchor="start"))
    # Panel B memory and latency.
    x1, y1, pw2, ph2 = 455, 42, 218, 180
    body.append(_svg_text(x1 + pw2 / 2, 20, "B. Memory and query cost", size=11, weight="700"))
    body.append(_line(x1, y1 + ph2, x1 + pw2, y1 + ph2))
    body.append(_line(x1, y1, x1, y1 + ph2))
    max_mem = max(r[5] for r in rows) or 1.0
    max_ms = max(r[4] for r in rows) or 1.0
    prev = None
    for i, (lab, _au, _re, _p99, ms, mem) in enumerate(rows):
        x = x1 + 16 + i * (pw2 - 32) / max(len(rows) - 1, 1)
        ym = y1 + ph2 - ph2 * mem / max_mem
        yl = y1 + ph2 - ph2 * ms / max_ms
        body.append(f'<circle cx="{x:.1f}" cy="{ym:.1f}" r="4" fill="#7aa6c2"/>')
        body.append(f'<rect x="{x-3:.1f}" y="{yl-3:.1f}" width="6" height="6" fill="#d09253"/>')
        if prev:
            body.append(_line(prev[0], prev[1], x, ym, "#7aa6c2", 1.0))
            body.append(_line(prev[0], prev[2], x, yl, "#d09253", 1.0))
        prev = (x, ym, yl)
        if i in [0, 1, 2, 6]:
            body.append(_svg_text(x, y1 + ph2 + 18, lab, size=8))
    body.append(_svg_text(x1 + pw2 / 2, 270, "Blue: memory size; orange: ms/flow", size=9))
    body.append(_svg_text(360, 304, "Attack-pollution row is a non-deployable diagnostic stress test.", size=11, weight="700"))
    _write_svg("fig7_memory_governance", width, height, "\n".join(body))
    print({"fig7_source": str(p), "rows": len(rows)})


def draw_runtime_scalability() -> None:
    p = ROOT / "results" / "runtime_unified" / "knn_scaling_runtime_unified_summary.csv"
    value_col = "batch_ms_per_flow_mean_mean"
    source_col = "source"
    if not p.exists():
        p = ROOT / "results" / "efficiency" / "ann_knn_scalability.csv"
        value_col = "query_ms_per_flow_mean"
        source_col = "memory_source"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("memory_size")
    if df.empty:
        return
    width, height = 360, 270
    x0, y0, pw, ph = 58, 28, 272, 158
    body: list[str] = []
    body.append(_line(x0, y0 + ph, x0 + pw, y0 + ph))
    body.append(_line(x0, y0, x0, y0 + ph))
    max_x = float(df["memory_size"].max())
    max_y = float(df[value_col].max()) * 1.12
    prev_by_source: dict[str, tuple[float, float]] = {}
    colors = {"real_subsample": "#7aa6c2", "real subsample": "#7aa6c2", "bootstrap_duplicate_memory": "#d09253", "duplicated memory": "#d09253"}
    for _, r in df.iterrows():
        src = str(r.get(source_col, "memory"))
        x = x0 + pw * _safe_float(r["memory_size"]) / max_x
        y = y0 + ph - ph * _safe_float(r[value_col]) / max_y
        color = colors.get(src, "#34424f")
        if src in prev_by_source:
            px, py = prev_by_source[src]
            body.append(_line(px, py, x, y, color, 1.4))
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#23313d" stroke-width="0.6"/>')
        prev_by_source[src] = (x, y)
    for tick in [0, 10000, 50000]:
        x = x0 + pw * tick / max_x
        body.append(_line(x, y0 + ph, x, y0 + ph + 4))
        body.append(_svg_text(x, y0 + ph + 18, f"{tick//1000}k", size=8))
    body.append(_svg_text(x0 + pw / 2, height - 20, "benign memory size", size=10))
    body.append(_svg_text(18, y0 + ph / 2, "ms/flow", size=10))
    body.append(f'<circle cx="{x0+8}" cy="{height-50}" r="4" fill="#7aa6c2"/>')
    body.append(_svg_text(x0 + 18, height - 46, "real subsample", size=9, anchor="start"))
    body.append(f'<circle cx="{x0+122}" cy="{height-50}" r="4" fill="#d09253"/>')
    body.append(_svg_text(x0 + 132, height - 46, "duplicated stress", size=9, anchor="start"))
    body.append(_svg_text(180, 16, "Exact KNN flow-ready scaling", size=12, weight="700"))
    _write_svg("fig8_runtime_scalability", width, height, "\n".join(body))
    print({"fig8_source": str(p), "max_memory": int(max_x), "max_ms_per_flow": _fmt(max_y / 1.12, 3)})


def build_runtime_scaling_table(results_dir: Path) -> None:
    p = results_dir / "runtime_unified" / "knn_scaling_runtime_unified_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lookup = {int(float(r["memory_size"])): r for _, r in df.iterrows()}
    lines = [
        r"\begin{tabularx}{\columnwidth}{lXcc}",
        r"\toprule",
        r"Category & Setting & Throughput / latency & Scope \\",
        r"\midrule",
        r"Artifact load & processed flow JSONL & 40,797 flows/s & existing Zeek-aligned flows \\",
        r"Artifact load & profile motif JSONL & 58,489 flows/s & cached motif artifacts \\",
        r"Diagnosis & KNN scoring to records & 8,710 flows/s & first 512 test flows \\",
        r"\midrule",
    ]
    for size in [1000, 10000, 50000]:
        row = lookup.get(size)
        if row is None:
            continue
        src = "real subsample" if "real" in str(row.get("source")) else "duplicated memory"
        lines.append(rf"KNN scaling & {size // 1000}k benign memory & {_fmt(row.get('batch_ms_per_flow_mean_mean'), 3)} ms/flow & exact KNN, {src} \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    (TABLE_DIR / "table_runtime_scaling_compact.tex").write_text("\n".join(lines), encoding="utf-8")


def build_ann_scalability_table(results_dir: Path) -> None:
    p = results_dir / "runtime_unified" / "knn_scaling_runtime_unified_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("memory_size")
    rows = []
    for _, r in df.iterrows():
        src = str(r.get("source", "")).replace("_", r"\_")
        rows.append(
            [
                "exact\\_knn",
                _fmt(r.get("memory_size"), 0),
                src,
                f"${_fmt(r.get('batch_ms_per_flow_mean_mean'), 3)}\\pm{_fmt(r.get('batch_ms_per_flow_mean_std'), 3)}$",
                f"${_fmt(r.get('memory_mb_mean'), 2)}\\pm{_fmt(r.get('memory_mb_std'), 2)}$",
            ]
        )
    (TABLE_DIR / "table_ann_scalability.tex").write_text(
        _latex_table(["Method", "Memory", "Source", "Batch ms/flow", "Memory MB"], rows),
        encoding="utf-8",
    )


def draw_unknown_recall(results_dir: Path) -> None:
    p = results_dir / "main_detection" / "unknown_best_settings_3seed_runs.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    attack_col = next((c for c in ["heldout_attack", "unknown_attack", "attack"] if c in df.columns), None)
    if attack_col is None:
        return
    g = df.groupby(attack_col)[["recall_at_0_1pct_fpr", "recall_at_1pct_fpr"]].mean().sort_index()
    draw_bar_chart("fig_unknown_low_fpr_recall", list(g.index), [("R@0.1%", g["recall_at_0_1pct_fpr"].tolist(), "#7aa6c2"), ("R@1%", g["recall_at_1pct_fpr"].tolist(), "#d09253")], ylabel="Recall", ymax=1.0)


def draw_calibration(results_dir: Path) -> None:
    p = results_dir / "motif_selection" / "motif_selection_conformal_calibration.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    sub = df[(df["motif_selection_strategy"] == "full_utility") & (df["target_alpha"].isin([0.01, 0.005, 0.001]))]
    if sub.empty:
        return
    g = sub.groupby("target_alpha")["realized_fpr"].mean().sort_index()
    labels = [f"a={a:g}" for a in g.index]
    realized = (g.to_numpy() * 100.0).tolist()
    target = ([float(a) * 100.0 for a in g.index])
    draw_bar_chart("fig_low_fpr_calibration", labels, [("target", target, "#8fbf8f"), ("realized", realized, "#b85c5c")], ylabel="FPR (%)")


def draw_memory(results_dir: Path) -> None:
    p = results_dir / "memory_optimization" / "summary_by_setting.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    pick = df[df["setting"].isin(["uniform_exact", "coreset_tail_preserving_0.5", "coreset_random_0.5", "oracle_pollution_5pct_attack_diagnostic"])]
    labels = pick["setting"].str.replace("oracle_pollution_5pct_attack_diagnostic", "5pct_pollution").str.replace("coreset_tail_preserving_0.5", "tail_core50").str.replace("coreset_random_0.5", "rand_core50").tolist()
    draw_bar_chart(
        "motivation_memory_pollution",
        labels,
        [
            ("AUROC", pick["auroc_mean"].astype(float).tolist(), "#7aa6c2"),
            ("R@1%", pick["recall_at_1pct_fpr_mean"].astype(float).tolist(), "#d09253"),
        ],
        ylabel="Score",
        ymax=1.0,
    )


def draw_heatmap() -> None:
    p = ROOT / "results" / "primitive_analysis" / "primitive_category_scale_summary.csv"
    if not p.exists():
        p = ROOT / "results" / "raw_rebuild_split_first" / "primitive_categories" / "primitive_category_scale_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    sub = df[df["class"].notna() & df["family"].notna() & df["class_family_trigger_rate"].notna()]
    if sub.empty:
        return
    pivot = sub.groupby(["class", "family"])["class_family_trigger_rate"].mean().unstack(fill_value=0.0)
    preferred_rows = [r for r in ["BENIGN", "Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"] if r in pivot.index]
    preferred_cols = [c for c in ["packet_shape", "burst_shape", "timing_rhythm", "direction_transition", "composite"] if c in pivot.columns]
    if not preferred_rows or not preferred_cols:
        return
    pivot = pivot.loc[preferred_rows, preferred_cols]
    labels_x = [c.replace("_", "\n") for c in pivot.columns]
    labels_y = list(pivot.index)
    data = pivot.to_numpy(dtype=float)
    body = []
    width, height = 356, 252
    x0, y0, cw, ch = 92, 42, 50, 28

    def fill_for(v: float) -> str:
        lo = np.array([246, 249, 252], dtype=float)
        hi = np.array([77, 132, 184], dtype=float)
        rgb = lo + (hi - lo) * max(0.0, min(1.0, float(v)))
        return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"

    for i, row in enumerate(data):
        for j, v in enumerate(row):
            x = x0 + j * cw
            y = y0 + i * ch
            color = fill_for(v)
            fg = "#ffffff" if v >= 0.62 else "#1f2a36"
            body.append(
                f'<rect x="{x}" y="{y}" width="{cw-2}" height="{ch-2}" '
                f'rx="2" fill="{color}" stroke="white" stroke-width="0.8"/>'
            )
            body.append(
                f'<text x="{x + (cw - 2) / 2}" y="{y + 17}" text-anchor="middle" '
                f'font-size="9.5" font-weight="700" font-family="Arial, Helvetica, sans-serif" '
                f'fill="{fg}">{v:.2f}</text>'
            )
    for j, lab in enumerate(labels_x):
        body.append(_svg_text(x0 + j * cw + (cw - 2) / 2, 18, lab, size=8, weight="700"))
    for i, lab in enumerate(labels_y):
        body.append(_svg_text(x0 - 8, y0 + i * ch + 17, lab, size=8.5, anchor="end", weight="700"))
    body.append(_svg_text(width / 2, 232, "Trigger rate by motif family (class-level mean)", size=8, weight="700"))
    _write_svg("motivation_primitive_heatmap", width, height, "\n".join(body))


def draw_online(results_dir: Path) -> None:
    p = results_dir / "memory_optimization" / "summary_by_setting.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    pick = df[(df["experiment_group"] == "indexed_retrieval") & (df["setting"].isin(["exact_reference", "coreset_tail_preserving_0.25", "coreset_tail_preserving_0.5", "coreset_tail_preserving_0.75"]))]
    if pick.empty:
        return
    labels = pick["setting"].str.replace("coreset_tail_preserving_", "tail").str.replace("exact_reference", "exact").tolist()
    draw_bar_chart("motivation_online_replay_evt", labels, [("ms/flow", pick["query_ms_per_flow_mean"].astype(float).tolist(), "#7aa6c2")], ylabel="ms/flow")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_memory_governance_table(results_dir)
    build_runtime_scaling_table(results_dir)
    build_ann_scalability_table(results_dir)
    build_dm_motif_table(results_dir)
    build_candidate_source_table(results_dir)
    build_selection_stage_table(results_dir)
    build_explanation_table(results_dir)
    draw_applied_workflow()
    draw_data_capability()
    draw_candidate_to_dictionary()
    draw_motif_transaction_evidence()
    draw_low_fpr_dashboard(results_dir)
    draw_calibration_alert_budget(results_dir)
    draw_memory_governance()
    draw_runtime_scalability()
    draw_framework()
    draw_algorithm()
    draw_token_example()
    draw_unknown_recall(results_dir)
    draw_calibration(results_dir)
    draw_memory(results_dir)
    draw_heatmap()
    draw_online(results_dir)
    print({"tables": str(TABLE_DIR), "figures": str(FIG_DIR)})


if __name__ == "__main__":
    main()
