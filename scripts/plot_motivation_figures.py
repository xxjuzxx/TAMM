#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PAPER_DIR = REPO_ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures"


def esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def text(
    x: float,
    y: float,
    value: Any,
    *,
    size: int = 12,
    anchor: str = "middle",
    weight: str = "400",
    color: str = "#243042",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, color: str = "#334155", width: float = 1.5) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"/>'


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "#ffffff",
    stroke: str = "#cbd5e1",
    sw: float = 1.0,
    rx: float = 4.0,
) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'


def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" '
        'stroke="#334155" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>'
    )


def svg_wrap(width: int, height: int, body: list[str]) -> str:
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/></marker>',
        '</defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    return "\n".join(head + body + ["</svg>", ""])


def write_svg(path: Path, width: int, height: int, body: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_wrap(width, height, body), encoding="utf-8")
    return path


def convert_pdf(svg_path: Path) -> Path:
    pdf_path = svg_path.with_suffix(".pdf")
    subprocess.run(
        ["cairosvg", str(svg_path), "-o", str(pdf_path)],
        check=True,
        cwd=str(REPO_ROOT),
    )
    return pdf_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def hbar(
    x: float,
    y: float,
    w: float,
    label: str,
    value: float,
    *,
    max_value: float = 1.0,
    color: str = "#1f77b4",
    suffix: str = "",
) -> list[str]:
    bar_w = max(0.0, min(w, w * value / max_value))
    return [
        text(x - 8, y + 12, label, size=11, anchor="end", color="#334155"),
        rect(x, y, w, 16, fill="#edf2f7", stroke="#e2e8f0", rx=3),
        rect(x, y, bar_w, 16, fill=color, stroke=color, rx=3),
        text(x + w + 8, y + 12, f"{value:.4f}{suffix}", size=10, anchor="start", color="#334155"),
    ]


def build_closedset_gap(out_dir: Path) -> Path:
    # Values are the current paper-table values from Table "Main detection results"
    # and Table "Main fixed low-FPR diagnosis results".
    closed = [
        ("RF multiclass", 0.9285),
        ("XGB multiclass", 0.9519),
        ("Composer multiclass", 0.9816),
        ("Binary head", 0.9932),
        ("+ anomaly feats", 0.9985),
    ]
    diagnosis = [
        ("FlowPrim AUROC", 0.9982, "#0f766e"),
        ("R@1%FPR", 0.9826, "#0f766e"),
        ("R@0.1%FPR", 0.5866, "#0f766e"),
        ("P99 realized FPR", 0.0258, "#ea580c"),
    ]
    width, height = 960, 430
    body = [
        text(width / 2, 28, "Closed-set scores do not settle low-FPR diagnosis", size=19, weight="700"),
        text(width / 2, 51, "Near-saturated supervised results still leave benign-tail calibration and alert-budget risk.", size=12, color="#5b6678"),
        text(245, 84, "Closed-set / supervised sanity", size=14, weight="700"),
        text(715, 84, "Low-FPR scores and calibration", size=14, weight="700"),
        line(480, 78, 480, 356, color="#cbd5e1", width=1),
    ]
    for i, (label, val) in enumerate(closed):
        body.extend(hbar(190, 113 + i * 42, 230, label, val, color="#2563eb"))
    body.append(text(245, 352, "Macro-F1 values from current paper table", size=11, color="#64748b"))
    for i, (label, val, color) in enumerate(diagnosis):
        y = 113 + i * 42
        body.extend(hbar(665, y, 190, label, val, color=color))
    body.append(rect(572, 288, 306, 54, fill="#fff1f2", stroke="#fb7185", sw=1.2, rx=6))
    body.append(text(725, 310, "BENIGN-val P99 alert budget", size=12, weight="700", color="#9f1239"))
    body.append(text(725, 330, "258.3 false alerts / 10k benign flows", size=13, weight="700", color="#9f1239"))
    body.append(text(715, 360, "False alerts are an alert-budget callout, not a comparable 0-1 score.", size=11, color="#64748b"))
    body.append(
        text(
            width / 2,
            392,
            "Diagnosis deployability depends on benign-tail thresholds, not only ranking or closed-set accuracy.",
            size=12,
            weight="700",
            color="#111827",
        )
    )
    return write_svg(out_dir / "motivation_closedset_gap.svg", width, height, body)


def color_for(value: float) -> str:
    lo = (238, 242, 246)
    hi = (25, 132, 136)
    t = max(0.0, min(1.0, value))
    rgb = tuple(round(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))
    return "#%02x%02x%02x" % rgb


def build_primitive_heatmap(out_dir: Path) -> Path:
    # Current paper-table values from the class-level behavior evidence table.
    rows = [
        ("BENIGN", [0.8735, 0.09, 0.7790, 0.0590]),
        ("Probe", [0.9940, 0.94, 0.0310, 0.0435]),
        ("DDoS", [0.0090, 0.00, 0.9595, 0.0020]),
        ("Botnet", [0.0000, 0.00, 0.1122, 0.9754]),
        ("BruteForce", [0.0005, 0.00, 0.1746, 0.9931]),
        ("WebAttack*", [0.0000, 0.00, 0.7143, 1.0000]),
    ]
    cols = ["SHORT", "SAME", "REPEAT", "DUP"]
    width, height = 560, 392
    x0, y0, cw, ch = 142, 82, 84, 36
    body = [
        text(width / 2, 28, "Primitive profiles are evidence, not labels", size=18, weight="700"),
        text(width / 2, 51, "Benign traffic also triggers primitives; profiles summarize evidence mixtures.", size=12, color="#5b6678"),
    ]
    for j, col in enumerate(cols):
        body.append(text(x0 + j * cw + cw / 2, 72, col, size=12, weight="700"))
    for i, (label, vals) in enumerate(rows):
        y = y0 + i * (ch + 4)
        body.append(text(x0 - 12, y + 24, label, size=12, anchor="end", weight="700"))
        for j, val in enumerate(vals):
            x = x0 + j * cw
            fill = color_for(val)
            body.append(rect(x, y, cw - 4, ch, fill=fill, stroke="#ffffff", sw=0.8, rx=2))
            fg = "#ffffff" if val >= 0.55 else "#0f172a"
            body.append(text(x + cw / 2 - 2, y + 24, f"{val:.2f}", size=13, weight="700", color=fg))
    body.append(text(width / 2, 342, "* WebAttack has only 28 held-out attack flows in the leave-one setup.", size=11, color="#b45309"))
    body.append(text(width / 2, 365, "Primitive trigger rates are audit evidence summaries, not deterministic attack signatures.", size=12, weight="700"))
    return write_svg(out_dir / "motivation_primitive_heatmap.svg", width, height, body)


def build_token_view_attribution(out_dir: Path) -> Path:
    rows = read_csv(ROOT / "results" / "primitive_categories" / "primitive_category_feature_attribution.csv")
    order = [
        ("profile_only", "Profile only"),
        ("structural_only", "Structural only"),
        ("packet_burst_only", "Packet+burst"),
        ("packet_burst_plus_profile", "+ profile"),
        ("packet_burst_plus_structural", "+ structural"),
        ("packet_burst_plus_profile_structural", "FlowPrim fixed"),
    ]
    by_view = {row["feature_view"]: row for row in rows}
    data = []
    for key, label in order:
        row = by_view[key]
        data.append((label, to_float(row["recall_at_1pct_fpr"]), to_float(row["val_p99_realized_fpr"])))
    width, height = 760, 420
    body = [
        text(width / 2, 28, "Token-view attribution for low-FPR diagnosis", size=18, weight="700"),
        text(width / 2, 51, "Structural primitives recover packet/burst separation as auditable motifs.", size=12, color="#5b6678"),
        text(295, 82, "Recall@1%FPR", size=13, weight="700"),
        text(635, 82, "P99 FPR", size=13, weight="700"),
    ]
    x, y0, w = 210, 106, 290
    for i, (label, recall, fpr) in enumerate(data):
        y = y0 + i * 44
        color = "#0f766e" if label != "FlowPrim fixed" else "#2563eb"
        body.extend(hbar(x, y, w, label, recall, color=color))
        body.append(text(610, y + 12, f"{fpr:.4f}", size=11, color="#9a3412", anchor="middle", weight="700"))
        body.append(rect(648, y + 2, min(80, fpr / 0.05 * 80), 12, fill="#ea580c", stroke="#ea580c", rx=2))
        body.append(rect(648, y + 2, 80, 12, fill="none", stroke="#fed7aa", rx=2))
    body.append(text(width / 2, 386, "Profile primitives help audit; structural primitives carry much stronger low-FPR separation.", size=12, weight="700"))
    return write_svg(out_dir / "motivation_token_view_attribution.svg", width, height, body)


def build_memory_pollution(out_dir: Path) -> Path:
    # Values are the Botnet leave-one diagnostic memory-pollution results
    # already reported in the current manuscript.
    points = [(0, 0.9516), (1, 0.7641), (5, 0.3547), (10, 0.3009)]
    width, height = 620, 360
    left, right, top, bottom = 82, 560, 74, 268

    def sx(xval: float) -> float:
        return left + xval / 10.0 * (right - left)

    def sy(yval: float) -> float:
        return bottom - yval / 1.0 * (bottom - top)

    body = [
        text(width / 2, 28, "Ungated memory updates can destroy diagnosis quality", size=18, weight="700"),
        text(width / 2, 51, "Botnet leave-one diagnostic pollution stress; attack labels are used only for this non-deployable stress test.", size=11, color="#5b6678"),
        line(left, bottom, right, bottom, color="#64748b"),
        line(left, top, left, bottom, color="#64748b"),
        text((left + right) / 2, 327, "Attack pollution admitted into benign memory (%)", size=12, weight="700"),
        text(20, (top + bottom) / 2, "AUROC", size=12, anchor="middle", weight="700"),
    ]
    for tick in [0, 2, 4, 6, 8, 10]:
        body.append(line(sx(tick), bottom, sx(tick), bottom + 5, color="#64748b"))
        body.append(text(sx(tick), bottom + 22, str(tick), size=10, color="#475467"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        body.append(line(left - 5, sy(tick), left, sy(tick), color="#64748b"))
        body.append(text(left - 10, sy(tick) + 4, f"{tick:.2f}", size=10, anchor="end", color="#475467"))
        body.append(line(left, sy(tick), right, sy(tick), color="#e2e8f0", width=0.8))
    path = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points))
    body.append(f'<path d="{path}" fill="none" stroke="#dc2626" stroke-width="2.5"/>')
    for xval, yval in points:
        body.append(f'<circle cx="{sx(xval):.1f}" cy="{sy(yval):.1f}" r="5" fill="#dc2626"/>')
        body.append(text(sx(xval), sy(yval) - 10, f"{yval:.4f}", size=10, weight="700", color="#991b1b"))
    body.append(text(width / 2, 300, "This motivates quarantine and reputation-gated memory governance.", size=12, weight="700", color="#111827"))
    return write_svg(out_dir / "motivation_memory_pollution.svg", width, height, body)


def build_online_replay_evt(out_dir: Path) -> Path:
    rows = read_csv(ROOT / "results" / "online_replay_knn" / "online_replay_aggregate.csv")
    rows = sorted(rows, key=lambda r: (r["memory_policy"], r["threshold_mode"]))
    labels = []
    for row in rows:
        policy = "full" if row["memory_policy"] == "full" else "coreset75"
        threshold = "EVT-P99" if row["threshold_mode"] == "evt_p99" else "P99"
        labels.append(f"{policy} {threshold}")
    width, height = 820, 430
    body = [
        text(width / 2, 28, "Artifact-order replay: calibration and memory tradeoffs", size=18, weight="700"),
        text(width / 2, 51, "Replay starts after behavior tokens are available; it is not end-to-end PCAP replay timing.", size=11, color="#5b6678"),
        text(290, 84, "False alerts / 10k benign", size=13, weight="700"),
        text(590, 84, "Throughput (flows/s)", size=13, weight="700"),
    ]
    y0 = 110
    for i, (row, label) in enumerate(zip(rows, labels)):
        y = y0 + i * 62
        alerts = to_float(row["false_alerts_per_10k_benign_mean"])
        recall = to_float(row["attack_recall_online_mean"])
        fps = to_float(row["throughput_flows_per_second_mean"])
        mem = to_float(row["memory_mib_estimate_mean"])
        body.extend(hbar(235, y, 170, label, alerts, max_value=300.0, color="#ea580c"))
        body.extend(hbar(535, y, 160, "", fps, max_value=15000.0, color="#2563eb", suffix=""))
        body.append(text(705, y + 12, f"recall {recall:.4f}, {mem:.2f} MiB", size=10, anchor="start", color="#475467"))
    body.append(text(width / 2, 380, "EVT-P99 lowers false alerts with negligible recall loss; the current coreset lowers memory but is not a speed claim.", size=12, weight="700"))
    return write_svg(out_dir / "motivation_online_replay_evt.svg", width, height, body)


def box(x: float, y: float, w: float, h: float, title: str, lines: list[str], fill: str, stroke: str) -> list[str]:
    body = [rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.4, rx=8), text(x + w / 2, y + 23, title, size=12, weight="700")]
    for i, item in enumerate(lines):
        body.append(text(x + w / 2, y + 47 + i * 16, item, size=9, color="#475467"))
    return body


def build_flowprim_framework(out_dir: Path) -> Path:
    width, height = 1040, 430
    xs = [26 + i * 166 for i in range(6)]
    y, bw, bh = 78, 148, 112
    body = [
        text(width / 2, 28, "FlowPrim Behavior-Only Framework", size=19, weight="700"),
        text(width / 2, 51, "Traffic behavior primitive mining for low-FPR malicious traffic diagnosis", size=12, color="#5b6678"),
    ]
    specs = [
        ("Traffic sources", ["PCAP / Zeek", "packet CSV", "flow CSV fallback"], "#eff6ff", "#2563eb"),
        ("Flow construction", ["bidirectional flows", "label alignment", "audit metadata"], "#ecfeff", "#0891b2"),
        ("Primitive mining", ["profile primitives", "structural motifs", "trigger provenance"], "#f0fdf4", "#16a34a"),
        ("Behavior tokens", ["global stats", "packet/burst tokens", "primitive tokens"], "#fff7ed", "#ea580c"),
        ("Behavior composer", ["Transformer", "supervised heads", "optional embedding"], "#f5f3ff", "#7c3aed"),
        ("Diagnosis heads", ["binary detection", "KNN benign memory", "P99 threshold"], "#fdf2f8", "#db2777"),
    ]
    for x, (title, lines, fill, stroke) in zip(xs, specs):
        body.extend(box(x, y, bw, bh, title, lines, fill, stroke))
    for i in range(5):
        body.append(arrow(xs[i] + bw + 2, y + bh / 2, xs[i + 1] - 5, y + bh / 2))
    body.extend(box(86, 242, 238, 86, "Shortcut control", ["No IP/time/5-tuple", "protocol/service tokens", "metadata only"], "#f8fafc", "#64748b"))
    body.extend(box(370, 242, 238, 86, "Fit-only-on-training rule", ["train-only vocab", "benign memory", "validation threshold"], "#f8fafc", "#64748b"))
    body.extend(box(654, 242, 238, 86, "Analyst-facing record", ["score + threshold", "primitives + tokens", "nearest evidence"], "#f8fafc", "#64748b"))
    body.append(arrow(xs[1] + bw / 2, y + bh + 8, 205, 237))
    body.append(arrow(xs[3] + bw / 2, y + bh + 8, 489, 237))
    body.append(arrow(xs[5] + bw / 2, y + bh + 8, 773, 237))
    body.append(text(width / 2, 374, "Reported low-FPR diagnosis uses behavior evidence: direction, length, timing, burst structure, and primitive triggers.", size=12, color="#475467"))
    body.append(text(width / 2, 395, "Protocol, service, host identifiers, absolute time, and five-tuples are retained only for compatibility and audit metadata.", size=12, color="#475467"))
    return write_svg(out_dir / "fig_flowprim_framework.svg", width, height, body)


def build_primitive_token_example(out_dir: Path) -> Path:
    width, height = 650, 380
    body = [
        text(width / 2, 27, "Primitive Mining and Behavior Tokenization Example", size=17, weight="700"),
        text(width / 2, 50, "Packet direction, length, and timing evidence becomes auditable behavior tokens.", size=12, color="#5b6678"),
    ]
    x0, y0, pkt_w, pkt_h = 62, 82, 66, 44
    packets = [
        ("C->S", "60B", "0ms", "#dbeafe"),
        ("C->S", "60B", "2ms", "#dbeafe"),
        ("C->S", "60B", "3ms", "#dbeafe"),
        ("S->C", "66B", "5ms", "#ecfdf5"),
        ("S->C", "66B", "6ms", "#ecfdf5"),
        ("S->C", "66B", "8ms", "#ecfdf5"),
    ]
    body.append(text(24, y0 + 26, "Packets", size=12, anchor="start", weight="700"))
    for i, (direction, length, iat, fill) in enumerate(packets):
        x = x0 + i * (pkt_w + 12)
        body.append(rect(x, y0, pkt_w, pkt_h, fill=fill, stroke="#2563eb", sw=1.1, rx=6))
        body.append(text(x + pkt_w / 2, y0 + 17, direction, size=10, weight="700"))
        body.append(text(x + pkt_w / 2, y0 + 32, f"{length}, {iat}", size=9, color="#475467"))
    body.append('<path d="M74,148 C150,171 226,171 304,148" stroke="#16a34a" stroke-width="2" fill="none"/>')
    body.append('<path d="M424,148 C472,171 520,171 568,148" stroke="#16a34a" stroke-width="2" fill="none"/>')
    body.append(text(198, 184, "REPEAT: same-direction equal-length run", size=11, weight="700", color="#166534"))
    body.append(text(496, 184, "DUP: adjacent repeated segment", size=11, weight="700", color="#166534"))
    tags = [
        ("Global", ["PKTN=6", "DUR_BIN=short", "DIR_RATIO=1.0"], "#eff6ff", "#2563eb"),
        ("Primitive", ["PRIM_PROFILE_REPEAT", "PRIM_PROFILE_DUP", "PRIM_STRUCT_REQ_RESP"], "#f0fdf4", "#16a34a"),
        ("Packet/Burst", ["D=C2S", "LEN=60", "IAT=small", "BURST=0"], "#fff7ed", "#ea580c"),
    ]
    for i, (title, lines, fill, stroke) in enumerate(tags):
        body.extend(box(62 + i * 180, 218, 158, 90, title, lines, fill, stroke))
    body.append(arrow(325, 198, 325, 211))
    body.append(text(width / 2, 340, "x_f = [CLS, global, primitive, packet/burst tokens]", size=13, weight="700", color="#111827"))
    body.append(text(width / 2, 362, "Audit metadata such as protocol, service, IPs, absolute time, and five-tuples is not emitted as behavior tokens.", size=11, color="#475467"))
    return write_svg(out_dir / "fig_primitive_token_example.svg", width, height, body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FlowPrim motivation and behavior-only figure assets.")
    parser.add_argument("--out-dir", default=str(FIG_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    svg_paths = [
        build_closedset_gap(out_dir),
        build_primitive_heatmap(out_dir),
        build_token_view_attribution(out_dir),
        build_memory_pollution(out_dir),
        build_online_replay_evt(out_dir),
        build_flowprim_framework(out_dir),
        build_primitive_token_example(out_dir),
    ]
    for svg_path in svg_paths:
        convert_pdf(svg_path)
    for svg_path in svg_paths:
        print(svg_path)
        print(svg_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
