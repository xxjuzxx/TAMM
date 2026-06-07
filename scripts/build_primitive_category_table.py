#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "primitive_categories" / "primitive_category_feature_attribution.csv"
DEFAULT_OUTPUT = ROOT.parent / "paper" / "tables" / "table_primitive_category_attribution.tex"

VIEW_LABELS = {
    "profile_only": ("Profile only", r"\texttt{PRIM\_PROFILE\_*}"),
    "structural_only": ("Structural only", r"\texttt{PRIM\_STRUCT\_*}"),
    "profile_plus_structural": ("Profile + structural", "profile + structural primitives"),
    "packet_burst_only": ("Packet+burst", "packet/burst tokens"),
    "packet_burst_plus_profile": ("Packet+burst + profile", r"packet/burst + \texttt{PRIM\_PROFILE\_*}"),
    "packet_burst_plus_structural": ("Packet+burst + structural", r"packet/burst + \texttt{PRIM\_STRUCT\_*}"),
    "packet_burst_plus_profile_structural": ("Packet+burst + both", "packet/burst + both primitive categories"),
}

ORDER = [
    "profile_only",
    "structural_only",
    "profile_plus_structural",
    "packet_burst_only",
    "packet_burst_plus_profile",
    "packet_burst_plus_structural",
    "packet_burst_plus_profile_structural",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _fmt(value: Any, std: Any | None = None, digits: int = 4) -> str:
    mean = _num(value)
    if mean is None:
        return "-"
    sigma = _num(std)
    if sigma is None:
        return f"{mean:.{digits}f}"
    return rf"${mean:.{digits}f}\pm{sigma:.{digits}f}$"


def build(rows: list[dict[str, str]]) -> str:
    by_view = {row["feature_view"]: row for row in rows}
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"View & Evidence & AUROC & FPR95 & R@0.1\%FPR & R@1\%FPR & P99 FPR \\",
        r"\midrule",
    ]
    for view in ORDER:
        row = by_view.get(view)
        if row is None:
            continue
        label, evidence = VIEW_LABELS[view]
        lines.append(
            " & ".join(
                [
                    label,
                    evidence,
                    _fmt(row.get("auroc"), row.get("auroc_std")),
                    _fmt(row.get("fpr95"), row.get("fpr95_std")),
                    _fmt(row.get("recall_at_0_1pct_fpr"), row.get("recall_at_0_1pct_fpr_std")),
                    _fmt(row.get("recall_at_1pct_fpr"), row.get("recall_at_1pct_fpr_std")),
                    _fmt(row.get("val_p99_realized_fpr"), row.get("val_p99_realized_fpr_std")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the profile/structural primitive category attribution LaTeX table.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    rows = _read_csv(Path(args.input))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(rows), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
