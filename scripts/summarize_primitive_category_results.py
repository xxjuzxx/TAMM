#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize primitive category experiment CSV artifacts.")
    parser.add_argument("--input", default="results/primitive_categories")
    parser.add_argument("--output", default="results/primitive_categories/primitive_category_summary.md")
    args = parser.parse_args()
    root = Path(args.input)
    feature = _read_csv(root / "primitive_category_feature_attribution.csv")
    metrics = _read_csv(root / "primitive_category_unknown_metrics.csv")
    coverage = _read_csv(root / "primitive_category_explanation_coverage.csv")
    scale = _read_csv(root / "primitive_category_scale_summary.csv")
    lift = _read_csv(root / "top_structural_primitives_by_lift.csv")
    runtime = _read_csv(root / "primitive_category_runtime.csv")

    by_view = {row.get("feature_view", ""): row for row in feature}
    profile_row = by_view.get("profile_only")
    structural_row = by_view.get("structural_only")
    pb = by_view.get("packet_burst_only")
    pbstruct = by_view.get("packet_burst_plus_structural")

    lines = [
        "# Primitive Category Result Summary",
        "",
        f"Input directory: `{root}`.",
        "",
        "## Feature Attribution",
        "",
        "| View | Runs | AUROC | FPR95 | R@0.1%FPR | R@1%FPR | P99 FPR | False alerts/10k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in feature:
        lines.append(
            "| {view} | {runs} | {auroc} | {fpr95} | {r01} | {r1} | {p99} | {fa} |".format(
                view=row.get("feature_view", ""),
                runs=row.get("runs", ""),
                auroc=_fmt(row.get("auroc")),
                fpr95=_fmt(row.get("fpr95")),
                r01=_fmt(row.get("recall_at_0_1pct_fpr")),
                r1=_fmt(row.get("recall_at_1pct_fpr")),
                p99=_fmt(row.get("val_p99_realized_fpr")),
                fa=_fmt(row.get("false_alerts_per_10k_benign"), 1),
            )
        )

    lines.extend(["", "## Hypothesis Checks", ""])
    if profile_row and structural_row:
        lines.append(
            "- Structural-only vs profile-only: AUROC {struct_a} vs {profile_a}; R@1%FPR {struct_r} vs {profile_r}; P99 FPR {struct_f} vs {profile_f}.".format(
                struct_a=_fmt(structural_row.get("auroc")),
                profile_a=_fmt(profile_row.get("auroc")),
                struct_r=_fmt(structural_row.get("recall_at_1pct_fpr")),
                profile_r=_fmt(profile_row.get("recall_at_1pct_fpr")),
                struct_f=_fmt(structural_row.get("val_p99_realized_fpr")),
                profile_f=_fmt(profile_row.get("val_p99_realized_fpr")),
            )
        )
    if structural_row and pb:
        lines.append(
            "- Gap to packet+burst: AUROC gap {ag}; R@1%FPR gap {rg}.".format(
                ag=_fmt(_num(pb.get("auroc")) - _num(structural_row.get("auroc"))),
                rg=_fmt(_num(pb.get("recall_at_1pct_fpr")) - _num(structural_row.get("recall_at_1pct_fpr"))),
            )
        )
    if pb and pbstruct:
        lines.append(
            "- Adding structural primitives to packet+burst: AUROC delta {ad}; R@1%FPR delta {rd}; P99 FPR delta {fd}.".format(
                ad=_fmt(_num(pbstruct.get("auroc")) - _num(pb.get("auroc"))),
                rd=_fmt(_num(pbstruct.get("recall_at_1pct_fpr")) - _num(pb.get("recall_at_1pct_fpr"))),
                fd=_fmt(_num(pbstruct.get("val_p99_realized_fpr")) - _num(pb.get("val_p99_realized_fpr"))),
            )
        )
    cover_rows = [
        row
        for row in coverage
        if row.get("structural_family", "") == "" and str(row.get("view_uses_structural", "")).lower() == "true"
    ]
    if cover_rows:
        avg_cov = sum(_num(row.get("explanation_coverage")) for row in cover_rows) / len(cover_rows)
        lines.append(f"- Mean alert explanation coverage with at least one structural primitive: {_fmt(avg_cov)}.")

    lines.extend(["", "## Scale And Runtime", ""])
    scale_main = [row for row in scale if row.get("family", "") == "" and row.get("class", "") == ""]
    if scale_main:
        avg_vocab = sum(_num(row.get("primitive_vocab_size")) for row in scale_main) / len(scale_main)
        avg_count = sum(_num(row.get("avg_primitive_count_per_flow")) for row in scale_main) / len(scale_main)
        lines.append(f"- Mean structural primitive vocabulary size across runs: {_fmt(avg_vocab, 1)}.")
        lines.append(f"- Mean structural primitive count per flow: {_fmt(avg_count, 2)}.")
    if runtime:
        lines.append(
            f"- Mean extraction ms/flow: {_fmt(sum(_num(row.get('extraction_ms_per_flow')) for row in runtime) / len(runtime), 4)}."
        )
        lines.append(
            f"- Mean total flow-ready ms/flow: {_fmt(sum(_num(row.get('total_flow_ready_ms_per_flow')) for row in runtime) / len(runtime), 4)}."
        )

    lines.extend(["", "## Top Structural Primitives By Lift", "", "| Primitive | Family | Attack | Seed | Attack rate | Benign rate | Lift |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in lift[:20]:
        lines.append(
            f"| `{row.get('primitive')}` | {row.get('family')} | {row.get('heldout_attack')} | {row.get('seed')} | {_fmt(row.get('attack_rate'))} | {_fmt(row.get('benign_rate'))} | {_fmt(row.get('attack_to_benign_lift'), 2)} |"
        )

    lines.extend(["", "## Artifact Counts", ""])
    lines.append(f"- Per-run metric rows: {len(metrics)}.")
    lines.append(f"- Coverage rows: {len(coverage)}.")
    lines.append(f"- Scale rows: {len(scale)}.")
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
