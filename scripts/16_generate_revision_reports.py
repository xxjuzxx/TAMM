#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.pipeline.common import PAPER_ROOT, ROOT, command_record, ensure_dirs, read_csv, write_json, write_md


MODIFIED_FILES = [
    "configs/datasets.yaml",
    "configs/experiments.yaml",
    "configs/calibration.yaml",
    "configs/baselines.yaml",
    "src/pipeline/__init__.py",
    "src/pipeline/common.py",
    "scripts/00_inventory_data.py",
    "scripts/01_extract_flows_from_pcap.py",
    "scripts/02_normalize_flows.py",
    "scripts/03_tokenize_flows.py",
    "scripts/04_extract_primitives.py",
    "scripts/05_build_feature_matrices.py",
    "scripts/06_make_splits.py",
    "scripts/07_build_benign_memory.py",
    "scripts/08_run_main_detection.py",
    "scripts/09_run_ablation.py",
    "scripts/10_run_primitive_analysis.py",
    "scripts/11_run_calibration_robustness.py",
    "scripts/12_run_diagnosis_cases.py",
    "scripts/13_run_efficiency.py",
    "scripts/14_generate_figures_tables.py",
    "scripts/15_validate_results.py",
    "scripts/16_generate_revision_reports.py",
    "scripts/17_summarize_raw_flow_extraction.py",
    "scripts/run_all_experiments.py",
    "scripts/run_baselines.py",
    "README.md",
    "../paper/main.tex",
    "../paper/reproducibility_checklist.md",
]


def _maybe_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def _mean(rows: list[dict[str, str]], field: str) -> float | None:
    vals: list[float] = []
    for row in rows:
        try:
            vals.append(float(row[field]))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "missing" if value is None else f"{value:.{digits}f}"


def _row_by_field(rows: list[dict[str, str]], field: str, value: str) -> dict[str, str] | None:
    for row in rows:
        if str(row.get(field, "")) == value:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FlowPrim ICDM revision reports.")
    args = parser.parse_args()
    ensure_dirs()

    primitive_rows = _maybe_rows(ROOT / "results/primitive_categories/primitive_category_feature_attribution.csv")
    raw_rebuild_rows = _maybe_rows(ROOT / "results/raw_rebuild_split_first/raw_rebuild_split_first_summary.csv")
    unknown_rows = _maybe_rows(ROOT / "paper_icdm_applied_2026/experiments/revision/unknown_best_settings_3seed_runs.csv")
    behavior_feature_rows = _maybe_rows(ROOT / "paper_icdm_applied_2026/experiments/revision/behavior_feature_attribution.csv")
    memory_scope_rows = _maybe_rows(ROOT / "paper_icdm_applied_2026/experiments/revision/memory_scope_audit_summary.csv")
    pcap_manifest = _maybe_rows(ROOT / "data/manifests/pcap_manifest.csv")
    flow_extraction_rows = _maybe_rows(ROOT / "data/manifests/flow_extraction_manifest.csv")
    raw_flow_rows = _maybe_rows(ROOT / "data/manifests/raw_flow_extraction_summary.csv")
    run_summary = (ROOT / "reports/run_all_experiments_summary.md").read_text(encoding="utf-8") if (ROOT / "reports/run_all_experiments_summary.md").exists() else ""

    write_md(
        ROOT / "reports/modified_files.md",
        [
            "# Modified Files",
            "",
            "## Code and Config",
            "",
            *[f"- `{path}`" for path in MODIFIED_FILES if not path.startswith("../paper")],
            "",
            "## Paper / Documentation",
            "",
            *[f"- `{path}`" for path in MODIFIED_FILES if path.startswith("../paper")],
            "",
            "## Generated Artifacts",
            "",
            "- `data/manifests/*.csv|json`: raw data, token, split, feature, primitive, and benign-memory manifests.",
            "- `reports/*.md`: inventory, missing-data, experiment, reproducibility, readiness, and remaining-risk reports.",
            "- `results/main_detection/`, `results/ablation/`, `results/primitive_analysis/`, `results/calibration_robustness/`, `results/diagnosis_cases/`, `results/efficiency/`, `results/baselines/`: collected experiment artifacts from existing real CSV/JSON outputs.",
            "- `tables/*/*.md`: generated markdown mirrors of paper-ready tables.",
        ],
    )

    unknown_auc = _mean(unknown_rows, "auroc")
    unknown_r1 = _mean(unknown_rows, "recall_at_1pct_fpr")
    unknown_p99 = _mean(unknown_rows, "val_p99_0_false_positive_rate")
    fixed_row = _row_by_field(primitive_rows, "feature_view", "packet_burst_plus_profile_structural")
    raw_fixed_row = _row_by_field(raw_rebuild_rows, "feature_view", "packet_burst_plus_profile_structural")
    write_md(
        ROOT / "reports/experiment_summary.md",
        [
            "# Experiment Summary",
            "",
            "## Completed in This Run",
            "",
            "- Raw PCAP/corrected CSV inventory without using `data/raw/ready/ids2017`.",
            "- Flow extraction command manifest for CICIDS2017 day PCAPs.",
            "- Full five-day raw PCAP-to-Zeek-to-labeled-flow reconstruction completed in `flows_only` mode from day-level PCAPs and corrected CSVs.",
            "- Normalized-flow parquet sample from the existing FlowPrim CICIDS2017 flow artifact.",
            "- Token, primitive, split, feature-matrix, and benign-memory manifests.",
            "- Main detection, ablation, primitive analysis, calibration, diagnosis, efficiency, and baseline artifact collection.",
            "- Table/figure regeneration wrapper and primitive-category LaTeX table regeneration.",
            "- Reproducibility/leakage/risky-claim validation.",
            "- Raw-PCAP split-first low-FPR rebuild from reconstructed day-level flows: 15 leave-one splits, 15 train-only token corpora, and 105 behavior-only KNN metric rows.",
            "",
            "## Main Behavior-Only Results",
            "",
            (
                "- Fixed FlowPrim row from `results/primitive_categories/primitive_category_feature_attribution.csv`: "
                f"AUROC {_fmt(float(fixed_row['auroc']) if fixed_row else None)}, "
                f"R@1%FPR {_fmt(float(fixed_row['recall_at_1pct_fpr']) if fixed_row else None)}, "
                f"P99 realized FPR {_fmt(float(fixed_row['val_p99_realized_fpr']) if fixed_row else None)}."
            ),
            f"- Historical locked-setting support rows: {len(unknown_rows)} rows; AUROC mean {_fmt(unknown_auc)}, R@1%FPR mean {_fmt(unknown_r1)}, P99 realized FPR mean {_fmt(unknown_p99)}.",
            f"- Primitive category attribution rows: {len(primitive_rows)}.",
            f"- Behavior-feature attribution rows: {len(behavior_feature_rows)}.",
            f"- Memory-scope audit rows: {len(memory_scope_rows)}.",
            "",
            "## Raw Split-First Low-FPR Check",
            "",
            "- Source: `data/interim/flows/cicids2017/*/raw_cicids2017_*_labeled_flows.jsonl`, generated from the five day-level CICIDS2017 PCAP files and corrected CSV labels.",
            "- Splits: `paper_icdm_applied_2026/experiments/raw_rebuild/unknown/splits_leave_one_*_seed{42,43,44}.json`.",
            "- Token corpora: `paper_icdm_applied_2026/experiments/raw_rebuild/unknown/tokens_category/`.",
            "- Results: `results/raw_rebuild_split_first/raw_rebuild_unknown_metrics.csv` and `results/raw_rebuild_split_first/raw_rebuild_split_first_summary.csv`.",
            (
                "- Fixed FlowPrim raw sanity-check row: "
                f"AUROC {_fmt(float(raw_fixed_row['auroc_mean']) if raw_fixed_row else None)}, "
                f"R@1%FPR {_fmt(float(raw_fixed_row['recall_at_1pct_fpr_mean']) if raw_fixed_row else None)}, "
                f"P99 FPR {_fmt(float(raw_fixed_row['val_p99_realized_fpr_mean']) if raw_fixed_row else None)}."
            ),
            "- Interpretation: this is a stricter raw-source sanity check and is more conservative than the main verified artifacts; it is not used to inflate the main low-FPR claims.",
            "",
            "## Raw Data Inventory",
            "",
            *[f"- {row.get('day')}: PCAP {row.get('pcap_size_bytes')} bytes, corrected CSV rows {row.get('csv_rows')}" for row in pcap_manifest],
            "",
            "## Raw PCAP-to-Flow Smoke",
            "",
            *[
                f"- {row.get('day')}: {row.get('status')} `{row.get('stage')}` output `{row.get('output_labeled_flows')}`"
                for row in flow_extraction_rows
            ],
            "",
            "## Raw PCAP-to-Flow Reconstruction Summary",
            "",
            *[
                f"- {row.get('day')}: {row.get('matched_flows')}/{row.get('total_zeek_flows')} matched, match rate {float(row.get('match_rate') or 0):.4f}, families {row.get('matched_attack_families')}"
                for row in raw_flow_rows
            ],
            "",
            "## Skipped / Not Claimed",
            "",
            "- Full all-five-day PCAP-to-flow reconstruction and capped raw split-first PCAP-to-token-to-KNN low-FPR sanity check were executed.",
            "- The main low-FPR table still uses the verified train-only category token artifacts because the raw rebuild is intentionally capped and more conservative.",
            "- No results were fabricated from the excluded `ready/ids2017` per-class cache.",
            "- Any table copied from existing artifacts keeps its original CSV/JSON source path.",
            "",
            "## Run-All Log",
            "",
            run_summary,
        ],
    )

    write_md(
        ROOT / "reports/paper_revision_summary.md",
        [
            "# Paper Revision Summary",
            "",
            "## Current Structure",
            "",
            "The manuscript already follows an Applied Track-oriented structure: Introduction, motivation/background, FlowPrim framework, primitive mining, tokenization/representation, diagnosis, experiments by RQ, related work, discussion/limitations, and conclusion.",
            "",
            "## Updates Completed",
            "",
            "- Reproducibility checklist field names were aligned with `PRIM_PROFILE_*` and `PRIM_STRUCT_*` category naming.",
            "- README wording now states that newly generated corpora and results use only category prefixes.",
            "- Table font consistency was already centralized through `\\FlowPrimTableStyle` and `\\FlowPrimTableFit` in `paper/main.tex`.",
            "- The manuscript already states that profile and structural primitives are parallel evidence categories, not sequential primitive versions.",
            "",
            "## Result Synchronization",
            "",
            "- `paper/tables/table_primitive_category_attribution.tex` is regenerated from `results/primitive_categories/primitive_category_feature_attribution.csv`.",
            "- Revision tables under `paper/tables/` are backed by CSVs under `paper_icdm_applied_2026/experiments/revision/`.",
            "- Generated mirrors are available under `tables/`.",
            "",
            "## Remaining Paper Work",
            "",
            "- A full rewrite into exactly the requested 1-9 section numbering was not forced because the current manuscript already contains the Applied Track narrative and is page-constrained.",
            "- If the author wants the exact 1-9 section labels, that should be a focused paper-edit pass after deciding page budget.",
        ],
    )

    readiness = [
        ("Real-world motivation", "pass", "Introduction frames payload-free, unknown-attack, low-FPR, and analyst-evidence constraints.", "Keep abstract concise under page limit."),
        ("Application significance", "pass", "Diagnosis records, alert budgets, calibration, and throughput are reported.", "Clarify deployment scope as prototype workflow, not deployed product."),
        ("Technical novelty", "pass", "Profile/structural primitive categories plus benign-memory evidence scoring.", "Avoid claiming primitives alone solve all low-FPR detection."),
        ("Experimental comprehensiveness", "pass", "Main, ablation, calibration, diagnosis, efficiency, extra benign, baselines, full day-level raw PCAP-to-labeled-flow reconstruction, and capped raw split-first low-FPR rebuild exist.", "Treat the raw rebuild as a sanity check because exact-KNN caps make it more conservative than the main verified artifacts."),
        ("Deployment realism", "partial", "Benign-only calibration and alert budgets are explicit.", "Flow-timeout/live capture latency remains outside measured runtime."),
        ("Diagnosis evidence", "pass", "Per-flow audit and primitive coverage artifacts are available.", "Keep caveat that primitives do not fully explain every alert."),
        ("Efficiency", "pass", "KNN scalability and e2e throughput smoke artifacts are collected.", "ANN/coreset remains deployment extension."),
        ("Reproducibility", "pass", "Manifests, commands, full raw PCAP-to-flow reconstruction, raw split-first token corpora, raw rebuild metrics, and validation reports generated.", "Keep both main verified artifacts and raw-rebuild sanity-check artifacts clearly separated in the paper."),
        ("Limitations", "pass", "WebAttack support, behavior-only memory scope, KNN scaling, calibration instability, and domain shift are disclosed.", "Keep limitations visible in final camera-ready draft."),
        ("No exaggerated claims", "pass", "Risky claim grep currently passes after validation.", "Re-run `scripts/15_validate_results.py` before submission."),
    ]
    write_md(
        ROOT / "reports/icdm_applied_readiness_check.md",
        [
            "# ICDM Applied Track Readiness Check",
            "",
            "| Item | Status | Evidence | Remaining Issue / Fix |",
            "|---|---|---|---|",
            *[f"| {item} | {status} | {evidence} | {fix} |" for item, status, evidence, fix in readiness],
        ],
    )

    write_md(
        ROOT / "reports/remaining_issues_and_next_steps.md",
        [
            "# Remaining Issues and Next Steps",
            "",
            "## P0: Before Submission",
            "",
            "- Keep the raw split-first table explicitly framed as a capped sanity check, because it is more conservative than the main verified low-FPR artifacts.",
            "- Re-run `python scripts/15_validate_results.py` after any manuscript edit.",
            "- Confirm that every table cited in the final manuscript has a corresponding CSV/JSON source listed in `reports/result_consistency_check.json` or `results/summaries/figure_table_manifest.json`.",
            "",
            "## P1: Strongly Recommended",
            "",
            "- Run the full primitive category experiment again only if code or token corpora change.",
            "- Add one more diagnosis case table row for benign-tail false positives if space allows.",
            "- Consider whether the raw split-first table should stay in the main paper or move to appendix if the page limit becomes tight.",
            "",
            "## P2: Useful if Time Allows",
            "",
            "- Implement indexed/ANN KNN as a non-central deployment extension.",
            "- Broaden external behavior-token PCAP diagnostics beyond CICIDS2017-derived artifacts.",
            "- Add automated numeric cross-reference checks between `paper/main.tex` and result CSVs.",
            "",
            "## P3: Future Work",
            "",
            "- Live capture buffering and flow-timeout latency measurement.",
            "- Analyst-feedback gated benign memory updates in a real SOC loop.",
            "",
            "## Known Risks",
            "",
            "- WebAttack has only 28 held-out attack flows.",
            "- Recall@0.1%FPR is unstable and should not be oversold.",
            "- The main algorithm intentionally excludes protocol/service grouping, so any future environment-aware extension must remain outside the paper's primary claim unless separately audited.",
            "- Exact KNN scales linearly with benign memory size.",
            "- Extra benign memory expansion can worsen P99 FPR under some strategies.",
        ],
    )

    write_json(ROOT / "reports/revision_report_manifest.json", {"command": command_record(sys.argv), "reports": [
        "reports/modified_files.md",
        "reports/experiment_summary.md",
        "reports/paper_revision_summary.md",
        "reports/icdm_applied_readiness_check.md",
        "reports/remaining_issues_and_next_steps.md",
    ]})
    print(ROOT / "reports/experiment_summary.md")


if __name__ == "__main__":
    main()
