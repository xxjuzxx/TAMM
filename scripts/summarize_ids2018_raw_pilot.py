#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize IDS2018 PCAP-level FlowPrim pilot artifacts.")
    parser.add_argument("--pilot-dir", default="outputs/ids2018_raw_tuesday_pilot")
    parser.add_argument("--split-dir", default="paper_icdm_applied_2026/experiments/ids2018_raw_tuesday_pilot/unknown")
    parser.add_argument("--token-dir", default="paper_icdm_applied_2026/experiments/ids2018_raw_tuesday_pilot/unknown/tokens_category")
    parser.add_argument("--result-dir", default="results/ids2018_raw_tuesday_pilot/primitive_categories")
    parser.add_argument("--out", default="results/ids2018_raw_tuesday_pilot/RESULT_SUMMARY.md")
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    split_dir = Path(args.split_dir)
    token_dir = Path(args.token_dir)
    result_dir = Path(args.result_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_stats = _read_json(pilot_dir / "label_stats.json")
    dataset_report = _read_json(pilot_dir / "dataset_report.json")
    split_rows = _read_csv(split_dir / "raw_rebuild_split_manifest.csv")
    token_manifest = _read_json(token_dir / "manifest.json")
    metrics = _read_csv(result_dir / "primitive_category_unknown_metrics.csv")
    attribution = _read_csv(result_dir / "primitive_category_feature_attribution.csv")
    runtime = _read_csv(result_dir / "primitive_category_runtime.csv")

    lines = [
        "# IDS2018 Raw Tuesday Pilot Summary",
        "",
        "This summary is generated from saved JSON/CSV artifacts. It reports an IDS2018 Tuesday PCAP-derived pilot using the FlowPrim split-first token/primitive/KNN path.",
        "",
        "## Label Alignment",
        "",
        f"- Total Zeek flows: `{label_stats.get('total_zeek_flows', '-')}`",
        f"- Matched flows: `{label_stats.get('matched_flows', '-')}`",
        f"- Unmatched flows: `{label_stats.get('unmatched_flows', '-')}`",
        f"- Match rate: `{_fmt(label_stats.get('match_rate'))}`",
        f"- Ambiguous rate: `{_fmt(label_stats.get('ambiguous_rate'))}`",
        f"- Offset seconds: `{label_stats.get('ids2018_time_offset_seconds', '-')}`",
        f"- Offset estimator: `{label_stats.get('offset_estimator', label_stats.get('ids2018_time_offset_mode', '-'))}`",
        f"- Match by family: `{json.dumps(label_stats.get('match_by_attack_family', {}), sort_keys=True)}`",
        f"- Labeled JSONL rows: `{_count_jsonl(pilot_dir / 'labeled_flows.jsonl')}`",
        "",
        "## Dataset Report",
        "",
        f"- Attack family counts: `{json.dumps(dataset_report.get('attack_family_counts', {}), sort_keys=True)}`",
        f"- Packet count summary: `{json.dumps(dataset_report.get('packet_count', {}), sort_keys=True)}`",
        "",
        "## Splits",
        "",
    ]
    if split_rows:
        headers = ["seed", "heldout_attack", "train", "val", "test_benign", "test_attack", "test_total"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in split_rows:
            lines.append("| " + " | ".join(str(row.get(key, "")) for key in headers) + " |")
    else:
        lines.append("- Split manifest missing or not generated.")
    lines.extend(
        [
            "",
            "## Token Corpus",
            "",
            f"- Token manifest rows: `{len(token_manifest.get('rows', [])) if token_manifest else 0}`",
            f"- Artifact prefix: `{token_manifest.get('artifact_prefix', '-') if token_manifest else '-'}`",
            "",
            "## Primitive/KNN Metrics",
            "",
        ]
    )
    if attribution:
        headers = [
            "feature_view",
            "runs",
            "auroc",
            "recall_at_1pct_fpr",
            "recall_at_0_1pct_fpr",
            "val_p99_realized_fpr",
            "false_alerts_per_10k_benign",
            "query_ms_per_flow",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in attribution:
            lines.append("| " + " | ".join(_fmt(row.get(key)) if key not in {"feature_view", "runs"} else str(row.get(key, "")) for key in headers) + " |")
    else:
        lines.append("- Primitive category attribution missing or not generated.")
    lines.extend(
        [
            "",
            "## Artifact Counts",
            "",
            f"- Per-run metric rows: `{len(metrics)}`",
            f"- Runtime rows: `{len(runtime)}`",
            "",
            "## Reproduce",
            "",
            "```bash",
            "bash scripts/00_run_zeek.sh \\",
            "  --input outputs/ids2018_raw_tuesday_pilot/pcaps \\",
            "  --out_dir outputs/ids2018_raw_tuesday_pilot/zeek \\",
            "  --ignore_checksums --continue_on_error",
            "",
            "python scripts/label_ids2018_flows.py \\",
            "  --zeek_logs 'outputs/ids2018_raw_tuesday_pilot/zeek/*/conn.log' \\",
            "  --label_csv data/raw/CSE-CIC-IDS2018_organized/processed_csv/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv \\",
            "  --out outputs/ids2018_raw_tuesday_pilot/labeled_flows.jsonl \\",
            "  --unmatched_out outputs/ids2018_raw_tuesday_pilot/unmatched_flows.jsonl \\",
            "  --stats_out outputs/ids2018_raw_tuesday_pilot/label_stats.json \\",
            "  --label_alignment_report outputs/ids2018_raw_tuesday_pilot/label_alignment_report.json \\",
            "  --dataset_report outputs/ids2018_raw_tuesday_pilot/dataset_report.json \\",
            "  --tolerance_seconds 2.0 --streaming-csv-filter --no-auto-time-offset --time-offset-seconds 14400",
            "",
            "python scripts/18_make_raw_rebuild_splits.py \\",
            "  --flow-glob 'data/interim/flows/ids2018/*/raw_ids2018_*_labeled_flows.jsonl' \\",
            "  --output-dir paper_icdm_applied_2026/experiments/ids2018_raw_tuesday_pilot/unknown \\",
            "  --attacks DDoS --seeds 42 43 44 --dataset-name CSE-CIC-IDS2018",
            "",
            "python scripts/19_build_raw_rebuild_token_corpora.py \\",
            "  --flow-glob 'data/interim/flows/ids2018/*/raw_ids2018_*_labeled_flows.jsonl' \\",
            "  --split-dir paper_icdm_applied_2026/experiments/ids2018_raw_tuesday_pilot/unknown \\",
            "  --output paper_icdm_applied_2026/experiments/ids2018_raw_tuesday_pilot/unknown/tokens_category \\",
            "  --attacks DDoS --seeds 42 43 44 --dataset-name CSE-CIC-IDS2018 --artifact-prefix ids2018",
            "",
            "python scripts/run_primitive_category_experiments.py \\",
            "  --output results/ids2018_raw_tuesday_pilot/primitive_categories \\",
            "  --token-dir paper_icdm_applied_2026/experiments/ids2018_raw_tuesday_pilot/unknown/tokens_category \\",
            "  --dataset-name CSE-CIC-IDS2018 --artifact-prefix ids2018 \\",
            "  --attacks DDoS --seeds 42 43 44",
            "```",
            "",
            "## Leakage Controls",
            "",
            "- Raw IPs, absolute timestamps, complete five-tuples, protocol, service, and ports are used only for label joining/audit metadata, not as behavior tokens or memory grouping keys.",
            "- Token vocabulary and structural primitive support filtering are train-only.",
            "- Train and validation splits are benign-only; DDoS labels are used only for offline test metrics.",
            "",
            "## Limitations",
            "",
            "- This is a Tuesday exact-join PCAP pilot, not a full all-day IDS2018 reproduction.",
            "- Public IDS2018 CSV files for most days lack complete five-tuples, so they cannot currently support IDS2017-equivalent exact flow-level joining without additional official metadata.",
            "- Results should not be written into the paper as full IDS2018 evidence unless the remaining days are aligned with equivalent label fidelity.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
