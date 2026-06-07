from __future__ import annotations

from typing import Any

from src.data.cicids2017_labeler import label_flows
from src.data.dataset_adapter import dataset_report, label_alignment_report, normalize_flows


def adapt_labeled_flows(flows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = normalize_flows(flows, dataset="CICIDS2017")
    return normalized, dataset_report(normalized, dataset="CICIDS2017")


def align_and_adapt_cicids2017(
    flows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    *,
    tolerance_seconds: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    labeled, unmatched, stats = label_flows(flows, label_rows, tolerance_seconds=tolerance_seconds)
    normalized = normalize_flows(labeled, dataset="CICIDS2017")
    time_deltas = [
        float(row.get("label_time_delta", 0.0))
        for row in normalized
        if row.get("label_time_delta") is not None
    ]
    alignment = label_alignment_report(
        dataset="CICIDS2017",
        total_zeek_flows=len(flows),
        matched_flows=len(normalized),
        unmatched_flows=len(unmatched),
        ambiguous_matches=int(stats.get("ambiguous_matches", 0)),
        time_deltas=time_deltas,
        matched_flows_rows=normalized,
        dropped_reason_counts=stats.get("dropped_reason_counts", {}),
        notes=[
            "Alignment fields are for labeling/reporting only and must not enter model tokens.",
            f"tolerance_seconds={tolerance_seconds}",
        ],
    )
    return normalized, unmatched, alignment, dataset_report(normalized, dataset="CICIDS2017")
