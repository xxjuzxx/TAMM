#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experiments" / "ccfa_baseline_retests" / "summary"
PAPER_TABLE_DIR = ROOT / "paper" / "flowprim_motif_memory_icdm" / "tables"


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return float(statistics.fmean(clean)), float(statistics.stdev(clean))


def fmt_pm(mean: Any, std: Any, digits: int = 4) -> str:
    try:
        m = float(mean)
        s = float(std)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(m):
        return "--"
    if not math.isfinite(s):
        s = 0.0
    return f"${m:.{digits}f}\\pm{s:.{digits}f}$"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def add_row(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    method: str,
    unit: str,
    runs: int,
    auroc_mean: float,
    auroc_std: float,
    fpr95_mean: float,
    fpr95_std: float,
    r001_mean: float,
    r001_std: float,
    r01_mean: float,
    r01_std: float,
    p99_mean: float | None = None,
    p99_std: float | None = None,
    status: str,
    table_tier: str,
) -> None:
    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "eval_unit": unit,
            "runs": int(runs),
            "auroc_mean": auroc_mean,
            "auroc_std": auroc_std,
            "fpr95_mean": fpr95_mean,
            "fpr95_std": fpr95_std,
            "recall_at_0.001_fpr_mean": r001_mean,
            "recall_at_0.001_fpr_std": r001_std,
            "recall_at_0.01_fpr_mean": r01_mean,
            "recall_at_0.01_fpr_std": r01_std,
            "val_p99_realized_fpr_mean": p99_mean,
            "val_p99_realized_fpr_std": p99_std,
            "official_reproduction_status": status,
            "table_tier": table_tier,
        }
    )


def tamm_row(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "TAMM repository" / "results" / "primitive_categories" / "primitive_category_unknown_metrics.csv"
    df = pd.read_csv(path)
    df = df[
        (df["feature_view"] == "packet_burst_plus_profile_structural")
        & (df.get("memory_scope", "global") == "global")
    ].copy()
    add_row(
        rows,
        dataset="CICIDS2017",
        method="TAMM fixed",
        unit="flow",
        runs=len(df),
        auroc_mean=mean_std(df["auroc"].tolist())[0],
        auroc_std=mean_std(df["auroc"].tolist())[1],
        fpr95_mean=mean_std(df["fpr95"].tolist())[0],
        fpr95_std=mean_std(df["fpr95"].tolist())[1],
        r001_mean=mean_std(df["recall_at_0_1pct_fpr"].tolist())[0],
        r001_std=mean_std(df["recall_at_0_1pct_fpr"].tolist())[1],
        r01_mean=mean_std(df["recall_at_1pct_fpr"].tolist())[0],
        r01_std=mean_std(df["recall_at_1pct_fpr"].tolist())[1],
        p99_mean=mean_std(df["val_p99_realized_fpr"].tolist())[0],
        p99_std=mean_std(df["val_p99_realized_fpr"].tolist())[1],
        status="native_tamm_protocol",
        table_tier="main",
    )


def aggregate_csv_row(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    dataset: str,
    methods: list[str] | None = None,
    table_tier: str,
    status: str | None = None,
) -> None:
    df = pd.read_csv(path)
    for _, item in df[df["heldout_attack"] == "Aggregate"].iterrows():
        method = str(item["method"])
        if methods is not None and method not in methods:
            continue
        add_row(
            rows,
            dataset=dataset,
            method=method,
            unit=str(item["eval_unit"]),
            runs=int(item["runs"]),
            auroc_mean=float(item["auroc_mean"]),
            auroc_std=float(item["auroc_std"]),
            fpr95_mean=float(item["fpr95_mean"]),
            fpr95_std=float(item["fpr95_std"]),
            r001_mean=float(item["recall_at_0.001_fpr_mean"]),
            r001_std=float(item["recall_at_0.001_fpr_std"]),
            r01_mean=float(item["recall_at_0.01_fpr_mean"]),
            r01_std=float(item["recall_at_0.01_fpr_std"]),
            p99_mean=float(item["val_p99_realized_fpr_mean"]) if "val_p99_realized_fpr_mean" in item and pd.notna(item["val_p99_realized_fpr_mean"]) else None,
            p99_std=float(item["val_p99_realized_fpr_std"]) if "val_p99_realized_fpr_std" in item and pd.notna(item["val_p99_realized_fpr_std"]) else None,
            status=status or str(item.get("official_reproduction_status", "adapted")),
            table_tier=table_tier,
        )


def write_latex(rows: list[dict[str, Any]], path: Path, *, dataset: str, tiers: set[str], caption_comment: str | None = None) -> None:
    selected = [row for row in rows if row["dataset"] == dataset and row["table_tier"] in tiers]
    tier_rank = {"main": 0, "strong_adapted": 1, "adapted_appendix": 2}
    selected.sort(key=lambda row: (tier_rank.get(str(row["table_tier"]), 99), str(row["method"])))
    lines = []
    if caption_comment:
        lines.append(f"% {caption_comment}")
    lines.extend(
        [
            "\\begin{tabular}{llccccc}",
            "\\toprule",
            "Method & Unit & Runs & AUROC & FPR95 & R@0.1\\%FPR & R@1\\%FPR \\\\",
            "\\midrule",
        ]
    )
    last_tier: str | None = None
    for row in selected:
        tier = str(row["table_tier"])
        if last_tier is not None and tier != last_tier:
            lines.append("\\midrule")
        last_tier = tier
        lines.append(
            f"{row['method']} & {row['eval_unit']} & {row['runs']} & "
            f"{fmt_pm(row['auroc_mean'], row['auroc_std'])} & "
            f"{fmt_pm(row['fpr95_mean'], row['fpr95_std'])} & "
            f"{fmt_pm(row['recall_at_0.001_fpr_mean'], row['recall_at_0.001_fpr_std'])} & "
            f"{fmt_pm(row['recall_at_0.01_fpr_mean'], row['recall_at_0.01_fpr_std'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def short_method_name(method: str) -> str:
    mapping = {
        "Kitsune flow AE ensemble": "Kitsune flow AE",
        "BSTS-Net embedding-KNN": "BSTS emb.-KNN",
        "BSTS-Net relation-window distance": "BSTS rel.-window",
        "TrafficFormer-style token representation KNN": "TrafficFormer-style",
        "Trident-style open-world prototypes": "Trident-style",
        "CADE-style contrastive AE": "CADE-style",
        "ContraMTD-style local-global contrastive": "ContraMTD-style",
        "HyperVision-style interaction graph KNN": "HyperVision-style",
        "RAPIER-style robust detector": "RAPIER-style",
        "Whisper-style frequency detector": "Whisper-style",
    }
    return mapping.get(method, method)


def metric_from_tamm(metric: str) -> pd.DataFrame:
    path = ROOT / "TAMM repository" / "results" / "primitive_categories" / "primitive_category_unknown_metrics.csv"
    df = pd.read_csv(path)
    df = df[
        (df["feature_view"] == "packet_burst_plus_profile_structural")
        & (df.get("memory_scope", "global") == "global")
    ].copy()
    rows: list[dict[str, Any]] = []
    for attack, group in df.groupby("heldout_attack", sort=False):
        mean, std = mean_std(group[metric].tolist())
        rows.append({"method": "TAMM fixed", "heldout_attack": attack, "runs": len(group), "mean": mean, "std": std})
    mean, std = mean_std(df[metric].tolist())
    rows.append({"method": "TAMM fixed", "heldout_attack": "Aggregate", "runs": len(df), "mean": mean, "std": std})
    return pd.DataFrame(rows)


def metric_from_aggregate_csv(path: Path, methods: list[str], source_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method"].isin(methods)].copy()
    return pd.DataFrame(
        {
            "method": df["method"],
            "heldout_attack": df["heldout_attack"],
            "runs": df["runs"],
            "mean": df[f"{source_col}_mean"],
            "std": df[f"{source_col}_std"],
        }
    )


def write_family_metric_table(
    frames: list[pd.DataFrame],
    path: Path,
    *,
    method_groups: list[list[str]],
    attack_order: list[str],
    metric_label: str,
    caption_comment: str | None = None,
) -> None:
    df = pd.concat(frames, ignore_index=True)
    row_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for _, item in df.iterrows():
        row_lookup[(str(item["method"]), str(item["heldout_attack"]))] = (float(item["mean"]), float(item["std"]))

    header_attacks = attack_order + ["Aggregate"]
    colspec = "l" + "c" * len(header_attacks)
    lines = []
    if caption_comment:
        lines.append(f"% {caption_comment}")
    lines.extend(
        [
            f"\\begin{{tabular}}{{{colspec}}}",
            "\\toprule",
            "Method & " + " & ".join(header_attacks) + " \\\\",
            "\\midrule",
        ]
    )
    for group_index, methods in enumerate(method_groups):
        if group_index > 0:
            lines.append("\\midrule")
        for method in methods:
            cells = []
            for attack in header_attacks:
                mean, std = row_lookup.get((method, attack), (float("nan"), float("nan")))
                cells.append(fmt_pm(mean, std))
            lines.append(f"{short_method_name(method)} & " + " & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"% Cell values are {metric_label}.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reproduction_notes(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Baseline Reproduction Status Notes",
        "",
        "These notes are intended to be mirrored in the paper text/table captions.",
        "",
        "| Method | Dataset | Status | Interpretation |",
        "|---|---|---|---|",
    ]
    interpretation = {
        "native_tamm_protocol": "Native TAMM/TMDD result under the paper's fixed leave-one-unknown protocol.",
        "tamm_protocol_flow_artifact_adapted": "Kitsune-style autoencoder ensemble adapted to local flow artifacts; not the official packet-level KitNET/AfterImage pipeline.",
        "tamm_protocol_adapted_core": "BSTS-Net core architecture/window idea adapted to local FlowPrim artifacts and TAMM splits; not an official full reproduction.",
        "tamm_protocol_adapted_no_official_code_found": "Algorithmic proxy of the paper idea under TAMM splits; no verified official code was available in this workspace at run time.",
        "tamm_protocol_adapted_endpoint_graph": "Endpoint/flow-interaction graph scoring adapted to TAMM flow artifacts.",
        "tamm_protocol_adapted_contrastive": "Local-global contrastive anomaly scoring adapted to benign-only TAMM training.",
        "tamm_protocol_adapted_benign_only": "Contrastive autoencoder adapted to benign-only TAMM training rather than the original known-class protocol.",
        "tamm_protocol_adapted_noisy_label_robust": "Robust/noisy-label detector idea adapted to benign-only anomaly scoring.",
        "tamm_protocol_adapted_frequency_features": "Frequency-domain scoring adapted to flow packet sketches.",
        "tamm_protocol_adapted_representation_proxy": "Representation baseline using local TAMM behavior-token artifacts; not the official pre-trained TrafficFormer model.",
    }
    seen = set()
    for row in sorted(rows, key=lambda r: (str(r["method"]), str(r["dataset"]))):
        key = (row["method"], row["dataset"], row["official_reproduction_status"])
        if key in seen:
            continue
        seen.add(key)
        status = str(row["official_reproduction_status"])
        lines.append(f"| {row['method']} | {row['dataset']} | `{status}` | {interpretation.get(status, 'Adapted same-protocol retest.')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    tamm_row(rows)
    aggregate_csv_row(
        rows,
        ROOT / "experiments" / "kitsune_baseline" / "tamm_protocol" / "kitsune_tamm_protocol_aggregate.csv",
        dataset="CICIDS2017",
        table_tier="main",
        status="tamm_protocol_flow_artifact_adapted",
    )
    aggregate_csv_row(
        rows,
        ROOT / "experiments" / "bsts_net_reproduction" / "tamm_protocol" / "bsts_tamm_protocol_aggregate.csv",
        dataset="CICIDS2017",
        methods=["BSTS-Net embedding-KNN", "BSTS-Net relation-window distance", "BSTS-Net native entropy rule"],
        table_tier="main",
        status="tamm_protocol_adapted_core",
    )
    aggregate_csv_row(
        rows,
        ROOT / "experiments" / "ccfa_baseline_retests" / "results_ids2017" / "ccfa_adapted_retests_aggregate.csv",
        dataset="CICIDS2017",
        methods=[
            "TrafficFormer-style token representation KNN",
            "Trident-style open-world prototypes",
            "CADE-style contrastive AE",
            "ContraMTD-style local-global contrastive",
            "HyperVision-style interaction graph KNN",
        ],
        table_tier="strong_adapted",
    )
    aggregate_csv_row(
        rows,
        ROOT / "experiments" / "ccfa_baseline_retests" / "results_ids2017" / "ccfa_adapted_retests_aggregate.csv",
        dataset="CICIDS2017",
        methods=["RAPIER-style robust detector", "Whisper-style frequency detector"],
        table_tier="adapted_appendix",
    )
    aggregate_csv_row(
        rows,
        ROOT / "experiments" / "ccfa_baseline_retests" / "results_ids2018" / "ccfa_adapted_retests_aggregate.csv",
        dataset="CSE-CIC-IDS2018",
        table_tier="adapted_appendix",
    )

    ids2017_public_methods = [
        "Kitsune flow AE ensemble",
        "BSTS-Net embedding-KNN",
        "BSTS-Net relation-window distance",
    ]
    ids2017_ccfa_methods = [
        "TrafficFormer-style token representation KNN",
        "Trident-style open-world prototypes",
        "CADE-style contrastive AE",
        "ContraMTD-style local-global contrastive",
        "HyperVision-style interaction graph KNN",
    ]
    ids2017_appendix_methods = [
        "RAPIER-style robust detector",
        "Whisper-style frequency detector",
    ]
    ids2018_ccfa_methods = [
        "TrafficFormer-style token representation KNN",
        "Trident-style open-world prototypes",
        "CADE-style contrastive AE",
        "ContraMTD-style local-global contrastive",
        "HyperVision-style interaction graph KNN",
        "RAPIER-style robust detector",
        "Whisper-style frequency detector",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUT_DIR / "paper_ready_baseline_comparison.csv")
    write_latex(
        rows,
        OUT_DIR / "table_ids2017_main_with_ccfa_adapted.tex",
        dataset="CICIDS2017",
        tiers={"main", "strong_adapted"},
        caption_comment="Rows after the second midrule are same-protocol adapted retests, not official full reproductions.",
    )
    write_latex(
        rows,
        OUT_DIR / "table_ids2017_appendix_all_adapted.tex",
        dataset="CICIDS2017",
        tiers={"main", "strong_adapted", "adapted_appendix"},
        caption_comment="Same-protocol IDS2017 comparison including appendix adapted baselines.",
    )
    write_latex(
        rows,
        OUT_DIR / "table_ids2018_external_adapted.tex",
        dataset="CSE-CIC-IDS2018",
        tiers={"adapted_appendix"},
        caption_comment="IDS2018 official-victim external split-first adapted retests; one seed available.",
    )
    write_family_metric_table(
        [
            metric_from_tamm("recall_at_1pct_fpr"),
            metric_from_aggregate_csv(
                ROOT / "experiments" / "kitsune_baseline" / "tamm_protocol" / "kitsune_tamm_protocol_aggregate.csv",
                ["Kitsune flow AE ensemble"],
                "recall_at_0.01_fpr",
            ),
            metric_from_aggregate_csv(
                ROOT / "experiments" / "bsts_net_reproduction" / "tamm_protocol" / "bsts_tamm_protocol_aggregate.csv",
                ["BSTS-Net embedding-KNN", "BSTS-Net relation-window distance"],
                "recall_at_0.01_fpr",
            ),
            metric_from_aggregate_csv(
                ROOT / "experiments" / "ccfa_baseline_retests" / "results_ids2017" / "ccfa_adapted_retests_aggregate.csv",
                ids2017_ccfa_methods + ids2017_appendix_methods,
                "recall_at_0.01_fpr",
            ),
        ],
        OUT_DIR / "table_ids2017_r1_family_breakdown_adapted.tex",
        method_groups=[
            ["TAMM fixed"] + ids2017_public_methods,
            ids2017_ccfa_methods,
            ids2017_appendix_methods,
        ],
        attack_order=["Botnet", "DDoS", "Probe", "WebAttack", "BruteForce"],
        metric_label="Recall@1\\%FPR",
        caption_comment="CICIDS2017 family-level Recall@1%FPR; adapted-style rows are not official full reproductions.",
    )
    write_family_metric_table(
        [
            metric_from_aggregate_csv(
                ROOT / "experiments" / "ccfa_baseline_retests" / "results_ids2018" / "ccfa_adapted_retests_aggregate.csv",
                ids2018_ccfa_methods,
                "recall_at_0.01_fpr",
            ),
        ],
        OUT_DIR / "table_ids2018_r1_family_breakdown_adapted.tex",
        method_groups=[ids2018_ccfa_methods],
        attack_order=["Botnet", "BruteForce", "DDoS", "DoS", "Infiltration", "WebAttack"],
        metric_label="Recall@1\\%FPR",
        caption_comment="CSE-CIC-IDS2018 official-victim family-level Recall@1%FPR; one seed available.",
    )
    write_reproduction_notes(rows, OUT_DIR / "baseline_reproduction_status_notes.md")

    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "table_ids2017_main_with_ccfa_adapted.tex",
        "table_ids2017_appendix_all_adapted.tex",
        "table_ids2018_external_adapted.tex",
        "table_ids2017_r1_family_breakdown_adapted.tex",
        "table_ids2018_r1_family_breakdown_adapted.tex",
    ]:
        (PAPER_TABLE_DIR / name).write_text((OUT_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
