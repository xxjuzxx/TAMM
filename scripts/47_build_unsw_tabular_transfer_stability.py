#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import write_json


DEFAULT_SETTINGS = [
    "zero_shot_cicids2017_to_unsw",
    "zero_shot_cicids2017_to_unsw_train_threshold",
    "unsw_train_to_unsw_test_sanity",
    "unsw_train_to_unsw_test_sanity_cal_macro",
    "unsw_train_to_unsw_test_sanity_cal_fpr05",
    "few_shot_unsw_01pct_warm_start",
    "few_shot_unsw_01pct_scratch",
    "few_shot_unsw_01pct_warm_start_cal_macro",
    "few_shot_unsw_01pct_warm_start_cal_fpr05",
    "few_shot_unsw_01pct_scratch_cal_macro",
    "few_shot_unsw_01pct_scratch_cal_fpr05",
    "few_shot_unsw_05pct_warm_start",
    "few_shot_unsw_05pct_scratch",
    "few_shot_unsw_05pct_warm_start_cal_macro",
    "few_shot_unsw_05pct_warm_start_cal_fpr05",
    "few_shot_unsw_05pct_scratch_cal_macro",
    "few_shot_unsw_05pct_scratch_cal_fpr05",
    "few_shot_unsw_10pct_warm_start",
    "few_shot_unsw_10pct_scratch",
    "few_shot_unsw_10pct_warm_start_cal_macro",
    "few_shot_unsw_10pct_warm_start_cal_fpr05",
    "few_shot_unsw_10pct_scratch_cal_macro",
    "few_shot_unsw_10pct_scratch_cal_fpr05",
]

METRIC_FIELDS = ["accuracy", "macro_f1", "weighted_f1", "auroc", "auprc", "fpr95"]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _seed_from_dir(path: Path) -> int:
    match = re.search(r"seed(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return 42


def _run_payload(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    data = _load_json(summary_path)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise TypeError(f"invalid rows payload in {summary_path}")
    return {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "seed": _seed_from_dir(run_dir),
        "rows_by_setting": {
            str(row["setting"]): row
            for row in rows
            if isinstance(row, dict) and row.get("setting")
        },
    }


def _aggregate_setting(setting: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_seed_rows: list[dict[str, Any]] = []
    for run in runs:
        row = run["rows_by_setting"].get(setting)
        if row is None:
            raise KeyError(f"missing setting {setting!r} in {run['summary_path']}")
        per_seed_rows.append(
            {
                "seed": int(run["seed"]),
                "setting": setting,
                "run_dir": run["run_dir"],
                "summary_path": run["summary_path"],
                "path": row.get("path"),
                "threshold": row.get("threshold"),
                "accuracy": row.get("accuracy"),
                "macro_f1": row.get("macro_f1"),
                "weighted_f1": row.get("weighted_f1"),
                "auroc": row.get("auroc"),
                "auprc": row.get("auprc"),
                "fpr95": row.get("fpr95"),
                "num_train": row.get("num_train"),
                "num_test": row.get("num_test"),
                "num_threshold_calibration": row.get("num_threshold_calibration"),
                "num_few_shot_train": row.get("num_few_shot_train"),
                "num_few_shot_calibration": row.get("num_few_shot_calibration"),
            }
        )

    aggregate: dict[str, Any] = {
        "setting": setting,
        "num_runs": len(per_seed_rows),
        "seeds": [row["seed"] for row in per_seed_rows],
        "per_seed_rows": per_seed_rows,
    }
    for field in METRIC_FIELDS + ["threshold"]:
        values = [row[field] for row in per_seed_rows if row.get(field) is not None]
        aggregate[field] = _mean_std([float(value) for value in values]) if values else None

    for field in ["num_train", "num_test", "num_threshold_calibration", "num_few_shot_train", "num_few_shot_calibration"]:
        values = [row[field] for row in per_seed_rows if row.get(field) is not None]
        if values and len({int(value) for value in values}) == 1:
            aggregate[field] = int(values[0])
        elif values:
            aggregate[field] = [int(value) for value in values]
        else:
            aggregate[field] = None

    return aggregate


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _fmt_pm(stats: dict[str, float] | None, digits: int = 4) -> str:
    if stats is None:
        return "-"
    return f"{stats['mean']:.{digits}f} +/- {stats['std']:.{digits}f}"


def _render_markdown(aggregates: list[dict[str, Any]], runs: list[dict[str, Any]], out_dir: Path) -> str:
    lines = [
        "# D3 UNSW-NB15 seed stability",
        "",
        "## Protocol",
        "",
        f"- Run dirs: {', '.join(run['run_dir'] for run in runs)}",
        "- Shared tabular semantics adapter, CICIDS2017 -> UNSW-NB15.",
        "- Seeds are derived from the run directory suffix when present; the base run directory is treated as seed 42.",
        "",
        "## Selected settings",
        "",
        "| Setting | Macro-F1 mean +/- std | Accuracy mean +/- std | Weighted-F1 mean +/- std | AUROC mean +/- std | AUPRC mean +/- std | FPR95 mean +/- std | Seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregates, key=lambda item: item["macro_f1"]["mean"] if item.get("macro_f1") else float("-inf"), reverse=True):
        lines.append(
            "| {setting} | {macro} | {acc} | {weighted} | {auroc} | {auprc} | {fpr95} | {seeds} |".format(
                setting=row["setting"],
                macro=_fmt_pm(row.get("macro_f1")),
                acc=_fmt_pm(row.get("accuracy")),
                weighted=_fmt_pm(row.get("weighted_f1")),
                auroc=_fmt_pm(row.get("auroc")),
                auprc=_fmt_pm(row.get("auprc")),
                fpr95=_fmt_pm(row.get("fpr95")),
                seeds=", ".join(str(seed) for seed in row["seeds"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-seed rows",
            "",
            "| Seed | Setting | Accuracy | Macro-F1 | Weighted-F1 | AUROC | AUPRC | FPR95 | Threshold |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted((seed_row for aggregate in aggregates for seed_row in aggregate["per_seed_rows"]), key=lambda item: (item["setting"], item["seed"])):
        lines.append(
            "| {seed} | {setting} | {acc} | {macro} | {weighted} | {auroc} | {auprc} | {fpr95} | {threshold} |".format(
                seed=row["seed"],
                setting=row.get("setting", "-"),
                acc=_fmt(row.get("accuracy")),
                macro=_fmt(row.get("macro_f1")),
                weighted=_fmt(row.get("weighted_f1")),
                auroc=_fmt(row.get("auroc")),
                auprc=_fmt(row.get("auprc")),
                fpr95=_fmt(row.get("fpr95")),
                threshold=_fmt(row.get("threshold")),
            )
        )

    lines.extend(
        [
            "",
            "## Note",
            "",
            f"- Output: `{out_dir}`",
            "- Use the aggregated table to decide whether source warm-start is consistently better than scratch on the same target fraction.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate D3 UNSW-NB15 transfer results across seeds.")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="Per-seed result directories that contain summary.json.")
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS, help="Settings to aggregate. Defaults to the D3 paper rows.")
    parser.add_argument("--out_dir", default="outputs/results/ccfa_d3_unsw_tabular_transfer_stability")
    args = parser.parse_args()

    runs = [_run_payload(Path(run_dir)) for run_dir in args.run_dirs]
    settings = list(dict.fromkeys(args.settings))
    aggregates = [_aggregate_setting(setting, runs) for setting in settings]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": {
            "base": "CICIDS2017 corrected CSV -> UNSW-NB15",
            "adapter": "shared_tabular_semantics_v1",
            "varying_factor": "seed",
            "settings": settings,
        },
        "run_dirs": args.run_dirs,
        "rows": aggregates,
    }
    write_json(payload, out_dir / "summary.json")
    markdown = _render_markdown(aggregates, runs, out_dir)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
