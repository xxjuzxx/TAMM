#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import median, quantiles
from typing import Any

import _bootstrap  # noqa: F401

from src.utils.io import read_jsonl, write_json
import json
from pathlib import Path


def _read_predictions(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return read_jsonl(p)
    with p.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("prediction file must contain a JSON list or JSONL rows")
    return [dict(row) for row in data]


def _key(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if value is None:
        return "null"
    return str(value)


def _numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    if len(values) >= 4:
        q = quantiles(values, n=4)
        p25, p75 = q[0], q[2]
    else:
        p25 = p75 = median(values)
    return {
        "count": len(values),
        "min": min(values),
        "p25": p25,
        "median": median(values),
        "p75": p75,
        "max": max(values),
    }


def _counter_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(_key(row.get(field)) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize classifier test prediction errors.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--true_label", default=None)
    parser.add_argument("--pred_label", default=None)
    parser.add_argument("--only_errors", action="store_true")
    parser.add_argument("--group_by", action="append", default=["label", "dataset_file", "service_key"])
    args = parser.parse_args()

    rows = _read_predictions(args.predictions)
    selected = rows
    if args.true_label is not None:
        selected = [row for row in selected if str(row.get("true_label")) == args.true_label]
    if args.pred_label is not None:
        selected = [row for row in selected if str(row.get("pred_label")) == args.pred_label]
    if args.only_errors:
        selected = [row for row in selected if row.get("true_label") != row.get("pred_label")]

    confusion = Counter((str(row.get("true_label")), str(row.get("pred_label"))) for row in rows)
    filtered_confusion = Counter((str(row.get("true_label")), str(row.get("pred_label"))) for row in selected)
    by_true_pred_group: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for group_field in args.group_by:
        grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for row in selected:
            grouped[(str(row.get("true_label")), str(row.get("pred_label")))][_key(row.get(group_field))] += 1
        by_true_pred_group[group_field] = {
            f"{true_label}->{pred_label}": dict(counts)
            for (true_label, pred_label), counts in sorted(grouped.items())
        }

    summary = {
        "input": args.predictions,
        "filters": {
            "true_label": args.true_label,
            "pred_label": args.pred_label,
            "only_errors": bool(args.only_errors),
        },
        "num_rows": len(rows),
        "num_selected": len(selected),
        "num_selected_correct": sum(1 for row in selected if row.get("true_label") == row.get("pred_label")),
        "pred_counts": _counter_by(selected, "pred_label"),
        "true_counts": _counter_by(selected, "true_label"),
        "raw_label_counts": _counter_by(selected, "label"),
        "dataset_file_counts": _counter_by(selected, "dataset_file"),
        "service_key_counts": _counter_by(selected, "service_key"),
        "numeric": {
            "packet_count": _numeric_summary(selected, "packet_count"),
            "token_count": _numeric_summary(selected, "token_count"),
            "pred_confidence": _numeric_summary(selected, "pred_confidence"),
        },
        "confusion_pairs_all": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
        "confusion_pairs_selected": {f"{a}->{b}": n for (a, b), n in sorted(filtered_confusion.items())},
        "by_true_pred_group": by_true_pred_group,
        "examples": selected[:20],
    }
    write_json(summary, args.out)
    print({"num_rows": len(rows), "num_selected": len(selected), "out": args.out})


if __name__ == "__main__":
    main()
