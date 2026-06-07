#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict

import _bootstrap  # noqa: F401

from src.utils.io import iter_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows_by_label: dict[str, list[dict]] = defaultdict(list)
    for row in iter_jsonl(args.scores):
        rows_by_label[str(row.get("label", "UNKNOWN"))].append(row)

    result = {}
    for label, rows in sorted(rows_by_label.items()):
        total = len(rows)
        positives = [row for row in rows if int(row.get("binary_label", 0)) == 1]
        negatives = [row for row in rows if int(row.get("binary_label", 0)) == 0]
        predicted_attack = [row for row in rows if int(row.get("prediction", 0)) == 1]
        scores = [float(row.get("anomaly_score", 0.0)) for row in rows]
        result[label] = {
            "total": total,
            "binary_label": "ATTACK" if positives else "BENIGN",
            "predicted_attack": len(predicted_attack),
            "recall_or_fpr": (len(predicted_attack) / total) if total else 0.0,
            "mean_score": sum(scores) / total if total else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "positives": len(positives),
            "negatives": len(negatives),
        }
    write_json(result, args.out)
    print(result)


if __name__ == "__main__":
    main()
