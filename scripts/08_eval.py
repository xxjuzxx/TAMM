#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json

from sklearn.metrics import classification_report, confusion_matrix

import _bootstrap  # noqa: F401

from src.evaluation.metrics import classification_metrics
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = sorted({row["true_label"] for row in rows} | {row["pred_label"] for row in rows})
    if args.task == "binary":
        labels = [label for label in ["BENIGN", "ATTACK"] if label in labels]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    y_true = [label_to_id[row["true_label"]] for row in rows]
    y_pred = [label_to_id[row["pred_label"]] for row in rows]
    report = {
        "metrics": classification_metrics(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "labels": labels,
        "num_rows": len(rows),
    }
    write_json(report, args.out)
    print(report["metrics"])


if __name__ == "__main__":
    main()
