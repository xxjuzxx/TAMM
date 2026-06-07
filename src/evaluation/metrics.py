from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _fpr_at_tpr(y_true: list[int], y_score: np.ndarray, target_tpr: float = 0.95) -> float | None:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ValueError:
        return None
    eligible = fpr[tpr >= target_tpr]
    if eligible.size == 0:
        return None
    return float(np.min(eligible))


def _tail_f1_metrics(y_true: list[int], y_pred: list[int]) -> tuple[float | None, float | None]:
    labels = sorted(set(y_true))
    if not labels:
        return None, None
    _, _, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    support_arr = np.asarray(support, dtype=np.float32)
    f1_arr = np.asarray(f1, dtype=np.float32)
    observed = support_arr > 0
    if not observed.any():
        return None, None
    observed_support = support_arr[observed]
    minority_cutoff = float(np.median(observed_support))
    minority_mask = observed & (support_arr <= minority_cutoff)
    minority_macro_f1 = float(np.mean(f1_arr[minority_mask] if minority_mask.any() else f1_arr[observed]))
    worst_class_f1 = float(np.min(f1_arr[observed]))
    return minority_macro_f1, worst_class_f1


def classification_metrics(y_true: list[int], y_pred: list[int], y_score: np.ndarray | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    minority_macro_f1, worst_class_f1 = _tail_f1_metrics(y_true, y_pred)
    metrics["minority_macro_f1"] = minority_macro_f1
    metrics["worst_class_f1"] = worst_class_f1
    if y_score is not None:
        labels = sorted(set(y_true))
        try:
            if y_score.shape[1] == 2:
                metrics["auroc"] = float(roc_auc_score(y_true, y_score[:, 1]))
                metrics["auprc"] = float(average_precision_score(y_true, y_score[:, 1]))
                metrics["fpr95"] = _fpr_at_tpr(y_true, y_score[:, 1], target_tpr=0.95)
            elif len(labels) > 2:
                metrics["auroc_ovr"] = float(roc_auc_score(y_true, y_score, multi_class="ovr"))
                y_true_bin = label_binarize(y_true, classes=labels)
                metrics["auprc_ovr"] = float(average_precision_score(y_true_bin, y_score, average="macro"))
        except ValueError:
            pass
    return metrics


def report_dict(y_true: list[int], y_pred: list[int], target_names: list[str] | None = None) -> dict[str, Any]:
    labels = list(range(len(target_names))) if target_names is not None else None
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def confusion(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred).tolist()
