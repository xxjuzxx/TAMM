from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import classification_metrics


def test_classification_metrics_reports_tail_and_fpr95() -> None:
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    y_score = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.1, 0.9],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )

    metrics = classification_metrics(y_true, y_pred, y_score)

    assert metrics["fpr95"] is not None
    assert 0.0 <= metrics["fpr95"] <= 1.0
    assert metrics["minority_macro_f1"] == pytest.approx(0.5)
    assert metrics["worst_class_f1"] == pytest.approx(0.5)
