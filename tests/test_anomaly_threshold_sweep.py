from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("anomaly_threshold_sweep", scripts_dir / "43_eval_anomaly_threshold_sweep.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_anomaly_threshold_sweep_reports_recall_under_fpr_budget() -> None:
    module = _load_module()
    rows = [
        {"binary_label": 0, "anomaly_score": 0.10},
        {"binary_label": 0, "anomaly_score": 0.20},
        {"binary_label": 0, "anomaly_score": 0.30},
        {"binary_label": 1, "anomaly_score": 0.70},
        {"binary_label": 1, "anomaly_score": 0.80},
    ]

    payload = module.summarize_scores(rows)

    assert payload["num_scores"] == 5
    assert payload["recall_at_1pct_fpr"]["false_positive_rate"] == 0.0
    assert payload["recall_at_1pct_fpr"]["attack_recall"] == 1.0
    assert payload["best_metric_metrics"]["macro_f1"] > 0.0
