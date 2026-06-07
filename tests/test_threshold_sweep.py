from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_threshold_sweep_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("threshold_sweep", scripts_dir / "15_eval_threshold_sweep.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maximin_threshold_optimizes_worst_calibration_panel() -> None:
    module = _load_threshold_sweep_module()
    items = [
        {"val_true": [0, 1], "val_scores": np.array([[0.60, 0.40], [0.40, 0.60]])},
        {"val_true": [0, 1], "val_scores": np.array([[0.80, 0.20], [0.20, 0.80]])},
    ]
    threshold, value = module._best_maximin_threshold(items, "macro_f1")
    assert 0.40 < threshold <= 0.60
    assert value == 1.0
