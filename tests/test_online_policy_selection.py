from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_policy_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("online_policy", scripts_dir / "17_select_online_policy.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(path: Path, macro_f1: float, fpr: float, recall: float, stateful: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokens": "tokens.pt",
        "checkpoint": "model.pt",
        "split": "temporal_stratified",
        "scope": "test",
        "threshold": 0.16,
        "threshold_source": "fixed",
        "num_eval_flows": 10,
        "summary": {
            "macro_f1": macro_f1,
            "weighted_f1": macro_f1,
            "accuracy": macro_f1,
            "false_positive_rate": fpr,
            "attack_recall_online": recall,
            "false_positives": int(fpr * 100),
            "true_positives": int(recall * 100),
            "confusion_matrix": [[100, int(fpr * 100)], [100 - int(recall * 100), int(recall * 100)]],
        },
    }
    if stateful:
        payload["stateful_service"] = {
            "enabled": True,
            "min_model_score": 0.01,
            "stateful_only_alerts": 2,
            "stateful_missing_key_flows": 0,
            "stateful_updated_flows": 10,
            "model_only_summary": {
                "macro_f1": macro_f1 - 0.01,
                "false_positive_rate": fpr,
                "attack_recall_online": recall - 0.01,
                "confusion_matrix": [[100, int(fpr * 100)], [100 - int((recall - 0.01) * 100), int((recall - 0.01) * 100)]],
            },
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_online_policy_selection_prefers_best_feasible_policy(tmp_path: Path) -> None:
    module = _load_policy_module()
    high_fpr = tmp_path / "high_fpr.json"
    best = tmp_path / "best.json"
    low_recall = tmp_path / "low_recall.json"
    _summary(high_fpr, macro_f1=0.999, fpr=0.02, recall=1.0)
    _summary(best, macro_f1=0.995, fpr=0.005, recall=0.998, stateful=True)
    _summary(low_recall, macro_f1=0.996, fpr=0.0, recall=0.990)

    rows = [
        module.flatten_policy(high_fpr, name="high_fpr"),
        module.flatten_policy(best, name="best"),
        module.flatten_policy(low_recall, name="low_recall"),
    ]
    selected = module.select_policy(rows, max_fpr=0.01, min_attack_recall=0.995)
    assert selected["name"] == "best"
    assert selected["stateful_enabled"] is True
    assert selected["model_only_attack_recall_online"] == 0.988


def test_online_policy_selection_returns_none_when_constraints_fail(tmp_path: Path) -> None:
    module = _load_policy_module()
    path = tmp_path / "policy.json"
    _summary(path, macro_f1=0.999, fpr=0.02, recall=0.990)
    rows = [module.flatten_policy(path, name="policy")]
    assert module.select_policy(rows, max_fpr=0.01, min_attack_recall=0.995) is None
