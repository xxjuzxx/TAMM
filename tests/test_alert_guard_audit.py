from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_audit_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("alert_guard_audit", scripts_dir / "19_audit_alert_guard.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alert_guard_audit_reports_blocked_labels_and_services(tmp_path: Path) -> None:
    module = _load_audit_module()
    scores = tmp_path / "scores.jsonl"
    rows = [
        {
            "binary_label_id": 1,
            "label": "FTP-Patator",
            "prediction": 0,
            "alert_guard_blocked_prediction": True,
            "alert_service_protocol": "TCP",
            "alert_service_port": "21",
        },
        {
            "binary_label_id": 0,
            "label": "Benign",
            "prediction": 0,
            "alert_guard_blocked_prediction": True,
            "alert_service_protocol": "UDP",
            "alert_service_port": "53",
        },
        {
            "binary_label_id": 1,
            "label": "DDoS",
            "prediction": 1,
            "alert_guard_blocked_prediction": False,
            "alert_service_protocol": "TCP",
            "alert_service_port": "80",
        },
    ]
    scores.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    out = module.audit_scores(scores)
    assert out["blocked_true_positives"] == 1
    assert out["blocked_false_positives"] == 1
    assert out["passed_true_positives"] == 1
    assert out["blocked_true_positive_labels"] == [{"key": "FTP-Patator", "count": 1}]
    assert out["blocked_true_positive_services"] == [{"key": "TCP|21", "count": 1}]
    assert out["blocked_false_positive_services"] == [{"key": "UDP|53", "count": 1}]
