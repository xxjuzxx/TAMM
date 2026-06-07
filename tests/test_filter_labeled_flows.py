from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _filter_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "29_filter_labeled_flows.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("filter_labeled_flows", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_filter_rows_can_select_attempted_webattack_only() -> None:
    module = _filter_module()
    rows = [
        {"label": "Web Attack - XSS", "raw_label": "Web Attack - XSS - Attempted", "binary_label": "ATTACK"},
        {"label": "Web Attack - XSS", "raw_label": "Web Attack - XSS", "binary_label": "ATTACK"},
        {"label": "DoS Hulk", "raw_label": "DoS Hulk - Attempted", "binary_label": "ATTACK"},
        {"label": "BENIGN", "raw_label": "BENIGN", "binary_label": "BENIGN"},
    ]

    selected = module.filter_rows(rows, merged_label_globs=["WebAttack"], raw_label_globs=["*Attempted*"])

    assert selected == [rows[0]]


def test_summarize_reports_selected_counts() -> None:
    module = _filter_module()
    rows = [
        {"label": "Web Attack - XSS", "raw_label": "Web Attack - XSS - Attempted", "binary_label": "ATTACK"},
        {"label": "BENIGN", "raw_label": "BENIGN", "binary_label": "BENIGN"},
    ]

    stats = module.summarize(rows, [rows[0]], type("Args", (), {
        "input": "in.jsonl",
        "out": "out.jsonl",
        "label_glob": [],
        "raw_label_glob": ["*Attempted*"],
        "merged_label_glob": ["WebAttack"],
        "binary_label_glob": [],
    })())

    assert stats["selected_rows"] == 1
    assert stats["selected_merged_label_counts"] == {"WebAttack": 1}
