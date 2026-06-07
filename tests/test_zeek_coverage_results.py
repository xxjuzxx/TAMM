from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "34_build_zeek_coverage_results.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("zeek_coverage_results", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_row_reads_binary_metrics_and_label_stats() -> None:
    module = _module()

    row = module._row(module.THURSDAY_INFILTRATION_DROP_RESULT)

    assert row["artifact_status"] == "archived"
    assert row["missing_paths"]
    assert row["selected_label_counts"] == {}


def test_markdown_includes_thursday_infiltration_auroc_and_counts() -> None:
    module = _module()
    rows = [module._row(module.MAIN_RESULT), module._row(module.V2_RESULT)]
    infiltration_rows = [
        module._row(module.THURSDAY_INFILTRATION_DROP_RESULT),
        module._row(module.THURSDAY_INFILTRATION_ATTACK_RESULT),
    ]
    infiltration = {
        "rows": infiltration_rows,
        "drop_selected_rows": infiltration_rows[0].get("selected_rows"),
        "drop_matched": infiltration_rows[0].get("matched"),
        "drop_match_rate": infiltration_rows[0].get("match_rate"),
        "drop_macro_f1": infiltration_rows[0].get("macro_f1"),
        "attack_selected_rows": infiltration_rows[1].get("selected_rows"),
        "attack_matched": infiltration_rows[1].get("matched"),
        "attack_match_rate": infiltration_rows[1].get("match_rate"),
        "attack_macro_f1": infiltration_rows[1].get("macro_f1"),
    }

    table = module._markdown(rows, infiltration)

    assert "archived" in table
    assert "Zeek-first coverage expansion diagnostics" in table
    assert "Thursday Infiltration binary diagnostics" in table
