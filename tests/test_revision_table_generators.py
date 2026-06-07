from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module(script_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_primitive_category_table_declares_profile_and_structural_views() -> None:
    module = _module("build_primitive_category_table.py")

    assert "profile_only" in module.VIEW_LABELS
    assert "structural_only" in module.VIEW_LABELS
    assert "packet_burst_plus_profile_structural" in module.VIEW_LABELS


def test_revision_experiment_uses_profile_feature_filter_names() -> None:
    module = _module("64_run_icdm_revision_experiments.py")
    view_filters = {row[2] for row in module.FEATURE_ATTRIBUTION_VIEWS}

    assert "profile_only" in view_filters
    assert "packet_burst_profile" in view_filters
    assert all("profile" in item or item in {"global_only", "packet", "packet_burst"} for item in view_filters)
