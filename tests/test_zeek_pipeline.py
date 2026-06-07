from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "26_run_zeek_pipeline.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("zeek_pipeline", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_zeek_logs_pairs_features_with_conn(tmp_path) -> None:
    module = _pipeline_module()
    run_dir = tmp_path / "SqlInjection"
    run_dir.mkdir()
    features = run_dir / "Features.log"
    conn = run_dir / "conn.log"
    features.write_text("{}", encoding="utf-8")
    conn.write_text("#separator \\x09", encoding="utf-8")

    assert module.collect_zeek_logs(tmp_path) == [features, conn]


def test_collect_zeek_logs_falls_back_to_conn_log(tmp_path) -> None:
    module = _pipeline_module()
    run_dir = tmp_path / "OnlyConn"
    run_dir.mkdir()
    conn = run_dir / "conn.log"
    conn.write_text("#separator \\x09", encoding="utf-8")

    assert module.collect_zeek_logs(tmp_path) == [conn]
