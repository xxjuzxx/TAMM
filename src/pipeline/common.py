from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a project dependency in normal runs.
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT.parent / "paper"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    for rel in [
        "data/raw/pcaps",
        "data/interim/flows",
        "data/interim/normalized_flows",
        "data/interim/tokens",
        "data/interim/primitive_events",
        "data/processed/feature_matrices",
        "data/processed/splits",
        "data/processed/benign_memory",
        "data/manifests",
        "results/main_detection",
        "results/ablation",
        "results/primitive_analysis",
        "results/calibration_robustness",
        "results/diagnosis_cases",
        "results/efficiency",
        "results/baselines",
        "results/summaries",
        "figures/pipeline",
        "figures/score_distributions",
        "figures/primitive_heatmaps",
        "figures/calibration_sensitivity",
        "figures/efficiency",
        "figures/diagnosis",
        "tables/dataset_protocol",
        "tables/main_results",
        "tables/ablation",
        "tables/efficiency",
        "tables/diagnosis_cases",
        "tables/primitive_analysis",
        "logs/extraction",
        "logs/experiments",
        "logs/paper_sync",
        "reports",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def read_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read FlowPrim pipeline configs")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_md(path: str | Path, lines: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def command_record(argv: list[str] | None = None) -> dict[str, Any]:
    argv = argv or sys.argv
    return {
        "created_at": utc_now(),
        "cwd": str(Path.cwd()),
        "command": shlex.join(argv),
        "python": sys.version.split()[0],
        "raw_ip_used_as_token": False,
        "absolute_time_used_as_token": False,
        "five_tuple_used_as_token": False,
    }


def run_command(
    cmd: list[str],
    *,
    log_path: str | Path | None = None,
    dry_run: bool = False,
    check: bool = True,
) -> dict[str, Any]:
    started = utc_now()
    record = {
        "command": shlex.join(cmd),
        "started_at": started,
        "dry_run": bool(dry_run),
        "returncode": 0,
    }
    if dry_run:
        if log_path:
            write_md(log_path, [f"$ {record['command']}", "dry_run: true"])
        return record
    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    record["returncode"] = int(proc.returncode)
    record["completed_at"] = utc_now()
    if log_path:
        write_md(log_path, [f"$ {record['command']}", "", proc.stdout])
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return record


def file_status(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    out: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if p.exists():
        stat = p.stat()
        out.update({"size_bytes": int(stat.st_size), "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat()})
    return out

