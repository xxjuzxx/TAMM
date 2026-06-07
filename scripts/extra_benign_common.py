from __future__ import annotations

import csv
import importlib.util
import json
import math
import pickle
import shlex
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from src.features.tokenizer import PrimitiveTrafficTokenizer, Vocabulary
from src.features.token_alias import is_burst_token, is_flow_summary_token, is_packet_token, is_profile_token
from src.utils.io import read_yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PAPER_TABLE_DIR = REPO / "paper" / "tables"
UNKNOWN_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "unknown"
TOKEN_DIR = UNKNOWN_DIR / "tokens_category"
EXTRA_ARTIFACT_DIR = ROOT / "artifacts" / "extra_benign"
EXTRA_RESULT_DIR = ROOT / "results"
EXTRA_SPLIT_DIR = ROOT / "splits"
EXTRA_REPORT_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "extra_benign"
SWEEP_PATH = ROOT / "scripts" / "52_sweep_anomaly_low_fpr.py"
DEFAULT_FLOW_SOURCE = ROOT / "outputs" / "processed" / "ccfa" / "cicids2017_interim_labeled_flows.jsonl"
DEFAULT_REFERENCE_TOKEN = TOKEN_DIR / "cicids2017_leave_one_botnet_anomaly_seed43_a3_full_rhythm.pt"

ATTACKS = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
}

BEST_SETTINGS = {
    "Botnet": {"feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_euclidean", "k": 3, "group_mode": "protocol"},
    "DDoS": {"feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "protocol"},
    "Probe": {"feature_filter": "all_no_special", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "protocol"},
    "WebAttack": {"feature_filter": "packet_burst", "transform": "binary_l2", "scorer": "knn_cosine", "k": 1, "group_mode": "global"},
    "BruteForce": {"feature_filter": "packet_burst_profile", "transform": "tfidf_l2", "scorer": "knn_cosine", "k": 3, "group_mode": "protocol"},
}

SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[MASK]"}
PRIMITIVE_NAMES = ("SHORT", "SAME", "PKT", "LOCAL", "REPEAT", "DUP")
CONTEXT_GROUP_PREFIXES = ("SVC", "PROTO_", "APP_", "STATE_", "CTX_")


def load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("flowprim_extra_benign_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load_sweep_module()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def command_used() -> str:
    return shlex.join(sys.argv)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2, sort_keys=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(val):
        return "-"
    return f"{val:.{digits}f}"


def fmt_pm(mean: Any, std: Any, digits: int = 4) -> str:
    return f"${fmt(mean, digits)}\\pm{fmt(std, digits)}$"


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def group_summary(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, items in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row = {key: value for key, value in zip(keys, key_values)}
        row["num_runs"] = len(items)
        for metric in metrics:
            vals = [safe_float(item.get(metric), float("nan")) for item in items]
            mean, std = mean_std(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    return out


def token_path(attack: str, seed: int) -> Path:
    slug = ATTACKS[attack]
    return TOKEN_DIR / f"cicids2017_leave_one_{slug}_anomaly_seed{seed}_a3_full_rhythm.pt"


def load_token_data(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_state(attack: str, seed: int) -> dict[str, Any]:
    setting = dict(BEST_SETTINGS[attack])
    data = load_token_data(token_path(attack, seed))
    labels = data["binary_labels"].cpu().numpy().astype(np.int64)
    train_idx = S._split_indices(data, "train")
    val_idx = S._split_indices(data, "val")
    test_idx = S._split_indices(data, "test")
    if not np.all(labels[train_idx] == 0):
        raise ValueError(f"Non-benign row in train split for {attack} seed {seed}")
    if not np.all(labels[val_idx] == 0):
        raise ValueError(f"Non-benign row in val split for {attack} seed {seed}")
    features, feature_stats = S._features(data, train_idx, feature_filter=setting["feature_filter"], transform=setting["transform"])
    groups = S._groups(data, setting["group_mode"])
    return {
        "attack": attack,
        "seed": seed,
        "setting": setting,
        "token_data": data,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "features": features,
        "feature_stats": feature_stats,
        "groups": groups,
    }


def id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    return {int(idx): str(token) for token, idx in vocab.items()}


def kept_token_ids(token_data: dict[str, Any], feature_filter: str) -> list[int]:
    mapping = id_to_token(token_data["vocab"])
    return sorted(idx for idx, token in mapping.items() if S._keep_token(token, feature_filter))


def raw_count_matrix_from_token_rows(token_rows: list[dict[str, Any]], token_data: dict[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
    vocab = {str(k): int(v) for k, v in token_data["vocab"].items()}
    vocab_size = len(vocab)
    unk = int(vocab.get("[UNK]", 4))
    counts = np.zeros((len(token_rows), vocab_size), dtype=np.float32)
    flow_ids: list[str] = []
    groups: list[str] = []
    for row_idx, row in enumerate(token_rows):
        flow_id = str(row["flow_id"])
        flow_ids.append(flow_id)
        proto = str(row.get("protocol") or row.get("service_protocol") or "unknown").lower()
        groups.append(proto if proto else "unknown")
        tokens = list(row.get("tokens") or [])[: int(token_data.get("max_len") or 512)]
        ids = [int(vocab.get(str(token), unk)) for token in tokens]
        if ids:
            counts[row_idx, :] = np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab_size)[:vocab_size]
    return counts, flow_ids, groups


def transform_extra_counts(counts: np.ndarray, token_data: dict[str, Any], train_idx: np.ndarray, feature_filter: str, transform: str) -> tuple[np.ndarray, list[int], list[str]]:
    kept = kept_token_ids(token_data, feature_filter)
    if not kept:
        raise ValueError(f"Feature filter kept no tokens: {feature_filter}")
    features = counts[:, kept].astype(np.float32, copy=True)
    base_counts = S._raw_counts(token_data, kept)
    if transform.startswith("binary"):
        features = (features > 0).astype(np.float32)
        features = S._normalize(features, transform.removeprefix("binary_") or "none")
    elif transform.startswith("tfidf"):
        train_counts = base_counts[train_idx]
        df = np.sum(train_counts > 0, axis=0)
        idf = np.log((1.0 + len(train_idx)) / (1.0 + df)) + 1.0
        features = features * idf.reshape(1, -1).astype(np.float32)
        features = S._normalize(features, transform.removeprefix("tfidf_") or "none")
    elif transform.startswith("count"):
        features = S._normalize(features, transform.removeprefix("count_") or "none")
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    mapping = id_to_token(token_data["vocab"])
    return features.astype(np.float32, copy=False), kept, [mapping[idx] for idx in kept]


def score_external(
    eval_features: np.ndarray,
    eval_groups: list[str],
    ref_features: np.ndarray,
    ref_groups: list[str],
    *,
    scorer: str,
    k: int,
) -> np.ndarray:
    if eval_features.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if ref_features.shape[0] == 0:
        return np.ones(eval_features.shape[0], dtype=np.float32)
    refs_by_group: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(ref_groups):
        refs_by_group[str(group)].append(idx)
    eval_by_group: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(eval_groups):
        eval_by_group[str(group)].append(idx)
    out = np.zeros(eval_features.shape[0], dtype=np.float32)
    for group, eval_idx in eval_by_group.items():
        ref_idx = refs_by_group.get(group) or list(range(ref_features.shape[0]))
        out[np.asarray(eval_idx, dtype=np.int64)] = S._score_against_refs(
            eval_features[np.asarray(eval_idx, dtype=np.int64)],
            ref_features[np.asarray(ref_idx, dtype=np.int64)],
            scorer,
            k,
        )
    return out


def metrics_for_scores(y_true: np.ndarray, scores: np.ndarray, calibration_scores: np.ndarray) -> dict[str, Any]:
    best = S._best_macro(y_true, scores)
    r01 = S._best_recall_under_fpr(y_true, scores, 0.001)
    r1 = S._best_recall_under_fpr(y_true, scores, 0.01)
    rank = S._rank_metrics(y_true, scores)
    out = {
        "macro_f1": best["macro_f1"],
        "best_threshold": best["threshold"],
        "auroc": rank["auroc"],
        "auprc": rank["auprc"],
        "fpr95": rank["fpr95"],
        "recall_at_0_1pct_fpr": r01["attack_recall"],
        "recall_at_1pct_fpr": r1["attack_recall"],
        "actual_fpr_at_0_1pct_fpr": r01["false_positive_rate"],
        "actual_fpr_at_1pct_fpr": r1["false_positive_rate"],
    }
    for label, percentile in [("p95", 95.0), ("p99", 99.0), ("p99_5", 99.5), ("p100", 100.0)]:
        threshold = float(np.percentile(calibration_scores, percentile)) if calibration_scores.size else float("inf")
        m = S._metrics_at_threshold(y_true, scores, threshold)
        out[f"{label}_threshold"] = threshold
        out[f"{label}_realized_fpr"] = m["false_positive_rate"]
        out[f"{label}_recall"] = m["attack_recall"]
        out[f"{label}_macro_f1"] = m["macro_f1"]
        out[f"{label}_false_alerts_per_10k_benign"] = m["false_positive_rate"] * 10000.0
    return out


def token_group(token: str) -> str:
    if is_profile_token(token):
        return "primitive"
    if is_burst_token(token):
        return "burst"
    if is_packet_token(token):
        return "packet"
    if is_flow_summary_token(token):
        return "global"
    if token.startswith(CONTEXT_GROUP_PREFIXES):
        return "context"
    return "other"


def dominant_token_group(row_counts: np.ndarray, kept_tokens: list[str]) -> str:
    scores: dict[str, float] = defaultdict(float)
    for value, token in zip(row_counts.tolist(), kept_tokens):
        if value:
            scores[token_group(token)] += float(value)
    if not scores:
        return "none"
    return max(scores.items(), key=lambda item: item[1])[0]


def protocol_of_flow(flow: dict[str, Any]) -> str:
    value = str(flow.get("protocol") or flow.get("proto") or "").lower()
    if value in {"6", "tcp"}:
        return "tcp"
    if value in {"17", "udp"}:
        return "udp"
    if value in {"1", "icmp"}:
        return "icmp"
    return value or "unknown"


def service_of_flow(flow: dict[str, Any]) -> str:
    service_key = flow.get("service_key")
    if isinstance(service_key, (list, tuple)) and len(service_key) >= 2:
        return str(service_key[1])
    if flow.get("dst_port") is not None:
        return str(flow.get("dst_port"))
    return ""


def primitive_flags(profile_row: dict[str, Any] | None) -> dict[str, int]:
    profile = (profile_row or {}).get("profile") or {}
    return {
        "primitive_SHORT": int(bool(profile.get("short"))),
        "primitive_SAME": int(bool(profile.get("same"))),
        "primitive_PKT": int(bool(profile.get("packet"))),
        "primitive_LOCAL": int(bool(profile.get("local"))),
        "primitive_REPEAT": int(bool(profile.get("repeat"))),
        "primitive_DUP": int(bool(profile.get("duplicate"))),
    }


def build_tokenizer_for_existing_vocab(token_data: dict[str, Any]) -> PrimitiveTrafficTokenizer:
    cfg = dict(token_data.get("tokenizer_config") or {})
    vocab = Vocabulary()
    vocab.token_to_id = {str(key): int(value) for key, value in token_data["vocab"].items()}
    return PrimitiveTrafficTokenizer(**cfg, vocab=vocab)


def default_profile_row(flow_id: str) -> dict[str, Any]:
    return {"flow_id": flow_id, "profile": {"short": None, "same": None, "packet": [], "local": [], "repeat": [], "duplicate": []}}


def render_simple_latex(rows: list[dict[str, Any]], columns: list[tuple[str, str]], path: Path, *, full_width: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * len(columns)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(header for _key, header in columns) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        values = []
        for key, _header in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value, 4)
            values.append(str(value).replace("_", "\\_").replace("%", "\\%"))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_memory_artifact(payload: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as handle:
        pickle.dump(payload, handle)


def load_extra_token_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def load_extra_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    return {str(row["flow_id"]): row for row in read_csv(path)}


def source_label(flow: dict[str, Any], fallback: str) -> str:
    raw = str(flow.get("label_source_file") or flow.get("dataset_file") or fallback)
    return Path(raw).stem if raw else fallback
