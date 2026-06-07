#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score, roc_curve


BENIGN = {"BENIGN", "Benign", "benign", "0", 0}
ATTACKS = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
}


class AutoEncoder(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        hidden = max(1, min(hidden, max(1, dim)))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stable_id(row: dict[str, Any], idx: int) -> str:
    raw = row.get("flow_id") or row.get("_row_id") or row.get("source_flow_id")
    if raw:
        return str(raw)
    parts = [
        str(row.get("src_ip", "")),
        str(row.get("src_port", "")),
        str(row.get("dst_ip", "")),
        str(row.get("dst_port", "")),
        str(row.get("start_ts", "")),
        str(idx),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]


def is_benign(row: dict[str, Any]) -> bool:
    return row.get("binary_label") in BENIGN or row.get("label") in BENIGN or row.get("attack_family") in BENIGN


def attack_family(row: dict[str, Any]) -> str:
    if is_benign(row):
        return "BENIGN"
    return str(row.get("attack_family") or row.get("label") or "ATTACK")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_flows_by_id(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    rows: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = stable_id(row, idx)
            row["_row_id"] = flow_id
            row["_attack_family"] = attack_family(row)
            rows[flow_id] = row
            counts[row["_attack_family"]] += 1
    return rows, counts


def rows_for_split(flow_by_id: dict[str, dict[str, Any]], split_payload: dict[str, Any], split: str) -> list[dict[str, Any]]:
    ids = split_payload["splits"][split]
    missing = [flow_id for flow_id in ids if flow_id not in flow_by_id]
    if missing:
        raise KeyError(f"{len(missing)} split ids missing from flow source; first={missing[:3]}")
    return [flow_by_id[flow_id] for flow_id in ids]


def labels_for(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([0 if is_benign(row) else 1 for row in rows], dtype=np.int32)


def numeric_list(values: Any) -> list[float]:
    out: list[float] = []
    if not isinstance(values, list):
        return out
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def bool_dirs(values: Any, n: int) -> list[bool]:
    if not isinstance(values, list):
        return [True] * n
    out: list[bool] = []
    for value in values[:n]:
        out.append(bool(value))
    while len(out) < n:
        out.append(True)
    return out


def entropy_from_counts(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(len(values))
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0))


def burst_stats(lens: list[float], dirs: list[bool]) -> list[float]:
    if not lens:
        return [0.0] * 8
    bursts: list[tuple[bool, list[float]]] = []
    cur_dir = dirs[0] if dirs else True
    cur_vals: list[float] = []
    for length, direction in zip(lens, dirs):
        if direction != cur_dir and cur_vals:
            bursts.append((cur_dir, cur_vals))
            cur_vals = []
            cur_dir = direction
        cur_vals.append(length)
    if cur_vals:
        bursts.append((cur_dir, cur_vals))
    burst_sizes = [sum(abs(v) for v in vals) for _, vals in bursts]
    burst_packets = [len(vals) for _, vals in bursts]
    orig_bursts = sum(1 for direction, _ in bursts if direction)
    resp_bursts = len(bursts) - orig_bursts
    transitions = max(0, len(bursts) - 1)
    return [
        float(len(bursts)),
        float(transitions),
        float(orig_bursts),
        float(resp_bursts),
        float(np.mean(burst_sizes)) if burst_sizes else 0.0,
        float(np.max(burst_sizes)) if burst_sizes else 0.0,
        float(np.mean(burst_packets)) if burst_packets else 0.0,
        float(np.max(burst_packets)) if burst_packets else 0.0,
    ]


def flow_features(row: dict[str, Any], max_packets: int) -> np.ndarray:
    lens = numeric_list(row.get("lens"))
    tss = numeric_list(row.get("tss"))
    dirs = bool_dirs(row.get("dirs"), len(lens))
    signed: list[float] = []
    for length, direction in zip(lens[:max_packets], dirs[:max_packets]):
        sign = 1.0 if direction else -1.0
        signed.append(sign * math.log1p(abs(length)))
    while len(signed) < max_packets:
        signed.append(0.0)

    iats = [max(0.0, b - a) for a, b in zip(tss[:-1], tss[1:])]
    abs_lens = [abs(v) for v in lens]
    orig_lens = [abs(length) for length, direction in zip(lens, dirs) if direction]
    resp_lens = [abs(length) for length, direction in zip(lens, dirs) if not direction]
    length_bins = [min(9, int(math.log2(max(1.0, abs(v))))) for v in lens]
    duration = float(row.get("duration") or ((tss[-1] - tss[0]) if len(tss) >= 2 else 0.0))
    repeated_ratio = 0.0
    if lens:
        repeated_ratio = 1.0 - (len(set(round(v, 3) for v in lens)) / max(len(lens), 1))

    stats = [
        float(len(lens)),
        duration,
        float(sum(abs_lens)),
        float(sum(orig_lens)),
        float(sum(resp_lens)),
        float(len(orig_lens) / max(len(lens), 1)),
        float(np.mean(abs_lens)) if abs_lens else 0.0,
        float(np.std(abs_lens)) if abs_lens else 0.0,
        float(np.min(abs_lens)) if abs_lens else 0.0,
        float(np.max(abs_lens)) if abs_lens else 0.0,
        float(np.percentile(abs_lens, 25)) if abs_lens else 0.0,
        float(np.percentile(abs_lens, 50)) if abs_lens else 0.0,
        float(np.percentile(abs_lens, 75)) if abs_lens else 0.0,
        float(np.mean(iats)) if iats else 0.0,
        float(np.std(iats)) if iats else 0.0,
        float(np.max(iats)) if iats else 0.0,
        float(np.percentile(iats, 90)) if iats else 0.0,
        entropy_from_counts(length_bins),
        repeated_ratio,
    ]
    stats.extend(burst_stats(lens, dirs))
    arr = np.asarray(signed + stats, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def feature_matrix(rows: list[dict[str, Any]], max_packets: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, max_packets + 27), dtype=np.float32)
    return np.stack([flow_features(row, max_packets) for row in rows], axis=0).astype(np.float32)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def transform_standard(x: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float) -> np.ndarray:
    z = (x - mean) / std
    z = np.clip(z, -clip, clip)
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def feature_groups(x_train: np.ndarray, max_group_size: int) -> list[list[int]]:
    dim = x_train.shape[1]
    if dim <= max_group_size:
        return [list(range(dim))]
    corr = np.corrcoef(x_train, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    variances = np.var(x_train, axis=0)
    order = list(np.argsort(-variances))
    remaining = set(range(dim))
    groups: list[list[int]] = []
    for seed in order:
        if seed not in remaining:
            continue
        candidates = [idx for idx in remaining if idx != seed]
        candidates.sort(key=lambda idx: corr[seed, idx], reverse=True)
        group = [seed] + candidates[: max(0, max_group_size - 1)]
        for idx in group:
            remaining.discard(idx)
        groups.append(sorted(group))
    return groups


def train_ae(
    x: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    hidden_ratio: float,
) -> tuple[AutoEncoder, list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dim = int(x.shape[1])
    hidden = max(1, int(math.ceil(dim * hidden_ratio)))
    model = AutoEncoder(dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    data = torch.tensor(x, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=False)
    losses: list[float] = []
    model.train()
    for _epoch in range(epochs):
        epoch_losses: list[float] = []
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
    return model, losses


def reconstruction_rmse(model: AutoEncoder, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch_np = x[start : start + batch_size]
            batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
            recon = model(batch).detach().cpu().numpy()
            rmse = np.sqrt(np.mean((recon - batch_np) ** 2, axis=1))
            scores.append(rmse.astype(np.float32))
    return np.concatenate(scores, axis=0)


def train_kitsune(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    max_packets: int,
    max_group_size: int,
    epochs: int,
    output_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    clip: float,
    hidden_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_train = feature_matrix(train_rows, max_packets)
    raw_val = feature_matrix(val_rows, max_packets)
    raw_test = feature_matrix(test_rows, max_packets)
    mean, std = fit_standardizer(raw_train)
    train_x = transform_standard(raw_train, mean, std, clip)
    val_x = transform_standard(raw_val, mean, std, clip)
    test_x = transform_standard(raw_test, mean, std, clip)
    groups = feature_groups(train_x, max_group_size)

    ensemble_models: list[AutoEncoder] = []
    ensemble_losses: list[list[float]] = []
    train_errors: list[np.ndarray] = []
    val_errors: list[np.ndarray] = []
    test_errors: list[np.ndarray] = []
    for group_idx, group in enumerate(groups):
        model, losses = train_ae(
            train_x[:, group],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + group_idx,
            device=device,
            hidden_ratio=hidden_ratio,
        )
        ensemble_models.append(model)
        ensemble_losses.append(losses)
        train_errors.append(reconstruction_rmse(model, train_x[:, group], batch_size, device))
        val_errors.append(reconstruction_rmse(model, val_x[:, group], batch_size, device))
        test_errors.append(reconstruction_rmse(model, test_x[:, group], batch_size, device))

    ens_train = np.stack(train_errors, axis=1).astype(np.float32)
    ens_val = np.stack(val_errors, axis=1).astype(np.float32)
    ens_test = np.stack(test_errors, axis=1).astype(np.float32)
    ens_mean, ens_std = fit_standardizer(ens_train)
    ens_train_z = transform_standard(ens_train, ens_mean, ens_std, clip)
    ens_val_z = transform_standard(ens_val, ens_mean, ens_std, clip)
    ens_test_z = transform_standard(ens_test, ens_mean, ens_std, clip)
    output_model, output_losses = train_ae(
        ens_train_z,
        epochs=output_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed + 10007,
        device=device,
        hidden_ratio=hidden_ratio,
    )
    val_scores = reconstruction_rmse(output_model, ens_val_z, batch_size, device)
    test_scores = reconstruction_rmse(output_model, ens_test_z, batch_size, device)
    meta = {
        "device": str(device),
        "input_dim": int(train_x.shape[1]),
        "ensemble_size": int(len(groups)),
        "max_group_size": int(max_group_size),
        "group_sizes": [len(group) for group in groups],
        "epochs": int(epochs),
        "output_epochs": int(output_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "clip": float(clip),
        "hidden_ratio": float(hidden_ratio),
        "ensemble_final_loss_mean": float(np.mean([losses[-1] for losses in ensemble_losses if losses])) if ensemble_losses else float("nan"),
        "output_final_loss": float(output_losses[-1]) if output_losses else float("nan"),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    return val_scores, test_scores, meta


def threshold_at(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def fixed_fpr_threshold(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    benign_scores = scores[labels == 0]
    return threshold_at(benign_scores, 1.0 - target_fpr)


def metrics_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int32)
    tn = int(np.sum((labels == 0) & (pred == 0)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    tp = int(np.sum((labels == 1) & (pred == 1)))
    fpr = fp / max(fp + tn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_alerts_per_10k_benign": float(fpr * 10000.0),
    }


def fpr_at_recall(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return float("nan")
    fpr, tpr, _thresholds = roc_curve(labels, scores)
    candidates = fpr[tpr >= target_recall]
    return float(np.min(candidates)) if candidates.size else 1.0


def metric_bundle(labels: np.ndarray, scores: np.ndarray, val_scores: np.ndarray) -> dict[str, float]:
    has_both = labels.size > 0 and len(set(labels.tolist())) == 2
    out: dict[str, float] = {
        "auroc": float(roc_auc_score(labels, scores)) if has_both else float("nan"),
        "auprc": float(average_precision_score(labels, scores)) if has_both else float("nan"),
        "fpr95": fpr_at_recall(labels, scores, 0.95),
    }
    for target in (0.001, 0.01, 0.05):
        threshold = fixed_fpr_threshold(scores, labels, target)
        metrics = metrics_at(labels, scores, threshold)
        key = str(target)
        out[f"recall_at_{key}_fpr"] = float(metrics["recall"])
        out[f"precision_at_{key}_fpr"] = float(metrics["precision"])
        out[f"actual_fpr_at_{key}_fpr"] = float(metrics["fpr"])
        out[f"macro_f1_at_{key}_fpr"] = float(f1_score(labels, (scores >= threshold).astype(np.int32), average="macro", zero_division=0))
    p99 = threshold_at(val_scores, 0.99)
    p99_metrics = metrics_at(labels, scores, p99)
    pred = (scores >= p99).astype(np.int32)
    out["val_p99_threshold"] = float(p99)
    out["val_p99_realized_fpr"] = float(p99_metrics["fpr"])
    out["val_p99_attack_recall"] = float(p99_metrics["recall"])
    out["val_p99_precision"] = float(precision_score(labels, pred, zero_division=0))
    out["val_p99_f1"] = float(p99_metrics["f1"])
    out["val_p99_macro_f1"] = float(f1_score(labels, pred, average="macro", zero_division=0))
    out["false_alerts_per_10k_benign"] = float(p99_metrics["false_alerts_per_10k_benign"])
    return out


def slug_for_attack(attack: str) -> str:
    return ATTACKS[attack]


def split_file_for(base_dir: Path, attack: str, seed: int) -> Path:
    return base_dir / f"splits_leave_one_{slug_for_attack(attack)}_anomaly_seed{seed}.json"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "auroc",
        "auprc",
        "fpr95",
        "recall_at_0.001_fpr",
        "recall_at_0.01_fpr",
        "recall_at_0.05_fpr",
        "actual_fpr_at_0.01_fpr",
        "val_p99_realized_fpr",
        "val_p99_attack_recall",
        "false_alerts_per_10k_benign",
    ]
    out: list[dict[str, Any]] = []
    group_keys = sorted({(row["method"], row["eval_unit"], row["heldout_attack"]) for row in rows})
    for method, eval_unit, heldout in group_keys:
        group = [row for row in rows if row["method"] == method and row["eval_unit"] == eval_unit and row["heldout_attack"] == heldout]
        item: dict[str, Any] = {"method": method, "eval_unit": eval_unit, "heldout_attack": heldout, "runs": len(group)}
        for metric in metrics:
            vals = []
            for row in group:
                try:
                    vals.append(float(row.get(metric, "nan")))
                except (TypeError, ValueError):
                    pass
            mean, std = mean_std(vals)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        out.append(item)
    for method, eval_unit in sorted({(row["method"], row["eval_unit"]) for row in rows}):
        group = [row for row in rows if row["method"] == method and row["eval_unit"] == eval_unit]
        item = {"method": method, "eval_unit": eval_unit, "heldout_attack": "Aggregate", "runs": len(group)}
        for metric in metrics:
            vals = []
            for row in group:
                try:
                    vals.append(float(row.get(metric, "nan")))
                except (TypeError, ValueError):
                    pass
            mean, std = mean_std(vals)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        out.append(item)
    return out


def fmt_pm(mean: Any, std: Any, digits: int = 4) -> str:
    try:
        m = float(mean)
        s = float(std)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(m):
        return "--"
    return f"${m:.{digits}f}\\pm{s:.{digits}f}$"


def write_latex_summary(agg_rows: list[dict[str, Any]], path: Path) -> None:
    selected = [row for row in agg_rows if row["heldout_attack"] == "Aggregate"]
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Method & Unit & Runs & AUROC & FPR95 & R@0.1\\%FPR & R@1\\%FPR \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['method']} & {row['eval_unit']} & {row['runs']} & "
            f"{fmt_pm(row['auroc_mean'], row['auroc_std'])} & "
            f"{fmt_pm(row['fpr95_mean'], row['fpr95_std'])} & "
            f"{fmt_pm(row['recall_at_0.001_fpr_mean'], row['recall_at_0.001_fpr_std'])} & "
            f"{fmt_pm(row['recall_at_0.01_fpr_mean'], row['recall_at_0.01_fpr_std'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_split(
    *,
    flow_by_id: dict[str, dict[str, Any]],
    source_counts: Counter[str],
    split_path: Path,
    output_dir: Path,
    max_packets: int,
    max_group_size: int,
    epochs: int,
    output_epochs: int,
    batch_size: int,
    lr: float,
    clip: float,
    hidden_ratio: float,
) -> dict[str, Any]:
    payload = read_json(split_path)
    seed = int(payload["seed"])
    heldout = str(payload["leave_label"])
    train = rows_for_split(flow_by_id, payload, "train")
    val = rows_for_split(flow_by_id, payload, "val")
    test = rows_for_split(flow_by_id, payload, "test")
    val_scores, test_scores, meta = train_kitsune(
        train,
        val,
        test,
        max_packets=max_packets,
        max_group_size=max_group_size,
        epochs=epochs,
        output_epochs=output_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        clip=clip,
        hidden_ratio=hidden_ratio,
    )
    labels = labels_for(test)
    row: dict[str, Any] = {
        "dataset": "CICIDS2017",
        "heldout_attack": heldout,
        "seed": seed,
        "split_file": str(split_path),
        "source_rows": int(sum(source_counts.values())),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "test_attack_flows": int(np.sum(labels == 1)),
        "test_benign_flows": int(np.sum(labels == 0)),
        "method": "Kitsune flow AE ensemble",
        "eval_unit": "flow",
        "official_reproduction_status": "tamm_protocol_flow_artifact_adapted",
        **meta,
        **metric_bundle(labels, test_scores, val_scores),
    }
    run_dir = output_dir / f"ids2017_leave_one_{slug_for_attack(heldout)}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    for item, label, score in zip(test, labels, test_scores):
        score_rows.append(
            {
                "flow_id": item["_row_id"],
                "label": int(label),
                "attack_family": item["_attack_family"],
                "score": float(score),
            }
        )
    write_csv(score_rows, run_dir / "scores.csv")
    write_csv([row], run_dir / "metrics.csv")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kitsune-style flow AE baseline under TAMM leave-one split protocol.")
    parser.add_argument("--flow-file", default="outputs/processed/ccfa/cicids2017_interim_labeled_flows.jsonl")
    parser.add_argument("--split-dir", default="paper_icdm_applied_2026/experiments/unknown")
    parser.add_argument("--output-dir", default="experiments/kitsune_baseline/tamm_protocol")
    parser.add_argument("--attacks", nargs="+", default=list(ATTACKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-packets", type=int, default=32)
    parser.add_argument("--max-group-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=8.0)
    parser.add_argument("--hidden-ratio", type=float, default=0.75)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    flow_by_id, source_counts = read_flows_by_id(Path(args.flow_file))
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for attack in args.attacks:
            split_path = split_file_for(Path(args.split_dir), attack, seed)
            print(f"running Kitsune flow AE {attack} seed={seed} split={split_path}", flush=True)
            row = evaluate_split(
                flow_by_id=flow_by_id,
                source_counts=source_counts,
                split_path=split_path,
                output_dir=output_dir,
                max_packets=args.max_packets,
                max_group_size=args.max_group_size,
                epochs=args.epochs,
                output_epochs=args.output_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                clip=args.clip,
                hidden_ratio=args.hidden_ratio,
            )
            rows.append(row)
            print(
                f"{attack} seed={seed} auroc={row['auroc']:.4f} "
                f"r@1%fpr={row['recall_at_0.01_fpr']:.4f} p99fpr={row['val_p99_realized_fpr']:.4f}",
                flush=True,
            )

    write_csv(rows, output_dir / "kitsune_tamm_protocol_runs.csv")
    agg = aggregate(rows)
    write_csv(agg, output_dir / "kitsune_tamm_protocol_aggregate.csv")
    write_latex_summary(agg, output_dir / "table_kitsune_tamm_protocol_summary.tex")
    print(f"wrote {output_dir / 'kitsune_tamm_protocol_runs.csv'}")
    print(f"wrote {output_dir / 'kitsune_tamm_protocol_aggregate.csv'}")


if __name__ == "__main__":
    main()
