#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


BENIGN = {"BENIGN", "Benign", "benign", "0", 0}


class BstsNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.ln1 = nn.Linear(128, 64)
        self.ln2 = nn.Linear(64, 32)
        self.ln3 = nn.Linear(32, 16)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ln3(x)
        return torch.sigmoid(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_once(x1), self.forward_once(x2), self.forward_once(anchor)


def is_benign(row: dict[str, Any]) -> bool:
    return row.get("binary_label") in BENIGN or row.get("attack_family") in BENIGN or row.get("label") in BENIGN


def attack_family(row: dict[str, Any]) -> str:
    if is_benign(row):
        return "BENIGN"
    return str(row.get("attack_family") or row.get("label") or "ATTACK")


def stable_id(row: dict[str, Any], idx: int) -> str:
    raw = row.get("flow_id")
    if raw:
        return str(raw)
    parts = [
        str(row.get("src_ip", "")),
        str(row.get("src_port", "")),
        str(row.get("dst_ip", "")),
        str(row.get("dst_port", "")),
        str(idx),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]


def port_value(row: dict[str, Any]) -> int:
    for key in ("dst_port", "dstport"):
        try:
            return int(float(row.get(key) or 0))
        except Exception:
            continue
    return 0


def flow_vector(row: dict[str, Any]) -> np.ndarray:
    lens = row.get("lens") or []
    vals = []
    for value in lens[:127]:
        try:
            vals.append(float(value))
        except Exception:
            vals.append(-1.0)
    while len(vals) < 127:
        vals.append(-1.0)
    vals.insert(0, float(port_value(row)))
    arr = np.asarray(vals, dtype=np.float32)
    arr[0] = min(arr[0], 65535.0) / 65535.0
    lens_part = arr[1:]
    lens_part[:] = np.sign(lens_part) * np.log1p(np.abs(lens_part))
    positive = lens_part[lens_part >= 0]
    scale = float(np.percentile(positive, 95)) if positive.size else 1.0
    if scale <= 0:
        scale = 1.0
    lens_part[:] = lens_part / scale
    return arr


def iter_flows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_row_id"] = stable_id(row, idx)
            row["_attack_family"] = attack_family(row)
            yield idx, row


def load_flows(path: Path, max_rows: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    full_counts: Counter[str] = Counter()
    for _idx, row in iter_flows(path):
        full_counts[row["_attack_family"]] += 1

    total_rows = sum(full_counts.values())
    if max_rows is None or total_rows <= max_rows:
        rows: list[dict[str, Any]] = []
        for _idx, row in iter_flows(path):
            rows.append(row)
        return rows, {
            "source_rows": int(total_rows),
            "source_family_counts": dict(sorted(full_counts.items())),
            "sampling": "full",
        }

    families = sorted(full_counts)
    base_quota = max(1, max_rows // max(1, len(families)))
    quotas = {family: min(full_counts[family], base_quota) for family in families}
    remaining = max_rows - sum(quotas.values())
    expandable = [family for family in families if full_counts[family] > quotas[family]]
    while remaining > 0 and expandable:
        for family in list(expandable):
            if remaining <= 0:
                break
            quotas[family] += 1
            remaining -= 1
            if quotas[family] >= full_counts[family]:
                expandable.remove(family)

    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: Counter[str] = Counter()
    for _idx, row in iter_flows(path):
        family = row["_attack_family"]
        quota = quotas[family]
        if quota <= 0:
            continue
        seen[family] += 1
        bucket = selected[family]
        if len(bucket) < quota:
            bucket.append(row)
            continue
        replace_idx = rng.randint(0, seen[family] - 1)
        if replace_idx < quota:
            bucket[replace_idx] = row

    rows = [row for family in families for row in selected[family]]
    rng.shuffle(rows)
    return rows, {
        "source_rows": int(total_rows),
        "source_family_counts": dict(sorted(full_counts.items())),
        "sampling": "stratified_reservoir",
        "sampling_seed": seed,
        "target_max_rows": max_rows,
        "sample_quotas": dict(sorted((family, int(quotas[family])) for family in quotas)),
    }


def split_rows(
    rows: list[dict[str, Any]],
    seed: int,
    train_benign: int,
    val_benign: int,
    test_benign: int,
    test_attack_per_family: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    benign = [r for r in rows if is_benign(r)]
    attacks_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not is_benign(row):
            attacks_by_family[row["_attack_family"]].append(row)
    rng.shuffle(benign)
    for items in attacks_by_family.values():
        rng.shuffle(items)
    needed_benign = train_benign + val_benign + test_benign
    if len(benign) < needed_benign:
        train_benign = max(1, int(len(benign) * 0.55))
        val_benign = max(1, int(len(benign) * 0.20))
        test_benign = max(1, len(benign) - train_benign - val_benign)
    train = benign[:train_benign]
    val = benign[train_benign : train_benign + val_benign]
    test = benign[train_benign + val_benign : train_benign + val_benign + test_benign]
    attack_counts: dict[str, int] = {}
    for family, items in sorted(attacks_by_family.items()):
        chosen = items[:test_attack_per_family]
        test.extend(chosen)
        attack_counts[family] = len(chosen)
    rng.shuffle(test)
    meta = {
        "train_benign": len(train),
        "val_benign": len(val),
        "test_benign": sum(1 for r in test if is_benign(r)),
        "test_attack_by_family": attack_counts,
        "test_attack": sum(attack_counts.values()),
    }
    return train, val, test, meta


def build_triplets(rows: list[dict[str, Any]], max_triplets: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("src_ip", "")), str(row.get("dst_port", "")))].append(row)
    positives = [items for items in grouped.values() if len(items) >= 2]
    all_rows = rows[:]
    if len(positives) < 1 or len(all_rows) < 3:
        raise ValueError("Not enough benign data to form BSTS-Net triplets")

    triplets: list[np.ndarray] = []
    attempts = 0
    while len(triplets) < max_triplets and attempts < max_triplets * 20:
        attempts += 1
        group = rng.choice(positives)
        a, p = rng.sample(group, 2)
        n = rng.choice(all_rows)
        if n in group and len(positives) > 1:
            continue
        triplets.append(np.stack([flow_vector(a), flow_vector(p), flow_vector(n)], axis=0))
    if not triplets:
        raise ValueError("No triplets sampled")
    return np.stack(triplets, axis=0).astype(np.float32)


def train_model(
    train_rows: list[dict[str, Any]],
    output_model: Path,
    seed: int,
    max_triplets: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[BstsNet, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    triplets = build_triplets(train_rows, max_triplets, seed)
    data = torch.tensor(triplets, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = BstsNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    losses: list[float] = []
    model.train()
    for _epoch in range(epochs):
        epoch_losses: list[float] = []
        for (batch,) in loader:
            batch = batch.to(device)
            out_a, out_p, out_n = model(batch[:, 0, :], batch[:, 1, :], batch[:, 2, :])
            loss = loss_fn(out_a, out_p, out_n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model)
    return model, {"device": str(device), "triplets": int(triplets.shape[0]), "losses": losses}


def embed_rows(model: BstsNet, rows: list[dict[str, Any]], batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    vectors = np.stack([flow_vector(r) for r in rows], axis=0).astype(np.float32)
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(vectors), batch_size):
            batch = torch.tensor(vectors[start : start + batch_size], dtype=torch.float32, device=device)
            emb = model.forward_once(batch).detach().cpu().numpy()
            outs.append(emb)
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 16), dtype=np.float32)


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(len(values))
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))


def window_features(rows: list[dict[str, Any]], emb: np.ndarray, win_size: int, step: int) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(str(row.get("dst_port", "")), str(row.get("src_ip", "")))].append(idx)

    feats: list[np.ndarray] = []
    labels: list[int] = []
    metas: list[dict[str, Any]] = []
    for (port, src_ip), indices in groups.items():
        indices.sort(key=lambda i: float((rows[i].get("tss") or [0.0])[0] if rows[i].get("tss") else 0.0))
        if len(indices) <= win_size:
            windows = [indices]
        else:
            windows = [indices[i - win_size : i] for i in range(win_size, len(indices) + 1, step)]
        for win in windows:
            if not win:
                continue
            e = emb[win]
            dst_ips = [str(rows[i].get("dst_ip", "")) for i in win]
            labels_win = [0 if is_benign(rows[i]) else 1 for i in win]
            feat = np.concatenate(
                [
                    np.sqrt(np.mean(np.square(e), axis=0)),
                    np.asarray([len(win) / max(win_size, 1), len(set(dst_ips)) / max(len(win), 1), entropy(dst_ips)], dtype=np.float32),
                ]
            )
            feats.append(feat.astype(np.float32))
            labels.append(int(any(labels_win)))
            metas.append({"port": port, "src_ip": src_ip, "window_size": len(win), "attack_flows": int(sum(labels_win))})
    if not feats:
        return np.zeros((0, 19), dtype=np.float32), [], []
    return np.stack(feats, axis=0), labels, metas


def train_window_centroids(model: BstsNet, train_rows: list[dict[str, Any]], win_size: int, class_nums: int, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    emb = embed_rows(model, train_rows, batch_size)
    x, _, _ = window_features(train_rows, emb, win_size, max(1, win_size // 2))
    if x.shape[0] == 0:
        raise ValueError("No benign windows generated")
    k = min(max(1, class_nums), x.shape[0])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(x)
    sizes = Counter(int(v) for v in labels)
    large = [idx for idx, _ in sizes.most_common(max(1, k - 1))]
    benign_centroids = km.cluster_centers_[large]
    return benign_centroids.astype(np.float32), {"train_windows": int(x.shape[0]), "k": int(k), "benign_centroids": int(benign_centroids.shape[0])}


def score_windows(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    diff = x[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(dist, axis=1).astype(np.float32)


def entropy_rule_predict(
    x: np.ndarray,
    labels: np.ndarray,
    metas: list[dict[str, Any]],
    class_nums: int,
    delta1: int,
    delta2: float,
    delta3: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pred = np.zeros(labels.shape[0], dtype=np.int32)
    scores = np.zeros(labels.shape[0], dtype=np.float32)
    by_port: dict[str, list[int]] = defaultdict(list)
    for idx, meta in enumerate(metas):
        by_port[str(meta.get("port", ""))].append(idx)

    skipped_small = 0
    skipped_popular = 0
    clustered_ports = 0
    for _port, idxs in by_port.items():
        if len(idxs) < class_nums:
            skipped_small += 1
            continue
        if len(idxs) > max(1.0, labels.shape[0] / max(delta2, 1e-6)):
            skipped_popular += 1
            continue
        clustered_ports += 1
        x_port = x[idxs]
        k = min(class_nums, len(idxs))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(x_port)
        cluster_entropy: dict[int, float] = {}
        for cluster_id in range(k):
            member_pos = [pos for pos, cid in enumerate(cluster_ids) if int(cid) == cluster_id]
            if not member_pos:
                cluster_entropy[cluster_id] = float("inf")
                continue
            # The public Detect.py uses destination-IP dispersion as the final suspicious-cluster cue.
            # Our window feature stores destination-IP entropy as the last column.
            cluster_entropy[cluster_id] = float(np.mean(x_port[member_pos, -1]))
        finite = [v for v in cluster_entropy.values() if math.isfinite(v)]
        if not finite:
            continue
        min_cluster = min(cluster_entropy, key=lambda c: cluster_entropy[c])
        max_entropy = max(finite)
        for cluster_id in range(k):
            member_global = [idxs[pos] for pos, cid in enumerate(cluster_ids) if int(cid) == cluster_id]
            cluster_score = max_entropy - cluster_entropy.get(cluster_id, max_entropy)
            for global_idx in member_global:
                scores[global_idx] = float(cluster_score)
            if len(member_global) < delta1:
                continue
            if cluster_id == min_cluster and len(member_global) >= delta3:
                pred[member_global] = 1

    meta = {
        "ports": len(by_port),
        "clustered_ports": clustered_ports,
        "skipped_small_ports": skipped_small,
        "skipped_popular_ports": skipped_popular,
        "class_nums": class_nums,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }
    return pred, scores, meta


def threshold_at(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def fixed_fpr_threshold(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    benign_scores = scores[labels == 0]
    return threshold_at(benign_scores, 1.0 - target_fpr)


def low_tail_fpr_threshold(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    benign_scores = scores[labels == 0]
    return threshold_at(benign_scores, target_fpr)


def metrics_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int32)
    if labels.size == 0:
        return {}
    tn = int(np.sum((labels == 0) & (pred == 0)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    tp = int(np.sum((labels == 1) & (pred == 1)))
    fpr = fp / max(fp + tn, 1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
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


def metrics_at_low_tail(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores <= threshold).astype(np.int32)
    if labels.size == 0:
        return {}
    tn = int(np.sum((labels == 0) & (pred == 0)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    tp = int(np.sum((labels == 1) & (pred == 1)))
    fpr = fp / max(fp + tn, 1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
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


def metrics_from_pred(labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if labels.size == 0:
        return {}
    tn = int(np.sum((labels == 0) & (pred == 0)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    tp = int(np.sum((labels == 1) & (pred == 1)))
    fpr = fp / max(fp + tn, 1)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_alerts_per_10k_benign": float(fpr * 10000.0),
    }


def evaluate_run(args: argparse.Namespace, flow_file: Path, dataset_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    rows, load_meta = load_flows(flow_file, args.max_rows, args.seed)
    train, val, test, split_meta = split_rows(
        rows,
        args.seed,
        args.train_benign,
        args.val_benign,
        args.test_benign,
        args.test_attack_per_family,
    )
    run_dir = Path(args.output_dir) / dataset_name
    model, train_meta = train_model(
        train,
        run_dir / f"bsts_adapted_seed{args.seed}.pt",
        args.seed,
        args.max_triplets,
        args.epochs,
        args.batch_size,
        args.lr,
    )
    centroids, centroid_meta = train_window_centroids(model, train, args.win_size, args.class_nums, args.batch_size)

    val_emb = embed_rows(model, val, args.batch_size)
    test_emb = embed_rows(model, test, args.batch_size)
    val_x, val_labels_raw, val_metas = window_features(val, val_emb, args.win_size, max(1, args.win_size // 2))
    test_x, test_labels_raw, test_metas = window_features(test, test_emb, args.win_size, max(1, args.win_size // 2))
    val_labels = np.asarray(val_labels_raw, dtype=np.int32)
    test_labels = np.asarray(test_labels_raw, dtype=np.int32)
    val_scores = score_windows(val_x, centroids)
    test_scores = score_windows(test_x, centroids)
    entropy_pred, entropy_scores, entropy_meta = entropy_rule_predict(
        test_x,
        test_labels,
        test_metas,
        args.class_nums,
        args.delta1,
        args.delta2,
        args.delta3,
    )

    val_p99_threshold = threshold_at(val_scores, 0.99)
    val_p99 = metrics_at(test_labels, test_scores, val_p99_threshold)
    val_p01_threshold = threshold_at(val_scores, 0.01)
    val_p01 = metrics_at_low_tail(test_labels, test_scores, val_p01_threshold)
    fixed: dict[str, dict[str, float]] = {}
    fixed_low_tail: dict[str, dict[str, float]] = {}
    for target in (0.001, 0.01, 0.05):
        th = fixed_fpr_threshold(test_scores, test_labels, target)
        fixed[str(target)] = metrics_at(test_labels, test_scores, th)
        low_th = low_tail_fpr_threshold(test_scores, test_labels, target)
        fixed_low_tail[str(target)] = metrics_at_low_tail(test_labels, test_scores, low_th)

    y_has_both = len(set(test_labels.tolist())) == 2
    auc = float(roc_auc_score(test_labels, test_scores)) if y_has_both else float("nan")
    auc_low_tail = float(roc_auc_score(test_labels, -test_scores)) if y_has_both else float("nan")
    auprc = float(average_precision_score(test_labels, test_scores)) if y_has_both else float("nan")
    auprc_low_tail = float(average_precision_score(test_labels, -test_scores)) if y_has_both else float("nan")
    entropy_auc = float(roc_auc_score(test_labels, entropy_scores)) if y_has_both else float("nan")
    entropy_auprc = float(average_precision_score(test_labels, entropy_scores)) if y_has_both else float("nan")

    score_rows = []
    for split_name, labels, scores, metas in (
        ("val", val_labels, val_scores, val_metas),
        ("test", test_labels, test_scores, test_metas),
    ):
        for idx, (label, score, meta) in enumerate(zip(labels, scores, metas)):
            row = {"split": split_name, "window_id": idx, "is_attack": int(label), "score": float(score), **meta}
            if split_name == "test":
                row["inverse_distance_score"] = float(-score)
                row["entropy_rule_score"] = float(entropy_scores[idx])
                row["entropy_rule_pred"] = int(entropy_pred[idx])
            score_rows.append(row)

    metrics = {
        "dataset": dataset_name,
        "method": "bsts_net_adapted_core",
        "flow_file": str(flow_file),
        "seed": args.seed,
        "official_reproduction_status": "adapted_core_reproduction",
        "limitations": [
            "Upstream testData.zip is a Git LFS pointer locally and is not usable without LFS data.",
            "Upstream repository does not include complete IDS2017/IDS2018 configs or pretrained checkpoints.",
            "This run preserves the BSTS-Net network shape and source-IP/destination-port window clustering idea, but uses local FlowPrim JSONL artifacts.",
        ],
        "counts": {
            "loaded_rows": len(rows),
            **load_meta,
            **split_meta,
            "val_windows": int(len(val_labels)),
            "test_windows": int(len(test_labels)),
            "test_attack_windows": int(np.sum(test_labels == 1)),
            "test_benign_windows": int(np.sum(test_labels == 0)),
        },
        "train": train_meta,
        "centroids": centroid_meta,
        "auroc": auc,
        "inverse_distance_auroc": auc_low_tail,
        "auprc": auprc,
        "inverse_distance_auprc": auprc_low_tail,
        "val_p99": val_p99,
        "val_p01_low_tail": val_p01,
        "fixed_fpr": fixed,
        "fixed_fpr_low_tail": fixed_low_tail,
        "entropy_rule": {
            **metrics_from_pred(test_labels, entropy_pred),
            "auroc": entropy_auc,
            "auprc": entropy_auprc,
            "meta": entropy_meta,
        },
        "config": vars(args),
        "elapsed_sec": time.perf_counter() - started,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"metrics_seed{args.seed}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (run_dir / f"scores_seed{args.seed}.csv").open("w", newline="", encoding="utf-8") as f:
        fields: list[str] = []
        for row in score_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(score_rows)
    return metrics


def flatten(metrics: dict[str, Any]) -> dict[str, Any]:
    fixed = metrics["fixed_fpr"]
    fixed_low = metrics["fixed_fpr_low_tail"]
    return {
        "dataset": metrics["dataset"],
        "method": metrics["method"],
        "seed": metrics["seed"],
        "loaded_rows": metrics["counts"]["loaded_rows"],
        "train_benign": metrics["counts"]["train_benign"],
        "val_benign": metrics["counts"]["val_benign"],
        "test_benign": metrics["counts"]["test_benign"],
        "test_attack": metrics["counts"]["test_attack"],
        "val_windows": metrics["counts"]["val_windows"],
        "test_windows": metrics["counts"]["test_windows"],
        "auroc": metrics["auroc"],
        "inverse_distance_auroc": metrics["inverse_distance_auroc"],
        "auprc": metrics["auprc"],
        "inverse_distance_auprc": metrics["inverse_distance_auprc"],
        "val_p99_fpr": metrics["val_p99"].get("fpr", float("nan")),
        "val_p99_precision": metrics["val_p99"].get("precision", float("nan")),
        "val_p99_recall": metrics["val_p99"].get("recall", float("nan")),
        "val_p99_f1": metrics["val_p99"].get("f1", float("nan")),
        "val_p01_low_tail_fpr": metrics["val_p01_low_tail"].get("fpr", float("nan")),
        "val_p01_low_tail_recall": metrics["val_p01_low_tail"].get("recall", float("nan")),
        "entropy_rule_fpr": metrics["entropy_rule"].get("fpr", float("nan")),
        "entropy_rule_precision": metrics["entropy_rule"].get("precision", float("nan")),
        "entropy_rule_recall": metrics["entropy_rule"].get("recall", float("nan")),
        "entropy_rule_f1": metrics["entropy_rule"].get("f1", float("nan")),
        "entropy_rule_auroc": metrics["entropy_rule"].get("auroc", float("nan")),
        "recall_at_0_1pct_fpr": fixed["0.001"].get("recall", float("nan")),
        "precision_at_0_1pct_fpr": fixed["0.001"].get("precision", float("nan")),
        "actual_fpr_at_0_1pct_fpr": fixed["0.001"].get("fpr", float("nan")),
        "recall_at_1pct_fpr": fixed["0.01"].get("recall", float("nan")),
        "precision_at_1pct_fpr": fixed["0.01"].get("precision", float("nan")),
        "actual_fpr_at_1pct_fpr": fixed["0.01"].get("fpr", float("nan")),
        "low_tail_recall_at_1pct_fpr": fixed_low["0.01"].get("recall", float("nan")),
        "low_tail_precision_at_1pct_fpr": fixed_low["0.01"].get("precision", float("nan")),
        "low_tail_actual_fpr_at_1pct_fpr": fixed_low["0.01"].get("fpr", float("nan")),
        "recall_at_5pct_fpr": fixed["0.05"].get("recall", float("nan")),
        "low_tail_recall_at_5pct_fpr": fixed_low["0.05"].get("recall", float("nan")),
        "elapsed_sec": metrics["elapsed_sec"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapted BSTS-Net core reproduction on local flow JSONL artifacts.")
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "FLOW_JSONL"), required=True)
    parser.add_argument("--output-dir", default="experiments/bsts_net_reproduction/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--train-benign", type=int, default=2500)
    parser.add_argument("--val-benign", type=int, default=1000)
    parser.add_argument("--test-benign", type=int, default=2000)
    parser.add_argument("--test-attack-per-family", type=int, default=2500)
    parser.add_argument("--max-triplets", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--win-size", type=int, default=10)
    parser.add_argument("--class-nums", type=int, default=6)
    parser.add_argument("--delta1", type=int, default=3)
    parser.add_argument("--delta2", type=float, default=100.0)
    parser.add_argument("--delta3", type=int, default=3)
    args = parser.parse_args()

    summaries = []
    for name, path in args.dataset:
        metrics = evaluate_run(args, Path(path), name)
        row = flatten(metrics)
        summaries.append(row)
        print(
            f"{name} seed={args.seed} auroc={row['auroc']:.4f} "
            f"val_p99_fpr={row['val_p99_fpr']:.4f} "
            f"r@1%fpr={row['recall_at_1pct_fpr']:.4f}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"bsts_adapted_summary_seed{args.seed}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
