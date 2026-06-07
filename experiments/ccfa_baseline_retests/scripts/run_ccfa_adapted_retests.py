#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors


SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}
BENIGN_VALUES = {"BENIGN", "Benign", "benign", "0", 0}

IDS2017_ATTACKS = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
}

IDS2018_ATTACKS = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "DoS": "dos",
    "Infiltration": "infiltration",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split_dir: Path
    token_dir: Path
    flow_file: Path | None
    selected_flow_dir: Path | None
    attacks: dict[str, str]
    default_seeds: list[int]
    status_note: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


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
    return row.get("binary_label") in BENIGN_VALUES or row.get("label") in BENIGN_VALUES or row.get("attack_family") in BENIGN_VALUES


def attack_family(row: dict[str, Any]) -> str:
    if is_benign(row):
        return "BENIGN"
    return str(row.get("attack_family") or row.get("label") or "ATTACK")


def read_flows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            flow_id = stable_id(row, idx)
            row["_row_id"] = flow_id
            row["_attack_family"] = attack_family(row)
            rows[flow_id] = row
    return rows


def token_path_for(spec: DatasetSpec, attack: str, seed: int) -> Path:
    slug = spec.attacks[attack]
    candidates = sorted(spec.token_dir.glob(f"*leave_one_{slug}_anomaly_seed{seed}_a3_full_rhythm.pt"))
    if not candidates:
        raise FileNotFoundError(f"No token artifact for {spec.name} {attack} seed={seed} in {spec.token_dir}")
    return candidates[0]


def split_path_for(spec: DatasetSpec, attack: str, seed: int) -> Path:
    slug = spec.attacks[attack]
    path = spec.split_dir / f"splits_leave_one_{slug}_anomaly_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def selected_flow_path_for(spec: DatasetSpec, attack: str, seed: int) -> Path | None:
    if spec.selected_flow_dir is None:
        return None
    slug = spec.attacks[attack]
    candidates = sorted(spec.selected_flow_dir.glob(f"*leave_one_{slug}_anomaly_seed{seed}_selected_flows.jsonl"))
    return candidates[0] if candidates else None


def load_token_data(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def split_indices(token_data: dict[str, Any], split: str) -> np.ndarray:
    return np.asarray([idx for idx, meta in enumerate(token_data.get("meta", [])) if meta.get("split") == split], dtype=np.int64)


def rows_in_token_order(token_data: dict[str, Any], flow_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for meta in token_data.get("meta", []):
        flow_id = str(meta.get("flow_id") or "")
        row = flow_by_id.get(flow_id)
        if row is None:
            missing.append(flow_id)
            row = {
                "_row_id": flow_id,
                "_attack_family": str(meta.get("attack_family") or meta.get("label") or "UNKNOWN"),
                "binary_label": meta.get("binary_label"),
                "attack_family": meta.get("attack_family"),
                "proto": "unknown",
                "protocol": "unknown",
                "lens": [],
                "dirs": [],
                "tss": [],
                "iats": [],
            }
        rows.append(row)
    if missing:
        raise KeyError(f"{len(missing)} flow ids from token artifact missing in flow source; first={missing[:3]}")
    return rows


def labels_from_token(token_data: dict[str, Any]) -> np.ndarray:
    return token_data["binary_labels"].cpu().numpy().astype(np.int32)


def numeric_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def bool_dirs(values: Any, n: int) -> list[bool]:
    if not isinstance(values, list):
        return [True] * n
    out = [bool(value) for value in values[:n]]
    while len(out) < n:
        out.append(True)
    return out


def entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(len(values))
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values() if count > 0))


def safe_stats(values: list[float], percentiles: list[int] | None = None) -> list[float]:
    if not values:
        pct = percentiles or []
        return [0.0, 0.0, 0.0, 0.0] + [0.0 for _ in pct]
    arr = np.asarray(values, dtype=np.float32)
    out = [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))]
    for pct in percentiles or []:
        out.append(float(np.percentile(arr, pct)))
    return out


def burst_stats(lens: list[float], dirs: list[bool]) -> list[float]:
    if not lens:
        return [0.0] * 10
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
    sizes = [sum(abs(v) for v in vals) for _direction, vals in bursts]
    counts = [len(vals) for _direction, vals in bursts]
    c2s = sum(1 for direction, _vals in bursts if direction)
    s2c = len(bursts) - c2s
    return [
        float(len(bursts)),
        float(max(0, len(bursts) - 1)),
        float(c2s),
        float(s2c),
        float(c2s / max(len(bursts), 1)),
        *safe_stats(sizes, [50]),
        *safe_stats(counts, [50])[:1],
    ][:10]


def local_flow_features(rows: list[dict[str, Any]], max_packets: int = 48) -> np.ndarray:
    feats: list[np.ndarray] = []
    for row in rows:
        lens = numeric_list(row.get("lens"))
        tss = numeric_list(row.get("tss"))
        iats = numeric_list(row.get("iats"))
        dirs = bool_dirs(row.get("dirs"), len(lens))
        signed = []
        for length, direction in zip(lens[:max_packets], dirs[:max_packets]):
            signed.append((1.0 if direction else -1.0) * math.log1p(abs(length)))
        while len(signed) < max_packets:
            signed.append(0.0)
        abs_lens = [abs(v) for v in lens]
        c2s_lens = [abs(v) for v, direction in zip(lens, dirs) if direction]
        s2c_lens = [abs(v) for v, direction in zip(lens, dirs) if not direction]
        if not iats and len(tss) >= 2:
            iats = [max(0.0, b - a) for a, b in zip(tss[:-1], tss[1:])]
        bins = [min(15, int(math.log2(max(1.0, abs(v))))) for v in lens]
        duration = float(row.get("duration") or ((tss[-1] - tss[0]) if len(tss) >= 2 else 0.0) or 0.0)
        packet_count = float(row.get("packet_count") or len(lens))
        byte_count = float(row.get("byte_count") or sum(abs_lens))
        stats = [
            packet_count,
            math.log1p(max(duration, 0.0)),
            math.log1p(max(byte_count, 0.0)),
            float(len(c2s_lens) / max(len(lens), 1)),
            float(sum(c2s_lens) / max(sum(abs_lens), 1.0)) if abs_lens else 0.0,
            entropy(bins),
            entropy([bool(v) for v in dirs]),
            float(len(set(bins)) / max(len(bins), 1)),
        ]
        stats.extend(safe_stats(abs_lens, [25, 50, 75, 90]))
        stats.extend(safe_stats(c2s_lens, [50]))
        stats.extend(safe_stats(s2c_lens, [50]))
        stats.extend(safe_stats(iats, [50, 90, 99]))
        stats.extend(burst_stats(lens, dirs))
        arr = np.asarray(signed + stats, dtype=np.float32)
        arr[~np.isfinite(arr)] = 0.0
        feats.append(arr)
    return np.stack(feats, axis=0).astype(np.float32)


def fft_features(rows: list[dict[str, Any]], length: int = 64, bins: int = 24) -> np.ndarray:
    feats: list[np.ndarray] = []
    for row in rows:
        lens = numeric_list(row.get("lens"))
        dirs = bool_dirs(row.get("dirs"), len(lens))
        iats = numeric_list(row.get("iats"))
        signed = np.zeros(length, dtype=np.float32)
        timing = np.zeros(length, dtype=np.float32)
        for idx, (packet_len, direction) in enumerate(zip(lens[:length], dirs[:length])):
            signed[idx] = (1.0 if direction else -1.0) * math.log1p(abs(packet_len))
        for idx, value in enumerate(iats[:length]):
            timing[idx] = math.log1p(max(float(value), 0.0))
        s_fft = np.abs(np.fft.rfft(signed))[:bins]
        t_fft = np.abs(np.fft.rfft(timing))[:bins]
        s_energy = float(np.sum(s_fft * s_fft))
        t_energy = float(np.sum(t_fft * t_fft))
        idxs = np.arange(len(s_fft), dtype=np.float32)
        s_centroid = float(np.sum(idxs * s_fft) / max(np.sum(s_fft), 1e-6))
        t_centroid = float(np.sum(idxs * t_fft) / max(np.sum(t_fft), 1e-6))
        arr = np.asarray(
            list(np.log1p(s_fft)) + list(np.log1p(t_fft)) + [math.log1p(s_energy), math.log1p(t_energy), s_centroid, t_centroid],
            dtype=np.float32,
        )
        arr[~np.isfinite(arr)] = 0.0
        feats.append(arr)
    return np.stack(feats, axis=0).astype(np.float32)


def endpoint_key(value: Any) -> str:
    return str(value or "NA")


def service_key(row: dict[str, Any]) -> str:
    proto = str(row.get("proto") or row.get("protocol") or "NA").upper()
    dst_port = str(row.get("dst_port") or "NA")
    return f"{proto}:{dst_port}"


def graph_features(rows: list[dict[str, Any]], train_idx: np.ndarray) -> np.ndarray:
    train_rows = [rows[int(idx)] for idx in train_idx.tolist()]
    src_counts: Counter[str] = Counter()
    dst_counts: Counter[str] = Counter()
    endpoint_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    service_counts: Counter[str] = Counter()
    src_service_counts: Counter[tuple[str, str]] = Counter()
    dst_service_counts: Counter[tuple[str, str]] = Counter()
    dst_ports_by_src: defaultdict[str, set[str]] = defaultdict(set)
    srcs_by_dst: defaultdict[str, set[str]] = defaultdict(set)
    dsts_by_src: defaultdict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        src = endpoint_key(row.get("src_ip"))
        dst = endpoint_key(row.get("dst_ip"))
        svc = service_key(row)
        src_counts[src] += 1
        dst_counts[dst] += 1
        endpoint_counts[src] += 1
        endpoint_counts[dst] += 1
        pair_counts[(src, dst)] += 1
        service_counts[svc] += 1
        src_service_counts[(src, svc)] += 1
        dst_service_counts[(dst, svc)] += 1
        dst_ports_by_src[src].add(str(row.get("dst_port") or "NA"))
        srcs_by_dst[dst].add(src)
        dsts_by_src[src].add(dst)
    total = max(len(train_rows), 1)
    out: list[list[float]] = []
    for row in rows:
        src = endpoint_key(row.get("src_ip"))
        dst = endpoint_key(row.get("dst_ip"))
        svc = service_key(row)
        values = [
            math.log1p(src_counts[src]),
            math.log1p(dst_counts[dst]),
            math.log1p(endpoint_counts[src]),
            math.log1p(endpoint_counts[dst]),
            math.log1p(pair_counts[(src, dst)]),
            math.log1p(service_counts[svc]),
            math.log1p(src_service_counts[(src, svc)]),
            math.log1p(dst_service_counts[(dst, svc)]),
            math.log1p(len(dst_ports_by_src[src])),
            math.log1p(len(srcs_by_dst[dst])),
            math.log1p(len(dsts_by_src[src])),
            1.0 if src_counts[src] == 0 else 0.0,
            1.0 if dst_counts[dst] == 0 else 0.0,
            1.0 if pair_counts[(src, dst)] == 0 else 0.0,
            1.0 if service_counts[svc] == 0 else 0.0,
            -math.log((service_counts[svc] + 1.0) / (total + len(service_counts) + 1.0)),
        ]
        out.append(values)
    return np.asarray(out, dtype=np.float32)


def token_bow_features(token_data: dict[str, Any], train_idx: np.ndarray, mode: str = "tfidf_l2") -> np.ndarray:
    input_ids = token_data["input_ids"].cpu().numpy()
    attention = token_data["attention_mask"].cpu().numpy()
    vocab = token_data["vocab"]
    id_to_token = {int(idx): str(token) for token, idx in vocab.items()}
    kept_ids = sorted(idx for idx, token in id_to_token.items() if token not in SPECIAL_TOKENS)
    rows = np.zeros((input_ids.shape[0], len(vocab)), dtype=np.float32)
    for row_idx in range(input_ids.shape[0]):
        active = input_ids[row_idx][attention[row_idx] > 0]
        if active.size:
            rows[row_idx] = np.bincount(active, minlength=len(vocab))[: len(vocab)]
    rows = rows[:, np.asarray(kept_ids, dtype=np.int64)]
    if mode.startswith("binary"):
        rows = (rows > 0).astype(np.float32)
    if mode.startswith("tfidf"):
        train_counts = rows[train_idx]
        df = np.sum(train_counts > 0, axis=0)
        idf = np.log((1.0 + len(train_idx)) / (1.0 + df)) + 1.0
        rows = rows * idf.reshape(1, -1).astype(np.float32)
    if mode.endswith("l2"):
        denom = np.linalg.norm(rows, axis=1, keepdims=True)
        rows = np.divide(rows, denom, out=np.zeros_like(rows), where=denom > 0)
    elif mode.endswith("l1"):
        denom = np.sum(np.abs(rows), axis=1, keepdims=True)
        rows = np.divide(rows, denom, out=np.zeros_like(rows), where=denom > 0)
    rows[~np.isfinite(rows)] = 0.0
    return rows.astype(np.float32)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(train_x: np.ndarray, *arrays: np.ndarray, clip: float = 8.0) -> tuple[np.ndarray, ...]:
    mean, std = fit_standardizer(train_x)
    out = []
    for x in (train_x, *arrays):
        z = (x - mean) / std
        z = np.clip(z, -clip, clip)
        z[~np.isfinite(z)] = 0.0
        out.append(z.astype(np.float32))
    return tuple(out)


def combine_features(*arrays: np.ndarray) -> np.ndarray:
    parts = [arr.astype(np.float32, copy=False) for arr in arrays if arr.size]
    return np.concatenate(parts, axis=1).astype(np.float32)


def knn_scores(train_x: np.ndarray, eval_x: np.ndarray, *, k: int = 5, metric: str = "euclidean") -> np.ndarray:
    if train_x.shape[0] == 0 or eval_x.shape[0] == 0:
        return np.zeros((eval_x.shape[0],), dtype=np.float32)
    kk = max(1, min(k, train_x.shape[0]))
    nbrs = NearestNeighbors(n_neighbors=kk, metric=metric, algorithm="auto")
    nbrs.fit(train_x)
    dist, _idx = nbrs.kneighbors(eval_x, return_distance=True)
    return np.mean(dist, axis=1).astype(np.float32)


def isolation_scores(train_x: np.ndarray, eval_x: np.ndarray, seed: int) -> np.ndarray:
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1)
    model.fit(train_x)
    return (-model.score_samples(eval_x)).astype(np.float32)


def prototype_scores(train_x: np.ndarray, eval_x: np.ndarray, seed: int, max_clusters: int = 12) -> np.ndarray:
    if train_x.shape[0] < 4:
        return knn_scores(train_x, eval_x, k=1)
    n_clusters = max(2, min(max_clusters, int(math.sqrt(train_x.shape[0] / 2.0)) + 1, train_x.shape[0] // 10))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(train_x)
    centers = km.cluster_centers_.astype(np.float32)
    variances: list[np.ndarray] = []
    priors: list[float] = []
    for cluster in range(n_clusters):
        subset = train_x[labels == cluster]
        if subset.shape[0] < 2:
            var = np.var(train_x, axis=0) + 1e-3
        else:
            var = np.var(subset, axis=0) + 1e-3
        variances.append(var.astype(np.float32))
        priors.append(float(subset.shape[0] / max(train_x.shape[0], 1)))
    scores = []
    for x in eval_x:
        dist = []
        for center, var, prior in zip(centers, variances, priors):
            mahal = float(np.mean(((x - center) ** 2) / var))
            dist.append(mahal - math.log(max(prior, 1e-6)))
        scores.append(min(dist))
    return np.asarray(scores, dtype=np.float32)


class AutoEncoder(nn.Module):
    def __init__(self, dim: int, latent: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def score_autoencoder(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    contrastive: bool,
    robust_trim: float,
    dropout_noise: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim = int(train_x.shape[1])
    latent = max(8, min(96, dim // 3 if dim >= 24 else dim))
    hidden = max(32, min(256, dim * 2))
    model = AutoEncoder(dim, latent, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_tensor = torch.tensor(train_x, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_tensor), batch_size=batch_size, shuffle=True, drop_last=False)
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for _epoch in range(max(1, epochs)):
        batch_losses: list[float] = []
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            aug = batch
            if dropout_noise > 0:
                mask = (torch.rand_like(batch) > dropout_noise).float()
                aug = batch * mask
            recon, z = model(aug)
            per_row = torch.mean((recon - batch) ** 2, dim=1)
            if 0.0 < robust_trim < 1.0 and per_row.numel() > 4:
                keep = max(1, int(math.ceil(per_row.numel() * robust_trim)))
                loss_recon = torch.topk(per_row, keep, largest=False).values.mean()
            else:
                loss_recon = per_row.mean()
            loss = loss_recon
            if contrastive:
                aug2 = batch
                if dropout_noise > 0:
                    mask2 = (torch.rand_like(batch) > dropout_noise).float()
                    aug2 = batch * mask2
                _recon2, z2 = model(aug2)
                z1 = F.normalize(z, dim=1)
                z2n = F.normalize(z2, dim=1)
                align = 1.0 - torch.mean(torch.sum(z1 * z2n, dim=1))
                var = torch.mean(torch.relu(0.5 - torch.std(z, dim=0)))
                loss = loss + 0.15 * align + 0.02 * var
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)) if batch_losses else float("nan"))

    def encode_and_recon(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        recon_errs: list[np.ndarray] = []
        zs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                batch_np = x[start : start + batch_size]
                batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
                recon, z = model(batch)
                err = torch.mean((recon - batch) ** 2, dim=1).detach().cpu().numpy()
                recon_errs.append(err.astype(np.float32))
                zs.append(z.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(recon_errs, axis=0), np.concatenate(zs, axis=0)

    train_recon, train_z = encode_and_recon(train_x)
    val_recon, val_z = encode_and_recon(val_x)
    test_recon, test_z = encode_and_recon(test_x)
    train_z_s, val_z_s, test_z_s = standardize(train_z, val_z, test_z, clip=8.0)
    val_knn = knn_scores(train_z_s, val_z_s, k=5)
    test_knn = knn_scores(train_z_s, test_z_s, k=5)
    recon_mean = float(np.mean(train_recon))
    recon_std = float(np.std(train_recon) if np.std(train_recon) > 1e-8 else 1.0)
    train_knn = knn_scores(train_z_s, train_z_s, k=min(6, train_z_s.shape[0]))
    knn_mean = float(np.mean(train_knn))
    knn_std = float(np.std(train_knn) if np.std(train_knn) > 1e-8 else 1.0)
    val_scores = ((val_recon - recon_mean) / recon_std + (val_knn - knn_mean) / knn_std).astype(np.float32)
    test_scores = ((test_recon - recon_mean) / recon_std + (test_knn - knn_mean) / knn_std).astype(np.float32)
    meta = {
        "device": str(device),
        "nn_dim": dim,
        "nn_hidden": hidden,
        "nn_latent": latent,
        "nn_epochs": int(epochs),
        "nn_final_loss": float(losses[-1]) if losses else float("nan"),
        "nn_elapsed_sec": float(time.perf_counter() - started),
    }
    return val_scores, test_scores, meta


def fpr_at_recall(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return float("nan")
    fpr, tpr, _thresholds = roc_curve(labels, scores)
    candidates = fpr[tpr >= target_recall]
    return float(np.min(candidates)) if candidates.size else 1.0


def threshold_for_target_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    benign = scores[labels == 0]
    if benign.size == 0:
        return float(np.max(scores) + 1e-12)
    return float(np.quantile(benign, 1.0 - target_fpr))


def metrics_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int32)
    tn = int(np.sum((labels == 0) & (pred == 0)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    tp = int(np.sum((labels == 1) & (pred == 1)))
    fpr = fp / max(fp + tn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
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


def metric_bundle(labels: np.ndarray, scores: np.ndarray, val_scores: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.int32)
    scores = scores.astype(np.float64)
    val_scores = val_scores.astype(np.float64)
    has_both = labels.size > 0 and len(set(labels.tolist())) == 2
    out: dict[str, float] = {
        "auroc": float(roc_auc_score(labels, scores)) if has_both else float("nan"),
        "auprc": float(average_precision_score(labels, scores)) if has_both else float("nan"),
        "fpr95": fpr_at_recall(labels, scores, 0.95),
    }
    for target, label in [(0.001, "0.001"), (0.01, "0.01"), (0.05, "0.05")]:
        thr = threshold_for_target_fpr(labels, scores, target)
        vals = metrics_at(labels, scores, thr)
        out[f"recall_at_{label}_fpr"] = float(vals["recall"])
        out[f"precision_at_{label}_fpr"] = float(vals["precision"])
        out[f"actual_fpr_at_{label}_fpr"] = float(vals["fpr"])
        out[f"macro_f1_at_{label}_fpr"] = float(f1_score(labels, (scores >= thr).astype(np.int32), average="macro", zero_division=0))
    p99 = float(np.quantile(val_scores, 0.99)) if val_scores.size else float("inf")
    p99_vals = metrics_at(labels, scores, p99)
    pred = (scores >= p99).astype(np.int32)
    out["val_p99_threshold"] = p99
    out["val_p99_realized_fpr"] = float(p99_vals["fpr"])
    out["val_p99_attack_recall"] = float(p99_vals["recall"])
    out["val_p99_precision"] = float(precision_score(labels, pred, zero_division=0))
    out["val_p99_f1"] = float(p99_vals["f1"])
    out["val_p99_macro_f1"] = float(f1_score(labels, pred, average="macro", zero_division=0))
    out["false_alerts_per_10k_benign"] = float(p99_vals["false_alerts_per_10k_benign"])
    return out


def method_scores(
    method: str,
    features: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    nn_epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str, str]:
    if method == "trident":
        x = combine_features(features["token_tfidf"], features["local"], features["graph"])
        train_x, val_x, test_x = standardize(x[train_idx], x[val_idx], x[test_idx], clip=8.0)
        return (
            prototype_scores(train_x, val_x, seed),
            prototype_scores(train_x, test_x, seed),
            {"score_model": "kmeans_diag_mahalanobis", "feature_view": "token+local+interaction"},
            "Trident-style open-world prototypes",
            "tamm_protocol_adapted_no_official_code_found",
        )
    if method == "hypervision":
        x = combine_features(features["graph"], features["local"])
        train_x, val_x, test_x = standardize(x[train_idx], x[val_idx], x[test_idx], clip=8.0)
        return (
            knn_scores(train_x, val_x, k=5),
            knn_scores(train_x, test_x, k=5),
            {"score_model": "interaction_graph_knn", "feature_view": "endpoint_interaction+local_flow"},
            "HyperVision-style interaction graph KNN",
            "tamm_protocol_adapted_endpoint_graph",
        )
    if method == "contramtd":
        x = combine_features(features["local"], features["graph"], features["token_binary"])
        train_x, val_x, test_x = standardize(x[train_idx], x[val_idx], x[test_idx], clip=8.0)
        val_scores, test_scores, meta = score_autoencoder(
            train_x,
            val_x,
            test_x,
            seed=seed + 101,
            epochs=nn_epochs,
            batch_size=batch_size,
            lr=1e-3,
            contrastive=True,
            robust_trim=1.0,
            dropout_noise=0.12,
        )
        meta.update({"score_model": "contrastive_local_global_ae", "feature_view": "local+interaction+token"})
        return val_scores, test_scores, meta, "ContraMTD-style local-global contrastive", "tamm_protocol_adapted_contrastive"
    if method == "cade":
        x = combine_features(features["token_tfidf"], features["local"])
        train_x, val_x, test_x = standardize(x[train_idx], x[val_idx], x[test_idx], clip=8.0)
        val_scores, test_scores, meta = score_autoencoder(
            train_x,
            val_x,
            test_x,
            seed=seed + 202,
            epochs=nn_epochs,
            batch_size=batch_size,
            lr=1e-3,
            contrastive=True,
            robust_trim=1.0,
            dropout_noise=0.18,
        )
        meta.update({"score_model": "contrastive_autoencoder_knn", "feature_view": "token+local"})
        return val_scores, test_scores, meta, "CADE-style contrastive AE", "tamm_protocol_adapted_benign_only"
    if method == "rapier":
        x = combine_features(features["token_tfidf"], features["local"], features["fft"])
        train_x, val_x, test_x = standardize(x[train_idx], x[val_idx], x[test_idx], clip=8.0)
        val_scores, test_scores, meta = score_autoencoder(
            train_x,
            val_x,
            test_x,
            seed=seed + 303,
            epochs=nn_epochs,
            batch_size=batch_size,
            lr=1e-3,
            contrastive=False,
            robust_trim=0.85,
            dropout_noise=0.20,
        )
        meta.update({"score_model": "trimmed_robust_autoencoder", "feature_view": "token+local+frequency"})
        return val_scores, test_scores, meta, "RAPIER-style robust detector", "tamm_protocol_adapted_noisy_label_robust"
    if method == "whisper":
        x = features["fft"]
        train_x, val_x, test_x = standardize(x[train_idx], x[val_idx], x[test_idx], clip=8.0)
        return (
            isolation_scores(train_x, val_x, seed),
            isolation_scores(train_x, test_x, seed),
            {"score_model": "frequency_domain_isolation_forest", "feature_view": "packet_fft"},
            "Whisper-style frequency detector",
            "tamm_protocol_adapted_frequency_features",
        )
    if method == "trafficformer":
        x = features["token_tfidf"]
        train_x, val_x, test_x = standardize(x[train_idx], x[val_idx], x[test_idx], clip=8.0)
        return (
            knn_scores(train_x, val_x, k=5, metric="cosine"),
            knn_scores(train_x, test_x, k=5, metric="cosine"),
            {"score_model": "token_representation_knn", "feature_view": "behavior_token_tfidf"},
            "TrafficFormer-style token representation KNN",
            "tamm_protocol_adapted_representation_proxy",
        )
    raise ValueError(f"Unsupported method: {method}")


def evaluate_run(
    spec: DatasetSpec,
    attack: str,
    seed: int,
    methods: list[str],
    out_dir: Path,
    nn_epochs: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    split_path = split_path_for(spec, attack, seed)
    token_path = token_path_for(spec, attack, seed)
    split_doc = read_json(split_path)
    token_data = load_token_data(token_path)
    flow_file = selected_flow_path_for(spec, attack, seed) or spec.flow_file
    if flow_file is None:
        raise FileNotFoundError(f"No flow source configured for {spec.name} {attack} seed={seed}")
    flow_by_id = read_flows(flow_file)
    rows = rows_in_token_order(token_data, flow_by_id)
    train_idx = split_indices(token_data, "train")
    val_idx = split_indices(token_data, "val")
    test_idx = split_indices(token_data, "test")
    labels = labels_from_token(token_data)

    features = {
        "local": local_flow_features(rows),
        "fft": fft_features(rows),
        "graph": graph_features(rows, train_idx),
        "token_tfidf": token_bow_features(token_data, train_idx, mode="tfidf_l2"),
        "token_binary": token_bow_features(token_data, train_idx, mode="binary_l2"),
    }
    test_labels = labels[test_idx]
    rows_out: list[dict[str, Any]] = []
    for method in methods:
        started = time.perf_counter()
        val_scores, test_scores, meta, display, status = method_scores(
            method,
            features,
            train_idx,
            val_idx,
            test_idx,
            seed,
            nn_epochs,
            batch_size,
        )
        metric = metric_bundle(test_labels, test_scores, val_scores)
        row: dict[str, Any] = {
            "dataset": spec.name,
            "heldout_attack": attack,
            "seed": seed,
            "split_file": str(split_path),
            "token_file": str(token_path),
            "flow_file": str(flow_file),
            "source_rows": len(flow_by_id),
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "test_attack_flows": int(np.sum(test_labels == 1)),
            "test_benign_flows": int(np.sum(test_labels == 0)),
            "method": display,
            "method_id": method,
            "eval_unit": "flow",
            "official_reproduction_status": status,
            "protocol_note": spec.status_note,
            "elapsed_sec": float(time.perf_counter() - started),
            **meta,
            **metric,
        }
        rows_out.append(row)
        run_dir = out_dir / "scores" / spec.name.lower().replace("-", "_") / f"leave_one_{spec.attacks[attack]}_seed{seed}"
        score_path = run_dir / f"{method}_scores.csv"
        score_rows = []
        for idx, label, score in zip(test_idx.tolist(), test_labels.tolist(), test_scores.tolist()):
            item = rows[int(idx)]
            score_rows.append(
                {
                    "flow_id": item.get("_row_id") or item.get("flow_id"),
                    "label": int(label),
                    "attack_family": item.get("_attack_family") or attack_family(item),
                    "score": float(score),
                }
            )
        write_csv(score_rows, score_path)
    return rows_out


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return float(statistics.fmean(clean)), float(statistics.stdev(clean))


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
        "elapsed_sec",
    ]
    out: list[dict[str, Any]] = []
    group_keys = sorted({(row["dataset"], row["method"], row["eval_unit"], row["heldout_attack"]) for row in rows})
    for dataset, method, unit, attack in group_keys:
        group = [row for row in rows if row["dataset"] == dataset and row["method"] == method and row["eval_unit"] == unit and row["heldout_attack"] == attack]
        item: dict[str, Any] = {"dataset": dataset, "method": method, "eval_unit": unit, "heldout_attack": attack, "runs": len(group)}
        item["official_reproduction_status"] = "; ".join(sorted({str(row["official_reproduction_status"]) for row in group}))
        for metric in metrics:
            vals = []
            for row in group:
                try:
                    vals.append(float(row.get(metric, "nan")))
                except (TypeError, ValueError):
                    pass
            m, s = mean_std(vals)
            item[f"{metric}_mean"] = m
            item[f"{metric}_std"] = s
        out.append(item)
    for dataset, method, unit in sorted({(row["dataset"], row["method"], row["eval_unit"]) for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset and row["method"] == method and row["eval_unit"] == unit]
        item = {"dataset": dataset, "method": method, "eval_unit": unit, "heldout_attack": "Aggregate", "runs": len(group)}
        item["official_reproduction_status"] = "; ".join(sorted({str(row["official_reproduction_status"]) for row in group}))
        for metric in metrics:
            vals = []
            for row in group:
                try:
                    vals.append(float(row.get(metric, "nan")))
                except (TypeError, ValueError):
                    pass
            m, s = mean_std(vals)
            item[f"{metric}_mean"] = m
            item[f"{metric}_std"] = s
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


def write_latex(agg: list[dict[str, Any]], path: Path, dataset: str) -> None:
    rows = [row for row in agg if row["dataset"] == dataset and row["heldout_attack"] == "Aggregate"]
    rows = sorted(rows, key=lambda r: str(r["method"]))
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Method & Unit & Runs & AUROC & FPR95 & R@0.1\\%FPR & R@1\\%FPR \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {row['eval_unit']} & {row['runs']} & "
            f"{fmt_pm(row['auroc_mean'], row['auroc_std'])} & "
            f"{fmt_pm(row['fpr95_mean'], row['fpr95_std'])} & "
            f"{fmt_pm(row['recall_at_0.001_fpr_mean'], row['recall_at_0.001_fpr_std'])} & "
            f"{fmt_pm(row['recall_at_0.01_fpr_mean'], row['recall_at_0.01_fpr_std'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(f):
        return "-"
    return f"{f:.{digits}f}"


def write_report(agg: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# CCF-A Adapted Baseline Retests",
        "",
        "All rows use the TAMM/TMDD leave-one-unknown protocol: benign-only train/validation, held-out attack only in test, anomaly score evaluation by AUROC/FPR95/low-FPR recall. Rows marked `adapted` are not official full reproductions of the cited systems.",
        "",
        "| Dataset | Method | Runs | AUROC | FPR95 | R@0.1%FPR | R@1%FPR | Val-P99 FPR | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted([r for r in agg if r["heldout_attack"] == "Aggregate"], key=lambda r: (str(r["dataset"]), str(r["method"]))):
        lines.append(
            "| {dataset} | {method} | {runs} | {auroc}+/-{auroc_std} | {fpr95}+/-{fpr95_std} | {r01}+/-{r01_std} | {r1}+/-{r1_std} | {p99}+/-{p99_std} | {status} |".format(
                dataset=row["dataset"],
                method=row["method"],
                runs=row["runs"],
                auroc=fmt(row.get("auroc_mean")),
                auroc_std=fmt(row.get("auroc_std")),
                fpr95=fmt(row.get("fpr95_mean")),
                fpr95_std=fmt(row.get("fpr95_std")),
                r01=fmt(row.get("recall_at_0.001_fpr_mean")),
                r01_std=fmt(row.get("recall_at_0.001_fpr_std")),
                r1=fmt(row.get("recall_at_0.01_fpr_mean")),
                r1_std=fmt(row.get("recall_at_0.01_fpr_std")),
                p99=fmt(row.get("val_p99_realized_fpr_mean")),
                p99_std=fmt(row.get("val_p99_realized_fpr_std")),
                status=row.get("official_reproduction_status", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_specs() -> dict[str, DatasetSpec]:
    ids2017_unknown = Path("paper_icdm_applied_2026/experiments/unknown")
    ids2018_unknown = Path("paper_icdm_applied_2026/experiments/ids2018_official_victim_external/unknown")
    return {
        "ids2017": DatasetSpec(
            name="CICIDS2017",
            split_dir=ids2017_unknown,
            token_dir=ids2017_unknown / "tokens_category",
            flow_file=Path("outputs/processed/ccfa/cicids2017_interim_labeled_flows.jsonl"),
            selected_flow_dir=None,
            attacks=IDS2017_ATTACKS,
            default_seeds=[42, 43, 44],
            status_note="IDS2017 corrected/Zeek flow artifacts, five leave-one unknown attacks, three seeds.",
        ),
        "ids2018": DatasetSpec(
            name="CSE-CIC-IDS2018",
            split_dir=ids2018_unknown,
            token_dir=ids2018_unknown / "tokens_category",
            flow_file=None,
            selected_flow_dir=ids2018_unknown / "selected_flows",
            attacks=IDS2018_ATTACKS,
            default_seeds=[43],
            status_note="IDS2018 official-victim external split-first artifacts; one seed is available for the six-family panel.",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCF-A adapted baselines under the TAMM/TMDD leave-one unknown protocol.")
    parser.add_argument("--datasets", nargs="+", default=["ids2017"], choices=["ids2017", "ids2018"])
    parser.add_argument("--methods", nargs="+", default=["trident", "hypervision", "contramtd", "cade"])
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default="experiments/ccfa_baseline_retests/results")
    parser.add_argument("--nn-epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    specs = dataset_specs()
    out_dir = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        spec = specs[dataset]
        attacks = args.attacks or list(spec.attacks)
        seeds = args.seeds or spec.default_seeds
        for seed in seeds:
            for attack in attacks:
                if attack not in spec.attacks:
                    raise KeyError(f"{attack} not configured for {dataset}; available={list(spec.attacks)}")
                split_path = spec.split_dir / f"splits_leave_one_{spec.attacks[attack]}_anomaly_seed{seed}.json"
                if not split_path.exists():
                    print(f"skip missing split: {split_path}", flush=True)
                    continue
                print(f"running {dataset} attack={attack} seed={seed} methods={','.join(args.methods)}", flush=True)
                rows = evaluate_run(
                    spec,
                    attack,
                    seed,
                    args.methods,
                    out_dir,
                    nn_epochs=args.nn_epochs,
                    batch_size=args.batch_size,
                )
                all_rows.extend(rows)
                for row in rows:
                    print(
                        f"  {row['method']}: auroc={row['auroc']:.4f} fpr95={row['fpr95']:.4f} "
                        f"r1={row['recall_at_0.01_fpr']:.4f} p99fpr={row['val_p99_realized_fpr']:.4f}",
                        flush=True,
                    )

    runs_path = out_dir / "ccfa_adapted_retests_runs.csv"
    agg_path = out_dir / "ccfa_adapted_retests_aggregate.csv"
    report_path = out_dir / "ccfa_adapted_retests_report.md"
    write_csv(all_rows, runs_path)
    agg = aggregate(all_rows)
    write_csv(agg, agg_path)
    write_report(agg, report_path)
    for dataset_name in sorted({row["dataset"] for row in all_rows}):
        safe = dataset_name.lower().replace("-", "_").replace(" ", "_")
        write_latex(agg, out_dir / f"table_{safe}_ccfa_adapted_retests.tex", dataset_name)
    print(json.dumps({"runs": len(all_rows), "runs_path": str(runs_path), "aggregate_path": str(agg_path), "report_path": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
