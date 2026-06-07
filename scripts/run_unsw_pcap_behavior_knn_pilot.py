#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import dpkt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from audit_unsw_pcap_label_alignment import DEFAULT_CSV_GLOB, DEFAULT_PCAP_DIR, _build_csv_index, _flow_key, _match_flows, _csv_paths  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "unsw_nb15_pcap_behavior_knn_pilot"


def _ip_text(raw: bytes) -> str:
    return ".".join(str(b) for b in raw)


def _canon_proto(p: int) -> str:
    return {6: "tcp", 17: "udp", 1: "icmp"}.get(int(p), str(p))


def _read_pcap_flows_with_sequences(path: Path, max_packets: int, max_seq: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flows: dict[tuple[str, int, str, int, str], dict[str, Any]] = {}
    packets = 0
    skipped = Counter()
    started = time.perf_counter()
    with path.open("rb") as handle:
        reader = dpkt.pcap.Reader(handle)
        linktype = reader.datalink()
        for ts, buf in reader:
            packets += 1
            if max_packets and packets > max_packets:
                break
            try:
                if linktype == dpkt.pcap.DLT_LINUX_SLL:
                    ip = dpkt.sll.SLL(buf).data
                elif hasattr(dpkt.pcap, "DLT_LINUX_SLL2") and linktype == dpkt.pcap.DLT_LINUX_SLL2:
                    ip = dpkt.sll2.SLL2(buf).data
                else:
                    ip = dpkt.ethernet.Ethernet(buf).data
                if not isinstance(ip, dpkt.ip.IP):
                    skipped["non_ipv4"] += 1
                    continue
                proto = _canon_proto(ip.p)
                if proto == "tcp" and isinstance(ip.data, dpkt.tcp.TCP):
                    sport, dport = int(ip.data.sport), int(ip.data.dport)
                elif proto == "udp" and isinstance(ip.data, dpkt.udp.UDP):
                    sport, dport = int(ip.data.sport), int(ip.data.dport)
                elif proto == "icmp":
                    sport, dport = 0, 0
                else:
                    skipped["unsupported_transport"] += 1
                    continue
                src = _ip_text(ip.src)
                dst = _ip_text(ip.dst)
                key = _flow_key(src, sport, dst, dport, proto)
                flow = flows.get(key)
                if flow is None:
                    flow = {
                        "srcip": key[0],
                        "sport": key[1],
                        "dstip": key[2],
                        "dsport": key[3],
                        "proto": proto,
                        "start_ts": float(ts),
                        "end_ts": float(ts),
                        "packet_count": 0,
                        "byte_count": 0,
                        "tss": [],
                        "lens": [],
                        "dirs": [],
                    }
                    flows[key] = flow
                direction_orig = src == flow["srcip"] and sport == flow["sport"]
                flow["end_ts"] = float(ts)
                flow["packet_count"] += 1
                flow["byte_count"] += int(len(buf))
                if len(flow["tss"]) < max_seq:
                    flow["tss"].append(float(ts))
                    flow["lens"].append(int(len(buf)))
                    flow["dirs"].append(0 if direction_orig else 1)
            except Exception:
                skipped["parse_error"] += 1
    rows = list(flows.values())
    for idx, row in enumerate(rows):
        row["flow_id"] = f"{path.stem}:{idx}"
        row["duration"] = float(row["end_ts"] - row["start_ts"])
    return rows, {
        "pcap": str(path),
        "linktype": int(linktype),
        "packets_seen": int(min(packets, max_packets) if max_packets else packets),
        "flows": len(rows),
        "skipped": dict(skipped),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _bin(value: float, edges: list[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype="float32"), float(value), side="right"))


def _flow_tokens(flow: dict[str, Any], *, max_positions: int) -> list[str]:
    lens = [int(x) for x in flow.get("lens", [])]
    dirs = [int(x) for x in flow.get("dirs", [])]
    tss = [float(x) for x in flow.get("tss", [])]
    tokens: list[str] = []
    pkt_n = int(flow.get("packet_count") or len(lens))
    byte_n = int(flow.get("byte_count") or sum(lens))
    duration = float(flow.get("duration") or 0.0)
    tokens.append(f"PKT_COUNT_BIN_{_bin(math.log1p(pkt_n), [1.0, 2.0, 4.0, 6.0, 8.0])}")
    tokens.append(f"BYTE_COUNT_BIN_{_bin(math.log1p(byte_n), [6.0, 8.0, 10.0, 12.0, 14.0])}")
    tokens.append(f"DURATION_BIN_{_bin(math.log1p(max(duration, 0.0)), [0.01, 0.1, 1.0, 3.0, 6.0])}")
    if pkt_n <= 4:
        tokens.append("SHORT_FLOW")
    if dirs:
        src_frac = 1.0 - (sum(dirs) / max(len(dirs), 1))
        if src_frac >= 0.85:
            tokens.append("SRC_DOMINANT")
        elif src_frac <= 0.15:
            tokens.append("DST_DOMINANT")
        else:
            tokens.append("BIDIRECTIONAL")
    length_edges = [64, 96, 160, 512, 1024, 1500]
    iat_edges = [0.001, 0.01, 0.1, 1.0, 10.0]
    seq_len = min(len(lens), max_positions)
    for idx in range(seq_len):
        tokens.append(f"PKT_{idx}_D{dirs[idx]}_L{_bin(lens[idx], length_edges)}")
        tokens.append(f"LENBIN_D{dirs[idx]}_L{_bin(lens[idx], length_edges)}")
        if idx > 0:
            iat = max(0.0, tss[idx] - tss[idx - 1])
            tokens.append(f"IATBIN_D{dirs[idx - 1]}{dirs[idx]}_T{_bin(iat, iat_edges)}")
            tokens.append(f"DIR_TRANS_{dirs[idx - 1]}_TO_{dirs[idx]}")
            tokens.append(f"LEN_TRANS_{_bin(lens[idx - 1], length_edges)}_TO_{_bin(lens[idx], length_edges)}")
    burst_count = 0
    if tss:
        burst_count = 1
        for idx in range(1, len(tss)):
            if tss[idx] - tss[idx - 1] > 0.1:
                burst_count += 1
    tokens.append(f"BURST_COUNT_BIN_{_bin(burst_count, [1, 2, 4, 8, 16])}")
    return tokens


def _build_matrix(flows: list[dict[str, Any]], vocab: dict[str, int] | None, *, max_positions: int) -> tuple[np.ndarray, dict[str, int], list[list[str]]]:
    token_rows = [_flow_tokens(flow, max_positions=max_positions) for flow in flows]
    if vocab is None:
        vocab = {tok: idx for idx, tok in enumerate(sorted({tok for row in token_rows for tok in row}))}
    matrix = np.zeros((len(flows), len(vocab)), dtype="float32")
    for row_idx, tokens in enumerate(token_rows):
        for token in tokens:
            idx = vocab.get(token)
            if idx is not None:
                matrix[row_idx, idx] += 1.0
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0)
    return matrix, vocab, token_rows


def _score_knn(query: np.ndarray, memory: np.ndarray, k: int, batch_size: int) -> np.ndarray:
    out = np.zeros(query.shape[0], dtype="float32")
    kk = max(1, min(int(k), memory.shape[0]))
    for start in range(0, len(query), batch_size):
        end = min(start + batch_size, len(query))
        dist = (1.0 - np.clip(query[start:end] @ memory.T, -1.0, 1.0)).astype("float32")
        part = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        vals = np.take_along_axis(dist, part, axis=1)
        out[start:end] = np.mean(vals, axis=1)
    return out


def _macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for cls in (0, 1):
        tp = int(np.sum((y_true == cls) & (pred == cls)))
        fp = int(np.sum((y_true != cls) & (pred == cls)))
        fn = int(np.sum((y_true == cls) & (pred != cls)))
        vals.append(float((2 * tp) / max(2 * tp + fp + fn, 1)))
    return float(sum(vals) / 2.0)


def _metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype("int64")
    benign = y_true == 0
    attack = y_true == 1
    fp = int(np.sum((pred == 1) & benign))
    tn = int(np.sum((pred == 0) & benign))
    tp = int(np.sum((pred == 1) & attack))
    fn = int(np.sum((pred == 0) & attack))
    return {
        "macro_f1": _macro_f1(y_true, pred),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "attack_recall": float(tp / max(tp + fn, 1)),
        "attack_precision": float(tp / max(tp + fp, 1)),
    }


def _best_recall_under_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in np.unique(scores):
        m = _metrics_at_threshold(y_true, scores, float(threshold))
        if float(m["false_positive_rate"]) <= target_fpr:
            if best is None or float(m["attack_recall"]) > float(best["attack_recall"]):
                best = {**m, "threshold": float(threshold)}
    if best is None:
        threshold = float(np.max(scores) + 1e-6)
        best = {**_metrics_at_threshold(y_true, scores, threshold), "threshold": threshold}
    return best


def _evaluate(y_true: np.ndarray, test_scores: np.ndarray, val_scores: np.ndarray) -> dict[str, Any]:
    p99_threshold = float(np.percentile(val_scores, 99.0))
    p99 = _metrics_at_threshold(y_true, test_scores, p99_threshold)
    r001 = _best_recall_under_fpr(y_true, test_scores, 0.001)
    r01 = _best_recall_under_fpr(y_true, test_scores, 0.01)
    row: dict[str, Any] = {}
    try:
        row["auroc"] = float(roc_auc_score(y_true, test_scores))
        row["auprc"] = float(average_precision_score(y_true, test_scores))
        fpr, tpr, _ = roc_curve(y_true, test_scores)
        eligible = fpr[tpr >= 0.95]
        row["fpr95"] = float(np.min(eligible)) if eligible.size else ""
    except ValueError:
        row.update({"auroc": "", "auprc": "", "fpr95": ""})
    row.update(
        {
            "recall_at_0_1pct_fpr": r001["attack_recall"],
            "recall_at_1pct_fpr": r01["attack_recall"],
            "p99_threshold": p99_threshold,
            "p99_realized_fpr": p99["false_positive_rate"],
            "false_alerts_per_10k_benign": float(p99["false_positive_rate"] * 10000.0),
            "p99_attack_recall": p99["attack_recall"],
            "p99_macro_f1": p99["macro_f1"],
        }
    )
    return row


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[Any, Any]:
    vals = []
    for row in rows:
        try:
            val = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return "", ""
    return float(np.mean(vals)), float(np.std(vals))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["heldout_attack"])].append(row)
    metrics = ["auroc", "auprc", "fpr95", "recall_at_0_1pct_fpr", "recall_at_1pct_fpr", "p99_realized_fpr", "false_alerts_per_10k_benign", "p99_attack_recall", "p99_macro_f1", "vocab_size", "memory_size", "query_ms_per_flow"]
    out: list[dict[str, Any]] = []
    for attack, items in sorted(groups.items()):
        row: dict[str, Any] = {"dataset": "UNSW-NB15-PCAP", "heldout_attack": attack, "runs": len(items), "seed_count": len({item["seed"] for item in items})}
        for metric in metrics:
            mean, std = _mean_std(items, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        out.append(row)
    row = {"dataset": "UNSW-NB15-PCAP", "heldout_attack": "Aggregate", "runs": len(rows), "seed_count": len({item["seed"] for item in rows})}
    for metric in metrics:
        mean, std = _mean_std(rows, metric)
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
    out.append(row)
    return out


def _sample_indices(indices: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    idx = indices.copy()
    rng.shuffle(idx)
    return np.sort(idx[: min(len(idx), int(cap))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UNSW-NB15 raw-PCAP packet-behavior KNN pilot.")
    parser.add_argument("--pcap-dir", default=str(DEFAULT_PCAP_DIR))
    parser.add_argument("--csv", nargs="+", default=[DEFAULT_CSV_GLOB])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pcaps", nargs="+", default=["1.pcap", "2.pcap", "3.pcap", "10.pcap"])
    parser.add_argument("--max-packets-per-pcap", type=int, default=500000)
    parser.add_argument("--tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--max-token-positions", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--benign-train", type=int, default=1500)
    parser.add_argument("--benign-val", type=int, default=700)
    parser.add_argument("--benign-test", type=int, default=1500)
    parser.add_argument("--attack-test-cap", type=int, default=800)
    parser.add_argument("--min-attack-test", type=int, default=20)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcap_dir = Path(args.pcap_dir)
    pcaps = [pcap_dir / item for item in args.pcaps]
    csv_paths: list[Path] = []
    for item in args.csv:
        csv_paths.extend(_csv_paths(item))
    csv_paths = [path for path in csv_paths if path.exists()]
    if not csv_paths:
        raise FileNotFoundError("No UNSW CSV files found.")

    index, csv_meta = _build_csv_index(csv_paths, chunksize=args.chunksize)
    all_flows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    for pcap in pcaps:
        flows, extract_meta = _read_pcap_flows_with_sequences(pcap, args.max_packets_per_pcap, args.max_seq)
        aligned, align_meta = _match_flows(flows, index, args.tolerance_seconds)
        matched = [row for row in aligned if row.get("alignment_status") == "matched"]
        all_flows.extend(matched)
        extraction_rows.append(
            {
                "pcap": str(pcap),
                "packets_seen": extract_meta["packets_seen"],
                "flows_extracted": extract_meta["flows"],
                "matched_flows": len(matched),
                "alignment_status_counts": json.dumps(align_meta["alignment_status_counts"], sort_keys=True),
                "matched_family_counts": json.dumps(align_meta["matched_family_counts"], sort_keys=True),
                "elapsed_seconds": extract_meta["elapsed_seconds"],
            }
        )

    if not all_flows:
        raise RuntimeError("No matched PCAP flows were produced.")
    family_counts = Counter(str(row.get("attack_family")) for row in all_flows)
    attacks = sorted(fam for fam, count in family_counts.items() if fam != "BENIGN" and count >= args.min_attack_test)
    metrics_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    flow_array = np.asarray(all_flows, dtype=object)
    benign_idx_all = np.asarray([idx for idx, row in enumerate(all_flows) if row.get("attack_family") == "BENIGN"], dtype=np.int64)
    benign_needed = int(args.benign_train + args.benign_val + args.benign_test)
    if len(benign_idx_all) < benign_needed:
        raise RuntimeError(f"Need {benign_needed} benign flows, found {len(benign_idx_all)}")
    started = time.perf_counter()
    for seed in args.seeds:
        rng = np.random.default_rng(int(seed))
        benign_idx = _sample_indices(benign_idx_all, benign_needed, rng)
        rng.shuffle(benign_idx)
        train_idx = benign_idx[: args.benign_train]
        val_idx = benign_idx[args.benign_train : args.benign_train + args.benign_val]
        test_benign_idx = benign_idx[args.benign_train + args.benign_val :]
        train_flows = flow_array[train_idx].tolist()
        val_flows = flow_array[val_idx].tolist()
        fit_start = time.perf_counter()
        train_x, vocab, _ = _build_matrix(train_flows, None, max_positions=args.max_token_positions)
        val_x, _vocab, _ = _build_matrix(val_flows, vocab, max_positions=args.max_token_positions)
        fit_seconds = time.perf_counter() - fit_start
        val_scores = _score_knn(val_x, train_x, args.k, args.score_batch_size)
        for attack in attacks:
            attack_idx_all = np.asarray([idx for idx, row in enumerate(all_flows) if row.get("attack_family") == attack], dtype=np.int64)
            attack_idx = _sample_indices(attack_idx_all, args.attack_test_cap, rng)
            test_idx = np.concatenate([test_benign_idx, attack_idx])
            test_flows = flow_array[test_idx].tolist()
            y_true = np.concatenate([np.zeros(len(test_benign_idx), dtype=np.int64), np.ones(len(attack_idx), dtype=np.int64)])
            test_x, _vocab, test_tokens = _build_matrix(test_flows, vocab, max_positions=args.max_token_positions)
            score_start = time.perf_counter()
            test_scores = _score_knn(test_x, train_x, args.k, args.score_batch_size)
            score_seconds = time.perf_counter() - score_start
            metric = _evaluate(y_true, test_scores, val_scores)
            metric.update(
                {
                    "dataset": "UNSW-NB15-PCAP",
                    "source_type": "raw_pcap_subset_17_2_2015",
                    "evidence_granularity": "packet_sequence_behavior_tokens",
                    "heldout_attack": attack,
                    "seed": int(seed),
                    "train_benign": int(len(train_idx)),
                    "val_benign": int(len(val_idx)),
                    "test_benign_count": int(len(test_benign_idx)),
                    "test_attack_count": int(len(attack_idx)),
                    "memory_size": int(train_x.shape[0]),
                    "vocab_size": int(train_x.shape[1]),
                    "fit_seconds": float(fit_seconds),
                    "score_seconds": float(score_seconds),
                    "query_ms_per_flow": float((score_seconds * 1000.0) / max(len(test_idx), 1)),
                    "attack_labels_used_for_threshold": False,
                    "attack_labels_used_for_memory": False,
                    "raw_ip_used_as_token": False,
                    "absolute_time_used_as_token": False,
                    "five_tuple_used_as_token": False,
                    "protocol_or_service_used_as_behavior": False,
                    "port_used_as_behavior": False,
                }
            )
            metrics_rows.append(metric)
            split_rows.append({"seed": seed, "heldout_attack": attack, "train_benign": len(train_idx), "val_benign": len(val_idx), "test_benign": len(test_benign_idx), "test_attack": len(attack_idx), "status": "ok"})
            print(json.dumps({k: metric[k] for k in ("seed", "heldout_attack", "auroc", "recall_at_1pct_fpr", "p99_realized_fpr")}, sort_keys=True))

    summary_rows = _summarize(metrics_rows)
    _write_csv(out_dir / "unsw_pcap_behavior_knn_metrics.csv", metrics_rows)
    _write_csv(out_dir / "unsw_pcap_behavior_knn_summary.csv", summary_rows)
    _write_csv(out_dir / "unsw_pcap_behavior_knn_splits.csv", split_rows)
    _write_csv(out_dir / "unsw_pcap_behavior_knn_extraction.csv", extraction_rows)
    _write_jsonl(out_dir / "unsw_pcap_behavior_knn_matched_flows_sample.jsonl", all_flows[:1000])
    _write_json(
        out_dir / "unsw_pcap_behavior_knn_manifest.json",
        {
            "pcaps": [str(path) for path in pcaps],
            "csv_paths": [str(path) for path in csv_paths],
            "max_packets_per_pcap": int(args.max_packets_per_pcap),
            "max_seq": int(args.max_seq),
            "max_token_positions": int(args.max_token_positions),
            "tolerance_seconds": float(args.tolerance_seconds),
            "csv_meta": csv_meta,
            "extraction_rows": extraction_rows,
            "matched_family_counts": dict(sorted(family_counts.items())),
            "attacks_run": attacks,
            "seeds": list(map(int, args.seeds)),
            "leakage_controls": {
                "vocabulary": "train_benign_only",
                "memory": "train_benign_only",
                "threshold": "validation_benign_only",
                "test_labels_used_for": "metrics_only",
                "five_tuple_and_absolute_time_used_for": "label_alignment_only",
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
                "protocol_or_service_used_as_behavior": False,
                "port_used_as_behavior": False,
            },
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )
    lines = [
        "# UNSW-NB15 Raw-PCAP Packet-behavior KNN Pilot",
        "",
        "This pilot uses packet sequences extracted from the local `17-2-2015` PCAP subset. Five-tuples and absolute timestamps are used only for CSV label alignment.",
        "",
        f"- PCAPs: {', '.join(args.pcaps)}",
        f"- Max packets per PCAP: {args.max_packets_per_pcap}",
        f"- Matched family counts: `{dict(sorted(family_counts.items()))}`",
        "",
        "| Held-out | Runs | AUROC | R@1%FPR | P99 FPR | Alerts/10k |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['heldout_attack']} | {row['runs']} | {float(row['auroc_mean']):.4f} | "
            f"{float(row['recall_at_1pct_fpr_mean']):.4f} | {float(row['p99_realized_fpr_mean']):.4f} | "
            f"{float(row['false_alerts_per_10k_benign_mean']):.1f} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "metrics_rows": len(metrics_rows), "summary_rows": len(summary_rows), "attacks_run": attacks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
