#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "local_external_pcaps"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canon_proto(proto: str) -> str:
    proto = str(proto or "").strip().lower()
    return {"6": "tcp", "17": "udp", "1": "icmp"}.get(proto, proto or "unknown")


def _flow_key(src: str, sport: str, dst: str, dport: str, proto: str) -> tuple[str, int, str, int, str]:
    def port(value: str) -> int:
        try:
            return int(float(value or 0))
        except Exception:
            return 0

    left = (str(src), port(sport))
    right = (str(dst), port(dport))
    proto = _canon_proto(proto)
    if left <= right:
        return (left[0], left[1], right[0], right[1], proto)
    return (right[0], right[1], left[0], left[1], proto)


def _label_for_path(dataset: str, path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    first = rel.parts[0]
    if dataset == "ustc_tfc2016":
        if first.lower() == "benign":
            return "BENIGN", path.stem.split("-")[0]
        return path.stem.split("-")[0], "malware"
    if dataset == "cic_ids2017_classwise":
        return ("BENIGN", "benign") if first.lower() == "benign" else (first, "attack")
    if dataset == "crossnet2021_scenarioA":
        return first, "application"
    raise ValueError(dataset)


def _iter_packet_csv(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            src = row.get("ip.src") or row.get("ipv6.src") or ""
            dst = row.get("ip.dst") or row.get("ipv6.dst") or ""
            if not src or not dst:
                continue
            tcp_s = row.get("tcp.srcport") or ""
            udp_s = row.get("udp.srcport") or ""
            tcp_d = row.get("tcp.dstport") or ""
            udp_d = row.get("udp.dstport") or ""
            proto = "tcp" if tcp_s or tcp_d else ("udp" if udp_s or udp_d else "unknown")
            yield (
                float(row.get("frame.time_epoch") or 0.0),
                src,
                tcp_s or udp_s or "0",
                dst,
                tcp_d or udp_d or "0",
                proto,
                int(float(row.get("frame.len") or 0)),
            )


def _iter_tshark(path: Path, max_packets: int):
    cmd = ["tshark"]
    if max_packets > 0:
        cmd += ["-c", str(max_packets)]
    cmd += [
        "-r",
        str(path),
        "-T",
        "fields",
        "-E",
        "separator=\t",
        "-e",
        "frame.time_epoch",
        "-e",
        "ip.src",
        "-e",
        "tcp.srcport",
        "-e",
        "udp.srcport",
        "-e",
        "ip.dst",
        "-e",
        "tcp.dstport",
        "-e",
        "udp.dstport",
        "-e",
        "ip.proto",
        "-e",
        "frame.len",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9:
            continue
        ts, src, tcp_s, udp_s, dst, tcp_d, udp_d, proto, length = parts[:9]
        if not src or not dst:
            continue
        yield (float(ts or 0.0), src, tcp_s or udp_s or "0", dst, tcp_d or udp_d or "0", _canon_proto(proto), int(float(length or 0)))
    proc.communicate()


def _extract_flows(dataset: str, root: Path, *, max_packets_per_file: int, prefer_csv: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files: list[Path] = []
    if prefer_csv:
        files = sorted(root.glob("*/*.csv"))
    if not files:
        seen: set[Path] = set()
        for pcap in sorted(root.rglob("*.pcap")):
            real = pcap.resolve()
            if real in seen:
                continue
            seen.add(real)
            files.append(pcap)

    all_flows: list[dict[str, Any]] = []
    extraction: list[dict[str, Any]] = []
    for path in files:
        family, role = _label_for_path(dataset, path, root)
        flows: dict[tuple[str, int, str, int, str], dict[str, Any]] = {}
        packets = 0
        started = time.perf_counter()
        iterator = _iter_packet_csv(path) if path.suffix.lower() == ".csv" else _iter_tshark(path, max_packets_per_file)
        for ts, src, sport, dst, dport, proto, length in iterator:
            packets += 1
            if path.suffix.lower() == ".csv" and max_packets_per_file > 0 and packets > max_packets_per_file:
                break
            key = _flow_key(src, sport, dst, dport, proto)
            flow = flows.get(key)
            if flow is None:
                flow = {
                    "dataset": dataset,
                    "source": str(path),
                    "family": family,
                    "role": role,
                    "src": key[0],
                    "sport": key[1],
                    "dst": key[2],
                    "dport": key[3],
                    "proto": key[4],
                    "start_ts": float(ts),
                    "end_ts": float(ts),
                    "packet_count": 0,
                    "byte_count": 0,
                    "tss": [],
                    "lens": [],
                    "dirs": [],
                }
                flows[key] = flow
            orig = str(src) == flow["src"] and int(float(sport or 0)) == int(flow["sport"])
            flow["end_ts"] = float(ts)
            flow["packet_count"] += 1
            flow["byte_count"] += int(length)
            if len(flow["tss"]) < 128:
                flow["tss"].append(float(ts))
                flow["lens"].append(int(length))
                flow["dirs"].append(0 if orig else 1)
        rows = list(flows.values())
        for idx, flow in enumerate(rows):
            flow["flow_id"] = f"{path.parent.name}:{path.stem}:{idx}"
            flow["duration"] = max(0.0, float(flow["end_ts"] - flow["start_ts"]))
        all_flows.extend(rows)
        extraction.append(
            {
                "dataset": dataset,
                "path": str(path),
                "family": family,
                "role": role,
                "packets_read": packets,
                "flows": len(rows),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        print(json.dumps(extraction[-1], sort_keys=True), flush=True)
    return all_flows, extraction


def _bin(value: float, edges: list[float]) -> int:
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def _tokens(flow: dict[str, Any], max_positions: int) -> list[str]:
    lens = [int(x) for x in flow.get("lens", [])]
    dirs = [int(x) for x in flow.get("dirs", [])]
    tss = [float(x) for x in flow.get("tss", [])]
    pkt = int(flow.get("packet_count") or len(lens))
    byt = int(flow.get("byte_count") or sum(lens))
    dur = float(flow.get("duration") or 0.0)
    toks = [
        f"PKT_COUNT_BIN_{_bin(math.log1p(pkt), [1, 2, 4, 6, 8])}",
        f"BYTE_COUNT_BIN_{_bin(math.log1p(byt), [6, 8, 10, 12, 14])}",
        f"DURATION_BIN_{_bin(math.log1p(max(dur, 0.0)), [0.01, 0.1, 1, 3, 6])}",
    ]
    if pkt <= 4:
        toks.append("SHORT_FLOW")
    if dirs:
        frac = 1.0 - sum(dirs) / max(1, len(dirs))
        toks.append("SRC_DOMINANT" if frac >= 0.85 else "DST_DOMINANT" if frac <= 0.15 else "BIDIRECTIONAL")
    for idx in range(min(len(lens), max_positions)):
        lb = _bin(lens[idx], [64, 96, 160, 512, 1024, 1500])
        toks.append(f"PKT_{idx}_D{dirs[idx]}_L{lb}")
        toks.append(f"LENBIN_D{dirs[idx]}_L{lb}")
        if idx > 0:
            prev = _bin(lens[idx - 1], [64, 96, 160, 512, 1024, 1500])
            iat = max(0.0, tss[idx] - tss[idx - 1])
            toks.append(f"IATBIN_D{dirs[idx-1]}{dirs[idx]}_T{_bin(iat, [0.001, 0.01, 0.1, 1, 10])}")
            toks.append(f"DIR_TRANS_{dirs[idx-1]}_TO_{dirs[idx]}")
            toks.append(f"LEN_TRANS_{prev}_TO_{lb}")
    bursts = 0
    if tss:
        bursts = 1
        for idx in range(1, len(tss)):
            if tss[idx] - tss[idx - 1] > 0.1:
                bursts += 1
    toks.append(f"BURST_COUNT_BIN_{_bin(bursts, [1, 2, 4, 8, 16])}")
    return toks


def _vectorize(flows: list[dict[str, Any]], vocab: dict[str, int] | None, max_positions: int):
    rows = [_tokens(flow, max_positions) for flow in flows]
    if vocab is None:
        vocab = {tok: idx for idx, tok in enumerate(sorted({tok for row in rows for tok in row}))}
    vecs = []
    for row in rows:
        counts: dict[int, float] = defaultdict(float)
        for tok in row:
            idx = vocab.get(tok)
            if idx is not None:
                counts[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        vecs.append({idx: val / norm for idx, val in counts.items()})
    return vecs, vocab


def _dist(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(i, 0.0) for i, v in a.items())
    return 1.0 - max(-1.0, min(1.0, dot))


def _score(query: list[dict[int, float]], memory: list[dict[int, float]], k: int) -> list[float]:
    kk = max(1, min(k, len(memory)))
    out = []
    for q in query:
        vals = sorted(_dist(q, m) for m in memory)[:kk]
        out.append(sum(vals) / kk)
    return out


def _auroc(y: list[int], s: list[float]) -> float | str:
    pairs = sorted(zip(s, y), key=lambda item: item[0])
    n0 = sum(1 for yy in y if yy == 0)
    n1 = sum(1 for yy in y if yy == 1)
    if n0 == 0 or n1 == 0:
        return ""
    ranks = []
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (rank + rank + (j - i) - 1) / 2
        ranks.extend([avg] * (j - i))
        rank += j - i
        i = j
    pos = sum(r for r, (_score_value, yy) in zip(ranks, pairs) if yy == 1)
    return (pos - n1 * (n1 + 1) / 2) / (n0 * n1)


def _at_threshold(y: list[int], s: list[float], threshold: float) -> dict[str, float]:
    fp = tn = tp = fn = 0
    for yy, score in zip(y, s):
        pred = score >= threshold
        if yy == 0 and pred:
            fp += 1
        elif yy == 0:
            tn += 1
        elif pred:
            tp += 1
        else:
            fn += 1
    fpr = fp / max(1, fp + tn)
    rec = tp / max(1, tp + fn)
    return {"false_positive_rate": fpr, "attack_recall": rec}


def _best_recall(y: list[int], s: list[float], target: float) -> float:
    best = 0.0
    for threshold in sorted(set(s)):
        m = _at_threshold(y, s, threshold)
        if m["false_positive_rate"] <= target:
            best = max(best, m["attack_recall"])
    return best


def _evaluate(y: list[int], s: list[float], val_scores: list[float]) -> dict[str, Any]:
    vals = sorted(val_scores)
    threshold = vals[min(len(vals) - 1, max(0, int(math.ceil(0.99 * len(vals))) - 1))] if vals else 1e9
    p99 = _at_threshold(y, s, threshold)
    return {
        "auroc": _auroc(y, s),
        "recall_at_0_1pct_fpr": _best_recall(y, s, 0.001),
        "recall_at_1pct_fpr": _best_recall(y, s, 0.01),
        "p99_threshold": threshold,
        "p99_realized_fpr": p99["false_positive_rate"],
        "false_alerts_per_10k_benign": p99["false_positive_rate"] * 10000,
        "p99_attack_recall": p99["attack_recall"],
    }


def _sample(items: list[int], cap: int, rng: random.Random) -> list[int]:
    values = list(items)
    rng.shuffle(values)
    return values[: min(len(values), cap)]


def _run_attack_detection(flows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_family: dict[str, list[int]] = defaultdict(list)
    for idx, flow in enumerate(flows):
        by_family[str(flow["family"])].append(idx)
    benign = by_family.get("BENIGN", [])
    needed = args.benign_train + args.benign_val + args.benign_test
    rows: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    for seed in args.seeds:
        rng = random.Random(seed)
        b = _sample(benign, needed, rng)
        if len(b) < needed:
            splits.append({"seed": seed, "heldout": "ALL", "status": "skipped", "reason": f"insufficient_benign_{len(b)}_need_{needed}"})
            continue
        train_idx = b[: args.benign_train]
        val_idx = b[args.benign_train : args.benign_train + args.benign_val]
        test_b_idx = b[args.benign_train + args.benign_val :]
        train_flows = [flows[i] for i in train_idx]
        val_flows = [flows[i] for i in val_idx]
        train_v, vocab = _vectorize(train_flows, None, args.max_token_positions)
        val_v, _ = _vectorize(val_flows, vocab, args.max_token_positions)
        val_scores = _score(val_v, train_v, args.k)
        for family, idxs in sorted(by_family.items()):
            if family == "BENIGN":
                continue
            if len(idxs) < args.min_attack_test:
                splits.append({"seed": seed, "heldout": family, "status": "skipped", "reason": f"low_support_{len(idxs)}"})
                continue
            attack_idx = _sample(idxs, args.attack_test_cap, rng)
            test_flows = [flows[i] for i in test_b_idx + attack_idx]
            test_v, _ = _vectorize(test_flows, vocab, args.max_token_positions)
            started = time.perf_counter()
            scores = _score(test_v, train_v, args.k)
            elapsed = time.perf_counter() - started
            y = [0] * len(test_b_idx) + [1] * len(attack_idx)
            metric = _evaluate(y, scores, val_scores)
            metric.update(
                {
                    "dataset": args.dataset,
                    "protocol": "attack_detection",
                    "heldout": family,
                    "seed": seed,
                    "train_benign": len(train_idx),
                    "val_benign": len(val_idx),
                    "test_benign": len(test_b_idx),
                    "test_attack": len(attack_idx),
                    "memory_size": len(train_v),
                    "vocab_size": len(vocab),
                    "query_ms_per_flow": elapsed * 1000 / max(1, len(test_v)),
                }
            )
            rows.append(metric)
            splits.append({"seed": seed, "heldout": family, "status": "ok", "test_attack": len(attack_idx)})
            print(json.dumps({k: metric[k] for k in ("dataset", "heldout", "seed", "auroc", "recall_at_1pct_fpr", "p99_realized_fpr")}, sort_keys=True), flush=True)
    return rows, splits


def _run_application_novelty(flows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_family: dict[str, list[int]] = defaultdict(list)
    for idx, flow in enumerate(flows):
        by_family[str(flow["family"])].append(idx)
    rows: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    known_train_each = max(20, args.known_train_per_class)
    known_val_each = max(10, args.known_val_per_class)
    known_test_each = max(10, args.known_test_per_class)
    for seed in args.seeds:
        for heldout, held_idxs in sorted(by_family.items()):
            if len(held_idxs) < args.min_attack_test:
                splits.append({"seed": seed, "heldout": heldout, "status": "skipped", "reason": f"low_support_{len(held_idxs)}"})
                continue
            rng = random.Random(seed * 1009 + hash(heldout) % 1000)
            train_idx: list[int] = []
            val_idx: list[int] = []
            test_known_idx: list[int] = []
            for fam, idxs in sorted(by_family.items()):
                if fam == heldout:
                    continue
                take = _sample(idxs, known_train_each + known_val_each + known_test_each, rng)
                if len(take) < known_val_each + known_test_each + 1:
                    continue
                train_idx.extend(take[:known_train_each])
                val_idx.extend(take[known_train_each : known_train_each + known_val_each])
                test_known_idx.extend(take[known_train_each + known_val_each : known_train_each + known_val_each + known_test_each])
            if not train_idx or not val_idx or not test_known_idx:
                splits.append({"seed": seed, "heldout": heldout, "status": "skipped", "reason": "insufficient_known_pool"})
                continue
            unknown_idx = _sample(held_idxs, args.attack_test_cap, rng)
            train_v, vocab = _vectorize([flows[i] for i in train_idx], None, args.max_token_positions)
            val_v, _ = _vectorize([flows[i] for i in val_idx], vocab, args.max_token_positions)
            val_scores = _score(val_v, train_v, args.k)
            test_idx = test_known_idx + unknown_idx
            test_v, _ = _vectorize([flows[i] for i in test_idx], vocab, args.max_token_positions)
            started = time.perf_counter()
            scores = _score(test_v, train_v, args.k)
            elapsed = time.perf_counter() - started
            y = [0] * len(test_known_idx) + [1] * len(unknown_idx)
            metric = _evaluate(y, scores, val_scores)
            metric.update(
                {
                    "dataset": args.dataset,
                    "protocol": "application_novelty",
                    "heldout": heldout,
                    "seed": seed,
                    "train_known": len(train_idx),
                    "val_known": len(val_idx),
                    "test_known": len(test_known_idx),
                    "test_unknown": len(unknown_idx),
                    "memory_size": len(train_v),
                    "vocab_size": len(vocab),
                    "query_ms_per_flow": elapsed * 1000 / max(1, len(test_v)),
                }
            )
            rows.append(metric)
            splits.append({"seed": seed, "heldout": heldout, "status": "ok", "test_unknown": len(unknown_idx)})
            print(json.dumps({k: metric[k] for k in ("dataset", "heldout", "seed", "auroc", "recall_at_1pct_fpr", "p99_realized_fpr")}, sort_keys=True), flush=True)
    return rows, splits


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["auroc", "recall_at_0_1pct_fpr", "recall_at_1pct_fpr", "p99_realized_fpr", "false_alerts_per_10k_benign", "p99_attack_recall", "vocab_size", "memory_size", "query_ms_per_flow"]
    out = []
    for heldout in sorted({str(r["heldout"]) for r in rows}) + ["Aggregate"]:
        items = rows if heldout == "Aggregate" else [r for r in rows if str(r["heldout"]) == heldout]
        if not items:
            continue
        row: dict[str, Any] = {"dataset": items[0]["dataset"], "protocol": items[0]["protocol"], "heldout": heldout, "runs": len(items)}
        for metric in metrics:
            vals = []
            for item in items:
                try:
                    vals.append(float(item[metric]))
                except Exception:
                    pass
            row[f"{metric}_mean"] = sum(vals) / len(vals) if vals else ""
            row[f"{metric}_std"] = (sum((x - row[f"{metric}_mean"]) ** 2 for x in vals) / len(vals)) ** 0.5 if vals else ""
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ustc_tfc2016", "cic_ids2017_classwise", "crossnet2021_scenarioA"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--prefer-csv", action="store_true")
    parser.add_argument("--max-packets-per-file", type=int, default=200000)
    parser.add_argument("--max-token-positions", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--benign-train", type=int, default=1500)
    parser.add_argument("--benign-val", type=int, default=600)
    parser.add_argument("--benign-test", type=int, default=1500)
    parser.add_argument("--attack-test-cap", type=int, default=1000)
    parser.add_argument("--min-attack-test", type=int, default=20)
    parser.add_argument("--known-train-per-class", type=int, default=120)
    parser.add_argument("--known-val-per-class", type=int, default=40)
    parser.add_argument("--known-test-per-class", type=int, default=60)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    flows, extraction = _extract_flows(args.dataset, Path(args.root), max_packets_per_file=args.max_packets_per_file, prefer_csv=args.prefer_csv)
    _write_csv(out / "extraction_manifest.csv", extraction)
    _write_json(out / "flow_inventory.json", {"dataset": args.dataset, "flows": len(flows), "family_counts": dict(sorted(Counter(str(f["family"]) for f in flows).items())), "args": vars(args)})
    if args.dataset == "crossnet2021_scenarioA":
        rows, splits = _run_application_novelty(flows, args)
    else:
        rows, splits = _run_attack_detection(flows, args)
    summary = _summary(rows)
    _write_csv(out / "metrics.csv", rows)
    _write_csv(out / "splits.csv", splits)
    _write_csv(out / "summary.csv", summary)
    _write_json(out / "manifest.json", {"dataset": args.dataset, "args": vars(args), "family_counts": dict(sorted(Counter(str(f["family"]) for f in flows).items())), "rows": len(rows)})
    print(json.dumps({"out": str(out), "flows": len(flows), "metrics_rows": len(rows)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
