#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import _bootstrap  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOW_GLOB = str(ROOT / "data" / "interim" / "flows" / "cicids2017" / "*" / "raw_cicids2017_*_labeled_flows.jsonl")
DEFAULT_OUTPUT_DIR = ROOT / "paper_icdm_applied_2026" / "experiments" / "raw_rebuild" / "unknown"
DEFAULT_REPORT = ROOT / "reports" / "raw_rebuild_split_first.md"

ATTACK_SLUG = {
    "Botnet": "botnet",
    "DDoS": "ddos",
    "DoS": "dos",
    "Probe": "probe",
    "WebAttack": "webattack",
    "BruteForce": "bruteforce",
    "Infiltration": "infiltration",
}


@dataclass(frozen=True)
class Candidate:
    flow_id: str
    label: str
    day: str
    proto: str
    start_ts: float
    source_path: str
    score: int


class TopK:
    """Keep the deterministic lowest-hash candidates without storing all raw flows."""

    def __init__(self, k: int) -> None:
        self.k = max(0, int(k))
        self._heap: list[tuple[int, str, Candidate]] = []
        self._seen: set[str] = set()

    def add(self, item: Candidate) -> None:
        if self.k <= 0 or item.flow_id in self._seen:
            return
        self._seen.add(item.flow_id)
        entry = (-int(item.score), item.flow_id, item)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
            return
        if entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def rows(self) -> list[Candidate]:
        return [entry[2] for entry in sorted(self._heap, key=lambda x: (-x[0], x[1]))]


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def _display_path(path: Path) -> str:
    """Return a repository-relative path when possible for portable manifests."""

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _canonical_family(row: dict[str, Any]) -> str:
    raw = str(row.get("attack_family") or row.get("label") or row.get("binary_label") or "").strip()
    norm = raw.lower().replace("-", "").replace("_", "").replace(" ", "")
    if norm in {"benign", "normal"}:
        return "BENIGN"
    if "bot" in norm:
        return "Botnet"
    if "ddos" in norm:
        return "DDoS"
    if "probe" in norm or "portscan" in norm or "scan" in norm:
        return "Probe"
    if "web" in norm or "sql" in norm or "xss" in norm:
        return "WebAttack"
    if "brute" in norm or "patator" in norm:
        return "BruteForce"
    if "heartbleed" in norm:
        return "Heartbleed"
    if "infiltration" in norm:
        return "Infiltration"
    if norm == "dos" or norm.startswith("dos"):
        return "DoS"
    return raw or "UNKNOWN"


def _stable_hash(seed: int, salt: str, flow_id: str) -> int:
    digest = hashlib.sha1(f"{int(seed)}|{salt}|{flow_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _day_from_path(path: Path) -> str:
    return path.parent.name.capitalize()


def _candidate(row: dict[str, Any], path: Path, seed: int, salt: str, label: str) -> Candidate | None:
    flow_id = str(row.get("flow_id") or "").strip()
    if not flow_id:
        return None
    return Candidate(
        flow_id=flow_id,
        label=label,
        day=str(row.get("day") or _day_from_path(path)),
        proto=str(row.get("proto") or row.get("protocol") or "unknown").lower(),
        start_ts=float(row.get("start_ts") or 0.0),
        source_path=str(path),
        score=_stable_hash(seed, salt, flow_id),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _label_counts(candidates: list[Candidate]) -> dict[str, int]:
    return dict(sorted(Counter(item.label for item in candidates).items()))


def _group_counts(candidates: list[Candidate], attr: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(item, attr)) for item in candidates).items()))


def _flow_id_digest(ids: Iterable[str]) -> str:
    return hashlib.sha1("\n".join(str(x) for x in ids).encode("utf-8")).hexdigest()


def _selected_payload(
    *,
    seed: int,
    attack: str,
    train: list[Candidate],
    val: list[Candidate],
    test_benign: list[Candidate],
    test_attack: list[Candidate],
    args: argparse.Namespace,
    raw_counts: dict[str, int],
    source_files: list[Path],
) -> dict[str, Any]:
    test = sorted(test_benign + test_attack, key=lambda x: (x.score, x.flow_id))
    splits = {
        "train": [item.flow_id for item in train],
        "val": [item.flow_id for item in val],
        "test": [item.flow_id for item in test],
    }
    selected_all = train + val + test
    label_counts = {
        "train": _label_counts(train),
        "val": _label_counts(val),
        "test": _label_counts(test),
    }
    return {
        "format": "flowprim_raw_rebuild_split_first_v1",
        "split": "leave_one_attack_out",
        "canonical_split": "raw_rebuild_split_first_leave_one_attack_out",
        "seed": int(seed),
        "leave_label": attack,
        "leave_one_mode": "anomaly",
        "splits": splits,
        "counts": {key: len(value) for key, value in splits.items()},
        "label_counts": label_counts,
        "flow_count": int(sum(len(value) for value in splits.values())),
        "raw_rebuild_full_label_counts": dict(sorted(raw_counts.items())),
        "raw_rebuild_source_files": [_display_path(path) for path in source_files],
        "raw_rebuild_source": args.source_description,
        "dataset": args.dataset_name,
        "excluded_known_attack_labels": sorted(label for label in raw_counts if label not in {"BENIGN", attack}),
        "held_out_count": int(len(test_attack)),
        "test_benign_count": int(len(test_benign)),
        "selection_caps": {
            "train_benign_cap": int(args.train_benign_cap),
            "val_benign_cap": int(args.val_benign_cap),
            "test_benign_cap": int(args.test_benign_cap),
            "attack_test_cap": int(args.attack_test_cap),
        },
        "selection_policy": "deterministic_lowest_sha1_hash_per_seed_after_raw_label_join",
        "split_first_controls": {
            "split_created_before_vocab": True,
            "split_created_before_structural_primitive_min_support": True,
            "train_split_benign_only": True,
            "validation_split_benign_only": True,
            "heldout_attack_used_for_threshold": False,
        },
        "token_safety": {
            "raw_ip_used_as_token": False,
            "absolute_time_used_as_token": False,
            "five_tuple_used_as_token": False,
        },
        "selected_day_counts": {
            "train": _group_counts(train, "day"),
            "val": _group_counts(val, "day"),
            "test": _group_counts(test, "day"),
        },
        "selected_proto_counts": {
            "train": _group_counts(train, "proto"),
            "val": _group_counts(val, "proto"),
            "test": _group_counts(test, "proto"),
        },
        "flow_id_digest": _flow_id_digest([flow_id for ids in splits.values() for flow_id in ids]),
        "notes": [
            "Known non-held-out attacks are excluded from this anomaly split rather than used for training or calibration.",
            "Caps make exact-KNN raw rebuild evaluation tractable; cap values are recorded in this split artifact.",
            "WebAttack remains low-support in the corrected raw labels and is not oversampled.",
        ],
    }


def make_splits(args: argparse.Namespace) -> dict[str, Any]:
    source_files = sorted(Path().glob(args.flow_glob) if not Path(args.flow_glob).is_absolute() else Path("/").glob(args.flow_glob.lstrip("/")))
    if not source_files:
        raise FileNotFoundError(f"No raw labeled flow files matched: {args.flow_glob}")

    attacks = list(args.attacks)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    raw_counts_by_seed: dict[int, Counter[str]] = {}
    for seed in args.seeds:
        benign_pool = TopK(args.train_benign_cap + args.val_benign_cap + args.test_benign_cap)
        attack_pools = {attack: TopK(args.attack_test_cap) for attack in attacks}
        raw_counts: Counter[str] = Counter()
        raw_day_counts: dict[str, Counter[str]] = defaultdict(Counter)

        for path in source_files:
            for row in _jsonl_rows(path):
                label = _canonical_family(row)
                raw_counts[label] += 1
                raw_day_counts[str(row.get("day") or _day_from_path(path))][label] += 1
                if label == "BENIGN":
                    cand = _candidate(row, path, seed, "benign_pool", label)
                    if cand is not None:
                        benign_pool.add(cand)
                elif label in attack_pools:
                    cand = _candidate(row, path, seed, f"attack:{label}", label)
                    if cand is not None:
                        attack_pools[label].add(cand)

        raw_counts_by_seed[int(seed)] = raw_counts
        benign = benign_pool.rows()
        train_end = min(args.train_benign_cap, len(benign))
        val_end = min(train_end + args.val_benign_cap, len(benign))
        test_end = min(val_end + args.test_benign_cap, len(benign))
        train = benign[:train_end]
        val = benign[train_end:val_end]
        test_benign = benign[val_end:test_end]

        for attack in attacks:
            slug = ATTACK_SLUG[attack]
            test_attack = attack_pools[attack].rows()
            payload = _selected_payload(
                seed=seed,
                attack=attack,
                train=train,
                val=val,
                test_benign=test_benign,
                test_attack=test_attack,
                args=args,
                raw_counts=dict(raw_counts),
                source_files=source_files,
            )
            split_path = out_dir / f"splits_leave_one_{slug}_anomaly_seed{seed}.json"
            _write_json(split_path, payload)
            row = {
                "seed": seed,
                "heldout_attack": attack,
                "split_path": _display_path(split_path),
                "train": len(train),
                "val": len(val),
                "test_benign": len(test_benign),
                "test_attack": len(test_attack),
                "test_total": len(test_benign) + len(test_attack),
                "raw_attack_available": raw_counts.get(attack, 0),
                "raw_benign_available": raw_counts.get("BENIGN", 0),
                "train_benign_only": True,
                "val_benign_only": True,
                "attack_labels_used_for_threshold": False,
                "raw_ip_used_as_token": False,
                "absolute_time_used_as_token": False,
                "five_tuple_used_as_token": False,
            }
            summary_rows.append(row)
            print(json.dumps(row, sort_keys=True))

        day_rows = [
            {"seed": seed, "day": day, "label": label, "count": count}
            for day, counts in sorted(raw_day_counts.items())
            for label, count in sorted(counts.items())
        ]
        _write_csv(out_dir / f"raw_rebuild_day_label_counts_seed{seed}.csv", day_rows)

    _write_csv(out_dir / "raw_rebuild_split_manifest.csv", summary_rows)
    _write_report(args, source_files, summary_rows, raw_counts_by_seed)
    return {"rows": summary_rows, "output_dir": str(out_dir), "source_files": [_display_path(p) for p in source_files]}


def _write_report(
    args: argparse.Namespace,
    source_files: list[Path],
    rows: list[dict[str, Any]],
    raw_counts_by_seed: dict[int, Counter[str]],
) -> None:
    lines = [
        "# Raw Rebuild Split-First Splits",
        "",
        f"This report records split-first validation splits for `{args.dataset_name}`. {args.source_description}",
        "",
        "## Inputs",
        "",
        *[f"- `{_display_path(path)}`" for path in source_files],
        "",
        "## Controls",
        "",
        "- Train and validation splits contain benign flows only.",
        "- Held-out attack labels are used only for offline test metrics.",
        "- The split is written before token vocabulary construction, structural primitive min-support filtering, benign memory construction, and threshold calibration.",
        "- Raw IP addresses, absolute timestamps, and complete five-tuples are not behavior tokens or primitives.",
        "- Known non-held-out attacks are excluded from each anomaly split.",
        "",
        "## Caps",
        "",
        f"- train benign: {args.train_benign_cap}",
        f"- validation benign: {args.val_benign_cap}",
        f"- test benign: {args.test_benign_cap}",
        f"- held-out attack: {args.attack_test_cap}",
        "",
        "## Raw Label Counts",
        "",
    ]
    for seed, counts in sorted(raw_counts_by_seed.items()):
        lines.append(f"Seed {seed}: " + ", ".join(f"{label}={count}" for label, count in sorted(counts.items())))
    lines.extend(["", "## Split Manifest", ""])
    if rows:
        headers = ["seed", "heldout_attack", "train", "val", "test_benign", "test_attack", "test_total", "raw_attack_available"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create split-first leave-one raw rebuild splits from PCAP-derived labeled flows.")
    parser.add_argument("--flow-glob", default=DEFAULT_FLOW_GLOB)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--attacks", nargs="+", default=list(ATTACK_SLUG))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--dataset-name", default="CICIDS2017")
    parser.add_argument(
        "--source-description",
        default="The inputs are PCAP-derived flow records joined to corrected CICIDS2017 CSV labels before split-first token/primitive fitting.",
    )
    parser.add_argument("--train-benign-cap", type=int, default=5000)
    parser.add_argument("--val-benign-cap", type=int, default=1000)
    parser.add_argument("--test-benign-cap", type=int, default=2000)
    parser.add_argument("--attack-test-cap", type=int, default=10000)
    args = parser.parse_args()
    summary = make_splits(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
