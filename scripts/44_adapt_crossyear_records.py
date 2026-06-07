#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

from src.data.dataset_adapter import dataset_report
from src.data.label_policy import binary_label_for
from src.utils.io import write_json, write_jsonl


SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
FAMILY_MAP = {
    "Benign": "BENIGN",
    "Bot": "Botnet",
    "DoS": "DoS",
    "Patator": "BruteForce",
    "Web": "WebAttack",
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _protocol_name(value: int) -> str:
    if int(value) == 6:
        return "tcp"
    if int(value) == 17:
        return "udp"
    return str(int(value)).lower()


def _flow_from_record(record: dict[str, Any], arrays: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    mask = arrays["packet_mask"][index].astype(bool)
    sizes = arrays["packet_sizes"][index][mask].astype(int)
    dirs_raw = arrays["packet_directions"][index][mask].astype(int)
    iats = arrays["packet_iats"][index][mask].astype(float)
    tss = np.cumsum(iats) if iats.size else np.asarray([], dtype=np.float64)
    label = str(record.get("label") or "UNKNOWN")
    attack_family = FAMILY_MAP.get(label, label)
    proto = _protocol_name(int(record.get("protocol", arrays["protocol"][index])))
    split = SPLIT_NAMES.get(int(record.get("split", arrays["split_id"][index])), "unknown")
    flow_id = str(record.get("flow_id") or f"crossyear:{index}")
    return {
        "flow_id": flow_id,
        "dataset": "CICIDS2017_2018_CROSSYEAR",
        "src_ip": "",
        "dst_ip": "",
        "src_port": "0",
        "dst_port": "0",
        "proto": proto,
        "protocol": proto.upper(),
        "lens": [int(abs(value)) for value in sizes.tolist()],
        "dirs": [1 if value >= 0 else 0 for value in dirs_raw.tolist()],
        "tss": [float(value) for value in tss.tolist()],
        "iats": [float(value) for value in iats.tolist()],
        "label": attack_family,
        "binary_label": binary_label_for(attack_family),
        "attack_family": attack_family,
        "day": "unknown",
        "split": split,
        "start_ts": float(record["time_first"]) if record.get("time_first") is not None else None,
        "packet_count": int(mask.sum()),
        "service_key": ["crossyear", proto.upper()],
        "meta": {
            "source_group": record.get("source_group"),
            "source_file": record.get("source_file"),
            "original_label": label,
            "category": record.get("category"),
            "split_id": int(record.get("split", arrays["split_id"][index])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt sampled CIC-IDS2017->CSE-CIC-IDS2018 records to FlowPrim flow schema.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out_flows", required=True)
    parser.add_argument("--out_splits", required=True)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()

    records = _read_jsonl(args.records)
    arrays = dict(np.load(args.arrays, allow_pickle=True))
    if len(records) != int(arrays["packet_mask"].shape[0]):
        raise ValueError(f"records/arrays row mismatch: {len(records)} vs {arrays['packet_mask'].shape[0]}")
    with Path(args.metadata).open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)

    flows = [_flow_from_record(record, arrays, index) for index, record in enumerate(records)]
    splits = {name: [] for name in ("train", "val", "test")}
    for flow in flows:
        split = str(flow.get("split"))
        if split in splits:
            splits[split].append(str(flow["flow_id"]))
    report = dataset_report(flows, dataset="CICIDS2017_2018_CROSSYEAR", source=args.records)
    report.update(
        {
            "source_metadata": {
                "dataset": source_metadata.get("dataset"),
                "split_mode": source_metadata.get("split_mode"),
                "max_records_per_label": source_metadata.get("max_records_per_label"),
                "source_root": source_metadata.get("source_root"),
            },
            "source_group_counts": dict(sorted(Counter(str(flow.get("meta", {}).get("source_group")) for flow in flows).items())),
            "split_counts": {name: len(ids) for name, ids in splits.items()},
            "adapter": "44_adapt_crossyear_records.py",
        }
    )
    write_jsonl(flows, args.out_flows)
    write_json({"splits": splits, "split_mode": "predefined_cross_year", "source": args.records}, args.out_splits)
    write_json(report, args.out_report)
    print({"num_flows": len(flows), "split_counts": report["split_counts"], "label_counts": report["attack_family_counts"]})


if __name__ == "__main__":
    main()
