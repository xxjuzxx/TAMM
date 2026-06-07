#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extra_benign_common import (
    DEFAULT_REFERENCE_TOKEN,
    EXTRA_ARTIFACT_DIR,
    ROOT,
    build_tokenizer_for_existing_vocab,
    default_profile_row,
    load_token_data,
    primitive_flags,
    protocol_of_flow,
    read_jsonl,
    read_yaml,
    rel,
    service_of_flow,
    source_label,
    write_csv,
    write_json,
    write_jsonl,
)


def _discover_existing_artifacts(pattern: str) -> list[tuple[Path, Path, Path | None]]:
    flows = sorted(ROOT.glob(pattern))
    out: list[tuple[Path, Path, Path | None]] = []
    for flow_path in flows:
        profile_path = Path(str(flow_path).replace("_labeled_flows.jsonl", "_profile_primitives.jsonl"))
        manifest_path = Path(str(flow_path).replace("_labeled_flows.jsonl", "_manifest.json"))
        if profile_path.exists():
            out.append((flow_path, profile_path, manifest_path if manifest_path.exists() else None))
    return out


def _flow_fingerprint(flow: dict[str, Any]) -> str:
    payload = {
        "start_ts": round(float(flow.get("start_ts") or 0.0), 6),
        "duration": round(float(flow.get("duration") or 0.0), 6),
        "protocol": protocol_of_flow(flow),
        "packet_count": int(flow.get("packet_count") or len(flow.get("lens") or [])),
        "lens": list(flow.get("lens") or [])[:32],
        "dirs": list(flow.get("dirs") or [])[:32],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _metadata_row(
    flow: dict[str, Any],
    profile_row: dict[str, Any] | None,
    tokens: list[str],
    *,
    source_dataset: str,
    source_type: str,
    capability_level: str,
) -> dict[str, Any]:
    flags = primitive_flags(profile_row)
    return {
        "flow_id": str(flow.get("flow_id")),
        "source_dataset": source_dataset,
        "source_type": source_type,
        "timestamp_start": flow.get("start_ts", ""),
        "protocol": protocol_of_flow(flow),
        "service": service_of_flow(flow),
        "packet_count": int(flow.get("packet_count") or len(flow.get("lens") or [])),
        "duration": float(flow.get("duration") or 0.0),
        "token_count": len(tokens),
        **flags,
        "capability_level": capability_level,
        "raw_ip_used_as_token": "false",
        "absolute_time_used_as_token": "false",
        "five_tuple_used_as_token": "false",
    }


def _empty_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        counts=np.zeros((0, 0), dtype=np.float32),
        flow_ids=np.asarray([], dtype=object),
        vocab_tokens=np.asarray([], dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare additional benign traffic for FlowPrim benign-memory/calibration experiments.")
    parser.add_argument("--extra-source", nargs="*", default=None, help="Existing labeled flow JSONL files or directories. Defaults to discovered zeek_benign artifacts.")
    parser.add_argument("--source-type", choices=["pcap", "zeek", "packet_csv", "flow_csv"], default="pcap")
    parser.add_argument("--config", default=str(ROOT / "configs" / "cicids2017.yaml"))
    parser.add_argument("--reference-token", default=str(DEFAULT_REFERENCE_TOKEN), help="Token corpus whose train-only vocabulary/config define the output token space.")
    parser.add_argument("--output-dir", default=str(EXTRA_ARTIFACT_DIR))
    parser.add_argument("--existing-pattern", default="outputs/processed/zeek_benign_*_pcap_pipeline_labeled_flows.jsonl")
    parser.add_argument("--source-pcap-dir", default="data/raw/CIC-IDS2017/pcaps", help="Original CIC-IDS2017 PCAP directory recorded for provenance.")
    parser.add_argument("--corrected-csv-dir", default="data/raw/CIC-IDS2017_corrected/CICIDS2017_improved", help="Corrected CICIDS2017 CSV label directory recorded for provenance.")
    parser.add_argument("--max-flows", type=int, default=None)
    args = parser.parse_args()

    del args.config  # Config is retained in CLI for reproducibility; existing artifacts already contain primitive rows.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_data = load_token_data(args.reference_token)
    tokenizer = build_tokenizer_for_existing_vocab(token_data)

    artifact_pairs: list[tuple[Path, Path, Path | None]] = []
    if args.extra_source:
        for item in args.extra_source:
            p = Path(item)
            if not p.is_absolute():
                p = ROOT / p
            if p.is_dir():
                for flow_path in sorted(p.glob("*_labeled_flows.jsonl")):
                    profile_path = Path(str(flow_path).replace("_labeled_flows.jsonl", "_profile_primitives.jsonl"))
                    if profile_path.exists():
                        artifact_pairs.append((flow_path, profile_path, None))
            elif p.exists():
                profile_path = Path(str(p).replace("_labeled_flows.jsonl", "_profile_primitives.jsonl"))
                if not profile_path.exists():
                    raise FileNotFoundError(f"Missing profile primitive artifact for {p}: {profile_path}")
                artifact_pairs.append((p, profile_path, None))
    else:
        artifact_pairs = _discover_existing_artifacts(args.existing_pattern)
    if not artifact_pairs:
        raise FileNotFoundError("No extra benign flow/profile primitive artifacts found. Run the PCAP/Zeek pipeline first or pass --extra-source.")

    flow_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_count = 0
    source_counts: dict[str, int] = {}
    for flow_path, profile_path, manifest_path in artifact_pairs:
        flows = read_jsonl(flow_path)
        profile_rows = {str(row.get("flow_id")): row for row in read_jsonl(profile_path)}
        source_dataset = source_label(flows[0] if flows else {}, flow_path.stem)
        if manifest_path and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                label_csvs = manifest.get("label_csv") or []
                if label_csvs:
                    source_dataset = Path(str(label_csvs[0])).stem
            except json.JSONDecodeError:
                pass
        for flow in flows:
            raw_count += 1
            label = str(flow.get("label") or flow.get("binary_label") or "").upper()
            if label != "BENIGN":
                continue
            fp = _flow_fingerprint(flow)
            if fp in seen:
                continue
            seen.add(fp)
            flow_id = str(flow.get("flow_id"))
            profile = profile_rows.get(flow_id, default_profile_row(flow_id))
            tokens = tokenizer.raw_flow_tokens(flow, profile)[: int(token_data.get("max_len") or tokenizer.max_len)]
            flow_rows.append(
                {
                    "flow_id": flow_id,
                    "source_dataset": source_dataset,
                    "source_type": args.source_type,
                    "start_ts": flow.get("start_ts"),
                    "end_ts": flow.get("end_ts"),
                    "duration": flow.get("duration"),
                    "protocol": protocol_of_flow(flow),
                    "packet_count": int(flow.get("packet_count") or len(flow.get("lens") or [])),
                    "label": "BENIGN",
                    "capability_level": "full_behavior" if args.source_type in {"pcap", "zeek", "packet_csv"} else "tabular_only",
                }
            )
            token_rows.append(
                {
                    "flow_id": flow_id,
                    "source_dataset": source_dataset,
                    "source_type": args.source_type,
                    "protocol": protocol_of_flow(flow),
                    "service": service_of_flow(flow),
                    "tokens": tokens,
                }
            )
            metadata_rows.append(
                _metadata_row(
                    flow,
                    profile,
                    tokens,
                    source_dataset=source_dataset,
                    source_type=args.source_type,
                    capability_level="full_behavior" if args.source_type in {"pcap", "zeek", "packet_csv"} else "tabular_only",
                )
            )
            source_counts[source_dataset] = source_counts.get(source_dataset, 0) + 1
            if args.max_flows is not None and len(flow_rows) >= args.max_flows:
                break
        if args.max_flows is not None and len(flow_rows) >= args.max_flows:
            break

    flows_path = output_dir / "extra_benign_flows.parquet"
    tokens_path = output_dir / "extra_benign_tokens.jsonl"
    hist_path = output_dir / "extra_benign_histograms.npz"
    metadata_path = output_dir / "extra_benign_metadata.csv"
    summary_path = output_dir / "extra_benign_prepare_summary.json"

    pd.DataFrame(flow_rows).to_parquet(flows_path, index=False)
    write_jsonl(token_rows, tokens_path)
    write_csv(metadata_rows, metadata_path)
    # Store full-vocabulary raw counts once. Experiment scripts transform these counts with the split-specific train-only IDF/filter.
    vocab = {str(k): int(v) for k, v in token_data["vocab"].items()}
    counts = np.zeros((len(token_rows), len(vocab)), dtype=np.float32)
    unk = int(vocab.get("[UNK]", 4))
    for row_idx, row in enumerate(token_rows):
        ids = [int(vocab.get(str(token), unk)) for token in row["tokens"]]
        if ids:
            counts[row_idx] = np.bincount(np.asarray(ids, dtype=np.int64), minlength=len(vocab))[: len(vocab)]
    np.savez_compressed(
        hist_path,
        counts=counts,
        flow_ids=np.asarray([row["flow_id"] for row in token_rows], dtype=object),
        vocab_tokens=np.asarray([token for token, _idx in sorted(vocab.items(), key=lambda kv: kv[1])], dtype=object),
    )
    capability_counts = {}
    if metadata_rows:
        capability_counts = {str(key): int(value) for key, value in pd.Series([row["capability_level"] for row in metadata_rows]).value_counts().items()}
    write_json(
        {
            "raw_flows_seen": raw_count,
            "dedup_benign_flows": len(flow_rows),
            "source_counts": source_counts,
            "source_type": args.source_type,
            "source_pcap_dir": args.source_pcap_dir,
            "corrected_csv_dir": args.corrected_csv_dir,
            "artifact_source_note": "Prepared from existing PCAP-to-Zeek benign slice artifacts derived from CIC-IDS2017 PCAPs and corrected CSV labels.",
            "capability_levels": capability_counts,
            "reference_token": rel(args.reference_token),
            "outputs": {
                "flows": rel(flows_path),
                "tokens": rel(tokens_path),
                "histograms": rel(hist_path),
                "metadata": rel(metadata_path),
            },
            "raw_ip_used_as_token": False,
            "absolute_time_used_as_token": False,
            "five_tuple_used_as_token": False,
        },
        summary_path,
    )
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
