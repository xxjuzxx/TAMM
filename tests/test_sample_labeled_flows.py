from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


def _sampler_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "25_sample_labeled_flows.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("sample_labeled_flows", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sample_labeled_flows_caps_each_label_and_sorts_by_time() -> None:
    module = _sampler_module()
    rows = [
        {"flow_id": "a1", "label": "BENIGN", "binary_label": "BENIGN", "start_ts": 3.0},
        {"flow_id": "a2", "label": "BENIGN", "binary_label": "BENIGN", "start_ts": 1.0},
        {"flow_id": "b1", "label": "DDoS", "binary_label": "ATTACK", "start_ts": 2.0},
        {"flow_id": "b2", "label": "DDoS", "binary_label": "ATTACK", "start_ts": 4.0},
        {"flow_id": "c1", "label": "PortScan", "binary_label": "ATTACK", "start_ts": 5.0},
    ]

    sampled = module.sample_labeled_flows(rows, per_label_limit=1, seed=7, sort_by_time=True)

    assert [row["start_ts"] for row in sampled] == sorted(row["start_ts"] for row in sampled)
    assert len(sampled) == 3
    assert sorted(row["label"] for row in sampled) == ["BENIGN", "DDoS", "PortScan"]


def test_sample_labeled_flows_can_cap_binary_labels_after_label_cap() -> None:
    module = _sampler_module()
    rows = [
        {"flow_id": "a1", "label": "BENIGN", "binary_label": "BENIGN", "start_ts": 1.0},
        {"flow_id": "a2", "label": "BENIGN", "binary_label": "BENIGN", "start_ts": 2.0},
        {"flow_id": "b1", "label": "DDoS", "binary_label": "ATTACK", "start_ts": 3.0},
        {"flow_id": "b2", "label": "PortScan", "binary_label": "ATTACK", "start_ts": 4.0},
        {"flow_id": "b3", "label": "SSH-Patator", "binary_label": "ATTACK", "start_ts": 5.0},
    ]

    sampled = module.sample_labeled_flows(rows, per_binary_label_limit=2, seed=7, sort_by_time=True)

    assert len(sampled) == 4
    assert sum(row["binary_label"] == "BENIGN" for row in sampled) == 2
    assert sum(row["binary_label"] == "ATTACK" for row in sampled) == 2


def test_binary_cap_label_balanced_keeps_minority_attack_labels() -> None:
    module = _sampler_module()
    rows = [
        *[
            {"flow_id": f"d{idx}", "label": "DDoS", "binary_label": "ATTACK", "start_ts": float(idx)}
            for idx in range(10)
        ],
        {"flow_id": "x1", "label": "Web Attack - XSS", "binary_label": "ATTACK", "start_ts": 20.0},
        {"flow_id": "s1", "label": "Web Attack - SQL Injection", "binary_label": "ATTACK", "start_ts": 21.0},
    ]

    sampled = module.sample_labeled_flows(
        rows,
        per_binary_label_limit=4,
        binary_cap_strategy="label_balanced",
        seed=7,
        sort_by_time=True,
    )

    labels = {row["label"] for row in sampled}
    assert len(sampled) == 4
    assert "Web Attack - XSS" in labels
    assert "Web Attack - SQL Injection" in labels


def test_summarize_reports_selected_source_counts() -> None:
    module = _sampler_module()
    rows = [
        {"flow_id": "a", "label": "BENIGN", "binary_label": "BENIGN", "source_file": "benign.jsonl"},
        {"flow_id": "b", "label": "DDoS", "binary_label": "ATTACK", "source_file": "attack.jsonl"},
    ]
    args = Namespace(
        seed=7,
        per_label_limit=None,
        per_binary_label_limit=None,
        binary_cap_strategy="label_balanced",
        exclude_label_from_source=[],
        sort_by_time=True,
    )

    stats = module.summarize(rows, rows, [Path("benign.jsonl"), Path("attack.jsonl")], args)

    assert stats["selected_source_counts"] == {"attack.jsonl": 1, "benign.jsonl": 1}
    assert stats["selected_binary_counts_by_source"] == {"attack.jsonl|ATTACK": 1, "benign.jsonl|BENIGN": 1}


def test_filter_labeled_flows_excludes_label_from_source_glob() -> None:
    module = _sampler_module()
    rows = [
        {"label": "BENIGN", "source_file": "outputs/processed/zeek_ddos_pcap_labeled_flows_expanded_drop.jsonl"},
        {"label": "DDoS", "source_file": "outputs/processed/zeek_ddos_pcap_labeled_flows_expanded_drop.jsonl"},
        {"label": "BENIGN", "source_file": "outputs/processed/zeek_benign_first5000_pcap_labeled_flows_smoke.jsonl"},
    ]

    filtered = module.filter_labeled_flows(rows, exclude_label_from_source=["BENIGN::*zeek_ddos*"])

    assert filtered == rows[1:]


def test_filter_labeled_flows_can_include_label_globs() -> None:
    module = _sampler_module()
    rows = [
        {"label": "BENIGN"},
        {"label": "DDoS"},
        {"label": "Web Attack - XSS"},
    ]

    filtered = module.filter_labeled_flows(rows, include_label=["Web Attack*", "DDoS"])

    assert filtered == rows[1:]


def test_dedupe_semantic_flows_drops_same_flow_from_multiple_sources() -> None:
    module = _sampler_module()
    rows = [
        {
            "flow_id": "source_a:a",
            "label": "Web Attack - SQL Injection",
            "start_ts": 1.1234564,
            "service_key": ["server", "80", "TCP"],
            "packet_count": 2,
            "lens": [1, 2],
            "dirs": [True, False],
        },
        {
            "flow_id": "source_b:a",
            "label": "Web Attack - SQL Injection",
            "start_ts": 1.1234564,
            "service_key": ["server", "80", "TCP"],
            "packet_count": 2,
            "lens": [1, 2],
            "dirs": [True, False],
        },
        {
            "flow_id": "source_b:b",
            "label": "Web Attack - SQL Injection",
            "start_ts": 2.0,
            "service_key": ["server", "80", "TCP"],
            "packet_count": 2,
            "lens": [1, 2],
            "dirs": [True, False],
        },
    ]

    deduped, dropped = module.dedupe_semantic_flows(rows)

    assert dropped == 1
    assert [row["flow_id"] for row in deduped] == ["source_a:a", "source_b:b"]
