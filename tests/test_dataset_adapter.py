from __future__ import annotations

from src.data.cicids2017_adapter import adapt_labeled_flows
from src.data.dataset_adapter import infer_day, label_alignment_report, normalize_attack_family, normalize_flow_schema


def test_normalize_attack_family_uses_canonical_families() -> None:
    assert normalize_attack_family("BENIGN") == "BENIGN"
    assert normalize_attack_family("BruteForce") == "BruteForce"
    assert normalize_attack_family("WebAttack") == "WebAttack"
    assert normalize_attack_family("Botnet") == "Botnet"
    assert normalize_attack_family("Probe") == "Probe"
    assert normalize_attack_family("FTP-Patator") == "BruteForce"
    assert normalize_attack_family("PortScan") == "Probe"
    assert normalize_attack_family("Bot") == "Botnet"
    assert normalize_attack_family("XSS") == "WebAttack"


def test_normalize_flow_schema_outputs_required_fields() -> None:
    flow = {
        "flow_id": "f1",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": 1234,
        "dst_port": 80,
        "protocol": "TCP",
        "lens": [40, 80],
        "dirs": [1, 0],
        "tss": [0.0, 0.2],
        "label": "FTP-Patator",
        "source_file": "Tuesday.csv",
    }
    row = normalize_flow_schema(flow)
    assert row["dataset"] == "CICIDS2017"
    assert row["attack_family"] == "BruteForce"
    assert row["binary_label"] == "ATTACK"
    assert row["day"] == "Tuesday"
    assert row["iats"] == [0.0, 0.2]


def test_infer_day_uses_known_day_names_only() -> None:
    assert infer_day({"day": "Monday"}) == "Monday"
    assert infer_day({"day": "Benign", "source_file": "/tmp/Benign/Benign.csv"}) == "unknown"
    assert infer_day({"label_source_file": "/tmp/monday.csv"}) == "Monday"


def test_dataset_report_contains_counts() -> None:
    flows, report = adapt_labeled_flows(
        [
            {"flow_id": "a", "lens": [1], "dirs": [1], "tss": [0.0], "label": "BENIGN"},
            {"flow_id": "b", "lens": [1], "dirs": [0], "tss": [0.0], "label": "DDoS"},
        ]
    )
    assert len(flows) == 2
    assert report["num_flows"] == 2
    assert report["binary_counts"] == {"ATTACK": 1, "BENIGN": 1}


def test_label_alignment_report_schema() -> None:
    report = label_alignment_report(
        dataset="CICIDS2017",
        total_zeek_flows=10,
        matched_flows=9,
        unmatched_flows=1,
        ambiguous_matches=1,
        time_deltas=[0.1, 0.2, 0.3],
        matched_flows_rows=[{"attack_family": "BENIGN"}, {"attack_family": "DDoS"}],
    )
    assert report["match_rate"] == 0.9
    assert report["ambiguous_rate"] == 1 / 9
    assert report["match_by_attack_family"] == {"BENIGN": 1, "DDoS": 1}
