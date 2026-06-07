from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "32_label_flows_by_ip.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("label_flows_by_ip", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_ipv4_addresses_filters_invalid_and_dedupes() -> None:
    module = _module()
    ips = module.extract_ipv4_addresses("bad 999.1.1.1 good 10.0.2.15 again 10.0.2.15 and 192.168.0.1")
    assert ips == ["10.0.2.15", "192.168.0.1"]


def test_label_flows_by_ip_marks_either_endpoint_as_attack() -> None:
    module = _module()
    flows = [
        {"flow_id": "a", "src_ip": "10.0.2.15", "dst_ip": "8.8.8.8"},
        {"flow_id": "b", "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2"},
        {"flow_id": "c", "src_ip": "3.3.3.3", "dst_ip": "10.0.2.15"},
    ]
    labeled, stats = module.label_flows_by_ip(flows, {"10.0.2.15"})
    assert [row["binary_label"] for row in labeled] == ["ATTACK", "BENIGN", "ATTACK"]
    assert labeled[0]["matched_malicious_ips"] == ["10.0.2.15"]
    assert stats["label_counts"] == {"BENIGN": 1, "Botnet": 2}
    assert stats["matched_malicious_ips"] == {"10.0.2.15": 2}


def test_stream_label_zeek_flows_by_ip_caps_each_label(tmp_path) -> None:
    module = _module()
    path = tmp_path / "conn.log"
    path.write_text(
        "\n".join(
            [
                "#separator \\x09",
                "#empty_field\t(empty)",
                "#unset_field\t-",
                "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tduration\torig_bytes\tresp_bytes\torig_pkts\tresp_pkts\tservice",
                "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tinterval\tcount\tcount\tcount\tcount\tstring",
                "1.0\tC1\t10.0.2.15\t1000\t8.8.8.8\t53\tudp\t0.1\t40\t80\t1\t1\tdns",
                "2.0\tC2\t10.0.2.15\t1001\t8.8.4.4\t53\tudp\t0.1\t40\t80\t1\t1\tdns",
                "3.0\tC3\t1.1.1.1\t1002\t2.2.2.2\t80\ttcp\t0.1\t40\t80\t1\t1\t-",
                "4.0\tC4\t3.3.3.3\t1003\t4.4.4.4\t80\ttcp\t0.1\t40\t80\t1\t1\t-",
            ]
        ),
        encoding="utf-8",
    )
    labeled, stats = module.stream_label_zeek_flows_by_ip([path], {"10.0.2.15"}, max_flows_per_label=1)
    assert len(labeled) == 2
    assert stats["label_counts"] == {"BENIGN": 1, "Botnet": 1}
    assert stats["input_rows"] == 3
    assert stats["capped_flows"] == 1
