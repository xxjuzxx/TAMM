from __future__ import annotations

from src.data.ids2018_schedule_labeler import (
    extract_ipv4s,
    label_ids2018_flows_by_schedule,
    windows_from_manifest,
)


def test_extract_ipv4s_discards_invalid_octets() -> None:
    assert extract_ipv4s("172.31.69.25 (Valid IP:18.217.21.148) 999.1.1.1") == [
        "172.31.69.25",
        "18.217.21.148",
    ]


def test_schedule_labeler_matches_bidirectional_attack_endpoint() -> None:
    manifest = {
        "source_url": "https://www.unb.ca/cic/datasets/ids-2018.html",
        "schedule_to_zeek_offset_seconds": 14400,
        "windows": [
            {
                "window_id": "w1",
                "attack_name": "Bot",
                "attack_family": "Botnet",
                "iso_date": "2018-03-02",
                "day_dir": "Friday-02-03-2018",
                "start_time": "10:11",
                "finish_time": "11:34",
                "attacker_ips": ["18.219.211.138"],
                "victim_ips": ["172.31.69.23"],
            }
        ],
    }
    windows = windows_from_manifest(manifest)
    flows = [
        {
            "flow_id": "f1",
            "src_ip": "172.31.69.23",
            "src_port": "443",
            "dst_ip": "18.219.211.138",
            "dst_port": "5050",
            "protocol": "TCP",
            "start_ts": windows[0].start_ts + 1,
            "end_ts": windows[0].start_ts + 2,
            "lens": [60, 80],
            "dirs": [1, 0],
            "tss": [0.0, 1.0],
            "iats": [0.0, 1.0],
            "day": "Friday-02-03-2018",
        }
    ]
    labeled, quarantine, alignment, _report = label_ids2018_flows_by_schedule(
        flows,
        windows,
        day="Friday-02-03-2018",
    )
    assert not quarantine
    assert labeled[0]["attack_family"] == "Botnet"
    assert labeled[0]["binary_label"] == "ATTACK"
    assert alignment["label_counts"]["Botnet"] == 1
    assert labeled[0]["meta"]["ids2018_label_protocol"] == "schedule_ip_time_window"


def test_schedule_labeler_quarantines_conflicting_windows() -> None:
    manifest = {
        "source_url": "https://www.unb.ca/cic/datasets/ids-2018.html",
        "schedule_to_zeek_offset_seconds": 0,
        "windows": [
            {
                "window_id": "w1",
                "attack_name": "Bot",
                "attack_family": "Botnet",
                "iso_date": "2018-03-02",
                "day_dir": "Friday-02-03-2018",
                "start_time": "10:00",
                "finish_time": "11:00",
                "attacker_ips": ["10.0.0.1"],
                "victim_ips": ["10.0.0.2"],
            },
            {
                "window_id": "w2",
                "attack_name": "DDoS",
                "attack_family": "DDoS",
                "iso_date": "2018-03-02",
                "day_dir": "Friday-02-03-2018",
                "start_time": "10:00",
                "finish_time": "11:00",
                "attacker_ips": ["10.0.0.1"],
                "victim_ips": ["10.0.0.2"],
            },
        ],
    }
    windows = windows_from_manifest(manifest)
    flows = [
        {
            "flow_id": "f1",
            "src_ip": "10.0.0.1",
            "src_port": "1",
            "dst_ip": "10.0.0.2",
            "dst_port": "2",
            "protocol": "TCP",
            "start_ts": windows[0].start_ts + 1,
            "end_ts": windows[0].start_ts + 2,
            "lens": [1],
            "dirs": [1],
            "tss": [0.0],
            "iats": [0.0],
            "day": "Friday-02-03-2018",
        }
    ]
    labeled, quarantine, alignment, _report = label_ids2018_flows_by_schedule(flows, windows)
    assert not labeled
    assert len(quarantine) == 1
    assert alignment["quarantine_flows"] == 1
