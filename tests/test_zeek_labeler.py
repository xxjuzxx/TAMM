from __future__ import annotations

from src.data.cicids2017_labeler import label_flows
from src.data.flow_aggregator import PacketRecord, aggregate_packets
from src.data.zeek_parser import aggregate_zeek_logs, iter_zeek_tsv


def test_bidirectional_packets_merge_into_one_flow() -> None:
    packets = [
        PacketRecord(1.0, "10.0.0.1", "1234", "10.0.0.2", "443", "TCP", 60, True),
        PacketRecord(1.2, "10.0.0.2", "443", "10.0.0.1", "1234", "TCP", 120, False),
    ]
    flows = aggregate_packets(packets)
    assert len(flows) == 1
    assert flows[0]["lens"] == [60, 120]
    assert flows[0]["packet_count"] == 2


def test_packets_with_different_zeek_uids_are_separate_flows() -> None:
    packets = [
        PacketRecord(1.0, "10.0.0.1", "1234", "10.0.0.2", "443", "TCP", 60, True, uid="C1"),
        PacketRecord(1.1, "10.0.0.2", "443", "10.0.0.1", "1234", "TCP", 120, False, uid="C1"),
        PacketRecord(2.0, "10.0.0.1", "1234", "10.0.0.2", "443", "TCP", 80, True, uid="C2"),
    ]
    flows = aggregate_packets(packets)
    assert len(flows) == 2
    assert [flow["packet_count"] for flow in flows] == [2, 1]
    assert [flow["uids"] for flow in flows] == [["C1", "C1"], ["C2"]]


def test_cicids_labeler_matches_reverse_tuple_with_time_tolerance() -> None:
    flow = aggregate_packets(
        [
            PacketRecord(100.0, "10.0.0.1", "1234", "10.0.0.2", "443", "TCP", 60, True),
            PacketRecord(100.5, "10.0.0.2", "443", "10.0.0.1", "1234", "TCP", 120, False),
        ]
    )[0]
    labels = [
        {
            "key": ("10.0.0.2", "443", "10.0.0.1", "1234", "TCP"),
            "reverse_key": ("10.0.0.1", "1234", "10.0.0.2", "443", "TCP"),
            "start_ts": 99.5,
            "end_ts": 101.0,
            "label": "DDoS",
            "source_file": "synthetic.csv",
        }
    ]
    labeled, unmatched, stats = label_flows([flow], labels, tolerance_seconds=2.0)
    assert len(labeled) == 1
    assert not unmatched
    assert labeled[0]["binary_label"] == "ATTACK"
    assert stats["match_rate"] == 1.0


def test_zeek_tsv_parser_reads_packet_rows(tmp_path) -> None:
    path = tmp_path / "packet.log"
    path.write_text(
        "\n".join(
            [
                "#separator \\x09",
                "#set_separator\t,",
                "#empty_field\t(empty)",
                "#unset_field\t-",
                "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tapplayerlength\tis_orig",
                "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tcount\tbool",
                "1.000\tC1\t10.0.0.1\t1234\t10.0.0.2\t443\ttcp\t60\tT",
                "1.100\tC1\t10.0.0.2\t443\t10.0.0.1\t1234\ttcp\t120\tF",
            ]
        ),
        encoding="utf-8",
    )
    rows = list(iter_zeek_tsv(path))
    assert rows[0]["id.orig_h"] == "10.0.0.1"
    flows = aggregate_zeek_logs([path])
    assert len(flows) == 1
    assert flows[0]["lens"] == [60, 120]


def test_zeek_conn_log_rows_are_converted_to_approximate_packets(tmp_path) -> None:
    path = tmp_path / "conn.log"
    path.write_text(
        "\n".join(
            [
                "#separator \\x09",
                "#empty_field\t(empty)",
                "#unset_field\t-",
                "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tduration\torig_bytes\tresp_bytes\torig_pkts\tresp_pkts\tservice",
                "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tinterval\tcount\tcount\tcount\tcount\tstring",
                "10.000\tC2\t10.0.0.1\t2345\t10.0.0.2\t80\ttcp\t1.0\t100\t300\t2\t3\thttp",
            ]
        ),
        encoding="utf-8",
    )
    flows = aggregate_zeek_logs([path])
    assert len(flows) == 1
    assert flows[0]["packet_count"] == 5
    assert flows[0]["lens"] == [50, 50, 100, 100, 100]
    assert flows[0]["appinfo"] == ["http"] * 5


def test_zeek_log_extension_can_contain_json_lines(tmp_path) -> None:
    path = tmp_path / "Features.log"
    path.write_text(
        "\n".join(
            [
                '{"uid":"C3","srcip":"10.0.0.1","srcport":3456,"dstip":"10.0.0.2","dstport":443,'
                '"is_orig":true,"applayerlength":60,"timestamp":20.0,"appinfo":"tls"}',
                '{"uid":"C3","srcip":"10.0.0.1","srcport":3456,"dstip":"10.0.0.2","dstport":443,'
                '"is_orig":false,"applayerlength":120,"timestamp":20.1,"appinfo":"tls"}',
            ]
        ),
        encoding="utf-8",
    )
    flows = aggregate_zeek_logs([path])
    assert len(flows) == 1
    assert flows[0]["lens"] == [60, 120]
    assert flows[0]["dirs"] == [True, False]


def test_features_log_can_use_conn_log_protocol_by_uid(tmp_path) -> None:
    features = tmp_path / "Features.log"
    features.write_text(
        "\n".join(
            [
                '{"uid":"C4","srcip":"10.0.0.1","srcport":5353,"dstip":"10.0.0.2","dstport":53,'
                '"is_orig":true,"applayerlength":44,"timestamp":30.0}',
                '{"uid":"C4","srcip":"10.0.0.2","srcport":53,"dstip":"10.0.0.1","dstport":5353,'
                '"is_orig":false,"applayerlength":92,"timestamp":30.1}',
            ]
        ),
        encoding="utf-8",
    )
    conn = tmp_path / "conn.log"
    conn.write_text(
        "\n".join(
            [
                "#separator \\x09",
                "#empty_field\t(empty)",
                "#unset_field\t-",
                "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tduration\torig_bytes\tresp_bytes\torig_pkts\tresp_pkts\tservice",
                "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tinterval\tcount\tcount\tcount\tcount\tstring",
                "30.000\tC4\t10.0.0.1\t5353\t10.0.0.2\t53\tudp\t0.1\t44\t92\t1\t1\tdns",
            ]
        ),
        encoding="utf-8",
    )
    flows = aggregate_zeek_logs([features, conn])
    assert len(flows) == 1
    assert flows[0]["protocol"] == "UDP"
    assert flows[0]["packet_count"] == 2
    assert flows[0]["appinfo"] == ["dns", "dns"]
