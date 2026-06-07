from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _slice_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "28_slice_pcap_by_label_windows.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("slice_pcap_by_label_windows", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_read_label_rows_applies_attempted_policy(tmp_path) -> None:
    module = _slice_module()
    csv_path = tmp_path / "wednesday.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Timestamp,Flow Duration,Label",
                "2017-07-05 13:43:23.194704,1000000,DoS Hulk",
                "2017-07-05 13:44:26.848472,1000000,DoS Hulk - Attempted",
            ]
        ),
        encoding="utf-8",
    )

    rows, stats = module.read_label_rows([csv_path], attempted_policy="drop")

    assert [row.label for row in rows] == ["DoS Hulk"]
    assert stats["skipped_attempted_rows"] == 1
    assert stats["label_counts"] == {"DoS Hulk": 1}


def test_build_windows_merges_nearby_rows_and_splits_gaps() -> None:
    module = _slice_module()
    rows = [
        module.LabelRow("labels.csv", 0, "DoS Hulk", "DoS Hulk", 100.0, 101.0),
        module.LabelRow("labels.csv", 1, "DoS Hulk", "DoS Hulk", 120.0, 121.0),
        module.LabelRow("labels.csv", 2, "DoS Hulk", "DoS Hulk", 400.0, 401.0),
    ]

    windows = module.build_windows(
        rows,
        Path("Wednesday.pcap"),
        Path("slices"),
        padding_seconds=5.0,
        merge_gap_seconds=60.0,
    )

    assert len(windows) == 2
    assert windows[0]["row_count"] == 2
    assert windows[0]["padded_start_ts"] == 95.0
    assert windows[0]["padded_end_ts"] == 125.0
    assert windows[1]["row_count"] == 1


def test_build_windows_can_use_flow_end_bounds_when_requested() -> None:
    module = _slice_module()
    rows = [
        module.LabelRow("labels.csv", 0, "DoS Hulk", "DoS Hulk", 100.0, 101.0),
        module.LabelRow("labels.csv", 1, "DoS Hulk", "DoS Hulk", 120.0, 121.0),
    ]

    windows = module.build_windows(
        rows,
        Path("Wednesday.pcap"),
        Path("slices"),
        padding_seconds=5.0,
        merge_gap_seconds=60.0,
        window_time_source="flow_end",
    )

    assert windows[0]["padded_end_ts"] == 126.0


def test_build_windows_can_cap_rows_per_window() -> None:
    module = _slice_module()
    rows = [
        module.LabelRow("labels.csv", idx, "DoS Hulk", "DoS Hulk", 100.0 + idx, 100.0 + idx)
        for idx in range(5)
    ]

    windows = module.build_windows(
        rows,
        Path("Wednesday.pcap"),
        Path("slices"),
        padding_seconds=0.0,
        merge_gap_seconds=60.0,
        max_rows_per_window=2,
    )

    assert [window["row_count"] for window in windows] == [2, 2, 1]


def test_filter_rows_supports_exact_and_glob_labels() -> None:
    module = _slice_module()
    rows = [
        module.LabelRow("labels.csv", 0, "DoS Hulk", "DoS Hulk", 100.0, 101.0),
        module.LabelRow("labels.csv", 1, "Web Attack - XSS", "Web Attack - XSS", 120.0, 121.0),
        module.LabelRow("labels.csv", 2, "BENIGN", "BENIGN", 400.0, 401.0),
    ]

    selected = module.filter_rows(rows, ["DoS Hulk"], ["Web Attack*"])

    assert [row.label for row in selected] == ["DoS Hulk", "Web Attack - XSS"]


def test_editcap_command_uses_utc_window_bounds() -> None:
    module = _slice_module()
    window = {
        "padded_start_utc": "2017-07-05T13:43:18.194704Z",
        "padded_end_utc": "2017-07-05T14:01:39.512067Z",
        "output_pcap": "slices/dos_hulk.pcap",
    }

    command = module.editcap_command(window, Path("Wednesday.pcap"), editcap_bin="editcap")

    assert command == [
        "editcap",
        "-F",
        "pcap",
        "-A",
        "2017-07-05T13:43:18.194704Z",
        "-B",
        "2017-07-05T14:01:39.512067Z",
        "Wednesday.pcap",
        "slices/dos_hulk.pcap",
    ]
