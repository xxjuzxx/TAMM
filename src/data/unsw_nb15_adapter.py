from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset_adapter import normalize_attack_family
from src.data.label_policy import apply_attempted_policy, binary_label_for, normalize_label


UNSW_ATTACK_FAMILY_MAP = {
    "normal": "BENIGN",
    "generic": "Generic",
    "exploits": "Exploit",
    "fuzzers": "Fuzzers",
    "dos": "DoS",
    "reconnaissance": "Probe",
    "analysis": "Analysis",
    "backdoor": "Backdoor",
    "shellcode": "Shellcode",
    "worms": "Worms",
}

PROTO_BUCKETS = ("tcp", "udp", "icmp", "other")
SERVICE_BUCKETS = ("none", "dns", "ftp", "http", "smtp", "ssh", "ssl", "snmp", "other")
STATE_BUCKETS = ("none", "fin", "con", "int", "req", "rst", "other")

NUMERIC_SHARED_COLUMNS = [
    "log_duration",
    "log_fwd_packets",
    "log_bwd_packets",
    "log_total_packets",
    "log_fwd_bytes",
    "log_bwd_bytes",
    "log_total_bytes",
    "log_packet_rate",
    "log_byte_rate",
    "log_fwd_packet_rate",
    "log_bwd_packet_rate",
    "log_fwd_byte_rate",
    "log_bwd_byte_rate",
    "log_flow_iat_mean",
    "log_flow_iat_std",
    "log_flow_iat_max",
    "log_flow_iat_min",
    "log_fwd_iat_mean",
    "log_bwd_iat_mean",
    "log_packet_length_mean",
    "log_packet_length_std",
    "log_packet_length_max",
    "log_packet_length_min",
    "log_fwd_packet_length_mean",
    "log_bwd_packet_length_mean",
    "down_up_ratio",
    "fwd_packet_fraction",
    "bwd_packet_fraction",
    "fwd_byte_fraction",
    "bwd_byte_fraction",
    "short_flow_flag",
]

SHARED_FEATURE_COLUMNS = (
    NUMERIC_SHARED_COLUMNS
    + [f"proto_{item}" for item in PROTO_BUCKETS]
    + [f"service_{item}" for item in SERVICE_BUCKETS]
    + [f"state_{item}" for item in STATE_BUCKETS]
)


def normalize_unsw_attack_family(value: object) -> str:
    normalized = normalize_label(value)
    return UNSW_ATTACK_FAMILY_MAP.get(normalized.lower(), "OtherAttack")


def _clean_cicids_labels(
    raw_labels: pd.Series,
    *,
    attempted_policy: str = "drop",
    drop_literal_label: bool = True,
) -> tuple[pd.Series, pd.Series]:
    labels = raw_labels.map(normalize_label)
    if drop_literal_label:
        labels = labels.mask(labels.str.lower().isin({"", "label", "nan", "none"}))
    resolved = labels.map(lambda item: apply_attempted_policy(item, attempted_policy) if pd.notna(item) else None)
    keep_mask = resolved.notna()
    kept = resolved.loc[keep_mask].map(normalize_label)
    return kept, keep_mask


def unsw_binary_labels(frame: pd.DataFrame) -> np.ndarray:
    if "label" in frame.columns:
        return pd.to_numeric(frame["label"], errors="coerce").fillna(1).astype(int).clip(0, 1).to_numpy(dtype=np.int64)
    if "attack_cat" not in frame.columns:
        raise ValueError("UNSW-NB15 frame requires either label or attack_cat")
    return np.array(
        [0 if normalize_unsw_attack_family(item) == "BENIGN" else 1 for item in frame["attack_cat"]],
        dtype=np.int64,
    )


def unsw_family_labels(frame: pd.DataFrame) -> list[str]:
    if "attack_cat" not in frame.columns:
        raise ValueError("UNSW-NB15 frame requires attack_cat for family labels")
    return [normalize_unsw_attack_family(item) for item in frame["attack_cat"]]


def cicids2017_binary_labels(raw_labels: pd.Series, attempted_policy: str = "drop") -> tuple[np.ndarray, pd.Series, pd.Series]:
    kept, keep_mask = _clean_cicids_labels(raw_labels, attempted_policy=attempted_policy)
    y = kept.map(lambda item: 0 if binary_label_for(item) == "BENIGN" else 1).astype(int).to_numpy(dtype=np.int64)
    return y, kept, keep_mask


def cicids2017_family_labels(raw_labels: pd.Series, attempted_policy: str = "drop") -> tuple[np.ndarray, pd.Series, pd.Series]:
    kept, keep_mask = _clean_cicids_labels(raw_labels, attempted_policy=attempted_policy)
    families = kept.map(normalize_attack_family)
    y = families.astype(str).to_numpy()
    return y, families, keep_mask


def cicids2018_binary_labels(raw_labels: pd.Series, attempted_policy: str = "drop") -> tuple[np.ndarray, pd.Series, pd.Series]:
    kept, keep_mask = _clean_cicids_labels(raw_labels, attempted_policy=attempted_policy)
    y = kept.map(lambda item: 0 if binary_label_for(item) == "BENIGN" else 1).astype(int).to_numpy(dtype=np.int64)
    return y, kept, keep_mask


def cicids2018_family_labels(raw_labels: pd.Series, attempted_policy: str = "drop") -> tuple[np.ndarray, pd.Series, pd.Series]:
    kept, keep_mask = _clean_cicids_labels(raw_labels, attempted_policy=attempted_policy)
    families = kept.map(normalize_attack_family)
    y = families.astype(str).to_numpy()
    return y, families, keep_mask


def read_csv_paths(paths: list[str | Path], max_rows_per_file: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, nrows=max_rows_per_file, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame["__source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No CSV files were provided")
    return pd.concat(frames, ignore_index=True, sort=False)


def shared_feature_columns() -> list[str]:
    return list(SHARED_FEATURE_COLUMNS)


def cicids2017_label_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if str(column).strip().lower() == "label":
            return column
    raise ValueError("Could not find CICIDS2017 Label column")


def cicids2018_label_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if str(column).strip().lower() == "label":
            return column
    raise ValueError("Could not find CICIDS2018 Label column")


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype("float64")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = np.divide(
        num.to_numpy(dtype="float64"),
        den.to_numpy(dtype="float64"),
        out=np.zeros(len(num), dtype="float64"),
        where=den.to_numpy(dtype="float64") > 0.0,
    )
    return pd.Series(out, index=num.index, dtype="float64")


def _log1p(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0.0)).astype("float64")


def _clip_ratio(series: pd.Series, max_value: float = 1_000.0) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=max_value).astype("float64")


def _one_hot(values: pd.Series, prefix: str, buckets: tuple[str, ...]) -> pd.DataFrame:
    normalized = values.astype(str).str.lower().where(values.astype(str).str.lower().isin(buckets), "other")
    if "none" in buckets:
        normalized = normalized.where(~values.isna(), "none")
    data = {f"{prefix}_{bucket}": (normalized == bucket).astype("float32") for bucket in buckets}
    return pd.DataFrame(data, index=values.index)


def _cic_proto(frame: pd.DataFrame) -> pd.Series:
    raw = _num(frame, "Protocol", default=-1).round().astype(int)
    return raw.map({6: "tcp", 17: "udp", 1: "icmp"}).fillna("other")


def _unsw_proto(frame: pd.DataFrame) -> pd.Series:
    raw = frame.get("proto", pd.Series("other", index=frame.index)).astype(str).str.lower().str.strip()
    return raw.where(raw.isin({"tcp", "udp", "icmp"}), "other")


def _service_from_ports(ports: pd.Series) -> pd.Series:
    port = pd.to_numeric(ports, errors="coerce").fillna(-1).astype(int)
    mapping = {
        20: "ftp",
        21: "ftp",
        22: "ssh",
        25: "smtp",
        53: "dns",
        80: "http",
        123: "other",
        161: "snmp",
        443: "ssl",
        465: "smtp",
        587: "smtp",
        8080: "http",
        8443: "ssl",
    }
    return port.map(mapping).fillna("other").where(port > 0, "none")


def _unsw_service(frame: pd.DataFrame) -> pd.Series:
    raw = frame.get("service", pd.Series("-", index=frame.index)).astype(str).str.lower().str.strip()
    raw = raw.replace({"-": "none", "ftp-data": "ftp", "ssl": "ssl"})
    return raw.where(raw.isin(SERVICE_BUCKETS), "other")


def _cic_state(frame: pd.DataFrame) -> pd.Series:
    proto = _cic_proto(frame)
    fin = _num(frame, "FIN Flag Count")
    syn = _num(frame, "SYN Flag Count")
    rst = _num(frame, "RST Flag Count")
    psh = _num(frame, "PSH Flag Count")
    ack = _num(frame, "ACK Flag Count")
    state = pd.Series("none", index=frame.index, dtype="object")
    tcp_mask = proto == "tcp"
    state.loc[tcp_mask & (fin > 0)] = "fin"
    state.loc[tcp_mask & (rst > 0) & (fin <= 0)] = "rst"
    state.loc[tcp_mask & (syn > 0) & (ack <= 0) & (fin <= 0) & (rst <= 0)] = "req"
    state.loc[tcp_mask & ((ack > 0) | (psh > 0)) & (fin <= 0) & (rst <= 0)] = "con"
    state.loc[tcp_mask & state.eq("none")] = "other"
    return state


def _unsw_state(frame: pd.DataFrame) -> pd.Series:
    raw = frame.get("state", pd.Series("none", index=frame.index)).astype(str).str.lower().str.strip()
    mapping = {"fin": "fin", "con": "con", "int": "int", "req": "req", "rst": "rst"}
    return raw.map(mapping).fillna("other")


def _assemble_features(
    *,
    index: pd.Index,
    duration_s: pd.Series,
    fwd_packets: pd.Series,
    bwd_packets: pd.Series,
    fwd_bytes: pd.Series,
    bwd_bytes: pd.Series,
    flow_iat_mean_s: pd.Series,
    flow_iat_std_s: pd.Series,
    flow_iat_max_s: pd.Series,
    flow_iat_min_s: pd.Series,
    fwd_iat_mean_s: pd.Series,
    bwd_iat_mean_s: pd.Series,
    packet_length_mean: pd.Series,
    packet_length_std: pd.Series,
    packet_length_max: pd.Series,
    packet_length_min: pd.Series,
    fwd_packet_length_mean: pd.Series,
    bwd_packet_length_mean: pd.Series,
    proto: pd.Series,
    service: pd.Series,
    state: pd.Series,
) -> pd.DataFrame:
    total_packets = fwd_packets + bwd_packets
    total_bytes = fwd_bytes + bwd_bytes
    numeric = pd.DataFrame(
        {
            "log_duration": _log1p(duration_s),
            "log_fwd_packets": _log1p(fwd_packets),
            "log_bwd_packets": _log1p(bwd_packets),
            "log_total_packets": _log1p(total_packets),
            "log_fwd_bytes": _log1p(fwd_bytes),
            "log_bwd_bytes": _log1p(bwd_bytes),
            "log_total_bytes": _log1p(total_bytes),
            "log_packet_rate": _log1p(_safe_div(total_packets, duration_s)),
            "log_byte_rate": _log1p(_safe_div(total_bytes, duration_s)),
            "log_fwd_packet_rate": _log1p(_safe_div(fwd_packets, duration_s)),
            "log_bwd_packet_rate": _log1p(_safe_div(bwd_packets, duration_s)),
            "log_fwd_byte_rate": _log1p(_safe_div(fwd_bytes, duration_s)),
            "log_bwd_byte_rate": _log1p(_safe_div(bwd_bytes, duration_s)),
            "log_flow_iat_mean": _log1p(flow_iat_mean_s),
            "log_flow_iat_std": _log1p(flow_iat_std_s),
            "log_flow_iat_max": _log1p(flow_iat_max_s),
            "log_flow_iat_min": _log1p(flow_iat_min_s),
            "log_fwd_iat_mean": _log1p(fwd_iat_mean_s),
            "log_bwd_iat_mean": _log1p(bwd_iat_mean_s),
            "log_packet_length_mean": _log1p(packet_length_mean),
            "log_packet_length_std": _log1p(packet_length_std),
            "log_packet_length_max": _log1p(packet_length_max),
            "log_packet_length_min": _log1p(packet_length_min),
            "log_fwd_packet_length_mean": _log1p(fwd_packet_length_mean),
            "log_bwd_packet_length_mean": _log1p(bwd_packet_length_mean),
            "down_up_ratio": _clip_ratio(_safe_div(bwd_packets, fwd_packets)),
            "fwd_packet_fraction": _clip_ratio(_safe_div(fwd_packets, total_packets), max_value=1.0),
            "bwd_packet_fraction": _clip_ratio(_safe_div(bwd_packets, total_packets), max_value=1.0),
            "fwd_byte_fraction": _clip_ratio(_safe_div(fwd_bytes, total_bytes), max_value=1.0),
            "bwd_byte_fraction": _clip_ratio(_safe_div(bwd_bytes, total_bytes), max_value=1.0),
            "short_flow_flag": (total_packets <= 4).astype("float32"),
        },
        index=index,
    )
    features = pd.concat(
        [
            numeric,
            _one_hot(proto, "proto", PROTO_BUCKETS),
            _one_hot(service, "service", SERVICE_BUCKETS),
            _one_hot(state, "state", STATE_BUCKETS),
        ],
        axis=1,
    )
    features = features.reindex(columns=SHARED_FEATURE_COLUMNS, fill_value=0.0)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def cicids2017_shared_features(frame: pd.DataFrame) -> pd.DataFrame:
    duration_s = _num(frame, "Flow Duration") / 1_000_000.0
    return _assemble_features(
        index=frame.index,
        duration_s=duration_s,
        fwd_packets=_num(frame, "Total Fwd Packet"),
        bwd_packets=_num(frame, "Total Bwd packets"),
        fwd_bytes=_num(frame, "Total Length of Fwd Packet"),
        bwd_bytes=_num(frame, "Total Length of Bwd Packet"),
        flow_iat_mean_s=_num(frame, "Flow IAT Mean") / 1_000_000.0,
        flow_iat_std_s=_num(frame, "Flow IAT Std") / 1_000_000.0,
        flow_iat_max_s=_num(frame, "Flow IAT Max") / 1_000_000.0,
        flow_iat_min_s=_num(frame, "Flow IAT Min") / 1_000_000.0,
        fwd_iat_mean_s=_num(frame, "Fwd IAT Mean") / 1_000_000.0,
        bwd_iat_mean_s=_num(frame, "Bwd IAT Mean") / 1_000_000.0,
        packet_length_mean=_num(frame, "Packet Length Mean"),
        packet_length_std=_num(frame, "Packet Length Std"),
        packet_length_max=_num(frame, "Packet Length Max"),
        packet_length_min=_num(frame, "Packet Length Min"),
        fwd_packet_length_mean=_num(frame, "Fwd Packet Length Mean"),
        bwd_packet_length_mean=_num(frame, "Bwd Packet Length Mean"),
        proto=_cic_proto(frame),
        service=_service_from_ports(frame.get("Dst Port", pd.Series(0, index=frame.index))),
        state=_cic_state(frame),
    )


def cicids2018_shared_features(frame: pd.DataFrame) -> pd.DataFrame:
    duration_s = _num(frame, "Flow Duration") / 1_000_000.0
    return _assemble_features(
        index=frame.index,
        duration_s=duration_s,
        fwd_packets=_num(frame, "Tot Fwd Pkts"),
        bwd_packets=_num(frame, "Tot Bwd Pkts"),
        fwd_bytes=_num(frame, "TotLen Fwd Pkts"),
        bwd_bytes=_num(frame, "TotLen Bwd Pkts"),
        flow_iat_mean_s=_num(frame, "Flow IAT Mean") / 1_000_000.0,
        flow_iat_std_s=_num(frame, "Flow IAT Std") / 1_000_000.0,
        flow_iat_max_s=_num(frame, "Flow IAT Max") / 1_000_000.0,
        flow_iat_min_s=_num(frame, "Flow IAT Min") / 1_000_000.0,
        fwd_iat_mean_s=_num(frame, "Fwd IAT Mean") / 1_000_000.0,
        bwd_iat_mean_s=_num(frame, "Bwd IAT Mean") / 1_000_000.0,
        packet_length_mean=_num(frame, "Pkt Len Mean"),
        packet_length_std=_num(frame, "Pkt Len Std"),
        packet_length_max=_num(frame, "Pkt Len Max"),
        packet_length_min=_num(frame, "Pkt Len Min"),
        fwd_packet_length_mean=_num(frame, "Fwd Pkt Len Mean"),
        bwd_packet_length_mean=_num(frame, "Bwd Pkt Len Mean"),
        proto=_cic_proto(frame),
        service=_service_from_ports(frame.get("Dst Port", pd.Series(0, index=frame.index))),
        state=_cic_state(frame),
    )


def unsw_nb15_shared_features(frame: pd.DataFrame) -> pd.DataFrame:
    duration_s = _num(frame, "dur")
    spkts = _num(frame, "spkts")
    dpkts = _num(frame, "dpkts")
    sbytes = _num(frame, "sbytes")
    dbytes = _num(frame, "dbytes")
    total_packets = spkts + dpkts
    smean = _num(frame, "smean")
    dmean = _num(frame, "dmean")
    weighted_mean = _safe_div((smean * spkts) + (dmean * dpkts), total_packets)
    weighted_var = _safe_div((spkts * (smean - weighted_mean) ** 2) + (dpkts * (dmean - weighted_mean) ** 2), total_packets)
    packet_min = pd.concat([smean.replace(0.0, np.nan), dmean.replace(0.0, np.nan)], axis=1).min(axis=1).fillna(0.0)
    packet_max = pd.concat([smean, dmean], axis=1).max(axis=1).fillna(0.0)
    mean_iat_s = _safe_div(duration_s, (total_packets - 1.0).clip(lower=1.0))
    return _assemble_features(
        index=frame.index,
        duration_s=duration_s,
        fwd_packets=spkts,
        bwd_packets=dpkts,
        fwd_bytes=sbytes,
        bwd_bytes=dbytes,
        flow_iat_mean_s=mean_iat_s,
        flow_iat_std_s=((_num(frame, "sjit") + _num(frame, "djit")) / 2.0) / 1000.0,
        flow_iat_max_s=pd.concat([_num(frame, "sinpkt"), _num(frame, "dinpkt")], axis=1).max(axis=1) / 1000.0,
        flow_iat_min_s=pd.concat([_num(frame, "sinpkt"), _num(frame, "dinpkt")], axis=1).min(axis=1) / 1000.0,
        fwd_iat_mean_s=_num(frame, "sinpkt") / 1000.0,
        bwd_iat_mean_s=_num(frame, "dinpkt") / 1000.0,
        packet_length_mean=weighted_mean,
        packet_length_std=np.sqrt(weighted_var.clip(lower=0.0)),
        packet_length_max=packet_max,
        packet_length_min=packet_min,
        fwd_packet_length_mean=smean,
        bwd_packet_length_mean=dmean,
        proto=_unsw_proto(frame),
        service=_unsw_service(frame),
        state=_unsw_state(frame),
    )
