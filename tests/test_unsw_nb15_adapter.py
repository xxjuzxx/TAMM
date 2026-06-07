from __future__ import annotations

import pandas as pd

from src.data.unsw_nb15_adapter import (
    cicids2017_shared_features,
    normalize_unsw_attack_family,
    shared_feature_columns,
    unsw_binary_labels,
    unsw_family_labels,
    unsw_nb15_shared_features,
)


def test_unsw_attack_family_mapping() -> None:
    assert normalize_unsw_attack_family("Normal") == "BENIGN"
    assert normalize_unsw_attack_family("Reconnaissance") == "Probe"
    assert normalize_unsw_attack_family("Exploits") == "Exploit"
    assert normalize_unsw_attack_family("Worms") == "Worms"


def test_unsw_labels_from_attack_cat_and_binary_label() -> None:
    frame = pd.DataFrame({"attack_cat": ["Normal", "Generic"], "label": [0, 1]})
    assert unsw_binary_labels(frame).tolist() == [0, 1]
    assert unsw_family_labels(frame) == ["BENIGN", "Generic"]


def test_shared_feature_adapters_have_same_columns() -> None:
    cic = pd.DataFrame(
        {
            "Protocol": [6],
            "Dst Port": [80],
            "Flow Duration": [1_000_000],
            "Total Fwd Packet": [2],
            "Total Bwd packets": [1],
            "Total Length of Fwd Packet": [100],
            "Total Length of Bwd Packet": [50],
            "Flow IAT Mean": [500_000],
            "Flow IAT Std": [100_000],
            "Flow IAT Max": [900_000],
            "Flow IAT Min": [100_000],
            "Fwd IAT Mean": [400_000],
            "Bwd IAT Mean": [300_000],
            "Packet Length Mean": [50],
            "Packet Length Std": [10],
            "Packet Length Max": [100],
            "Packet Length Min": [20],
            "Fwd Packet Length Mean": [50],
            "Bwd Packet Length Mean": [50],
            "ACK Flag Count": [1],
        }
    )
    unsw = pd.DataFrame(
        {
            "dur": [1.0],
            "proto": ["tcp"],
            "service": ["http"],
            "state": ["FIN"],
            "spkts": [2],
            "dpkts": [1],
            "sbytes": [100],
            "dbytes": [50],
            "sinpkt": [400],
            "dinpkt": [300],
            "sjit": [100],
            "djit": [100],
            "smean": [50],
            "dmean": [50],
        }
    )

    cic_features = cicids2017_shared_features(cic)
    unsw_features = unsw_nb15_shared_features(unsw)

    assert cic_features.columns.tolist() == shared_feature_columns()
    assert unsw_features.columns.tolist() == shared_feature_columns()
    assert cic_features.loc[0, "proto_tcp"] == 1.0
    assert unsw_features.loc[0, "service_http"] == 1.0
