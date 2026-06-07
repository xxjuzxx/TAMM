from __future__ import annotations

import copy
import numpy as np
import torch

from src.training.session_aggregation_trainer import (
    SessionAggregationClassifier,
    build_session_windows,
    merge_flow_metadata,
    train_session_classifier,
)
from src.models.behavior_composer import BehaviorComposer


def _token_data() -> dict:
    labels = ["app_a", "app_a", "app_a", "app_b", "app_b", "app_b"]
    return {
        "input_ids": torch.tensor(
            [
                [1, 4, 5, 0],
                [1, 4, 6, 0],
                [1, 4, 7, 0],
                [1, 8, 9, 0],
                [1, 8, 10, 0],
                [1, 8, 11, 0],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 6, dtype=torch.long),
        "token_type_ids": torch.zeros((6, 4), dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        "label_to_id": {"app_a": 0, "app_b": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4, "X": 5, "Y": 6, "Z": 7, "P": 8, "Q": 9, "R": 10, "S": 11},
        "max_len": 4,
        "meta": [
            {"flow_id": f"f{i}", "label": labels[i], "start_ts": float(i), "service_key": ["1.1.1.1", "80", "TCP"]}
            for i in range(6)
        ],
    }


def test_build_session_windows_shapes_and_labels() -> None:
    token_data = _token_data()
    labels = token_data["labels"]
    dataset, stats, session_meta = build_session_windows(
        token_data,
        labels,
        np.array([0, 1, 2, 3, 4, 5]),
        group_by="service_key",
        window_n=2,
        stride=1,
        min_flows=2,
        min_purity=0.0,
    )

    assert dataset.tensors[0].shape == (5, 2, 4)
    assert dataset.tensors[3].shape == (5, 2)
    assert stats["num_sessions"] == 5
    assert len(session_meta) == 5
    assert all(item["flow_count"] == 2 for item in session_meta)


def test_session_windows_can_use_merged_src_dst_metadata() -> None:
    token_data = _token_data()
    stripped = copy.deepcopy(token_data)
    for row in stripped["meta"]:
        row.pop("src_ip", None)
        row.pop("dst_ip", None)
        row.pop("protocol", None)
    flow_rows = [
        {"flow_id": "f0", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "protocol": "TCP"},
        {"flow_id": "f1", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "protocol": "TCP"},
        {"flow_id": "f2", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "protocol": "TCP"},
        {"flow_id": "f3", "src_ip": "10.0.0.3", "dst_ip": "10.0.0.4", "protocol": "UDP"},
        {"flow_id": "f4", "src_ip": "10.0.0.3", "dst_ip": "10.0.0.4", "protocol": "UDP"},
        {"flow_id": "f5", "src_ip": "10.0.0.3", "dst_ip": "10.0.0.4", "protocol": "UDP"},
    ]

    merged, summary = merge_flow_metadata(stripped, flow_rows)
    dataset, stats, session_meta = build_session_windows(
        merged,
        merged["labels"],
        np.array([0, 1, 2, 3, 4, 5]),
        group_by="src_dst_pair",
        window_n=2,
        stride=1,
        min_flows=2,
        min_purity=0.0,
    )

    assert summary["matched_metadata_rows"] == 6
    assert stats["num_groups"] == 2
    assert stats["num_sessions"] == 4
    assert dataset.tensors[0].shape == (4, 2, 4)
    assert {tuple(row["group_key"]) for row in session_meta} == {
        ("src_dst_pair", "10.0.0.1", "10.0.0.2"),
        ("src_dst_pair", "10.0.0.3", "10.0.0.4"),
    }


def test_session_classifier_forward_shape() -> None:
    token_data = _token_data()
    encoder = BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=2,
        max_seq_len=4,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        dropout=0.0,
        pooling_strategy="mean",
    )
    model = SessionAggregationClassifier(
        encoder,
        hidden_size=16,
        num_classes=2,
        dropout=0.0,
        flow_representation="shared_encode",
        session_pooling_strategy="mean",
        encoder_frozen=True,
    )
    input_ids = torch.tensor([[[1, 4, 5, 0], [1, 8, 9, 0]]], dtype=torch.long)
    attention_mask = torch.tensor([[[1, 1, 1, 0], [1, 1, 1, 0]]], dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    flow_mask = torch.tensor([[1, 1]], dtype=torch.long)

    logits = model(input_ids, attention_mask, token_type_ids, flow_mask)

    assert logits.shape == (1, 2)


def test_session_classifier_handles_padded_flow_without_nan() -> None:
    token_data = _token_data()
    encoder = BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=2,
        max_seq_len=4,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        dropout=0.0,
        pooling_strategy="class_aware_attentive",
    )
    model = SessionAggregationClassifier(
        encoder,
        hidden_size=16,
        num_classes=2,
        dropout=0.0,
        flow_representation="class_aware_summary",
        session_pooling_strategy="mean",
        encoder_frozen=True,
    )
    input_ids = torch.tensor([[[1, 4, 5, 0], [0, 0, 0, 0]]], dtype=torch.long)
    attention_mask = torch.tensor([[[1, 1, 1, 0], [0, 0, 0, 0]]], dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    flow_mask = torch.tensor([[1, 0]], dtype=torch.long)

    logits = model(input_ids, attention_mask, token_type_ids, flow_mask)

    assert torch.isfinite(logits).all()


def test_session_classifier_transformer_attentive_forward_shape() -> None:
    token_data = _token_data()
    encoder = BehaviorComposer(
        vocab_size=len(token_data["vocab"]),
        num_classes=2,
        max_seq_len=4,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        dropout=0.0,
        pooling_strategy="mean",
    )
    model = SessionAggregationClassifier(
        encoder,
        hidden_size=16,
        num_classes=2,
        dropout=0.0,
        flow_representation="shared_encode",
        session_pooling_strategy="transformer_attentive",
        session_window_n=3,
        session_transformer_layers=1,
        session_transformer_heads=2,
        session_transformer_intermediate_size=32,
        encoder_frozen=True,
    )
    input_ids = torch.tensor([[[1, 4, 5, 0], [1, 8, 9, 0], [0, 0, 0, 0]]], dtype=torch.long)
    attention_mask = torch.tensor([[[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]], dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    flow_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    logits = model(input_ids, attention_mask, token_type_ids, flow_mask)

    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_train_session_classifier_uses_separate_eval_purity() -> None:
    token_data = _token_data()
    rows = 30
    labels = [0] * 15 + [1] * 15
    token_data = {
        "input_ids": token_data["input_ids"][torch.tensor([0, 1, 2, 3, 4, 5] * 5)],
        "attention_mask": token_data["attention_mask"][torch.tensor([0, 1, 2, 3, 4, 5] * 5)],
        "token_type_ids": token_data["token_type_ids"][torch.tensor([0, 1, 2, 3, 4, 5] * 5)],
        "labels": torch.tensor(labels, dtype=torch.long),
        "binary_labels": torch.tensor(labels, dtype=torch.long),
        "label_to_id": {"app_a": 0, "app_b": 1},
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "vocab": token_data["vocab"],
        "max_len": token_data["max_len"],
        "meta": [
            {
                "flow_id": f"f{i}",
                "label": "app_a" if labels[i] == 0 else "app_b",
                "start_ts": float(i),
                "service_key": ["host", str(i % 3), "TCP"],
            }
            for i in range(rows)
        ],
    }
    config = {
        "seed": 7,
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 32,
            "dropout": 0.0,
            "max_seq_len": 4,
            "pooling": "mean",
            "session_group_by": "service_host_proto",
            "session_window_n": 3,
            "session_stride": 1,
            "session_min_flows": 2,
            "session_train_min_purity": 0.8,
            "session_eval_min_purity": 0.0,
            "session_pooling_strategy": "mean",
            "session_flow_representation": "shared_encode",
        },
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "loss": "ce",
        },
    }

    result = train_session_classifier(copy.deepcopy(token_data), config, task="multiclass", split="stratified")

    assert result.metrics["session_train_min_purity"] == 0.8
    assert result.metrics["session_eval_min_purity"] == 0.0
    assert result.metrics["session_stats"]["test"]["dropped_low_purity_sessions"] == 0
