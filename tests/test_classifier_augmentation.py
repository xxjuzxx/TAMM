from __future__ import annotations

import pytest
import numpy as np
import torch

from src.models.behavior_composer import BehaviorComposer
from src.training.classifier_trainer import (
    ClassPrototypeRegularizer,
    ContextAwareClassifier,
    FocalLoss,
    shared_representation_preservation_loss,
    supervised_contrastive_loss,
    _augment_indices,
    _balanced_sampler_for_labels,
    _build_extra_train_parts,
    _build_train_parts,
    _encoder_parameter_names,
    _merged_labels,
    _raw_label_group_ids,
    _temporal_stratified_by_group_indices,
    train_classifier,
)


def _token_data(num_rows: int = 8) -> dict:
    binary_labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)
    return {
        "input_ids": torch.arange(num_rows * 4, dtype=torch.long).reshape(num_rows, 4),
        "attention_mask": torch.ones((num_rows, 4), dtype=torch.long),
        "token_type_ids": torch.zeros((num_rows, 4), dtype=torch.long),
        "binary_labels": binary_labels,
        "labels": binary_labels.clone(),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }


def test_behavior_composer_pooling_strategies_forward_shapes() -> None:
    input_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 3, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)

    for strategy in ("cls", "mean", "attentive"):
        model = BehaviorComposer(
            vocab_size=8,
            num_classes=3,
            max_seq_len=4,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            intermediate_size=32,
            dropout=0.0,
            pooling_strategy=strategy,
        )
        logits = model(input_ids, attention_mask, token_type_ids)
        embedding = model.encode(input_ids, attention_mask, token_type_ids)
        assert logits.shape == (2, 3)
        assert embedding.shape == (2, 16)


def test_behavior_composer_class_aware_pooling_forward_shape() -> None:
    input_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 3, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    model = BehaviorComposer(
        vocab_size=8,
        num_classes=4,
        max_seq_len=4,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        dropout=0.0,
        pooling_strategy="class_aware_attentive",
    )

    logits = model(input_ids, attention_mask, token_type_ids)
    embedding = model.encode(input_ids, attention_mask, token_type_ids)
    token_hidden = model.encode_tokens(input_ids, attention_mask, token_type_ids)
    class_embeddings = model.class_aware_encode(token_hidden, attention_mask)

    assert logits.shape == (2, 4)
    assert embedding.shape == (2, 16)
    assert class_embeddings.shape == (2, 4, 16)


def test_class_aware_encoder_freeze_keeps_head_trainable() -> None:
    model = BehaviorComposer(
        vocab_size=8,
        num_classes=4,
        max_seq_len=4,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        dropout=0.0,
        pooling_strategy="class_aware_attentive",
    )
    encoder_names = _encoder_parameter_names(model)
    for name, parameter in model.named_parameters():
        if name in encoder_names:
            parameter.requires_grad = False

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert "class_queries" in trainable
    assert "class_logit_weight" in trainable
    assert "class_logit_bias" in trainable
    assert not any(name.startswith("encoder.") for name in trainable)


def test_behavior_composer_residual_class_aware_pooling_keeps_mean_encode() -> None:
    input_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 3, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    model = BehaviorComposer(
        vocab_size=8,
        num_classes=4,
        max_seq_len=4,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        dropout=0.0,
        pooling_strategy="residual_class_aware",
        class_aware_alpha=0.5,
        cls_beta=0.1,
    )
    model.eval()

    logits = model(input_ids, attention_mask, token_type_ids)
    token_hidden = model.encode_tokens(input_ids, attention_mask, token_type_ids)
    mean_embedding = model.mean_pool(token_hidden, attention_mask)
    residual_embedding = model.residual_class_aware_pool(token_hidden, attention_mask)
    encode_embedding = model.encode(input_ids, attention_mask, token_type_ids)
    classification_embedding = model.classification_embedding(input_ids, attention_mask, token_type_ids)

    assert logits.shape == (2, 4)
    assert encode_embedding.shape == (2, 16)
    assert residual_embedding.shape == (2, 16)
    assert torch.allclose(encode_embedding, mean_embedding, atol=1e-5)
    assert torch.allclose(classification_embedding, residual_embedding, atol=1e-5)


def test_shared_representation_preservation_loss_smoke() -> None:
    current = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    teacher = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    cosine_loss = shared_representation_preservation_loss(current, teacher, mode="cosine")
    mse_loss = shared_representation_preservation_loss(current, teacher, mode="mse")

    assert cosine_loss.item() >= 0.0
    assert mse_loss.item() >= 0.0


def test_augment_indices_default_to_train_attacks_only() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0])
    train_idx = np.array([0, 1, 2, 3])
    selected = _augment_indices(labels, train_idx, task="binary", attack_only=True, fraction=1.0, seed=7)
    assert selected.tolist() == [1, 3]


def test_build_train_parts_appends_aligned_augmented_attacks() -> None:
    base = _token_data()
    aug = _token_data()
    labels = base["binary_labels"]
    train_idx = np.array([0, 1, 2, 3])
    parts, info = _build_train_parts(
        base,
        labels,
        labels.numpy(),
        train_idx,
        task="binary",
        augment_token_data=[aug],
        augment_attack_only=True,
        augment_fraction=1.0,
        augment_fractions=None,
        seed=7,
    )
    assert len(parts) == 2
    assert parts[0][2].tolist() == [0, 1, 2, 3]
    assert parts[1][2].tolist() == [1, 3]
    assert info["num_augmented_train"] == 2


def test_build_train_parts_rejects_vocab_mismatch() -> None:
    base = _token_data()
    aug = _token_data()
    aug["vocab"] = {**aug["vocab"], "EXTRA": 99}
    try:
        _build_train_parts(
            base,
            base["binary_labels"],
            base["binary_labels"].numpy(),
            np.array([0, 1, 2, 3]),
            task="binary",
            augment_token_data=[aug],
            augment_attack_only=True,
            augment_fraction=1.0,
            augment_fractions=None,
            seed=7,
        )
    except ValueError as exc:
        assert "same vocabulary" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched vocab")


def test_build_train_parts_supports_per_source_fractions() -> None:
    base = _token_data()
    aug_a = _token_data()
    aug_b = _token_data()
    parts, info = _build_train_parts(
        base,
        base["binary_labels"],
        base["binary_labels"].numpy(),
        np.array([0, 1, 2, 3]),
        task="binary",
        augment_token_data=[aug_a, aug_b],
        augment_attack_only=True,
        augment_fraction=1.0,
        augment_fractions=[1.0, 0.5],
        seed=7,
    )
    assert len(parts) == 3
    assert parts[1][2].tolist() == [1, 3]
    assert len(parts[2][2]) == 1
    assert info["fractions"] == [1.0, 0.5]
    assert info["num_augmented_train"] == 3
    assert info["num_augmented_train_by_source"] == [2, 1]


def test_build_train_parts_rejects_mismatched_fraction_count() -> None:
    base = _token_data()
    try:
        _build_train_parts(
            base,
            base["binary_labels"],
            base["binary_labels"].numpy(),
            np.array([0, 1, 2, 3]),
            task="binary",
            augment_token_data=[_token_data(), _token_data()],
            augment_attack_only=True,
            augment_fraction=1.0,
            augment_fractions=[1.0],
            seed=7,
        )
    except ValueError as exc:
        assert "augment_fractions length" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched per-source fractions")


def test_augment_seed_changes_per_source_fraction_sampling() -> None:
    labels = np.array([0, 1, 0, 1, 1, 1])
    train_idx = np.array([0, 1, 2, 3, 4, 5])
    selected_a = _augment_indices(labels, train_idx, task="binary", attack_only=True, fraction=0.5, seed=7)
    selected_b = _augment_indices(labels, train_idx, task="binary", attack_only=True, fraction=0.5, seed=8)
    assert selected_a.tolist() != selected_b.tolist()
    assert selected_a.size == selected_b.size == 2


def test_build_extra_train_parts_accepts_different_row_count() -> None:
    base = _token_data()
    extra = _token_data(num_rows=4)
    extra["binary_labels"] = torch.tensor([1, 1, 0, 1], dtype=torch.long)
    extra["labels"] = extra["binary_labels"].clone()

    parts, info = _build_extra_train_parts(
        base,
        [extra],
        task="binary",
        target_label_to_id={"BENIGN": 0, "ATTACK": 1},
        fraction=1.0,
        seed=7,
    )

    assert len(parts) == 1
    assert parts[0][2].tolist() == [0, 1, 2, 3]
    assert parts[0][1].tolist() == [1, 1, 0, 1]
    assert info["num_extra_train"] == 4


def test_build_extra_train_parts_remaps_multiclass_merged_labels() -> None:
    base = _token_data()
    extra = {
        **_token_data(num_rows=3),
        "meta": [
            {"label": "Web Attack - XSS"},
            {"label": "DoS Hulk"},
            {"label": "BENIGN"},
        ],
    }

    parts, info = _build_extra_train_parts(
        base,
        [extra],
        task="multiclass_merged",
        target_label_to_id={"BENIGN": 0, "DoS": 1, "WebAttack": 2},
        fraction=1.0,
        seed=7,
    )

    assert parts[0][1].tolist() == [2, 1, 0]
    assert info["num_extra_train"] == 3


def test_build_extra_train_parts_rejects_unknown_labels() -> None:
    base = _token_data()
    extra = {
        **_token_data(num_rows=1),
        "meta": [{"label": "Heartbleed"}],
    }

    with pytest.raises(ValueError, match="labels not present"):
        _build_extra_train_parts(
            base,
            [extra],
            task="multiclass_merged",
            target_label_to_id={"BENIGN": 0, "DoS": 1},
            fraction=1.0,
            seed=7,
        )


def test_multiclass_merged_uses_shared_cicids_label_policy() -> None:
    token_data = {
        "meta": [
            {"label": "BENIGN"},
            {"label": "Web Attack - XSS"},
            {"label": "Web Attack - SQL Injection"},
            {"label": "FTP-Patator"},
            {"label": "SSH-Patator"},
            {"label": "DoS Slowhttptest"},
        ]
    }
    labels, label_to_id = _merged_labels(token_data)

    assert label_to_id == {"BENIGN": 0, "BruteForce": 1, "DoS": 2, "WebAttack": 3}
    assert labels.tolist() == [0, 3, 3, 1, 1, 2]


def test_raw_label_temporal_group_split_keeps_each_exact_label_in_test() -> None:
    labels = np.array([0] * 5 + [1] * 5 + [1] * 5)
    groups = np.array([0] * 5 + [1] * 5 + [2] * 5)
    order = np.arange(15)

    train_idx, val_idx, test_idx = _temporal_stratified_by_group_indices(
        labels,
        groups,
        val_ratio=0.2,
        test_ratio=0.2,
        order_values=order,
    )

    assert train_idx.tolist() == [0, 1, 2, 5, 6, 7, 10, 11, 12]
    assert val_idx.tolist() == [3, 8, 13]
    assert test_idx.tolist() == [4, 9, 14]


def test_raw_label_group_ids_use_exact_labels_from_meta() -> None:
    token_data = {
        "input_ids": torch.zeros((3, 4), dtype=torch.long),
        "meta": [
            {"label": "Web Attack - XSS"},
            {"label": "Web Attack - SQL Injection"},
            {"label": "Web Attack - XSS"},
        ],
    }

    group_ids = _raw_label_group_ids(token_data)

    assert group_ids[0] == group_ids[2]
    assert group_ids[0] != group_ids[1]


def test_train_classifier_accepts_service_context_features() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 8, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 8, dtype=torch.long),
        "token_type_ids": torch.zeros((8, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
        "meta": [
            {"flow_id": "a", "label": "Benign", "binary_label": "BENIGN", "start_ts": 0.0, "service_context": {"recent_count": 0, "recent_short": 0, "recent_packets": 0, "short_ratio": 0.0, "last_gap": 0.0}},
            {"flow_id": "b", "label": "Benign", "binary_label": "BENIGN", "start_ts": 1.0, "service_context": {"recent_count": 1, "recent_short": 1, "recent_packets": 2, "short_ratio": 1.0, "last_gap": 1.0}},
            {"flow_id": "c", "label": "Attack", "binary_label": "ATTACK", "start_ts": 2.0, "service_context": {"recent_count": 2, "recent_short": 1, "recent_packets": 4, "short_ratio": 0.5, "last_gap": 1.0}},
            {"flow_id": "d", "label": "Attack", "binary_label": "ATTACK", "start_ts": 3.0, "service_context": {"recent_count": 3, "recent_short": 2, "recent_packets": 6, "short_ratio": 0.66, "last_gap": 1.0}},
            {"flow_id": "e", "label": "Benign", "binary_label": "BENIGN", "start_ts": 4.0, "service_context": {"recent_count": 0, "recent_short": 0, "recent_packets": 0, "short_ratio": 0.0, "last_gap": 0.0}},
            {"flow_id": "f", "label": "Benign", "binary_label": "BENIGN", "start_ts": 5.0, "service_context": {"recent_count": 1, "recent_short": 1, "recent_packets": 2, "short_ratio": 1.0, "last_gap": 1.0}},
            {"flow_id": "g", "label": "Attack", "binary_label": "ATTACK", "start_ts": 6.0, "service_context": {"recent_count": 2, "recent_short": 1, "recent_packets": 4, "short_ratio": 0.5, "last_gap": 1.0}},
            {"flow_id": "h", "label": "Attack", "binary_label": "ATTACK", "start_ts": 7.0, "service_context": {"recent_count": 3, "recent_short": 2, "recent_packets": 6, "short_ratio": 0.66, "last_gap": 1.0}},
        ],
    }
    config = {
        "seed": 7,
        "training": {"epochs": 1, "batch_size": 2, "val_ratio": 0.25, "test_ratio": 0.25, "learning_rate": 1e-3, "weight_decay": 0.0},
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }
    result = train_classifier(token_data, config, use_service_context=True)
    assert result.metrics["use_service_context"] is True
    assert "threshold" in result.metrics
    assert result.predictions


def test_train_classifier_accepts_anomaly_features() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 8, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 8, dtype=torch.long),
        "token_type_ids": torch.zeros((8, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
        "meta": [
            {"flow_id": f"f{i}", "label": "Benign" if i < 4 else "Attack", "binary_label": "BENIGN" if i < 4 else "ATTACK", "start_ts": float(i), "service_key": ["1.1.1.1", "80", "TCP"]}
            for i in range(8)
        ],
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "ce",
            "anomaly_features": {
                "score_method": "cosine",
                "normalize": "l2",
                "include_special": False,
                "use_service_prototype_distance": True,
                "use_class_prototype_distance": False,
                "anomaly_feature_dim": 8,
            },
        },
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 32,
            "dropout": 0.0,
            "max_seq_len": 4,
            "use_anomaly_features": True,
            "use_service_prototype_distance": True,
            "use_class_prototype_distance": False,
            "anomaly_feature_dim": 8,
        },
    }

    result = train_classifier(token_data, config, use_anomaly_features=True)

    assert result.metrics["use_anomaly_features"] is True
    assert result.metrics["anomaly_features"]["use_service_prototype_distance"] is True
    assert "anomaly_features" in result.metrics
    assert result.predictions


def test_train_classifier_anomaly_features_fallback_without_benign_label() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 5, 6, 0]] * 15, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 15, dtype=torch.long),
        "token_type_ids": torch.zeros((15, 4), dtype=torch.long),
        "labels": torch.tensor([0] * 5 + [1] * 5 + [2] * 5, dtype=torch.long),
        "binary_labels": torch.tensor([1] * 15, dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"app_a": 0, "app_b": 1, "app_c": 2},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4, "PRIM_PROFILE_SHORT_FLOW": 5, "PRIM_PROFILE_REPEAT_SEG": 6},
        "max_len": 4,
        "meta": [
            {"flow_id": f"f{i}", "label": ["app_a", "app_b", "app_c"][i // 5], "start_ts": float(i), "service_key": ["svc", str(i % 2)]}
            for i in range(15)
        ],
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 3,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "ce",
            "anomaly_features": {
                "score_method": "cosine",
                "normalize": "l2",
                "include_special": False,
                "use_service_prototype_distance": True,
                "use_class_prototype_distance": False,
                "anomaly_feature_dim": 8,
            },
        },
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 32,
            "dropout": 0.0,
            "max_seq_len": 4,
            "use_anomaly_features": True,
            "use_service_prototype_distance": True,
            "anomaly_feature_dim": 8,
        },
    }

    result = train_classifier(token_data, config, task="multiclass", split="temporal_stratified")

    assert result.metrics["use_anomaly_features"] is True
    assert result.metrics["anomaly_features"]["prototype_scope"] == "all_train_fallback"
    assert result.metrics["anomaly_features"]["benign_label_id"] is None
    assert result.metrics["anomaly_features"]["warnings"]
    assert result.predictions


def test_train_classifier_accepts_hierarchical_classifier() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 12, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 12, dtype=torch.long),
        "token_type_ids": torch.zeros((12, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "DoS": 1, "PortScan": 2},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "ce",
            "use_hierarchical_classifier": True,
            "lambda_binary": 0.5,
            "lambda_coarse": 1.0,
            "hierarchical_gated_inference": True,
        },
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }

    result = train_classifier(token_data, config, task="multiclass")

    assert result.metrics["use_hierarchical_classifier"] is True
    assert result.metrics["hierarchical_classifier"]["lambda_binary"] == 0.5
    assert result.predictions


def test_train_classifier_hierarchical_coarse_only_without_benign_label() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 15, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 15, dtype=torch.long),
        "token_type_ids": torch.zeros((15, 4), dtype=torch.long),
        "binary_labels": torch.tensor([1] * 15, dtype=torch.long),
        "labels": torch.tensor([0] * 5 + [1] * 5 + [2] * 5, dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"app_a": 0, "app_b": 1, "app_c": 2},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 3,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "ce",
            "use_hierarchical_classifier": True,
            "lambda_binary": 0.0,
            "lambda_coarse": 1.0,
            "hierarchical_gated_inference": False,
        },
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }

    result = train_classifier(token_data, config, task="multiclass", split="temporal_stratified")

    assert result.metrics["use_hierarchical_classifier"] is True
    assert result.metrics["hierarchical_classifier"]["binary_loss_enabled"] is False
    assert result.metrics["hierarchical_classifier"]["benign_label_id"] is None
    assert result.predictions


def test_train_classifier_accepts_stat_fusion() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 5, 6, 0], [1, 7, 8, 0], [1, 5, 9, 0], [1, 10, 11, 0], [1, 5, 6, 0], [1, 7, 8, 0], [1, 5, 9, 0], [1, 10, 11, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 8, dtype=torch.long),
        "token_type_ids": torch.zeros((8, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[MASK]": 3,
            "[UNK]": 4,
            "PRIM_PROFILE_SHORT_FLOW": 5,
            "BURST_SINGLE": 6,
            "PRIM_PROFILE_SAME_LEN": 7,
            "PRIM_PROFILE_REPEAT_SEG": 8,
            "PRIM_PROFILE_DUP_SEG": 9,
            "PKT_LEN_1": 10,
            "PKT_IAT_1": 11,
        },
        "max_len": 4,
        "meta": [
            {"flow_id": f"f{i}", "label": "Benign" if i in {0, 1, 4, 5} else "Attack", "packet_count": float(i + 1), "token_count": 3, "start_ts": float(i)}
            for i in range(8)
        ],
    }
    config = {
        "seed": 7,
        "training": {"epochs": 1, "batch_size": 2, "val_ratio": 0.25, "test_ratio": 0.25, "learning_rate": 1e-3, "weight_decay": 0.0, "loss": "ce"},
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 32,
            "dropout": 0.0,
            "max_seq_len": 4,
            "use_stat_fusion": True,
            "stat_feature_names": ["packet_count", "token_count", "profile_short_count", "profile_repeat_count"],
            "stat_mlp_dim": 4,
        },
    }

    result = train_classifier(token_data, config)

    assert result.metrics["use_stat_fusion"] is True
    assert result.metrics["stat_fusion"]["feature_dim"] == 4
    assert result.predictions


def test_train_classifier_accepts_class_aware_pooling() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 12, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 12, dtype=torch.long),
        "token_type_ids": torch.zeros((12, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "DoS": 1, "PortScan": 2},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }
    config = {
        "seed": 7,
        "training": {"epochs": 1, "batch_size": 2, "val_ratio": 0.25, "test_ratio": 0.25, "learning_rate": 1e-3, "weight_decay": 0.0, "loss": "ce"},
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 32,
            "dropout": 0.0,
            "max_seq_len": 4,
            "pooling_strategy": "class_aware_attentive",
            "class_aware_pooling": True,
        },
    }

    result = train_classifier(token_data, config, task="multiclass")

    assert result.metrics["pooling_strategy"] == "class_aware_attentive"
    assert result.metrics["class_aware_pooling"] is True
    assert result.predictions


def test_train_classifier_accepts_class_aware_pooling_with_stat_fusion() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 5, 6, 0], [1, 7, 8, 0], [1, 5, 9, 0], [1, 10, 11, 0], [1, 5, 6, 0], [1, 7, 8, 0], [1, 5, 9, 0], [1, 10, 11, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 8, dtype=torch.long),
        "token_type_ids": torch.zeros((8, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[MASK]": 3,
            "[UNK]": 4,
            "PRIM_PROFILE_SHORT_FLOW": 5,
            "BURST_SINGLE": 6,
            "PRIM_PROFILE_SAME_LEN": 7,
            "PRIM_PROFILE_REPEAT_SEG": 8,
            "PRIM_PROFILE_DUP_SEG": 9,
            "PKT_LEN_1": 10,
            "PKT_IAT_1": 11,
        },
        "max_len": 4,
        "meta": [
            {"flow_id": f"f{i}", "label": "Benign" if i in {0, 1, 4, 5} else "Attack", "packet_count": float(i + 1), "token_count": 3, "start_ts": float(i)}
            for i in range(8)
        ],
    }
    config = {
        "seed": 7,
        "training": {"epochs": 1, "batch_size": 2, "val_ratio": 0.25, "test_ratio": 0.25, "learning_rate": 1e-3, "weight_decay": 0.0, "loss": "ce"},
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 32,
            "dropout": 0.0,
            "max_seq_len": 4,
            "pooling_strategy": "class_aware_attentive",
            "class_aware_pooling": True,
            "use_stat_fusion": True,
            "stat_feature_names": ["packet_count", "token_count"],
            "stat_mlp_dim": 4,
        },
    }

    result = train_classifier(token_data, config)

    assert result.metrics["class_aware_pooling"] is True
    assert result.metrics["use_stat_fusion"] is True
    assert result.predictions


def test_balanced_sampler_uses_inverse_class_frequency() -> None:
    labels = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    sampler = _balanced_sampler_for_labels(labels, num_classes=2, seed=7)
    weights = sampler.weights.tolist()

    assert weights[:3] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert weights[3] == pytest.approx(1.0)


def test_focal_loss_accepts_class_weights() -> None:
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.5]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    criterion = FocalLoss(gamma=2.0, weight=torch.tensor([1.0, 2.0]))

    loss = criterion(logits, labels)

    assert float(loss.item()) > 0.0


def test_supervised_contrastive_loss_handles_missing_positives() -> None:
    features = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss = supervised_contrastive_loss(features, labels)

    assert float(loss.item()) == pytest.approx(0.0)


def test_train_classifier_records_balanced_sampling_and_focal_loss() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 12, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 12, dtype=torch.long),
        "token_type_ids": torch.zeros((12, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "weighted_focal",
            "sampling": "class_balanced",
            "focal_gamma": 1.5,
        },
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }

    result = train_classifier(token_data, config)

    assert result.metrics["loss"] == "weighted_focal"
    assert result.metrics["sampling"] == "class_balanced"
    assert result.metrics["focal_gamma"] == 1.5
    assert {"true_label", "pred_label", "scores"}.issubset(result.predictions[0])


def test_train_classifier_accepts_supcon_and_proto_regularization() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 12, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 12, dtype=torch.long),
        "token_type_ids": torch.zeros((12, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "ce",
            "use_supcon": True,
            "lambda_supcon": 0.01,
            "supcon_temperature": 0.2,
            "use_class_proto_loss": True,
            "lambda_class_proto": 0.01,
        },
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }

    result = train_classifier(token_data, config)

    assert result.metrics["use_supcon"] is True
    assert result.metrics["use_class_proto_loss"] is True
    assert result.metrics["supcon"]["lambda_supcon"] == 0.01
    assert result.metrics["class_proto_loss"]["lambda_class_proto"] == 0.01


def test_context_classifier_app_projection_keeps_encoder_encode_shared() -> None:
    input_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 3, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    encoder = BehaviorComposer(
        vocab_size=8,
        num_classes=3,
        max_seq_len=4,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        dropout=0.0,
        pooling_strategy="class_aware_attentive",
    )
    model = ContextAwareClassifier(
        encoder,
        hidden_size=16,
        num_classes=3,
        dropout=0.0,
        use_app_projection=True,
        app_projection_dim=12,
    )
    model.eval()

    shared = encoder.encode(input_ids, attention_mask, token_type_ids)
    app = model.app_features(input_ids, attention_mask, token_type_ids)
    logits = model(input_ids, attention_mask, token_type_ids)

    assert shared.shape == (2, 16)
    assert app.shape == (2, 12)
    assert logits.shape == (2, 3)
    assert torch.allclose(encoder.encode(input_ids, attention_mask, token_type_ids), shared, atol=1e-5)


def test_context_classifier_app_projection_can_use_class_aware_summary() -> None:
    input_ids = torch.tensor([[1, 2, 3, 0], [1, 4, 3, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)
    encoder = BehaviorComposer(
        vocab_size=8,
        num_classes=3,
        max_seq_len=4,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        dropout=0.0,
        pooling_strategy="class_aware_attentive",
    )
    model = ContextAwareClassifier(
        encoder,
        hidden_size=16,
        num_classes=3,
        dropout=0.0,
        use_app_projection=True,
        app_projection_dim=12,
        app_projection_input="class_aware_summary",
    )

    app = model.app_features(input_ids, attention_mask, token_type_ids)
    logits = model(input_ids, attention_mask, token_type_ids)

    assert app.shape == (2, 12)
    assert logits.shape == (2, 3)


def test_app_projection_aux_losses_are_finite_with_singletons() -> None:
    features = torch.randn(4, 12, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    proto = ClassPrototypeRegularizer(num_classes=4, feature_dim=12)

    supcon = supervised_contrastive_loss(features, labels, temperature=0.2)
    proto_loss = proto(features, labels)

    assert torch.isfinite(supcon)
    assert torch.isfinite(proto_loss)


def test_train_classifier_accepts_app_projection_supcon_and_proto_aliases() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 12, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 12, dtype=torch.long),
        "token_type_ids": torch.zeros((12, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "ce",
            "use_supcon": True,
            "supcon_weight": 0.01,
            "supcon_temperature": 0.2,
            "use_app_proto_reg": True,
            "app_proto_weight": 0.01,
            "detach_shared_for_aux_loss": True,
        },
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 32,
            "dropout": 0.0,
            "max_seq_len": 4,
            "pooling_strategy": "class_aware_attentive",
            "class_aware_pooling": True,
            "use_app_projection": True,
            "app_projection_dim": 12,
        },
    }

    result = train_classifier(token_data, config)

    assert result.metrics["use_app_projection"] is True
    assert result.metrics["app_projection"]["dim"] == 12
    assert result.metrics["app_projection"]["detach_shared_for_aux_loss"] is True
    assert result.metrics["supcon"]["supcon_weight"] == 0.01
    assert result.metrics["class_proto_loss"]["app_proto_weight"] == 0.01
    assert result.metrics["class_proto_loss"]["feature_space"] == "app_projection"


def test_train_classifier_records_encoder_learning_rate() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 12, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 12, dtype=torch.long),
        "token_type_ids": torch.zeros((12, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "learning_rate": 1e-3,
            "encoder_learning_rate": 1e-4,
            "weight_decay": 0.0,
            "loss": "ce",
        },
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }

    result = train_classifier(token_data, config)

    assert result.metrics["learning_rate"] == 1e-3
    assert result.metrics["encoder_learning_rate"] == 1e-4
    assert result.metrics["head_learning_rate"] == 1e-3


def test_train_classifier_supports_staged_encoder_unfreeze() -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 12, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 12, dtype=torch.long),
        "token_type_ids": torch.zeros((12, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "ce",
            "freeze_encoder_epochs": 1,
        },
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }

    result = train_classifier(token_data, config)

    assert result.metrics["freeze_encoder_epochs"] == 1
    assert result.metrics["encoder_frozen_initially"] is True
    assert result.metrics["encoder_frozen_finally"] is False
    assert result.metrics["encoder_unfreeze_epoch"] == 2
    assert result.history[0]["encoder_frozen"] is True
    assert result.history[1]["encoder_frozen"] is False


@pytest.mark.parametrize(
    ("loss_type", "metric_key"),
    [
        ("class_balanced_ce", "cb_beta"),
        ("logit_adjusted_ce", "logit_adjust_tau"),
        ("ldam", "ldam_max_m"),
    ],
)
def test_train_classifier_accepts_long_tail_losses(loss_type: str, metric_key: str) -> None:
    token_data = {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 12, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 12, dtype=torch.long),
        "token_type_ids": torch.zeros((12, 4), dtype=torch.long),
        "binary_labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "labels": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"Benign": 0, "Attack": 1},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4},
        "max_len": 4,
    }
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss_type": loss_type,
            "cb_beta": 0.99,
            "logit_adjust_tau": 0.5,
            "ldam_max_m": 0.3,
            "ldam_scale": 5.0,
        },
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }

    result = train_classifier(token_data, config)

    assert result.metrics["loss"] == loss_type
    assert metric_key in result.metrics
    assert result.predictions
