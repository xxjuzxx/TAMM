from __future__ import annotations

import torch

from src.training.fixed_split_classifier import train_fixed_split_classifier


def _token_data() -> dict:
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 0, 1, 2], dtype=torch.long)
    splits = ["train", "train", "train", "train", "train", "train", "val", "test", "test"]
    return {
        "input_ids": torch.tensor([[1, 4, 5, 0]] * len(labels), dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * len(labels), dtype=torch.long),
        "token_type_ids": torch.zeros((len(labels), 4), dtype=torch.long),
        "binary_labels": (labels > 0).long(),
        "labels": labels,
        "binary_label_to_id": {"BENIGN": 0, "ATTACK": 1},
        "label_to_id": {"BENIGN": 0, "Botnet": 1, "Probe": 2},
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4, "X": 5},
        "max_len": 4,
        "meta": [{"flow_id": f"f{i}", "split": split, "label": str(int(label))} for i, (split, label) in enumerate(zip(splits, labels.tolist()))],
    }


def test_fixed_split_classifier_records_balanced_focal_settings() -> None:
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "weighted_focal",
            "sampling": "class_balanced",
            "focal_gamma": 1.5,
        },
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }

    result = train_fixed_split_classifier(_token_data(), config, task="multiclass")

    assert result.metrics["loss"] == "weighted_focal"
    assert result.metrics["sampling"] == "class_balanced"
    assert result.metrics["focal_gamma"] == 1.5
    assert result.predictions


def test_fixed_split_classifier_accepts_logit_adjusted_loss() -> None:
    config = {
        "seed": 7,
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss_type": "logit_adjusted_ce",
            "logit_adjust_tau": 0.7,
        },
        "model": {"hidden_size": 16, "num_layers": 1, "num_heads": 2, "intermediate_size": 32, "dropout": 0.0, "max_seq_len": 4},
    }

    result = train_fixed_split_classifier(_token_data(), config, task="multiclass")

    assert result.metrics["loss"] == "logit_adjusted_ce"
    assert result.metrics["logit_adjust_tau"] == 0.7
