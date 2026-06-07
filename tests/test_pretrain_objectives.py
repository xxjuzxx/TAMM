from __future__ import annotations

import numpy as np
import torch

from src.models.behavior_composer import BehaviorComposer
from src.training.pretrain_trainer import (
    BehaviorComposerPretrainer,
    _build_context_positive_map,
    _flow_view,
    _mask_inputs,
    _nt_xent_loss,
    _paragraph_pair_views,
    _segment_views,
)


def _vocab() -> dict[str, int]:
    return {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "A": 4, "B": 5, "C": 6, "D": 7}


def test_flow_view_keeps_shape_and_special_tokens() -> None:
    vocab = _vocab()
    input_ids = torch.tensor([[1, 4, 5, 6, 2, 0], [1, 5, 6, 7, 2, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long)

    view_ids, view_mask = _flow_view(
        input_ids,
        attention_mask,
        vocab=vocab,
        token_drop_prob=0.5,
        span_drop_prob=0.2,
        span_drop_length=2,
        mask_prob=0.5,
    )

    assert view_ids.shape == input_ids.shape
    assert view_mask.shape == attention_mask.shape
    assert torch.equal(view_ids[:, 0], input_ids[:, 0])
    assert torch.equal(view_ids[:, 4], input_ids[:, 4])
    assert torch.equal(view_mask[:, 0], attention_mask[:, 0])


def test_span_mask_inputs_returns_finite_labels() -> None:
    vocab = _vocab()
    input_ids = torch.tensor([[1, 4, 5, 6, 2, 0], [1, 5, 6, 7, 2, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long)
    special_ids = {vocab[token] for token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]}

    corrupted, labels, replaced = _mask_inputs(
        input_ids,
        attention_mask,
        vocab_size=len(vocab),
        mask_id=vocab["[MASK]"],
        special_ids=special_ids,
        mlm_probability=0.5,
        replace_probability=0.1,
        mask_strategy="span",
        span_length=2,
    )

    assert corrupted.shape == input_ids.shape
    assert labels.shape == input_ids.shape
    assert replaced.shape == input_ids.shape
    assert torch.all(labels[:, 0] == -100)
    assert torch.all(labels[:, 4] == -100)


def test_nt_xent_loss_is_finite_for_batch_and_zero_for_singleton() -> None:
    z_a = torch.randn(4, 8)
    z_b = torch.randn(4, 8)
    loss = _nt_xent_loss(z_a, z_b, temperature=0.2)
    assert torch.isfinite(loss)
    singleton = _nt_xent_loss(z_a[:1], z_b[:1], temperature=0.2)
    assert torch.isfinite(singleton)
    assert singleton.item() == 0.0


def test_segment_views_split_flow_tokens_without_dropping_specials() -> None:
    vocab = _vocab()
    input_ids = torch.tensor([[1, 4, 5, 6, 7, 2, 0], [1, 5, 2, 0, 0, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

    view_a, mask_a, view_b, mask_b = _segment_views(
        input_ids,
        attention_mask,
        vocab=vocab,
        min_segment_tokens=2,
    )

    assert view_a.shape == input_ids.shape
    assert view_b.shape == input_ids.shape
    assert torch.equal(view_a[:, 0], input_ids[:, 0])
    assert torch.equal(view_b[:, 0], input_ids[:, 0])
    assert torch.equal(view_a[:, 5], input_ids[:, 5])
    assert torch.equal(view_b[:, 5], input_ids[:, 5])
    assert mask_a[0, 1:5].sum().item() == 2
    assert mask_b[0, 1:5].sum().item() == 2
    assert mask_a[1, 1].item() == 1
    assert mask_b[1, 1].item() == 1


def test_pretrainer_flow_contrastive_embedding_shape() -> None:
    encoder = BehaviorComposer(
        vocab_size=len(_vocab()),
        num_classes=2,
        max_seq_len=6,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        dropout=0.0,
        pooling_strategy="mean",
    )
    model = BehaviorComposerPretrainer(encoder, hidden_size=16, vocab_size=len(_vocab()))
    input_ids = torch.tensor([[1, 4, 5, 6, 2, 0], [1, 5, 6, 7, 2, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids)

    embeddings = model.contrastive_embedding(input_ids, attention_mask, token_type_ids)

    assert embeddings.shape == (2, 16)


def test_build_context_positive_map_uses_ordered_neighbors_without_leakage() -> None:
    token_data = {
        "meta": [
            {"flow_id": "a", "start_ts": 1.0, "service_key": ["1.1.1.1", "80", "TCP"]},
            {"flow_id": "b", "start_ts": 2.0, "service_key": ["1.1.1.1", "80", "TCP"]},
            {"flow_id": "c", "start_ts": 3.0, "service_key": ["1.1.1.1", "80", "TCP"]},
            {"flow_id": "d", "start_ts": 1.5, "service_key": ["2.2.2.2", "443", "TCP"]},
            {"flow_id": "e", "start_ts": 2.5, "service_key": ["2.2.2.2", "443", "TCP"]},
            {"flow_id": "f", "start_ts": 3.5, "service_key": ["2.2.2.2", "443", "TCP"]},
        ]
    }

    positive_map, stats = _build_context_positive_map(
        token_data,
        group_by="service_host_proto",
        positive_mode="next",
        indices=np.array([0, 1, 2, 3, 4, 5]),
    )

    assert positive_map[0] == 1
    assert positive_map[1] == 2
    assert positive_map[2] == 1
    assert positive_map[3] == 4
    assert positive_map[4] == 5
    assert positive_map[5] == 4
    assert stats["num_context_groups"] == 2
    assert stats["num_context_flows_with_positive"] == 6


def test_paragraph_pair_views_keep_separator_and_shapes() -> None:
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[FLOW_SEP]": 4, "A": 5, "B": 6, "C": 7}
    input_ids = torch.tensor([[1, 5, 6, 4, 7, 2, 0], [1, 6, 4, 5, 7, 2, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.long)

    view_a, mask_a, view_b, mask_b = _paragraph_pair_views(
        input_ids,
        attention_mask,
        vocab=vocab,
        paragraph_sep_id=vocab["[FLOW_SEP]"],
        min_tokens_per_view=2,
    )

    assert view_a.shape == input_ids.shape
    assert view_b.shape == input_ids.shape
    assert mask_a.shape == attention_mask.shape
    assert mask_b.shape == attention_mask.shape
    assert torch.equal(view_a[:, 0], input_ids[:, 0])
    assert torch.equal(view_b[:, 0], input_ids[:, 0])
