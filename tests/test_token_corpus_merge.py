from __future__ import annotations

import torch

from src.data.token_corpus import TokenCorpusSource, infer_source_name, merge_token_corpora


def _corpus_a() -> dict:
    return {
        "input_ids": torch.tensor([[1, 5, 6, 2], [1, 6, 0, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((2, 4), dtype=torch.long),
        "meta": [
            {"flow_id": "a1", "label": "L1", "binary_label": "ATTACK", "start_ts": 1.0},
            {"flow_id": "a2", "label": "L1", "binary_label": "ATTACK", "start_ts": 2.0, "service_key": ["host", "80", "TCP"]},
        ],
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4, "A": 5, "B": 6},
        "max_len": 4,
        "profile_mode": "packet",
        "use_service_context": False,
        "record_service_context": False,
    }


def _corpus_b() -> dict:
    return {
        "input_ids": torch.tensor([[1, 5, 6, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "token_type_ids": torch.zeros((1, 4), dtype=torch.long),
        "meta": [
            {"flow_id": "b1", "label": "L2", "binary_label": "BENIGN", "start_ts": 3.0, "dataset_file": "b.csv"},
        ],
        "vocab": {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4, "B": 5, "C": 6},
        "max_len": 4,
        "profile_mode": "packet",
        "use_service_context": True,
        "record_service_context": True,
    }


def test_infer_source_name_strips_token_suffix() -> None:
    assert infer_source_name("crossnetA_tokens_500_packet_profile_rerun_20260517.pt") == "crossnetA"
    assert infer_source_name("cicids2017_tokens_2k_packet_profile.pt") == "cicids2017"


def test_merge_token_corpora_unifies_vocab_and_meta() -> None:
    merged, stats = merge_token_corpora(
        [TokenCorpusSource(path="a.pt", name="alpha", corpus=_corpus_a()), TokenCorpusSource(path="b.pt", name="beta", corpus=_corpus_b())],
        target_max_len=3,
    )

    inverse = {idx: token for token, idx in merged["vocab"].items()}
    assert merged["input_ids"].shape == (3, 3)
    assert merged["attention_mask"].shape == (3, 3)
    assert [inverse[int(idx)] for idx in merged["input_ids"][0].tolist()] == ["[CLS]", "A", "B"]
    assert merged["meta"][0]["source_dataset"] == "alpha"
    assert merged["meta"][0]["service_key"] == ["SOURCE", "alpha", "GLOBAL"]
    assert merged["meta"][1]["service_key"] == ["host", "80", "TCP"]
    assert merged["meta"][2]["source_dataset"] == "beta"
    assert merged["meta"][2]["dataset_file"] == "b.csv"
    assert merged["meta"][2]["merged_truncated"] is True
    assert stats["num_sources"] == 2
    assert stats["num_rows"] == 3
    assert stats["vocab_size"] >= 8
    assert stats["truncated_rows_total"] == 2


def test_merge_token_corpora_rejects_source_name_mismatch() -> None:
    try:
        merge_token_corpora(
            [TokenCorpusSource(path="a.pt", name="alpha", corpus=_corpus_a())],
            source_names=["alpha", "beta"],
        )
    except ValueError as exc:
        assert "source_names length" in str(exc)
    else:
        raise AssertionError("expected source_names length mismatch to fail")


def test_merge_token_corpora_can_preserve_first_vocab_ids() -> None:
    merged, stats = merge_token_corpora(
        [TokenCorpusSource(path="a.pt", name="alpha", corpus=_corpus_a()), TokenCorpusSource(path="b.pt", name="beta", corpus=_corpus_b())],
        target_max_len=4,
        preserve_first_vocab=True,
    )

    assert merged["vocab"]["A"] == _corpus_a()["vocab"]["A"]
    assert merged["vocab"]["B"] == _corpus_a()["vocab"]["B"]
    assert merged["vocab"]["C"] >= len(_corpus_a()["vocab"])
    assert stats["preserve_first_vocab"] is True
