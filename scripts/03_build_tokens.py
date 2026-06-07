#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import torch

from src.features.tokenizer import PrimitiveTrafficTokenizer, Vocabulary, build_token_dataset
from src.utils.io import read_jsonl, read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", required=True)
    parser.add_argument("--profile_primitives", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--config", default="configs/cicids2017.yaml")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--disable_profile_tokens", dest="disable_profile_tokens", action="store_true")
    parser.add_argument("--profile_mode", dest="profile_mode", choices=["none", "packet", "summary", "full"], default=None)
    parser.add_argument("--use_service_context", action="store_true")
    parser.add_argument("--record_service_context", action="store_true")
    parser.add_argument("--use_service_tokens", action="store_true")
    parser.add_argument("--service_context_window_seconds", type=float, default=None)
    parser.add_argument("--use_burst_shape_tokens", action="store_true")
    parser.add_argument("--use_first_k_signature", action="store_true")
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--use_context_profile_tokens", dest="use_context_profile_tokens", action="store_true")
    parser.add_argument("--use_transition_profile_tokens", dest="use_transition_profile_tokens", action="store_true")
    parser.add_argument("--base_vocab", default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config).get("tokenizer", {})
    if args.max_len is not None:
        cfg["max_len"] = args.max_len
    if args.disable_profile_tokens:
        cfg["use_profile_tokens"] = False
    if args.profile_mode is not None:
        cfg["profile_mode"] = args.profile_mode
    if args.use_service_context:
        cfg["use_service_context"] = True
    if args.record_service_context:
        cfg["record_service_context"] = True
    if args.use_service_tokens:
        cfg["use_service_tokens"] = True
    if args.service_context_window_seconds is not None:
        cfg["service_context_window_seconds"] = args.service_context_window_seconds
    if args.use_burst_shape_tokens:
        cfg["use_burst_shape_tokens"] = True
    if args.use_first_k_signature:
        cfg["use_first_k_signature"] = True
    if args.first_k is not None:
        cfg["first_k"] = args.first_k
    if args.use_context_profile_tokens:
        cfg["use_context_profile_tokens"] = True
        cfg["record_service_context"] = True
    if args.use_transition_profile_tokens:
        cfg["use_transition_profile_tokens"] = True
    vocab = None
    if args.base_vocab:
        with open(args.base_vocab, "r", encoding="utf-8") as handle:
            token_to_id = json.load(handle)
        vocab = Vocabulary()
        vocab.token_to_id = {str(key): int(value) for key, value in token_to_id.items()}
    tokenizer = PrimitiveTrafficTokenizer(**cfg, vocab=vocab)
    flows = read_jsonl(args.flows)
    profile_rows = read_jsonl(args.profile_primitives)
    token_data, stats = build_token_dataset(flows, profile_rows, tokenizer)
    torch.save(token_data, args.out)
    write_json(token_data["vocab"], args.vocab)
    stats_out = args.out.rsplit(".", 1)[0] + "_stats.json"
    write_json(stats, stats_out)
    manifest_out = args.out.rsplit(".", 1)[0] + "_manifest.json"
    write_json(
        {
            "flows": args.flows,
            "profile_primitives": args.profile_primitives,
            "out": args.out,
            "vocab": args.vocab,
            "config_path": args.config,
            "tokenizer_config": cfg,
            "base_vocab": args.base_vocab,
            "manifest_out": manifest_out,
            "use_service_context": bool(args.use_service_context),
            "record_service_context": bool(args.record_service_context),
            "use_service_tokens": bool(args.use_service_tokens),
            "service_context_window_seconds": args.service_context_window_seconds,
            "use_burst_shape_tokens": bool(args.use_burst_shape_tokens),
            "use_first_k_signature": bool(args.use_first_k_signature),
            "first_k": args.first_k,
            "use_context_profile_tokens": bool(args.use_context_profile_tokens),
            "use_transition_profile_tokens": bool(args.use_transition_profile_tokens),
            "raw_ip_used_as_token": False,
            "absolute_time_used_as_token": False,
            "five_tuple_used_as_token": False,
        },
        manifest_out,
    )
    print(stats)


if __name__ == "__main__":
    main()
