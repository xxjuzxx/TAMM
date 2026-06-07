#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import torch

from src.data.leakage_check import assert_no_leakage, build_leakage_report
from src.utils.io import write_json


def _read_json(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True)
    parser.add_argument("--profile_manifest", default=None)
    parser.add_argument("--token_manifest", default=None)
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--tokens", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow_port_tokens", action="store_true")
    args = parser.parse_args()

    split_payload = _read_json(args.splits)
    vocab = _read_json(args.vocab) if args.vocab else None
    token_data = torch.load(args.tokens, map_location="cpu", weights_only=False) if args.tokens else None
    report = build_leakage_report(
        split_payload=split_payload,
        profile_manifest=_read_json(args.profile_manifest),
        token_manifest=_read_json(args.token_manifest),
        vocab=vocab,
        token_data=token_data,
        allow_port_tokens=bool(args.allow_port_tokens),
    )
    write_json(report, args.out)
    assert_no_leakage(report)
    print(report)


if __name__ == "__main__":
    main()
