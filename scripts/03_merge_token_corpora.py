#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.data.token_corpus import TokenCorpusSource, infer_source_name, merge_token_corpora
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_rows_per_source", type=int, default=None)
    parser.add_argument("--source_names", nargs="*", default=None)
    parser.add_argument(
        "--preserve_first_vocab",
        action="store_true",
        help="Keep the first input vocabulary ids fixed and append only tokens unseen in that base vocab.",
    )
    args = parser.parse_args()

    sources = []
    for idx, path in enumerate(args.inputs):
        corpus = torch.load(path, map_location="cpu", weights_only=False)
        name = args.source_names[idx] if args.source_names is not None and idx < len(args.source_names) else infer_source_name(path)
        sources.append(TokenCorpusSource(path=str(path), name=str(name), corpus=corpus))

    merged, stats = merge_token_corpora(
        sources,
        target_max_len=args.max_len,
        max_rows_per_source=args.max_rows_per_source,
        source_names=args.source_names,
        preserve_first_vocab=args.preserve_first_vocab,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)

    vocab_path = Path(args.vocab) if args.vocab is not None else out_path.with_suffix("").with_name(out_path.stem + "_vocab.json")
    stats_path = Path(args.stats) if args.stats is not None else out_path.with_suffix("").with_name(out_path.stem + "_stats.json")
    manifest_path = Path(args.manifest) if args.manifest is not None else out_path.with_suffix("").with_name(out_path.stem + "_manifest.json")
    write_json(merged["vocab"], vocab_path)
    write_json(stats, stats_path)
    write_json(
        {
            "command": shlex.join(sys.argv),
            "script": Path(__file__).name,
            "inputs": args.inputs,
            "source_names": args.source_names,
            "out": str(out_path),
            "vocab": str(vocab_path),
            "stats": str(stats_path),
            "manifest": str(manifest_path),
            "max_len": args.max_len,
            "max_rows_per_source": args.max_rows_per_source,
            "preserve_first_vocab": bool(args.preserve_first_vocab),
            "num_sources": stats["num_sources"],
            "num_rows": stats["num_rows"],
            "vocab_size": stats["vocab_size"],
        },
        manifest_path,
    )
    print(stats)


if __name__ == "__main__":
    main()
