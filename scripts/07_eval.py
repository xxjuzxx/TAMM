#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation entry point placeholder for the MVP.")
    parser.add_argument("--help_only", action="store_true")
    parser.parse_args()
    print("Use scripts/05_train_classifier.py for MVP evaluation outputs.")


if __name__ == "__main__":
    main()
