#!/usr/bin/env bash
set -euo pipefail

MODE="quick"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-quick}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--mode quick|full]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "quick" && "$MODE" != "full" ]]; then
  echo "--mode must be quick or full" >&2
  exit 2
fi

python scripts/64_run_icdm_revision_experiments.py --mode "$MODE"
