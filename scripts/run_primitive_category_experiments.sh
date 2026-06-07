#!/usr/bin/env bash
set -euo pipefail

MODE="quick"
OUTPUT="results/primitive_categories"
TOKEN_DIR="paper_icdm_applied_2026/experiments/unknown/tokens_category"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --token-dir)
      TOKEN_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."

if [[ "$MODE" == "quick" ]]; then
  python scripts/run_primitive_category_experiments.py \
    --output "$OUTPUT" \
    --token-dir "$TOKEN_DIR" \
    --attacks Botnet DDoS \
    --seeds 43 \
    --feature_views \
      profile_only \
      structural_only \
      profile_plus_structural \
      packet_burst_only \
      packet_burst_plus_structural \
    --transform binary_l2 \
    --scorer knn_cosine \
    --k 3 \
    --min_support 5
elif [[ "$MODE" == "full" ]]; then
  python scripts/run_primitive_category_experiments.py \
    --output "$OUTPUT" \
    --token-dir "$TOKEN_DIR" \
    --attacks Botnet DDoS Probe WebAttack BruteForce \
    --seeds 42 43 44 \
    --transform binary_l2 \
    --scorer knn_cosine \
    --k 3 \
    --min_support 5
else
  echo "Unsupported mode: $MODE" >&2
  exit 2
fi

python scripts/summarize_primitive_category_results.py --input "$OUTPUT" --output "$OUTPUT/primitive_category_summary.md"
