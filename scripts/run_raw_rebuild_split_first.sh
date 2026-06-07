#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${MODE}" == "quick" ]]; then
  SEEDS=(43)
  ATTACKS=(Botnet DDoS Probe WebAttack BruteForce)
  TRAIN_CAP=3000
  VAL_CAP=600
  TEST_BENIGN_CAP=1000
  ATTACK_CAP=5000
elif [[ "${MODE}" == "full" ]]; then
  SEEDS=(42 43 44)
  ATTACKS=(Botnet DDoS Probe WebAttack BruteForce)
  TRAIN_CAP=5000
  VAL_CAP=1000
  TEST_BENIGN_CAP=2000
  ATTACK_CAP=10000
else
  echo "Usage: bash scripts/run_raw_rebuild_split_first.sh [quick|full] [extra args for primitive category runner]" >&2
  exit 2
fi

python scripts/18_make_raw_rebuild_splits.py \
  --seeds "${SEEDS[@]}" \
  --attacks "${ATTACKS[@]}" \
  --train-benign-cap "${TRAIN_CAP}" \
  --val-benign-cap "${VAL_CAP}" \
  --test-benign-cap "${TEST_BENIGN_CAP}" \
  --attack-test-cap "${ATTACK_CAP}"

python scripts/19_build_raw_rebuild_token_corpora.py \
  --seeds "${SEEDS[@]}" \
  --attacks "${ATTACKS[@]}" \
  --no-write-selected-flows

python scripts/run_primitive_category_experiments.py \
  --output results/raw_rebuild_split_first/primitive_categories \
  --token-dir paper_icdm_applied_2026/experiments/raw_rebuild/unknown/tokens_category \
  --attacks "${ATTACKS[@]}" \
  --seeds "${SEEDS[@]}" \
  --transform binary_l2 \
  --scorer knn_cosine \
  --k 3 \
  --min_support 5 \
  "$@"

python scripts/20_summarize_raw_rebuild_split_first.py
