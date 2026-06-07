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

python scripts/prepare_extra_benign.py \
  --source-type pcap \
  --output-dir artifacts/extra_benign

python scripts/gate_extra_benign.py \
  --attack Botnet \
  --seed 43 \
  --gate-policy below_p99_5 \
  --output-dir results

python scripts/split_extra_benign.py \
  --split-mode temporal \
  --output-dir splits \
  --results-dir results

if [[ "$MODE" == "quick" ]]; then
  python scripts/measure_extra_benign_zeek_throughput.py
  python experiments/run_extra_benign_attribution.py --mode quick
  python experiments/run_extra_benign_calibration_scaling.py --mode quick
  python experiments/run_extra_benign_memory_strategies.py --mode quick
  python scripts/build_extra_benign_summary.py --include-throughput
else
  python scripts/measure_extra_benign_zeek_throughput.py
  python experiments/run_extra_benign_attribution.py --mode full
  python experiments/run_extra_benign_calibration_scaling.py --mode full
  python experiments/run_extra_benign_memory_strategies.py --mode full
  python scripts/build_extra_benign_memory.py \
    --memory-strategy low_risk_only \
    --max-extra-memory 2000 \
    --output-memory artifacts/memory/knn_memory_old_plus_extra.pkl \
    --output-manifest artifacts/memory/memory_manifest.json
  python scripts/build_extra_benign_summary.py --include-throughput
fi
