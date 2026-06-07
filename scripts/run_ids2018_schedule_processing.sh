#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DAY="${1:-Friday-02-03-2018}"
PCAP_ROOT="${IDS2018_ORGANIZED_ROOT:-data/raw/CSE-CIC-IDS2018_organized}/extracted_pcaps"
PCAP_INPUT="$PCAP_ROOT/$DAY/pcap"
OUT_BASE="$ROOT/outputs/ids2018_schedule_full/$DAY"
INTERIM_BASE="$ROOT/data/interim/flows/ids2018/$DAY"
ZEEK_SCRIPT="${ZEEK_SCRIPT:-$ROOT/../third_party/BSTS-Net/AllFeas.zeek}"

if [[ ! -d "$PCAP_INPUT" ]]; then
  echo "Missing IDS2018 PCAP input directory: $PCAP_INPUT" >&2
  exit 2
fi

cd "$ROOT"
python scripts/build_ids2018_attack_schedule.py

mkdir -p "$OUT_BASE" "$INTERIM_BASE"

bash scripts/00_run_zeek.sh \
  --input "$PCAP_INPUT" \
  --out_dir "$OUT_BASE/zeek" \
  --zeek_script "$ZEEK_SCRIPT" \
  --ignore_checksums \
  --continue_on_error

python scripts/label_ids2018_by_schedule.py \
  --zeek_logs "$OUT_BASE/zeek/*/Features.log" "$OUT_BASE/zeek/*/conn.log" \
  --schedule configs/ids2018_attack_schedule.yaml \
  --day "$DAY" \
  --out "$OUT_BASE/labeled_flows.jsonl" \
  --quarantine_out "$OUT_BASE/quarantine_flows.jsonl" \
  --stats_out "$OUT_BASE/label_stats.json" \
  --label_alignment_report "$OUT_BASE/label_alignment_report.json" \
  --dataset_report "$OUT_BASE/dataset_report.json"

cp "$OUT_BASE/labeled_flows.jsonl" "$INTERIM_BASE/raw_ids2018_${DAY}_schedule_labeled_flows.jsonl"
cp "$OUT_BASE/quarantine_flows.jsonl" "$INTERIM_BASE/raw_ids2018_${DAY}_schedule_quarantine_flows.jsonl"
cp "$OUT_BASE/label_stats.json" "$INTERIM_BASE/raw_ids2018_${DAY}_schedule_label_stats.json"
cp "$OUT_BASE/label_alignment_report.json" "$INTERIM_BASE/raw_ids2018_${DAY}_schedule_label_alignment_report.json"
cp "$OUT_BASE/dataset_report.json" "$INTERIM_BASE/raw_ids2018_${DAY}_schedule_dataset_report.json"

echo "IDS2018 schedule processing finished for $DAY"
echo "Output: $OUT_BASE"
echo "Canonical interim: $INTERIM_BASE"
