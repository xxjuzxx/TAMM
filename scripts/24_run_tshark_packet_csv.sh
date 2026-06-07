#!/usr/bin/env bash
set -euo pipefail

INPUT=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$INPUT" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 --input PCAP_OR_DIR --out_dir DIR" >&2
  exit 2
fi

if ! command -v tshark >/dev/null 2>&1; then
  echo "tshark is not installed or not on PATH" >&2
  exit 127
fi

mkdir -p "$OUT_DIR"

if [[ -d "$INPUT" ]]; then
  mapfile -t PCAPS < <(find -L "$INPUT" -type f \( -name '*.pcap' -o -name '*.pcapng' \) | sort)
else
  PCAPS=("$INPUT")
fi

if [[ "${#PCAPS[@]}" -eq 0 ]]; then
  echo "No pcap/pcapng files found under $INPUT" >&2
  exit 2
fi

for pcap in "${PCAPS[@]}"; do
  label="$(basename "$(dirname "$pcap")")"
  stem="$(basename "${pcap%.*}")"
  if [[ "$label" == "." || "$label" == "/" || "$label" == "$(basename "$INPUT")" && ! -d "$INPUT" ]]; then
    out_path="$OUT_DIR/${stem}.csv"
  else
    mkdir -p "$OUT_DIR/$label"
    out_path="$OUT_DIR/$label/${stem}.csv"
  fi
  tshark -r "$pcap" \
    -T fields \
    -E header=y \
    -E separator=, \
    -E quote=d \
    -e frame.time_epoch \
    -e frame.len \
    -e ip.src \
    -e ip.dst \
    -e ipv6.src \
    -e ipv6.dst \
    -e ip.ttl \
    -e ipv6.hlim \
    -e tcp.srcport \
    -e tcp.dstport \
    -e tcp.flags.urg \
    -e tcp.flags.ack \
    -e tcp.flags.push \
    -e tcp.flags.reset \
    -e tcp.flags.syn \
    -e tcp.flags.fin \
    -e udp.srcport \
    -e udp.dstport \
    > "$out_path"
  echo "$out_path"
done
