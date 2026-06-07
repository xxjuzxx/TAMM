#!/usr/bin/env bash
set -euo pipefail

INPUT=""
OUT_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
CICFLOW_HOME="${CICFLOW_HOME:-$REPO_ROOT/third_party/CICFlowMeter-runtime/CICFlowMeter-4.0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --cicflow_home) CICFLOW_HOME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$INPUT" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 --input PCAP_OR_DIR --out_dir DIR [--cicflow_home DIR]" >&2
  exit 2
fi

if [[ ! -d "$CICFLOW_HOME/lib" || ! -d "$CICFLOW_HOME/lib/native" ]]; then
  echo "CICFlowMeter runtime not found: $CICFLOW_HOME" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
NATIVE_DIR="$CICFLOW_HOME/lib/native"

if [[ ! -e "$NATIVE_DIR/libpcap.so" && -e /usr/lib/x86_64-linux-gnu/libpcap.so.0.8 ]]; then
  ln -s /usr/lib/x86_64-linux-gnu/libpcap.so.0.8 "$NATIVE_DIR/libpcap.so"
fi

CLASSPATH=""
for jar in "$CICFLOW_HOME"/lib/*.jar; do
  if [[ -z "$CLASSPATH" ]]; then
    CLASSPATH="$jar"
  else
    CLASSPATH="$CLASSPATH:$jar"
  fi
done

export LD_LIBRARY_PATH="$NATIVE_DIR:${LD_LIBRARY_PATH:-}"
exec java -Djava.library.path="$NATIVE_DIR" -cp "$CLASSPATH" cic.cs.unb.ca.ifm.Cmd "$INPUT" "$OUT_DIR"
