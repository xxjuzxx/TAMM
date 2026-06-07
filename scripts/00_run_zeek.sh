#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

PCAP_INPUT=""
OUT_DIR=""
ZEEK_SCRIPT=""
IGNORE_CHECKSUMS=0
CONTINUE_ON_ERROR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pcap_dir|--input) PCAP_INPUT="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --zeek_script) ZEEK_SCRIPT="$2"; shift 2 ;;
    --ignore_checksums) IGNORE_CHECKSUMS=1; shift ;;
    --continue_on_error) CONTINUE_ON_ERROR=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOCAL_ZEEK_HOME="${ZEEK_HOME:-$WORKSPACE_ROOT/third_party/zeek_local/opt/zeek}"
LOCAL_ZEEK_LIB_DIR="$WORKSPACE_ROOT/third_party/zeek_local/usr/lib/x86_64-linux-gnu"

if [[ -x "$LOCAL_ZEEK_HOME/bin/zeek" ]]; then
  export PATH="$LOCAL_ZEEK_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$LOCAL_ZEEK_LIB_DIR:$LOCAL_ZEEK_HOME/lib:${LD_LIBRARY_PATH:-}"
  export ZEEKPATH="${ZEEKPATH:-$LOCAL_ZEEK_HOME/share/zeek:$LOCAL_ZEEK_HOME/share/zeek/policy:$LOCAL_ZEEK_HOME/share/zeek/site:$LOCAL_ZEEK_HOME/share/zeek/builtin-plugins}"
  export ZEEK_PLUGIN_PATH="${ZEEK_PLUGIN_PATH:-$LOCAL_ZEEK_HOME/lib/zeek/plugins}"
elif ! command -v zeek >/dev/null 2>&1; then
  echo "zeek is not installed or not on PATH, and local Zeek was not found at $LOCAL_ZEEK_HOME" >&2
  exit 127
fi

if [[ -z "$PCAP_INPUT" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 --input PCAP_OR_DIR --out_dir DIR [--zeek_script SCRIPT] [--ignore_checksums]" >&2
  exit 2
fi

if [[ -n "$ZEEK_SCRIPT" ]]; then
  ZEEK_SCRIPT="$(readlink -f "$ZEEK_SCRIPT")"
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(readlink -f "$OUT_DIR")"

ZEEK_ARGS=()
if [[ "$IGNORE_CHECKSUMS" -eq 1 ]]; then
  ZEEK_ARGS+=("-C")
fi

if [[ -f "$PCAP_INPUT" ]]; then
  PCAPS=("$PCAP_INPUT")
else
  mapfile -d '' PCAPS < <(
    find -L "$PCAP_INPUT" -type f \
      ! -name '.flowprim_extract_complete.json' \
      ! -path '*/logs/*' \
      ! -path '*/__MACOSX/*' \
      -print0 \
      | sort -z
  )
fi

input_root="$(readlink -f "$PCAP_INPUT")"
for pcap_item in "${PCAPS[@]}"; do
  pcap_item_abs="$(readlink -f "$pcap_item")"
  pcap="$pcap_item_abs"
  if command -v file >/dev/null 2>&1; then
    if ! file -b "$pcap" | grep -Eqi 'pcap(ng)? capture file'; then
      echo "Skipping non-pcap file: $pcap"
      continue
    fi
  fi
  if [[ -d "$PCAP_INPUT" ]]; then
    rel="${pcap_item#"$PCAP_INPUT"/}"
    base="$(basename "$rel")"
    dir="$(dirname "$rel")"
    if [[ "$base" == *.pcap || "$base" == *.pcapng || "$base" == *.cap ]]; then
      base="${base%.*}"
    fi
    if [[ "$dir" == "." ]]; then
      name="$base"
    else
      name="$dir/$base"
    fi
  else
    base="$(basename "$pcap")"
    if [[ "$base" == *.pcap || "$base" == *.pcapng || "$base" == *.cap ]]; then
      base="${base%.*}"
    fi
    name="$base"
  fi
  run_dir="$OUT_DIR/$name"
  mkdir -p "$run_dir"
  echo "Running Zeek: $pcap -> $run_dir"
  pushd "$run_dir" >/dev/null
  if [[ -n "$ZEEK_SCRIPT" ]]; then
    set +e
    zeek "${ZEEK_ARGS[@]}" -r "$pcap" "$ZEEK_SCRIPT"
    status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      echo "Zeek failed for $pcap with exit status $status" >&2
      if [[ "$CONTINUE_ON_ERROR" -ne 1 ]]; then
        exit "$status"
      fi
      echo "$pcap" >> "$OUT_DIR/zeek_failed_inputs.txt"
    fi
  else
    set +e
    zeek "${ZEEK_ARGS[@]}" -r "$pcap"
    status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      echo "Zeek failed for $pcap with exit status $status" >&2
      if [[ "$CONTINUE_ON_ERROR" -ne 1 ]]; then
        exit "$status"
      fi
      echo "$pcap" >> "$OUT_DIR/zeek_failed_inputs.txt"
    fi
  fi
  popd >/dev/null
done
