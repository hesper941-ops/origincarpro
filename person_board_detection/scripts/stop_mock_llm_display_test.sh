#!/usr/bin/env bash
set -euo pipefail

PID_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid-file) PID_FILE="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PID_FILE" ]]; then
  latest_root="/root/intelligent_car_ws/test_logs/person_board_stage2"
  PID_FILE="$(find "$latest_root" -mindepth 2 -maxdepth 2 -name test_processes.pid -type f 2>/dev/null | sort | tail -n 1)"
fi
if [[ -z "$PID_FILE" || ! -f "$PID_FILE" ]]; then
  echo "No stage-2 PID file found" >&2
  exit 1
fi

mapfile -t records < "$PID_FILE"
validated_groups=()
for ((index=${#records[@]}-1; index>=0; index--)); do
  read -r pid expected_start label <<< "${records[$index]}"
  if [[ "$label" == "keep_camera" ]]; then
    echo "Keeping camera pid=$pid by request"
    continue
  fi
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/stat" ]] || continue
  actual_start="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
  if [[ "$actual_start" != "$expected_start" ]]; then
    echo "Skip reused PID $pid ($label)" >&2
    continue
  fi
  validated_groups+=("$pid $label")
  echo "Stopping $label pid=$pid"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
done
sleep 1
for record in "${validated_groups[@]}"; do
  read -r pid label <<< "$record"
  echo "Force stopping $label pid=$pid"
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
done
