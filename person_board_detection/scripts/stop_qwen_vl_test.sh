#!/usr/bin/env bash
set -euo pipefail
pkill -TERM -f 'person_board_qwen_vl_worker' 2>/dev/null || true
pkill -TERM -f 'fake_qwen_vl_server.py' 2>/dev/null || true
pkill -TERM -f 'person_board_display' 2>/dev/null || true
sleep 1
