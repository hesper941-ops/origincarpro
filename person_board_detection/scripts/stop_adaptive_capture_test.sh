#!/usr/bin/env bash

set -u
set -o pipefail

LOG_ROOT="/root/intelligent_car_ws/test_logs/person_board"
PID_FILE=""

usage() {
    printf '用法：bash %s [--pid-file /path/to/test_processes.pid]\n' "$0"
}

while (($#)); do
    case "$1" in
        --pid-file)
            (($# >= 2)) || { usage; exit 2; }
            PID_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf '未知参数：%s\n' "$1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${PID_FILE}" ]]; then
    PID_FILE=$(find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -name test_processes.pid \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
fi

[[ -n "${PID_FILE}" && -r "${PID_FILE}" ]] || {
    printf '未找到可读的测试 PID 文件。\n' >&2
    exit 1
}

printf '仅清理 PID 文件记录的测试进程：%s\n' "${PID_FILE}"
while IFS='|' read -r role pid expected_start; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || continue
    [[ -n "${expected_start}" && -r "/proc/${pid}/stat" ]] || continue
    actual_start=$(awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null || true)
    if [[ "${actual_start}" != "${expected_start}" ]]; then
        printf '跳过 %-18s PID=%s（PID 已被复用或进程已结束）\n' "${role}" "${pid}"
        continue
    fi
    printf '停止 %-18s PID/PGID=%s\n' "${role}" "${pid}"
    kill -TERM -- "-${pid}" 2>/dev/null || true
done <"${PID_FILE}"

printf '清理命令完成；未使用宽泛 pkill。\n'
