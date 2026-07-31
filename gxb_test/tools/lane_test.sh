#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="${INTELLIGENT_CAR_WS:-/root/intelligent_car_ws}"
LOG_ROOT="${WORKSPACE}/test_logs"
CURRENT_FILE="${LOG_ROOT}/.lane_test_current"
WEB_PORT=8093
PROCESS_NAMES=(perception controller controller_watcher single_green_watcher)

latest_log_dir() {
  if [[ -f "${CURRENT_FILE}" ]]; then
    local recorded
    recorded="$(head -n 1 "${CURRENT_FILE}")"
    if [[ -d "${recorded}" ]]; then
      printf '%s\n' "${recorded}"
      return 0
    fi
  fi
  find "${LOG_ROOT}" -maxdepth 1 -type d -name 'lane_*' -print 2>/dev/null \
    | sort | tail -n 1
}

source_ros() {
  [[ -f /opt/tros/humble/setup.bash ]] || {
    echo "ERROR: /opt/tros/humble/setup.bash not found" >&2
    return 1
  }
  [[ -f "${WORKSPACE}/install/setup.bash" ]] || {
    echo "ERROR: ${WORKSPACE}/install/setup.bash not found" >&2
    return 1
  }

  # ROS/TROS setup 脚本会读取若干未预先定义的环境变量。
  # 加载期间临时关闭 nounset，加载完成后立即恢复。
  set +u

  # shellcheck disable=SC1091
  if ! source /opt/tros/humble/setup.bash; then
    set -u
    echo "ERROR: failed to source TROS environment" >&2
    return 1
  fi

  # shellcheck disable=SC1090
  if ! source "${WORKSPACE}/install/setup.bash"; then
    set -u
    echo "ERROR: failed to source workspace environment" >&2
    return 1
  fi

  set -u
}

pid_alive() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

known_residuals() {
  command -v pgrep >/dev/null 2>&1 || return 0
  pgrep -af '(^|/)(lane_perception_pipeline\.py|5_lane_path_controller\.py|watch_controller_v2\.py|watch_single_green_status\.py)( |$)' || true
}

show_load() {
  if command -v uptime >/dev/null 2>&1; then
    uptime
  elif [[ -r /proc/loadavg ]]; then
    echo "load_average=$(cat /proc/loadavg)"
  else
    echo "load_average=unavailable"
  fi
}

port_in_use() {
  ss -ltn 2>/dev/null | awk -v port=":${WEB_PORT}" '$4 ~ port "$" {found=1} END {exit !found}'
}

start_one() {
  local name="$1"
  local log_dir="$2"
  shift 2

  # Always load ROS inside the detached shell as well.  `bash +u` makes the
  # child safe even when nounset arrived through SHELLOPTS; source_ros then
  # restores nounset after both setup files have finished loading.
  export WORKSPACE
  export -f source_ros
  setsid bash +u -c '
    set -eo pipefail
    source_ros
    exec "$@"
  ' "lane-test-${name}" "$@" >"${log_dir}/${name}.log" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" >"${log_dir}/${name}.pid"
  echo "started ${name} pid=${pid}"
}

start_all() {
  source_ros
  [[ -e /dev/video0 ]] || {
    echo "ERROR: /dev/video0 not found" >&2
    return 1
  }
  if port_in_use; then
    echo "ERROR: Web port ${WEB_PORT} is already in use" >&2
    return 1
  fi
  local residuals
  residuals="$(known_residuals)"
  if [[ -n "${residuals}" ]]; then
    echo "ERROR: matching lane-test processes already exist; run stop first:" >&2
    printf '%s\n' "${residuals}" >&2
    return 1
  fi
  mkdir -p "${LOG_ROOT}"
  local stamp log_dir
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_dir="${LOG_ROOT}/lane_${stamp}"
  mkdir -p "${log_dir}"
  printf '%s\n' "${log_dir}" >"${CURRENT_FILE}"
  {
    echo "repo_root=${REPO_ROOT}"
    echo "workspace=${WORKSPACE}"
    echo "web_port=${WEB_PORT}"
    echo "perception=python3 gxb_test/lane_perception_pipeline.py"
    echo "controller=python3 gxb_test/5_lane_path_controller.py --ros-args -p dry_run:=true"
    echo "watch_rate_hz=1"
  } >"${log_dir}/startup_parameters.txt"
  git -C "${REPO_ROOT}" rev-parse HEAD >"${log_dir}/git_commit.txt"
  show_load >"${log_dir}/system_load_start.txt"

  start_one perception "${log_dir}" \
    python3 "${REPO_ROOT}/gxb_test/lane_perception_pipeline.py"
  start_one controller "${log_dir}" \
    python3 "${REPO_ROOT}/gxb_test/5_lane_path_controller.py" \
      --ros-args -p dry_run:=true
  start_one controller_watcher "${log_dir}" \
    python3 "${SCRIPT_DIR}/watch_controller_v2.py"
  start_one single_green_watcher "${log_dir}" \
    python3 "${SCRIPT_DIR}/watch_single_green_status.py"
  sleep 1
  for name in "${PROCESS_NAMES[@]}"; do
    local pid
    pid="$(cat "${log_dir}/${name}.pid")"
    if ! pid_alive "${pid}"; then
      echo "ERROR: ${name} exited during startup; see ${log_dir}/${name}.log" >&2
      stop_all
      return 1
    fi
  done
  echo "Web: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${WEB_PORT}/"
  echo "Logs: ${log_dir}"
  echo "Safety: controller is dry_run=true; no vehicle-speed publisher is started."
}

stop_all() {
  local log_dir
  log_dir="$(latest_log_dir || true)"
  if [[ -n "${log_dir}" ]]; then
    for name in "${PROCESS_NAMES[@]}"; do
      local pid_file="${log_dir}/${name}.pid"
      [[ -f "${pid_file}" ]] || continue
      local pid
      pid="$(cat "${pid_file}")"
      if pid_alive "${pid}"; then
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
      fi
    done
    for _attempt in 1 2 3 4 5; do
      local any_alive=0
      for name in "${PROCESS_NAMES[@]}"; do
        [[ -f "${log_dir}/${name}.pid" ]] || continue
        pid_alive "$(cat "${log_dir}/${name}.pid")" && any_alive=1
      done
      [[ ${any_alive} -eq 0 ]] && break
      sleep 1
    done
    for name in "${PROCESS_NAMES[@]}"; do
      [[ -f "${log_dir}/${name}.pid" ]] || continue
      local remaining_pid
      remaining_pid="$(cat "${log_dir}/${name}.pid")"
      if pid_alive "${remaining_pid}"; then
        kill -KILL -- "-${remaining_pid}" 2>/dev/null \
          || kill -KILL "${remaining_pid}" 2>/dev/null \
          || true
      fi
    done
  fi
  local residual pid command
  while IFS= read -r residual; do
    [[ -n "${residual}" ]] || continue
    pid="${residual%% *}"
    command="${residual#* }"
    case "${command}" in
      *lane_perception_pipeline.py*|*5_lane_path_controller.py*|*watch_controller_v2.py*|*watch_single_green_status.py*)
        kill -TERM "${pid}" 2>/dev/null || true
        ;;
    esac
  done < <(known_residuals)
  sleep 1
  while IFS= read -r residual; do
    [[ -n "${residual}" ]] || continue
    pid="${residual%% *}"
    command="${residual#* }"
    case "${command}" in
      *lane_perception_pipeline.py*|*5_lane_path_controller.py*|*watch_controller_v2.py*|*watch_single_green_status.py*)
        kill -KILL "${pid}" 2>/dev/null || true
        ;;
    esac
  done < <(known_residuals)
  echo "lane test processes stopped"
}

topic_publishers() {
  local topic="$1"
  ros2 topic info "${topic}" 2>/dev/null | awk '/Publisher count:/ {print $3}' || echo "unavailable"
}

show_status() {
  local log_dir
  log_dir="$(latest_log_dir || true)"
  echo "log_dir=${log_dir:-none}"
  if [[ -n "${log_dir}" ]]; then
    for name in "${PROCESS_NAMES[@]}"; do
      local pid=""
      [[ -f "${log_dir}/${name}.pid" ]] && pid="$(cat "${log_dir}/${name}.pid")"
      if pid_alive "${pid}"; then
        ps -p "${pid}" -o pid=,stat=,%cpu=,%mem=,cmd=
      else
        echo "${name}: pid=${pid:-none} stopped"
      fi
    done
  fi
  if command -v ros2 >/dev/null 2>&1; then
    echo "publishers centerline_path=$(topic_publishers /gxb_test/pipeline/centerline_path)"
    echo "publishers pipeline_status=$(topic_publishers /gxb_test/pipeline/status)"
    echo "publishers controller_status=$(topic_publishers /gxb_test/controller/status)"
  else
    echo "ROS topic publisher counts: unavailable (ros2 not sourced)"
  fi
  echo "web_port_${WEB_PORT}=$([[ $(port_in_use; echo $?) -eq 0 ]] && echo listening || echo closed)"
  show_load
}

show_logs() {
  local log_dir
  log_dir="$(latest_log_dir || true)"
  [[ -n "${log_dir}" ]] || { echo "No lane test logs found"; return 1; }
  echo "log_dir=${log_dir}"
  echo "perception summary:"
  grep -E 'centerline|single_green|DP|mode=|ERROR|Traceback' "${log_dir}/perception.log" 2>/dev/null | tail -n 30 || true
  echo "controller READY/BLOCKED counts:"
  printf 'READY=%s BLOCKED=%s\n' \
    "$(grep -c 'CONTROLLER READY' "${log_dir}/controller_watcher.log" 2>/dev/null || true)" \
    "$(grep -c 'CONTROLLER BLOCKED' "${log_dir}/controller_watcher.log" 2>/dev/null || true)"
  echo "mode counts:"
  grep -Eho 'mode=[^ ]+' "${log_dir}/controller_watcher.log" "${log_dir}/single_green_watcher.log" 2>/dev/null | sort | uniq -c || true
  echo "single-green/DP tail:"
  tail -n 30 "${log_dir}/single_green_watcher.log" 2>/dev/null || true
  echo "exceptions:"
  grep -RniE 'ERROR|Exception|Traceback|FATAL' "${log_dir}" --include='*.log' 2>/dev/null | tail -n 30 || true
}

tail_logs() {
  local log_dir
  log_dir="$(latest_log_dir || true)"
  [[ -n "${log_dir}" ]] || { echo "No lane test logs found"; return 1; }
  echo "Following logs in ${log_dir} (Ctrl-C to stop tail only)"
  tail -n 20 -F \
    "${log_dir}/perception.log" \
    "${log_dir}/controller.log" \
    "${log_dir}/controller_watcher.log" \
    "${log_dir}/single_green_watcher.log"
}

pack_logs() {
  local log_dir stamp archive
  log_dir="$(latest_log_dir || true)"
  [[ -n "${log_dir}" ]] || { echo "No lane test logs found"; return 1; }
  stamp="$(date +%Y%m%d_%H%M%S)"
  archive="${LOG_ROOT}/lane_test_${stamp}.tar.gz"
  show_status >"${log_dir}/status.txt"
  show_load >"${log_dir}/system_load_pack.txt"
  tar -C "${log_dir}" -czf "${archive}" \
    perception.log controller.log controller_watcher.log \
    single_green_watcher.log status.txt git_commit.txt \
    startup_parameters.txt system_load_start.txt system_load_pack.txt
  echo "archive=${archive}"
}

usage() {
  echo "Usage: bash gxb_test/tools/lane_test.sh {start|stop|status|logs|tail|pack}"
}

command_name="${1:-}"
case "${command_name}" in
  start) start_all ;;
  stop) stop_all ;;
  status) source_ros 2>/dev/null || true; show_status ;;
  logs) show_logs ;;
  tail) tail_logs ;;
  pack) source_ros 2>/dev/null || true; pack_logs ;;
  *) usage; exit 2 ;;
esac
