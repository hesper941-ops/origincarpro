#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="${INTELLIGENT_CAR_WS:-/root/intelligent_car_ws}"
LOG_ROOT="${WORKSPACE}/test_logs"
CURRENT_FILE="${LOG_ROOT}/.lane_closed_loop_current"
GATE_SCRIPT="${REPO_ROOT}/gxb_test/lane_control_gate.py"
PROCESS_NAMES=(base perception controller gate)

source_ros() {
  set +u
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
  fi
  if [[ -f /opt/tros/humble/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/tros/humble/setup.bash
  fi
  if [[ -f /root/install/ackermann_msgs/share/ackermann_msgs/package.bash ]]; then
    # shellcheck disable=SC1091
    source /root/install/ackermann_msgs/share/ackermann_msgs/package.bash
  fi
  # shellcheck disable=SC1090
  source "${WORKSPACE}/install/local_setup.bash"
  set -u
}

latest_log_dir() {
  if [[ -f "${CURRENT_FILE}" ]]; then
    local path
    path="$(head -n 1 "${CURRENT_FILE}")"
    [[ -d "${path}" ]] && { printf '%s\n' "${path}"; return; }
  fi
  find "${LOG_ROOT}" -maxdepth 1 -type d -name 'lane_closed_loop_*' \
    -print 2>/dev/null | sort | tail -n 1
}

pid_alive() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

pid_matches_label() {
  local pid="$1" label="$2" command_line
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  case "${label}" in
    base) [[ "${command_line}" == *"ros2 launch origincar_base base_serial.launch.py"* || "${command_line}" == *"/origincar_base_node"* ]] ;;
    perception) [[ "${command_line}" == *lane_perception_pipeline.py* ]] ;;
    controller) [[ "${command_line}" == *5_lane_path_controller.py* ]] ;;
    gate) [[ "${command_line}" == *lane_control_gate.py* ]] ;;
    *) return 1 ;;
  esac
}

existing_closed_loop_details() {
  local pattern matches found=false
  for pattern in \
    'origincar_base_node' \
    'lane_perception_pipeline.py' \
    '5_lane_path_controller.py' \
    'lane_control_gate.py'; do
    matches="$(pgrep -af -- "${pattern}" 2>/dev/null || true)"
    if [[ -n "${matches}" ]]; then
      printf '%s\n' "${matches}"
      found=true
    fi
  done
  if fuser -s /dev/ttyACM0 2>/dev/null; then
    echo "serial_owner=/dev/ttyACM0"
    fuser -v /dev/ttyACM0 2>&1 || true
    found=true
  fi
  matches="$(ss -ltnp 2>/dev/null | grep ':8093 ' || true)"
  if [[ -n "${matches}" ]]; then
    echo "web_port_owner=8093"
    printf '%s\n' "${matches}"
    found=true
  fi
  [[ "${found}" == false ]]
}

start_one() {
  local name="$1" log_dir="$2"
  shift 2
  export WORKSPACE ROS_DOMAIN_ID ROS_LOCALHOST_ONLY
  export -f source_ros
  setsid bash +u -c '
    set -eo pipefail
    source_ros
    exec "$@"
  ' "lane-closed-loop-${name}" "$@" >"${log_dir}/${name}.log" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" >"${log_dir}/${name}.pid"
  echo "started ${name} pid=${pid}"
}

topic_count() {
  local topic="$1" label="$2"
  ros2 topic info "${topic}" 2>/dev/null \
    | awk -v label="${label}" '$1 == label && $2 == "count:" {print $3}'
}

send_gate_control() {
  local command="$1"
  timeout 2 ros2 topic pub --once /gxb_test/lane_control_gate/control \
    std_msgs/msg/String "{data: '{\"command\":\"${command}\"}'}" \
    >/dev/null 2>&1 || true
}

zero_burst() {
  local seconds="$1" log_file="$2"
  python3 "${GATE_SCRIPT}" --zero --seconds "${seconds}" \
    --topic /cmd_vel >"${log_file}" 2>&1
}

process_group_alive() {
  local pgid="${1:-}"
  [[ "${pgid}" =~ ^[0-9]+$ ]] || return 1
  pgrep -g "${pgid}" >/dev/null 2>&1
}

process_group_matches_label() {
  local pgid="$1" label="$2" member

  while IFS= read -r member; do
    [[ "${member}" =~ ^[0-9]+$ ]] || continue
    if pid_matches_label "${member}" "${label}"; then
      return 0
    fi
  done < <(pgrep -g "${pgid}" 2>/dev/null || true)

  return 1
}

stop_group() {
  local pid="$1" label="$2" cleanup_log="$3"

  # start_one() 使用 setsid，因此记录 PID 同时也是 PGID。
  # ros2 launch leader 可能先退出，但子节点仍留在原进程组。
  if ! pid_alive "${pid}" && ! process_group_alive "${pid}"; then
    return 0
  fi

  for signal_name in INT TERM KILL; do
    if ! pid_alive "${pid}" && ! process_group_alive "${pid}"; then
      break
    fi

    echo "signal=${signal_name} pid=${pid} label=${label}" \
      >>"${cleanup_log}"

    kill -"${signal_name}" -- "-${pid}" 2>/dev/null \
      || kill -"${signal_name}" "${pid}" 2>/dev/null \
      || true

    [[ "${signal_name}" == KILL ]] || sleep 2
  done
}

stop_registered_processes() {
  local cleanup_log="$1" name pid_file pid seen=" "

  for name in gate controller perception base; do
    while IFS= read -r pid_file; do
      pid="$(head -n 1 "${pid_file}" 2>/dev/null || true)"

      [[ "${pid}" =~ ^[0-9]+$ ]] || continue
      [[ "${seen}" == *" ${pid} "* ]] && continue
      seen+="${pid} "

      if pid_alive "${pid}"; then
        if pid_matches_label "${pid}" "${name}"; then
          stop_group "${pid}" "${name}" "${cleanup_log}"
        else
          echo "skip_pid_mismatch pid=${pid} label=${name} file=${pid_file}" \
            >>"${cleanup_log}"
        fi

      elif process_group_alive "${pid}" \
        && process_group_matches_label "${pid}" "${name}"; then

        # leader 已退出，但我们确认该 PGID 中仍存在同类闭环子进程。
        echo "leader_dead_group_alive pgid=${pid} label=${name}" \
          >>"${cleanup_log}"

        stop_group "${pid}" "${name}" "${cleanup_log}"
      fi

    done < <(
      find "${LOG_ROOT}" -maxdepth 2 -type f -name "${name}.pid" \
        -print 2>/dev/null | sort -r
    )
  done
}

start_all() {
  local allow_motion=false
  [[ "${1:-}" == "--allow-motion" ]] && allow_motion=true
  source_ros
  [[ -e /dev/ttyACM0 ]] || { echo "ERROR: /dev/ttyACM0 missing" >&2; return 1; }
  [[ -e /dev/video0 ]] || { echo "ERROR: /dev/video0 missing" >&2; return 1; }
  mkdir -p "${LOG_ROOT}"
  local existing_details
  if ! existing_details="$(existing_closed_loop_details)"; then
    echo "ERROR: EXISTING CLOSED LOOP INSTANCE DETECTED" >&2
    printf '%s\n' "${existing_details}" >&2
    echo "Run first: bash gxb_test/tools/lane_closed_loop.sh stop" >&2
    return 1
  fi
  local existing_publishers
  existing_publishers="$(topic_count /cmd_vel Publisher || true)"
  existing_publishers="${existing_publishers:-0}"
  if [[ "${existing_publishers}" != 0 ]]; then
    echo "ERROR: /cmd_vel already has ${existing_publishers} publisher(s)" >&2
    return 1
  fi
  local stamp log_dir motion_value
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_dir="${LOG_ROOT}/lane_closed_loop_${stamp}"
  mkdir -p "${log_dir}"
  printf '%s\n' "${log_dir}" >"${CURRENT_FILE}"
  motion_value=false
  [[ "${allow_motion}" == true ]] && motion_value=true
  {
    echo "started_at=$(date --iso-8601=seconds)"
    echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
    echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"
    echo "motion_enabled=${motion_value}"
    echo "max_linear_normal=0.50"
    echo "max_linear_hybrid=0.40"
    echo "max_linear_degraded=0.20"
    echo "max_linear_recovery=0.25"
    echo "max_angular_normal=0.60"
    echo "max_angular_degraded=0.35"
    echo "max_angular_recovery=0.30"
    echo "serial=/dev/ttyACM0"
    echo "baud=921600"
  } >"${log_dir}/startup_parameters.txt"
  python3 -m py_compile \
    "${REPO_ROOT}/gxb_test/lane_perception_pipeline.py" \
    "${REPO_ROOT}/gxb_test/5_lane_path_controller.py" \
    "${GATE_SCRIPT}" >"${log_dir}/static_checks.log" 2>&1
  echo "py_compile=PASS" >>"${log_dir}/static_checks.log"

  start_one base "${log_dir}" \
    ros2 launch origincar_base base_serial.launch.py
  if ! python3 "${GATE_SCRIPT}" --wait-odom --timeout 12 \
    >"${log_dir}/feedback_probe.log" 2>&1; then
    echo "ERROR: base feedback did not become ready" >&2
    stop_all
    return 1
  fi

  start_one perception "${log_dir}" \
    python3 "${REPO_ROOT}/gxb_test/lane_perception_pipeline.py"
  sleep 1
  start_one controller "${log_dir}" \
    python3 "${REPO_ROOT}/gxb_test/5_lane_path_controller.py" \
      --ros-args \
      -p dry_run:=true \
      -p status_timeout_sec:=1.00 \
      -p normal_suggested_linear_x:=0.50 \
      -p degraded_suggested_linear_x:=0.35 \
      -p short_path_suggested_linear_x:=0.20 \
      -p recovery_suggested_linear_x:=0.32
  sleep 1
  start_one gate "${log_dir}" \
    python3 "${GATE_SCRIPT}" --ros-args \
      -p motion_enabled:="${motion_value}" \
      -p telemetry_log_path:="${log_dir}/feedback.log" \
      -p pipeline_timeout_sec:=1.20 \
      -p max_linear_normal:=0.50 \
      -p max_linear_hybrid:=0.40 \
      -p max_linear_degraded:=0.20 \
      -p max_linear_recovery:=0.25 \
      -p max_angular:=0.60 \
      -p max_angular_degraded:=0.35 \
      -p max_angular_recovery:=0.30
  sleep 2
  local name pid
  for name in "${PROCESS_NAMES[@]}"; do
    pid="$(cat "${log_dir}/${name}.pid")"
    if ! pid_alive "${pid}"; then
      echo "ERROR: ${name} exited during startup" >&2
      stop_all
      return 1
    fi
  done
  echo "log_dir=${log_dir}"
  echo "NORMAL MAX V=0.50 m/s"
  echo "HYBRID MAX V=0.40 m/s"
  echo "DEGRADED MAX V=0.20 m/s"
  echo "RECOVERY MAX V=0.25 m/s"
  echo "MAX W=0.60 rad/s (DEGRADED=0.35 RECOVERY=0.30)"
  if [[ "${allow_motion}" == true ]]; then
    echo "MOTION_ARMED=true; gate remains WAIT_READY until four fresh frames"
  else
    echo "MOTION_ARMED=false; zero-output smoke mode"
  fi
}

run_report() {
  local log_dir
  log_dir="$(latest_log_dir || true)"
  [[ -n "${log_dir}" ]] || { echo "ERROR: no run log" >&2; return 1; }
  python3 "${GATE_SCRIPT}" --report \
    --report-log "${log_dir}/feedback.log" --output-dir "${log_dir}"
}

stop_all() {
  source_ros || true
  local log_dir cleanup_log
  log_dir="$(latest_log_dir || true)"
  [[ -n "${log_dir}" ]] || { echo "no active/logged run"; return 0; }
  cleanup_log="${log_dir}/cleanup.log"
  : >"${cleanup_log}"
  send_gate_control stop
  zero_burst 2.0 "${log_dir}/stop_zero.log" || true
  stop_registered_processes "${cleanup_log}"
  sleep 1
  {
    echo "cmd_vel_publishers=$(topic_count /cmd_vel Publisher || echo 0)"
    echo "serial_owner_begin"
    fuser -v /dev/ttyACM0 2>&1 || true
    echo "serial_owner_end"
    echo "residual_begin"
    pgrep -af 'lane_control_gate.py|5_lane_path_controller.py|lane_perception_pipeline.py|origincar_base_node' || true
    echo "residual_end"
  } >>"${cleanup_log}"
  run_report >/dev/null || true
  echo "stopped; log_dir=${log_dir}"
}

estop_all() {
  source_ros
  local log_dir zero_pid gate_pid=""
  log_dir="$(latest_log_dir || true)"
  [[ -n "${log_dir}" ]] || { echo "ERROR: no active run" >&2; return 1; }
  zero_burst 5.0 "${log_dir}/estop_zero.log" &
  zero_pid=$!
  send_gate_control estop
  sleep 0.2
  if [[ -f "${log_dir}/gate.pid" ]]; then
    gate_pid="$(cat "${log_dir}/gate.pid")"
    stop_group "${gate_pid}" gate "${log_dir}/cleanup.log"
  fi
  wait "${zero_pid}"
  echo "ESTOP complete: gate stopped, zero Twist published for 5 seconds"
}

show_status() {
  source_ros || true
  local log_dir name pid
  log_dir="$(latest_log_dir || true)"
  echo "log_dir=${log_dir:-none}"
  if [[ -n "${log_dir}" ]]; then
    for name in "${PROCESS_NAMES[@]}"; do
      pid=""
      [[ -f "${log_dir}/${name}.pid" ]] && pid="$(cat "${log_dir}/${name}.pid")"
      if pid_alive "${pid}"; then
        ps -p "${pid}" -o pid=,stat=,%cpu=,%mem=,cmd=
      else
        echo "${name}: stopped"
      fi
    done
    tail -n 1 "${log_dir}/feedback.log" 2>/dev/null || true
  fi
  echo "cmd_vel_publishers=$(topic_count /cmd_vel Publisher || echo 0)"
  echo "cmd_vel_subscribers=$(topic_count /cmd_vel Subscription || echo 0)"
  fuser -v /dev/ttyACM0 2>&1 || true
}

show_logs() {
  local log_dir
  log_dir="$(latest_log_dir || true)"
  [[ -n "${log_dir}" ]] || { echo "ERROR: no logs" >&2; return 1; }
  echo "log_dir=${log_dir}"
  for file in base perception controller gate cleanup; do
    echo "=== ${file}.log ==="
    tail -n 20 "${log_dir}/${file}.log" 2>/dev/null || true
  done
  [[ -f "${log_dir}/summary.json" ]] && cat "${log_dir}/summary.json"
}

tail_logs() {
  local log_dir
  log_dir="$(latest_log_dir || true)"
  [[ -n "${log_dir}" ]] || { echo "ERROR: no logs" >&2; return 1; }
  tail -n 30 -F "${log_dir}/base.log" "${log_dir}/perception.log" \
    "${log_dir}/controller.log" "${log_dir}/gate.log" \
    "${log_dir}/feedback.log"
}

pack_logs() {
  local log_dir archive
  log_dir="$(latest_log_dir || true)"
  [[ -n "${log_dir}" ]] || { echo "ERROR: no logs" >&2; return 1; }
  run_report >/dev/null || true
  archive="${LOG_ROOT}/$(basename "${log_dir}").tar.gz"
  tar -C "$(dirname "${log_dir}")" -czf "${archive}" "$(basename "${log_dir}")"
  sha256sum "${archive}" >"${archive}.sha256"
  echo "archive=${archive}"
}

usage() {
  cat <<'EOF'
Usage: bash gxb_test/tools/lane_closed_loop.sh COMMAND

Commands:
  start                 Start all nodes in zero-output smoke mode
  start --allow-motion  Explicitly arm closed-loop motion after 4 ready frames
  stop                  Zero for >=2 s, then stop gate/controller/perception/base
  estop                 Independent 5 s zero burst and close the gate immediately
  status                Show processes, latest gate state, /cmd_vel and serial state
  tail                  Follow all main logs
  logs                  Show recent logs and summary
  report                Generate summary.json and summary.csv
  pack                  Generate report and compressed archive
EOF
}

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-86}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

case "${1:-}" in
  start) shift; start_all "${1:-}" ;;
  stop) stop_all ;;
  estop) estop_all ;;
  status) show_status ;;
  tail) tail_logs ;;
  logs) show_logs ;;
  report) run_report ;;
  pack) pack_logs ;;
  *) usage; exit 2 ;;
esac
