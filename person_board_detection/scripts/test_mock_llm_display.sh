#!/usr/bin/env bash
set -euo pipefail

MODE="replay"
TIMEOUT_SEC=120
EVENT_ID="person_board_stage2_001"
START_CAMERA="auto"
KEEP_CAMERA=false
NO_BUILD=false
DISPLAY_OVERRIDE=""
CAPTURE_DIR="/root/intelligent_car_ws/runtime/person_board/latest_capture"
LOG_ROOT="/root/intelligent_car_ws/test_logs/person_board_stage2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --event-id) EVENT_ID="${2:-}"; shift 2 ;;
    --start-camera) START_CAMERA="${2:-}"; shift 2 ;;
    --keep-camera) KEEP_CAMERA=true; shift ;;
    --no-build) NO_BUILD=true; shift ;;
    --display) DISPLAY_OVERRIDE="${2:-}"; shift 2 ;;
    --capture-dir) CAPTURE_DIR="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$MODE" == "replay" || "$MODE" == "full" ]] || {
  echo "--mode must be replay or full" >&2; exit 2;
}
[[ "$TIMEOUT_SEC" =~ ^[0-9]+$ && "$TIMEOUT_SEC" -gt 0 ]] || {
  echo "--timeout must be a positive integer" >&2; exit 2;
}
[[ "$START_CAMERA" == "auto" ]] || {
  echo "--start-camera currently only supports auto" >&2; exit 2;
}
[[ "$CAPTURE_DIR" == /* ]] || {
  echo "--capture-dir must be an absolute path" >&2; exit 2;
}

WORKSPACE="${PERSON_BOARD_WORKSPACE:-/root/intelligent_car_ws}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="$(git -C "$PACKAGE_ROOT" rev-parse --show-toplevel)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$STAMP"
PID_FILE="$LOG_DIR/test_processes.pid"
SUMMARY="$LOG_DIR/summary.txt"
mkdir -p "$LOG_DIR"
: > "$PID_FILE"
RESULT="FAIL"
FAILURE_REASON="test_interrupted"
BUILD_RESULT="SKIPPED"
MANIFEST_VALID=false
IMAGES_VALID=false
CAPTURE_BATCH_SENT=false
CLEANED_UP=false

source_ros_environment() {
  local include_workspace="${1:-false}"
  local nounset_was_enabled=0
  local source_rc=0
  case "$-" in
    *u*) nounset_was_enabled=1 ;;
  esac
  set +u
  # shellcheck disable=SC1091
  source /opt/tros/humble/setup.bash || source_rc=$?
  if [[ "$source_rc" -eq 0 && "$include_workspace" == true ]]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE/install/setup.bash" || source_rc=$?
  fi
  if [[ "$nounset_was_enabled" -eq 1 ]]; then
    set -u
  fi
  return "$source_rc"
}

if [[ -f /opt/tros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  if ! source_ros_environment false; then
    echo "Failed to source /opt/tros/humble/setup.bash" >&2
    exit 1
  fi
else
  echo "Missing /opt/tros/humble/setup.bash" >&2
  exit 1
fi
ROS_BASE_SOURCE="set +u; source /opt/tros/humble/setup.bash; set -u"
ROS_WORKSPACE_SOURCE="set +u; source /opt/tros/humble/setup.bash; source '$WORKSPACE/install/setup.bash'; set -u"

record_pid() {
  local pid="$1" label="$2" start
  start="$(awk '{print $22}' "/proc/$pid/stat")"
  printf '%s %s %s\n' "$pid" "$start" "$label" >> "$PID_FILE"
}

start_group() {
  local label="$1" logfile="$2" command="$3"
  setsid bash -lc "$command" > "$logfile" 2>&1 &
  local pid=$!
  record_pid "$pid" "$label"
  echo "$pid"
}

wait_for_node() {
  local node_name="$1" end=$((SECONDS + $2))
  while (( SECONDS < end )); do
    ros2 node list 2>/dev/null | grep -Fxq "$node_name" && return 0
    sleep 0.5
  done
  return 1
}

camera_has_data() {
  timeout 8 ros2 topic hz /aurora/rgb/image_raw --window 2 \
    > "$LOG_DIR/camera_probe.log" 2>&1 || true
  grep -q 'average rate:' "$LOG_DIR/camera_probe.log"
}

stop_own_processes() {
  if [[ -s "$PID_FILE" ]]; then
    bash "$PACKAGE_ROOT/scripts/stop_mock_llm_display_test.sh" \
      --pid-file "$PID_FILE" >/dev/null 2>&1 || true
  fi
}

write_summary() {
  local branch commit display_alive=false worker_alive=false
  branch="$(git -C "$SOURCE_ROOT" branch --show-current 2>/dev/null || true)"
  commit="$(git -C "$SOURCE_ROOT" rev-parse HEAD 2>/dev/null || true)"
  [[ -n "${DISPLAY_PID:-}" ]] && kill -0 "$DISPLAY_PID" 2>/dev/null && display_alive=true
  [[ -n "${WORKER_PID:-}" ]] && kill -0 "$WORKER_PID" 2>/dev/null && worker_alive=true
  cat > "$SUMMARY" <<EOF
git_branch=$branch
git_commit=$commit
test_mode=$MODE
build_result=$BUILD_RESULT
package_discovery=$(ros2 pkg prefix person_board_detection 2>/dev/null || echo FAILED)
DISPLAY=${DISPLAY_OVERRIDE:-${DISPLAY:-auto}}
XAUTHORITY=${XAUTHORITY:-auto-discovery}
display_started=${DISPLAY_PID:+true}
mock_worker_started=${WORKER_PID:+true}
manifest_valid=$MANIFEST_VALID
images_valid=$IMAGES_VALID
capture_batch_sent_or_received=$CAPTURE_BATCH_SENT
validating_seen=$(grep -q VALIDATING "$LOG_DIR/llm_status.log" 2>/dev/null && echo true || echo false)
analyzing_seen=$(grep -q ANALYZING "$LOG_DIR/llm_status.log" 2>/dev/null && echo true || echo false)
llm_result_success=$(grep -Eq '"success"[[:space:]]*:[[:space:]]*true' "$LOG_DIR/llm_result.log" 2>/dev/null && echo true || echo false)
request_id_chain_consistent=$(grep -q "${REQUEST_ID:-not_set}" "$LOG_DIR/llm_result.log" 2>/dev/null && grep -q "${REQUEST_ID:-not_set}" "$LOG_DIR/display_status.log" 2>/dev/null && echo true || echo false)
display_status_text_correct=$(grep -q '检测到人形立牌' "$LOG_DIR/display_status.log" 2>/dev/null && echo true || echo false)
worker_alive_at_end=$worker_alive
display_alive_at_end=$display_alive
result=$RESULT
failure_reason=$FAILURE_REASON
EOF
}

cleanup() {
  if [[ "$CLEANED_UP" == true ]]; then
    return
  fi
  CLEANED_UP=true
  [[ -f "$LOG_DIR/display_launcher.log" ]] && \
    cp "$LOG_DIR/display_launcher.log" "$LOG_DIR/display.log"
  if [[ ! -f "$LOG_DIR/node_list_after.txt" ]]; then
    ros2 node list > "$LOG_DIR/node_list_after.txt" 2>&1 || true
  fi
  write_summary || true
  echo "log_dir=$LOG_DIR"
  echo "summary=$SUMMARY"
  stop_own_processes
}

handle_signal() {
  RESULT="FAIL"
  FAILURE_REASON="test_interrupted"
  exit 130
}

trap cleanup EXIT
trap handle_signal INT TERM

cd "$SOURCE_ROOT"
{
  git status -sb
  git log -1 --oneline --decorate
  git remote -v
} > "$LOG_DIR/git_info.txt"
ros2 node list > "$LOG_DIR/node_list_before.txt" 2>&1 || true

if [[ "$NO_BUILD" == false ]]; then
  cd "$WORKSPACE"
  if colcon build --base-paths "$SOURCE_ROOT" --symlink-install \
      --packages-select person_board_detection \
      --event-handlers console_direct+ > "$LOG_DIR/build.log" 2>&1; then
    BUILD_RESULT="PASS"
  else
    BUILD_RESULT="FAIL"
    FAILURE_REASON="build_failed"
    exit 1
  fi
  cd "$SOURCE_ROOT"
else
  echo "Build skipped by --no-build" > "$LOG_DIR/build.log"
fi

if [[ ! -f "$WORKSPACE/install/setup.bash" ]]; then
  FAILURE_REASON="workspace_setup_missing"
  exit 1
fi
if ! source_ros_environment true; then
  FAILURE_REASON="workspace_setup_source_failed"
  exit 1
fi

if ! ros2 pkg prefix person_board_detection >/dev/null 2>&1; then
  FAILURE_REASON="package_not_discovered"
  exit 1
fi

display_command="cd '$SOURCE_ROOT';"
if [[ -n "$DISPLAY_OVERRIDE" ]]; then
  display_command+=" export DISPLAY='$DISPLAY_OVERRIDE';"
fi
display_command+=" bash person_board_detection/scripts/run_person_board_display.sh"
DISPLAY_PID="$(start_group display "$LOG_DIR/display_launcher.log" "$display_command")"
start_group display_status "$LOG_DIR/display_status.log" \
  "$ROS_WORKSPACE_SOURCE; ros2 topic echo /person_board/display_status std_msgs/msg/String --qos-durability transient_local --full-length" >/dev/null

deadline=$((SECONDS + TIMEOUT_SEC))
while ! grep -Eq 'WAITING|STARTING' "$LOG_DIR/display_status.log" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    FAILURE_REASON="display_start_timeout"; exit 1
  fi
  kill -0 "$DISPLAY_PID" 2>/dev/null || {
    FAILURE_REASON="display_exited"; exit 1;
  }
  sleep 0.2
done

WORKER_PID="$(start_group mock_worker "$LOG_DIR/mock_worker.log" \
  "$ROS_WORKSPACE_SOURCE; ros2 launch person_board_detection person_board_mock_llm.launch.py allowed_capture_directory:='$CAPTURE_DIR'")"
start_group llm_status "$LOG_DIR/llm_status.log" \
  "$ROS_WORKSPACE_SOURCE; ros2 topic echo /person_board/llm_status std_msgs/msg/String --qos-durability transient_local --full-length" >/dev/null
start_group llm_result "$LOG_DIR/llm_result.log" \
  "$ROS_WORKSPACE_SOURCE; ros2 topic echo /person_board/llm_result std_msgs/msg/String --qos-durability transient_local --full-length" >/dev/null
sleep 1

if [[ "$MODE" == "full" ]]; then
  start_group capture_batch "$LOG_DIR/capture_batch.log" \
    "$ROS_WORKSPACE_SOURCE; ros2 topic echo /person_board/capture_batch std_msgs/msg/String --full-length" >/dev/null
  start_group capture_status "$LOG_DIR/capture_status.log" \
    "$ROS_WORKSPACE_SOURCE; ros2 topic echo /person_board/status std_msgs/msg/String --full-length" >/dev/null
  start_group capture_done "$LOG_DIR/capture_done.log" \
    "$ROS_WORKSPACE_SOURCE; ros2 topic echo /person_board/capture_done std_msgs/msg/String --full-length" >/dev/null
  echo "请将人形立牌逐渐靠近相机，等待三张裁剪完成。"

  if camera_has_data; then
    echo "Reusing existing Aurora camera" > "$LOG_DIR/camera.log"
  else
    camera_label="camera"
    [[ "$KEEP_CAMERA" == true ]] && camera_label="keep_camera"
    CAMERA_PID="$(start_group "$camera_label" "$LOG_DIR/camera.log" \
      "$ROS_BASE_SOURCE; ros2 launch deptrum-ros-driver-aurora930 aurora930_launch.py")"
    camera_deadline=$((SECONDS + 30))
    until camera_has_data; do
      if (( SECONDS >= camera_deadline )); then
        FAILURE_REASON="camera_start_timeout"; exit 1
      fi
      kill -0 "$CAMERA_PID" 2>/dev/null || {
        FAILURE_REASON="camera_exited"; exit 1;
      }
      sleep 1
    done
  fi

  capture_parent="$(dirname "$CAPTURE_DIR")"
  capture_name="$(basename "$CAPTURE_DIR")"
  installed_capture_config="$(ros2 pkg prefix person_board_detection)/share/person_board_detection/config/person_board_capture.yaml"
  detector_command="$ROS_WORKSPACE_SOURCE; ros2 run person_board_detection person_board_detector --ros-args --params-file '$installed_capture_config' -p runtime_directory:='$capture_parent' -p fixed_capture_subdirectory:='$capture_name'"
  DETECTOR_PID="$(start_group detector "$LOG_DIR/detector.log" "$detector_command")"
  if ! wait_for_node /person_board_detector 20; then
    FAILURE_REASON="detector_start_timeout"; exit 1
  fi
  control_payload="$(printf '{data: '\''{\"command\":\"start\",\"event_id\":\"%s\"}'\''}' "$EVENT_ID")"
  ros2 topic pub --once /person_board/control std_msgs/msg/String \
    "$control_payload" > "$LOG_DIR/control_start.log" 2>&1

  capture_deadline=$((SECONDS + TIMEOUT_SEC))
  while ! grep -Eq '"success"[[:space:]]*:[[:space:]]*true' \
      "$LOG_DIR/capture_done.log" 2>/dev/null; do
    if (( SECONDS >= capture_deadline )); then
      FAILURE_REASON="capture_timeout"; exit 1
    fi
    sleep 0.5
  done
  grep -q "$EVENT_ID" "$LOG_DIR/capture_batch.log" || {
    FAILURE_REASON="capture_batch_event_id_mismatch"; exit 1;
  }
fi

for required in manifest.json crop_01.jpg crop_02.jpg crop_03.jpg; do
  if [[ ! -s "$CAPTURE_DIR/$required" ]]; then
    FAILURE_REASON="capture_file_missing_$required"; exit 1
  fi
done

python3 - "$CAPTURE_DIR" "$LOG_DIR/capture_batch_sent.json" "$LOG_DIR/manifest_copy.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path

capture_dir = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
manifest_copy = Path(sys.argv[3])
manifest_path = capture_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = [str(capture_dir / f"crop_{index:02d}.jpg") for index in range(1, 4)]
if manifest.get("image_paths") != expected or len(manifest.get("frames", [])) != 3:
    raise SystemExit("manifest protocol mismatch")
payload = {
    "event_id": manifest["event_id"],
    "request_id": manifest["request_id"],
    "manifest_path": str(manifest_path),
    "image_paths": expected,
}
output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
shutil.copy2(manifest_path, manifest_copy)
PY
MANIFEST_VALID=true
IMAGES_VALID=true
REQUEST_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["request_id"])' "$LOG_DIR/capture_batch_sent.json")"

if [[ "$MODE" == "replay" ]]; then
  ros_payload="$(python3 -c 'import json,sys; print(json.dumps({"data":open(sys.argv[1],encoding="utf-8").read()},ensure_ascii=False))' "$LOG_DIR/capture_batch_sent.json")"
  ros2 topic pub --once /person_board/capture_batch std_msgs/msg/String "$ros_payload" >/dev/null
fi
CAPTURE_BATCH_SENT=true
deadline=$((SECONDS + TIMEOUT_SEC))

for required_state in VALIDATING ANALYZING SUCCEEDED; do
  while ! grep -q "$required_state" "$LOG_DIR/llm_status.log" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      FAILURE_REASON="llm_state_timeout_$required_state"; exit 1
    fi
    kill -0 "$WORKER_PID" 2>/dev/null || {
      FAILURE_REASON="worker_exited"; exit 1;
    }
    sleep 0.2
  done
done
while ! grep -Eq '"success"[[:space:]]*:[[:space:]]*true' "$LOG_DIR/llm_result.log" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    FAILURE_REASON="llm_result_timeout"; exit 1
  fi
  sleep 0.2
done
while ! grep -q 'SHOWING_RESULT' "$LOG_DIR/display_status.log" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    FAILURE_REASON="display_result_timeout"; exit 1
  fi
  sleep 0.2
done

grep -q "$REQUEST_ID" "$LOG_DIR/llm_result.log" || {
  FAILURE_REASON="result_request_id_mismatch"; exit 1;
}
grep -q "$REQUEST_ID" "$LOG_DIR/display_status.log" || {
  FAILURE_REASON="display_request_id_mismatch"; exit 1;
}
grep -q '检测到人形立牌' "$LOG_DIR/display_status.log" || {
  FAILURE_REASON="display_text_mismatch"; exit 1;
}
kill -0 "$WORKER_PID" 2>/dev/null || {
  FAILURE_REASON="worker_not_persistent"; exit 1;
}
kill -0 "$DISPLAY_PID" 2>/dev/null || {
  FAILURE_REASON="display_not_persistent"; exit 1;
}
ros2 node list > "$LOG_DIR/node_list_after.txt" 2>&1 || true
RESULT="PASS"
FAILURE_REASON=""
echo "PASS: person-board stage-2 $MODE chain"
