#!/usr/bin/env bash
set -euo pipefail

# Isolate the complete moving test from publishers on the site-wide ROS graph.
export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=1

SPEED=0.10
MAX_DISTANCE=3.0
HARD_TIMEOUT=35
WORKSPACE=/root/intelligent_car_ws
CAPTURE_DIR="$WORKSPACE/runtime/person_board/latest_capture"
LOG_ROOT="$WORKSPACE/test_logs/person_board_qwen_vl_real_full"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="$(git -C "$PACKAGE_ROOT" rev-parse --show-toplevel)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$STAMP"
EVENT_ID="person_board_real_full_$STAMP"
READY_FILE="$LOG_DIR/drive_guard.ready"
ARM_FILE="$LOG_DIR/drive_guard.arm"
PID_FILE="$LOG_DIR/processes.pid"
mkdir -p "$LOG_DIR"
: > "$PID_FILE"
CLEANED=false
OWN_BASE=false
OWN_CAMERA=false
RESULT=FAIL
FAILURE_REASON=test_interrupted

source_ros() {
  set +u
  source /opt/tros/humble/setup.bash
  source /root/install/ackermann_msgs/share/ackermann_msgs/package.bash
  source "$WORKSPACE/install/setup.bash"
  set -u
}

record_pid() {
  local pid="$1" label="$2"
  printf '%s %s\n' "$pid" "$label" >> "$PID_FILE"
}

start_group() {
  local label="$1" logfile="$2" command="$3"
  setsid bash -lc "$command" > "$logfile" 2>&1 &
  local pid=$!
  record_pid "$pid" "$label"
  echo "$pid"
}

stop_zero() {
  source_ros
  timeout 3 ros2 topic pub --rate 20 /cmd_vel geometry_msgs/msg/Twist     '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'     > "$LOG_DIR/final_zero.log" 2>&1 || true
  for _ in 1 2 3 4 5; do
    timeout 2 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist       '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'       >/dev/null 2>&1 || true
  done
}

cleanup() {
  if [[ "$CLEANED" == true ]]; then return; fi
  CLEANED=true
  stop_zero
  if [[ -s "$PID_FILE" ]]; then
    tac "$PID_FILE" | while read -r pid label; do
      if [[ "$label" == base && "$OWN_BASE" != true ]]; then continue; fi
      if [[ "$label" == camera && "$OWN_CAMERA" != true ]]; then continue; fi
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
  fi
  sleep 1
  stop_zero
  echo "log_dir=$LOG_DIR"
}
trap cleanup EXIT
trap 'FAILURE_REASON=signal; exit 130' INT TERM

cd "$SOURCE_ROOT"
{
  git status -sb
  git status --short
  git branch --show-current
  git rev-parse HEAD
  git rev-parse origin/feature/person-board-qwen-vl
} > "$LOG_DIR/git_info.txt"
[[ "$(git branch --show-current)" == feature/person-board-qwen-vl ]]
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/feature/person-board-qwen-vl)" ]]
[[ -z "$(git status --short)" ]]

[[ -f /root/.config/person_board_llm/env ]]
[[ "$(stat -c '%a' /root/.config/person_board_llm/env)" == 600 ]]
set +u
source /root/.config/person_board_llm/env
set -u
: "${DASHSCOPE_API_KEY:?API key missing}"
: "${PERSON_BOARD_LLM_BASE_URL:?Base URL missing}"
: "${PERSON_BOARD_LLM_MODEL:?model missing}"
[[ "$PERSON_BOARD_LLM_MODEL" == qwen3-vl-flash ]]
python3 -c 'import os,urllib.parse; print("api_key_configured=true"); print("endpoint_host="+str(urllib.parse.urlparse(os.environ["PERSON_BOARD_LLM_BASE_URL"]).hostname)); print("model="+os.environ["PERSON_BOARD_LLM_MODEL"])' > "$LOG_DIR/api_config_sanitized.txt"

source_ros

# Fail closed before connecting the base if another publisher is visible even
# inside the isolated test graph. The drive guard must later be the sole
# /cmd_vel publisher.
ros2 topic info /cmd_vel -v > "$LOG_DIR/cmd_vel_preflight.txt" 2>&1 || true
CMD_PUBLISHERS="$(awk '/Publisher count:/ {print $3}' "$LOG_DIR/cmd_vel_preflight.txt")"
CMD_PUBLISHERS="${CMD_PUBLISHERS:-0}"
if [[ "$CMD_PUBLISHERS" != 0 ]]; then
  FAILURE_REASON=external_cmd_vel_publisher
  echo "unexpected /cmd_vel publishers=$CMD_PUBLISHERS" >&2
  exit 1
fi
python3 -m py_compile "$SCRIPT_DIR/real_full_drive_guard.py"
bash -n "$0"
echo "py_compile=PASS" > "$LOG_DIR/static_checks.txt"
echo "bash_n=PASS" >> "$LOG_DIR/static_checks.txt"
printf 'ROS_DOMAIN_ID=%s\nROS_LOCALHOST_ONLY=%s\n' \
  "$ROS_DOMAIN_ID" "$ROS_LOCALHOST_ONLY" > "$LOG_DIR/ros_isolation.txt"

python3 - "$CAPTURE_DIR" "$LOG_DIR/old_capture.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,out=Path(sys.argv[1]),Path(sys.argv[2])
data={}
for name in ("crop_01.jpg","crop_02.jpg","crop_03.jpg","manifest.json"):
    path=root/name
    data[name]={"exists":path.exists(),
                "mtime_ns":path.stat().st_mtime_ns if path.exists() else None,
                "sha256":hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
out.write_text(json.dumps(data,indent=2)+"\n")
PY

if ros2 node list 2>/dev/null | grep -Eq '/person_board_(detector|qwen_vl_worker|real_full_drive_guard)'; then
  echo "conflicting person_board node already running" >&2
  FAILURE_REASON=conflicting_node
  exit 1
fi

ROS_ENV="set +u; source /opt/tros/humble/setup.bash; source /root/install/ackermann_msgs/share/ackermann_msgs/package.bash; source '$WORKSPACE/install/setup.bash'; set -u"

if timeout 3 ros2 topic echo --once /odom nav_msgs/msg/Odometry >/dev/null 2>&1 &&
   ros2 topic info /cmd_vel 2>/dev/null | grep -Eq 'Subscription count: [1-9]'; then
  echo "reuse_base=true" > "$LOG_DIR/base_reuse.txt"
else
  BASE_PID="$(start_group base "$LOG_DIR/base.log"     "$ROS_ENV; ros2 launch origincar_base origincar_bringup.launch.py")"
  OWN_BASE=true
fi

BASE_DEADLINE=$((SECONDS + 20))
while ! timeout 2 ros2 topic echo --once /odom nav_msgs/msg/Odometry   > "$LOG_DIR/odom_probe.txt" 2>&1; do
  (( SECONDS < BASE_DEADLINE )) || { FAILURE_REASON=odom_start_timeout; exit 1; }
done
ros2 topic info /cmd_vel -v > "$LOG_DIR/cmd_vel_info.txt" 2>&1
grep -Eq 'Subscription count: [1-9]' "$LOG_DIR/cmd_vel_info.txt" || {
  FAILURE_REASON=cmd_vel_no_subscriber; exit 1;
}

if timeout 4 ros2 topic echo --once /aurora/rgb/image_raw sensor_msgs/msg/Image   >/dev/null 2>&1; then
  echo "reuse_camera=true" > "$LOG_DIR/camera_reuse.txt"
else
  CAMERA_PID="$(start_group camera "$LOG_DIR/camera.log"     "$ROS_ENV; ros2 launch deptrum-ros-driver-aurora930 aurora930_launch.py")"
  OWN_CAMERA=true
fi
CAMERA_DEADLINE=$((SECONDS + 25))
while ! timeout 3 ros2 topic echo --once /aurora/rgb/image_raw sensor_msgs/msg/Image   > "$LOG_DIR/camera_probe.txt" 2>&1; do
  (( SECONDS < CAMERA_DEADLINE )) || { FAILURE_REASON=camera_start_timeout; exit 1; }
done

start_group display_status "$LOG_DIR/display_status.log"   "$ROS_ENV; ros2 topic echo /person_board/display_status std_msgs/msg/String --qos-durability transient_local --full-length" >/dev/null
start_group llm_status "$LOG_DIR/llm_status.log"   "$ROS_ENV; ros2 topic echo /person_board/llm_status std_msgs/msg/String --qos-durability transient_local --full-length" >/dev/null
start_group llm_result "$LOG_DIR/llm_result.log"   "$ROS_ENV; ros2 topic echo /person_board/llm_result std_msgs/msg/String --qos-durability transient_local --full-length" >/dev/null
start_group capture_status "$LOG_DIR/capture_status.log"   "$ROS_ENV; ros2 topic echo /person_board/capture_status std_msgs/msg/String --full-length" >/dev/null
start_group capture_done "$LOG_DIR/capture_done.log"   "$ROS_ENV; ros2 topic echo /person_board/capture_done std_msgs/msg/String --full-length" >/dev/null
start_group capture_batch "$LOG_DIR/capture_batch.log"   "$ROS_ENV; ros2 topic echo /person_board/capture_batch std_msgs/msg/String --full-length" >/dev/null
start_group detected "$LOG_DIR/detected.log"   "$ROS_ENV; ros2 topic echo /person_board/detected std_msgs/msg/Bool" >/dev/null
start_group box "$LOG_DIR/box.log"   "$ROS_ENV; ros2 topic echo /person_board/box std_msgs/msg/Int32MultiArray" >/dev/null
start_group score "$LOG_DIR/score.log"   "$ROS_ENV; ros2 topic echo /person_board/score std_msgs/msg/Float32" >/dev/null
start_group inference "$LOG_DIR/inference_ms.log"   "$ROS_ENV; ros2 topic echo /person_board/inference_ms std_msgs/msg/Float32" >/dev/null

export PERSON_BOARD_WORKSPACE="$WORKSPACE"
if ros2 node list 2>/dev/null | grep -Fxq /person_board_display; then
  echo "reuse_display=true" > "$LOG_DIR/display.log"
else
  DISPLAY_PID="$(start_group display "$LOG_DIR/display.log"   "cd '$SOURCE_ROOT'; export PERSON_BOARD_WORKSPACE='$WORKSPACE'; bash person_board_detection/scripts/run_person_board_display.sh")"
fi
QWEN_PID="$(start_group qwen "$LOG_DIR/qwen_worker.log"   "$ROS_ENV; export DASHSCOPE_API_KEY; export PERSON_BOARD_LLM_BASE_URL; export PERSON_BOARD_LLM_MODEL; ros2 launch person_board_detection person_board_qwen_vl.launch.py")"
DETECTOR_PID="$(start_group detector "$LOG_DIR/detector.log"   "$ROS_ENV; ros2 launch person_board_detection person_board_capture.launch.py")"

READY_DEADLINE=$((SECONDS + 30))
while ! grep -q 'PRELOADED_IDLE' "$LOG_DIR/capture_status.log" 2>/dev/null; do
  (( SECONDS < READY_DEADLINE )) || { FAILURE_REASON=detector_not_ready; exit 1; }
  kill -0 "$DETECTOR_PID" 2>/dev/null || { FAILURE_REASON=detector_exited; exit 1; }
  sleep 0.2
done
while ! grep -Eq 'WAITING|STARTING' "$LOG_DIR/display_status.log" 2>/dev/null; do
  (( SECONDS < READY_DEADLINE )) || { FAILURE_REASON=display_not_ready; exit 1; }
  sleep 0.2
done
while ! grep -q '"state":"IDLE"' "$LOG_DIR/llm_status.log" 2>/dev/null; do
  (( SECONDS < READY_DEADLINE )) || { FAILURE_REASON=qwen_not_ready; exit 1; }
  sleep 0.2
done

GUARD_PID="$(start_group guard "$LOG_DIR/drive_guard.log"   "$ROS_ENV; python3 '$SCRIPT_DIR/real_full_drive_guard.py' --speed '$SPEED' --max-distance '$MAX_DISTANCE' --hard-timeout '$HARD_TIMEOUT' --timeline '$LOG_DIR/drive_timeline.jsonl' --summary '$LOG_DIR/drive_summary.json' --ready-file '$READY_FILE' --arm-file '$ARM_FILE'")"
GUARD_READY_DEADLINE=$((SECONDS + 12))
while [[ ! -f "$READY_FILE" ]]; do
  (( SECONDS < GUARD_READY_DEADLINE )) || { FAILURE_REASON=drive_guard_not_ready; exit 1; }
  kill -0 "$GUARD_PID" 2>/dev/null || { FAILURE_REASON=drive_guard_exited; exit 1; }
  sleep 0.1
done

# Final ready gate immediately before any non-zero /cmd_vel.
timeout 2 ros2 topic echo --once /odom nav_msgs/msg/Odometry > "$LOG_DIR/final_odom_ready.txt"
ros2 topic info /cmd_vel -v > "$LOG_DIR/final_cmd_ready.txt"
grep -Eq 'Subscription count: [1-9]' "$LOG_DIR/final_cmd_ready.txt"
timeout 2 ros2 topic echo --once /aurora/rgb/image_raw sensor_msgs/msg/Image   > "$LOG_DIR/final_camera_ready.txt"
echo "ready_for_nonzero=true" > "$LOG_DIR/final_ready.txt"

CONTROL="{data: '{\"command\":\"start\",\"event_id\":\"$EVENT_ID\"}'}"
ros2 topic pub --once /person_board/control std_msgs/msg/String "$CONTROL"   > "$LOG_DIR/control_start.txt" 2>&1
sleep 0.2
touch "$ARM_FILE"

DRIVE_DEADLINE=$((SECONDS + HARD_TIMEOUT + 12))
while kill -0 "$GUARD_PID" 2>/dev/null; do
  (( SECONDS < DRIVE_DEADLINE )) || {
    FAILURE_REASON=guard_hard_timeout
    kill -TERM -- "-$GUARD_PID" 2>/dev/null || true
    exit 1
  }
  sleep 0.2
done
wait "$GUARD_PID" || true
stop_zero

CHAIN_DEADLINE=$((SECONDS + 20))
while (( SECONDS < CHAIN_DEADLINE )); do
  grep -q '"success":true' "$LOG_DIR/llm_result.log" 2>/dev/null &&
  grep -q 'SHOWING_RESULT' "$LOG_DIR/display_status.log" 2>/dev/null && break
  grep -q '"state":"FAILED"' "$LOG_DIR/llm_status.log" 2>/dev/null && break
  sleep 0.2
done

python3 - "$CAPTURE_DIR" "$LOG_DIR" "$EVENT_ID" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,out,event=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
old=json.loads((out/"old_capture.json").read_text())
current={}
for name in ("crop_01.jpg","crop_02.jpg","crop_03.jpg","manifest.json"):
    path=root/name
    current[name]={"exists":path.exists(),
      "mtime_ns":path.stat().st_mtime_ns if path.exists() else None,
      "sha256":hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
      "changed":path.exists() and old[name]["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()}
(out/"new_capture.json").write_text(json.dumps(current,indent=2)+"\n")
manifest=json.loads((root/"manifest.json").read_text())
(out/"manifest_final.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
assert manifest["event_id"] == event
assert all(current[name]["changed"] for name in ("crop_01.jpg","crop_02.jpg","crop_03.jpg","manifest.json"))
PY

python3 - "$LOG_DIR" "$CAPTURE_DIR" <<'PY'
import json,re,sys
from pathlib import Path
out,capture=Path(sys.argv[1]),Path(sys.argv[2])
drive=json.loads((out/"drive_summary.json").read_text())
manifest=json.loads((capture/"manifest.json").read_text())
worker=(out/"qwen_worker.log").read_text()
result_text=(out/"llm_result.log").read_text()
display_text=(out/"display_status.log").read_text()
match=re.search(r'http_status=(\d+).*elapsed_ms=([0-9.]+)',worker)
result_match=re.search(r"data: '(.+)'",result_text)
result=json.loads(result_match.group(1)) if result_match else {}
display_records=[]
for item in re.findall(r"data: '(.+)'",display_text):
    try: display_records.append(json.loads(item))
    except json.JSONDecodeError: pass
display=next((x for x in reversed(display_records) if x.get("state")=="SHOWING_RESULT"),{})
checks={
 "moved":drive["final_distance_m"]>0.05,
 "distance_safe":drive["final_distance_m"]<=3.15,
 "detected":drive["detected"],
 "three_new_images":len(manifest.get("frames",[]))==3,
 "llm_success":result.get("success") is True,
 "http_200":bool(match and match.group(1)=="200"),
 "recognizable":result.get("recognizable") is True,
 "description_le_50":len(result.get("result_text",""))<=50,
 "display_showing":display.get("state")=="SHOWING_RESULT",
 "request_id_consistent":manifest.get("request_id")==result.get("request_id")==display.get("request_id"),
}
result_name="PASS" if all(checks.values()) else (
 "DRIVE_PASS_DETECTION_NOT_TRIGGERED" if checks["moved"] and not checks["detected"] else "FAIL")
summary={"result":result_name,"checks":checks,"drive":drive,
 "event_id":manifest.get("event_id"),"request_id":manifest.get("request_id"),
 "selected_image":result.get("selected_image_path"),
 "selected_quality_score":result.get("selected_quality_score"),
 "http_status":int(match.group(1)) if match else None,
 "api_attempt_latency_ms":float(match.group(2)) if match else None,
 "model":result.get("model"),"recognizable":result.get("recognizable"),
 "description":result.get("result_text"),"llm_result":result,
 "display_status":display}
(out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
(out/"summary.txt").write_text("\n".join([
 f"result={result_name}",f"final_distance_m={drive['final_distance_m']}",
 f"drive_elapsed_sec={drive['drive_elapsed_sec']}",f"stop_reason={drive['stop_reason']}",
 f"detected={str(drive['detected']).lower()}",f"event_id={manifest.get('event_id')}",
 f"request_id={manifest.get('request_id')}",f"selected_image={result.get('selected_image_path')}",
 f"http_status={match.group(1) if match else ''}",f"model={result.get('model')}",
 f"recognizable={str(result.get('recognizable')).lower()}",
 f"description={result.get('result_text')}",f"display_state={display.get('state')}",
])+"\n")
if result_name!="PASS": raise SystemExit(1)
PY

if grep -RIlE 'sk-[A-Za-z0-9_-]{10,}|Authorization: Bearer [^$]|data:image/jpeg;base64,[A-Za-z0-9+/]{100,}' "$LOG_DIR" > "$LOG_DIR/security_hits.txt"; then
  FAILURE_REASON=sensitive_log_detected
  exit 1
fi
echo "sensitive_pattern_hits=false" > "$LOG_DIR/security_scan.txt"
RESULT=PASS
FAILURE_REASON=
cat "$LOG_DIR/summary.txt"
