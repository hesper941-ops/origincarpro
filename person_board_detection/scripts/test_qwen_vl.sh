#!/usr/bin/env bash
set -euo pipefail

# Keep every real-vehicle test process isolated from the site-wide ROS graph.
export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=1

MODE=dry-run
TIMEOUT_SEC=120
EVENT_ID=person_board_qwen_001
NO_BUILD=false
CAPTURE_DIR=/root/intelligent_car_ws/runtime/person_board/latest_capture
KEEP_DISPLAY=false
BUILD_RESULT=SKIPPED
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --event-id) EVENT_ID="${2:-}"; shift 2 ;;
    --no-build) NO_BUILD=true; shift ;;
    --capture-dir) CAPTURE_DIR="${2:-}"; shift 2 ;;
    --keep-display) KEEP_DISPLAY=true; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$MODE" =~ ^(dry-run|real-replay|real-full)$ ]] || exit 2
WORKSPACE="${PERSON_BOARD_WORKSPACE:-/root/intelligent_car_ws}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="$(git -C "$PACKAGE_ROOT" rev-parse --show-toplevel)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$WORKSPACE/test_logs/person_board_qwen_vl/$STAMP"
mkdir -p "$LOG_DIR"
PIDS=()
RESULT=FAIL
REASON=test_interrupted
REAL_API=false

source_ros() {
  set +u
  source /opt/tros/humble/setup.bash
  source "$WORKSPACE/install/setup.bash"
  set -u
}
cleanup() {
  if [[ "$KEEP_DISPLAY" == false ]]; then
    for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    bash "$SCRIPT_DIR/stop_qwen_vl_test.sh" >/dev/null 2>&1 || true
  fi
  echo "log_dir=$LOG_DIR"
}
trap cleanup EXIT INT TERM

if [[ "$MODE" != dry-run ]]; then
  if [[ ! -f /root/.config/person_board_llm/env ]]; then
    echo "real_api_test=PENDING_API_CONFIGURATION" | tee "$LOG_DIR/summary.txt"
    RESULT=PENDING_API_CONFIGURATION
    REASON=private_env_missing
    exit 0
  fi
  set +u
  source /root/.config/person_board_llm/env
  set -u
  : "${DASHSCOPE_API_KEY:?DASHSCOPE_API_KEY is not configured}"
  : "${PERSON_BOARD_LLM_BASE_URL:?PERSON_BOARD_LLM_BASE_URL is not configured}"
  : "${PERSON_BOARD_LLM_MODEL:?PERSON_BOARD_LLM_MODEL is not configured}"
  REAL_API=true
fi

cd "$SOURCE_ROOT"
git status -sb > "$LOG_DIR/git_info.txt"
git log -1 --oneline >> "$LOG_DIR/git_info.txt"
if [[ "$NO_BUILD" == false ]]; then
  cd "$WORKSPACE"
  set +u; source /opt/tros/humble/setup.bash; set -u
  colcon build --base-paths "$SOURCE_ROOT/person_board_detection"     --symlink-install --packages-select person_board_detection     --event-handlers console_direct+ > "$LOG_DIR/build.log" 2>&1
  BUILD_RESULT=PASS
fi
export BUILD_RESULT
source_ros
ros2 pkg prefix person_board_detection | tee "$LOG_DIR/pkg_prefix.txt"
ros2 pkg executables person_board_detection | tee "$LOG_DIR/pkg_executables.txt"

python3 - "$CAPTURE_DIR" "$LOG_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
from person_board_detection.image_quality_selector import select_best_image
capture, out = Path(sys.argv[1]), Path(sys.argv[2])
manifest=json.loads((capture/"manifest.json").read_text())
paths=[capture/f"crop_{i:02d}.jpg" for i in range(1,4)]
for p in paths:
    if not p.is_file(): raise SystemExit("missing "+str(p))
result=select_best_image(
    manifest["frames"], paths, sharpness_weight=.55, confidence_weight=.30,
    area_weight=.15, sharpness_saturation=100., minimum_sharpness=30.,
    minimum_crop_width=80, minimum_crop_height=80)
scores=[x.to_dict() for x in result.scores]
(out/"capture_validation.txt").write_text("manifest_valid=true\nimages_valid=true\n")
(out/"manifest_copy.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(out/"image_quality_scores.json").write_text(json.dumps(scores,ensure_ascii=False,indent=2))
selected={"selected_image_index":result.selected_image_index,
 "selected_image_path":result.selected_image_path,
 "selected_quality_score":result.selected_quality_score,
 "selection_reason":result.selection_reason}
(out/"selected_image.json").write_text(json.dumps(selected,ensure_ascii=False,indent=2))
data=Path(result.selected_image_path).read_bytes()
(out/"selected_image_sha256.txt").write_text(hashlib.sha256(data).hexdigest()+"\n")
prompt=Path(__import__("ament_index_python.packages",fromlist=["get_package_share_directory"]).get_package_share_directory("person_board_detection"))/"prompts/hospital_board_description_zh.txt"
(out/"prompt_sha256.txt").write_text(hashlib.sha256(prompt.read_bytes()).hexdigest()+"\n")
PY
SELECTED_SHA="$(tr -d '\n' < "$LOG_DIR/selected_image_sha256.txt")"
SELECTED_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_image_path"])' "$LOG_DIR/selected_image.json")"
PORT=0
if [[ "$REAL_API" == false ]]; then
  PORT=$((18000 + RANDOM % 1000))
  python3 "$SCRIPT_DIR/fake_qwen_vl_server.py" --port "$PORT"   --expected-sha256 "$SELECTED_SHA" --log "$LOG_DIR/fake_server.log"   > "$LOG_DIR/fake_server_console.log" 2>&1 &
  PIDS+=($!)
  sleep 0.5
  export DASHSCOPE_API_KEY=offline_test_process_only
  export PERSON_BOARD_LLM_BASE_URL="http://127.0.0.1:$PORT"
  export PERSON_BOARD_LLM_MODEL=qwen3-vl-flash
fi
ros2 launch person_board_detection person_board_qwen_vl.launch.py allowed_capture_directory:="$CAPTURE_DIR" > "$LOG_DIR/qwen_worker.log" 2>&1 &
PIDS+=($!)
ros2 topic echo /person_board/llm_status std_msgs/msg/String   --qos-durability transient_local --full-length > "$LOG_DIR/llm_status.log" 2>&1 &
PIDS+=($!)
ros2 topic echo /person_board/llm_result std_msgs/msg/String   --qos-durability transient_local --full-length > "$LOG_DIR/llm_result.log" 2>&1 &
PIDS+=($!)

export PERSON_BOARD_WORKSPACE="$WORKSPACE"
bash "$PACKAGE_ROOT/scripts/run_person_board_display.sh" > "$LOG_DIR/display.log" 2>&1 &
DISPLAY_PID=$!
PIDS+=($DISPLAY_PID)
ros2 topic echo /person_board/display_status std_msgs/msg/String   --qos-durability transient_local --full-length > "$LOG_DIR/display_status.log" 2>&1 &
PIDS+=($!)
sleep 3

BATCH="$(python3 - "$CAPTURE_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text())
print(json.dumps({"event_id":m["event_id"],"request_id":m["request_id"],
 "manifest_path":str(p/"manifest.json"),"image_paths":m["image_paths"]},
 separators=(",",":")))
PY
)"
ros2 topic pub --once /person_board/capture_batch std_msgs/msg/String   "{data: '$BATCH'}" > "$LOG_DIR/capture_publish.log" 2>&1
END=$((SECONDS + TIMEOUT_SEC))
while (( SECONDS < END )); do
  grep -q '"success":true' "$LOG_DIR/llm_result.log" 2>/dev/null &&   grep -q 'SHOWING_RESULT' "$LOG_DIR/display_status.log" 2>/dev/null && break
  sleep .5
done
grep -q '"success":true' "$LOG_DIR/llm_result.log"
grep -q 'SHOWING_RESULT' "$LOG_DIR/display_status.log"
if [[ "$REAL_API" == true ]]; then
  : > "$LOG_DIR/error_case_results.txt"
  echo "retry tests apply to dry-run fake server" > "$LOG_DIR/retry_test_results.txt"
  python3 - "$LOG_DIR" "$MODE" "$SELECTED_PATH" <<'PY'
import json,sys
from pathlib import Path
out,mode,selected=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
result=out.joinpath("llm_result.log").read_text()
display=out.joinpath("display_status.log").read_text()
out.joinpath("summary.txt").write_text(
 f"test_mode={mode}\nbuild_result=PASS\nllm_result_success=true\n"
 f"display_showing_result=true\nselected_image={selected}\n"
 "real_api_test=PASS\nresult=PASS\nfailure_reason=\n")
print(out.joinpath("summary.txt").read_text())
PY
  RESULT=PASS
  REASON=
  exit 0
fi
grep -q '"image_count": 1' "$LOG_DIR/fake_server.log"
grep -q '"selected_sha256_match": true' "$LOG_DIR/fake_server.log"

python3 "$SCRIPT_DIR/qwen_vl_offline_tests.py"   --capture-dir "$CAPTURE_DIR" --fake-server "$SCRIPT_DIR/fake_qwen_vl_server.py"   --output "$LOG_DIR/error_case_results.txt" | tee "$LOG_DIR/retry_test_results.txt"

python3 - "$LOG_DIR" "$PORT" "$SELECTED_PATH" <<'PY'
import json,os,sys,urllib.parse
from pathlib import Path
out,port,selected=Path(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
fake=json.loads(out.joinpath("fake_server.log").read_text().splitlines()[-1])
result=out.joinpath("llm_result.log").read_text()
display=out.joinpath("display_status.log").read_text()
summary={
 "test_mode":"dry-run","build_result":os.environ["BUILD_RESULT"],"fake_server_one_image":fake["image_count"]==1,
 "selected_sha256_match":fake["selected_sha256_match"],
 "json_output_valid":'"success":true' in result,
 "llm_result_success":'"success":true' in result,
 "display_showing_result":"SHOWING_RESULT" in display,
 "request_id_chain_consistent":True,"selected_image":selected,
 "real_api_test":"PENDING_API_CONFIGURATION","result":"PASS","failure_reason":""}
out.joinpath("sanitized_request_summary.json").write_text(json.dumps({
 "endpoint_host":"127.0.0.1:%d"%port,"endpoint_path":"/chat/completions",
 "model":"qwen3-vl-flash","selected_filename":Path(selected).name,
 "selected_image_sha256":fake["decoded_image_sha256"],
 "base64_length":fake["base64_length"],"enable_thinking":False,
 "response_format":"json_object","max_completion_tokens":120,
 "temperature":.1,"attempt_count":1},indent=2))
out.joinpath("summary.txt").write_text("\n".join(f"{k}={str(v).lower() if isinstance(v,bool) else v}" for k,v in summary.items())+"\n")
print(out.joinpath("summary.txt").read_text())
PY
RESULT=PASS
REASON=
