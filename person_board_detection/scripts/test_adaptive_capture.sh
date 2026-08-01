#!/usr/bin/env bash

set -u
set -o pipefail

# Keep every real-vehicle test process isolated from the site-wide ROS graph.
export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=1

EXPECTED_BRANCH="feature/person-board-adaptive-capture"
WORKSPACE="/root/intelligent_car_ws"
SOURCE_ROOT="${WORKSPACE}/src"
DEFAULT_CAPTURE_DIR="${WORKSPACE}/runtime/person_board/latest_capture"
LOG_ROOT="${WORKSPACE}/test_logs/person_board"

EVENT_ID=""
TOTAL_TIMEOUT=90
START_CAMERA="auto"
KEEP_CAMERA=false
NO_BUILD=false
CAPTURE_DIR="${DEFAULT_CAPTURE_DIR}"

declare -A PROCESS_PIDS=()
declare -A PROCESS_START_TIMES=()

RESULT="FAIL"
FAILURE_REASON="测试尚未完成"
BUILD_OK="否"
PACKAGE_OK="否"
AURORA_OK="否"
IDLE_OK="否"
IDLE_NO_IMAGE_SUB="否"
ACTIVE_IMAGE_SUB="否"
TARGET_DETECTED="否"
LAST_STATE="未知"
IMAGES_OK="否"
MANIFEST_OK="否"
TMP_REMAINS="未知"
CAPTURE_DONE_OK="否"
DETECTOR_CLEAN_EXIT="否"
INFERENCE_SUMMARY="无数据"
BRANCH="未知"
HEAD_COMMIT="未知"
REMOTE_COMMIT="未知"
STARTED_CAMERA=false
LOG_DIR=""
MAIN_LOG=""
PID_FILE=""
START_EPOCH=0
DEADLINE=0
SUMMARY_WRITTEN=false
CLEANUP_DONE=false

usage() {
    cat <<'EOF'
用法：bash person_board_detection/scripts/test_adaptive_capture.sh [参数]

参数：
  --event-id ID          测试事件 ID（默认 person_board_YYYYMMDD_HHMMSS）
  --timeout SEC          总超时秒数（默认 90）
  --start-camera auto    图像不存在时自动启动 Aurora930
  --keep-camera          测试结束后保留本脚本启动的相机
  --no-build             跳过 py_compile 和 colcon build
  --capture-dir DIR      固定裁剪目录
  -h, --help             显示帮助
EOF
}

fail() {
    FAILURE_REASON="$1"
    RESULT="FAIL"
    printf '\n[FAIL] %s\n' "${FAILURE_REASON}" >&2
    exit 1
}

remaining_seconds() {
    local now
    now=$(date +%s)
    local remaining=$((DEADLINE - now))
    if ((remaining < 1)); then
        printf '0\n'
    else
        printf '%s\n' "${remaining}"
    fi
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "缺少命令：$1"
}

source_environment() {
    [[ -r /opt/tros/humble/setup.bash ]] || fail "缺少 /opt/tros/humble/setup.bash"
    [[ -r "${WORKSPACE}/install/setup.bash" ]] || fail "缺少 ${WORKSPACE}/install/setup.bash"
    set +u
    # shellcheck disable=SC1091
    source /opt/tros/humble/setup.bash || {
        set -u
        fail "source TROS 环境失败"
    }
    # shellcheck disable=SC1091
    source "${WORKSPACE}/install/setup.bash" || {
        set -u
        fail "source 工作空间环境失败"
    }
    set -u
}

record_process() {
    local role="$1"
    local pid="$2"
    local start_time=""
    if [[ -r "/proc/${pid}/stat" ]]; then
        start_time=$(awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null || true)
    fi
    PROCESS_PIDS["${role}"]="${pid}"
    PROCESS_START_TIMES["${role}"]="${start_time}"
    printf '%s|%s|%s\n' "${role}" "${pid}" "${start_time}" >>"${PID_FILE}"
}

start_group() {
    local role="$1"
    local logfile="$2"
    shift 2
    setsid "$@" > >(tee -a "${logfile}") 2>&1 &
    local pid=$!
    record_process "${role}" "${pid}"
    printf '[进程] %-18s PID/PGID=%s 日志=%s\n' "${role}" "${pid}" "${logfile}"
}

same_process() {
    local role="$1"
    local pid="${PROCESS_PIDS[${role}]:-}"
    local expected="${PROCESS_START_TIMES[${role}]:-}"
    [[ -n "${pid}" && -r "/proc/${pid}/stat" ]] || return 1
    local actual
    actual=$(awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null || true)
    [[ -n "${expected}" && "${actual}" == "${expected}" ]]
}

stop_group() {
    local role="$1"
    local pid="${PROCESS_PIDS[${role}]:-}"
    [[ -n "${pid}" ]] || return 0
    if ! same_process "${role}"; then
        return 0
    fi
    kill -TERM -- "-${pid}" 2>/dev/null || true
    local end=$((SECONDS + 5))
    while same_process "${role}" && ((SECONDS < end)); do
        sleep 0.2
    done
    if same_process "${role}"; then
        kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
}

cleanup() {
    [[ "${CLEANUP_DONE}" == false ]] || return 0
    CLEANUP_DONE=true
    local role
    for role in status_echo batch_echo done_echo detected_echo score_echo box_echo camera_hz inference_hz; do
        stop_group "${role}"
    done
    stop_group detector
    if [[ "${STARTED_CAMERA}" == true && "${KEEP_CAMERA}" == false ]]; then
        stop_group camera
    fi
}

write_summary() {
    [[ -n "${LOG_DIR}" ]] || return 0
    [[ "${SUMMARY_WRITTEN}" == false ]] || return 0
    SUMMARY_WRITTEN=true
    if [[ -f "${LOG_DIR}/capture_status.log" ]]; then
        local rates
        rates=$(grep -o '"current_inference_hz"[[:space:]]*:[[:space:]]*[0-9.]*' \
            "${LOG_DIR}/capture_status.log" 2>/dev/null | sort -u | tr '\n' ' ' || true)
        [[ -z "${rates}" ]] || INFERENCE_SUMMARY="${rates}"
    fi
    cat >"${LOG_DIR}/summary.txt" <<EOF
Git 分支: ${BRANCH}
Git 提交: ${HEAD_COMMIT}
远端提交: ${REMOTE_COMMIT}
构建是否成功: ${BUILD_OK}
包发现是否成功: ${PACKAGE_OK}
Aurora 是否正常: ${AURORA_OK}
PRELOADED_IDLE 是否正常: ${IDLE_OK}
空闲时是否无图像订阅: ${IDLE_NO_IMAGE_SUB}
start 后是否建立图像订阅: ${ACTIVE_IMAGE_SUB}
是否检测到目标: ${TARGET_DETECTED}
最后状态: ${LAST_STATE}
推理频率变化摘要: ${INFERENCE_SUMMARY}
三张图片是否生成: ${IMAGES_OK}
manifest 是否有效: ${MANIFEST_OK}
是否残留 tmp: ${TMP_REMAINS}
capture_done 是否成功: ${CAPTURE_DONE_OK}
detector 是否干净退出: ${DETECTOR_CLEAN_EXIT}
总测试结论: ${RESULT}
失败原因: ${FAILURE_REASON}
事件 ID: ${EVENT_ID}
固定裁剪目录: ${CAPTURE_DIR}
测试开始时间: $(date --date="@${START_EPOCH}" '+%Y-%m-%d %H:%M:%S %z' 2>/dev/null || printf '%s' "${START_EPOCH}")
测试结束时间: $(date '+%Y-%m-%d %H:%M:%S %z')
日志目录: ${LOG_DIR}
EOF
}

on_exit() {
    local status=$?
    cleanup
    write_summary
    if [[ -n "${LOG_DIR}" ]]; then
        printf '\n测试结论：%s\n' "${RESULT}"
        printf '摘要：%s\n' "${LOG_DIR}/summary.txt"
        if [[ "${RESULT}" != "PASS" ]]; then
            printf '失败原因：%s\n' "${FAILURE_REASON}"
        fi
    fi
    if [[ "${RESULT}" == "PASS" ]]; then
        return 0
    fi
    [[ ${status} -ne 0 ]] && return "${status}"
    return 1
}

on_signal() {
    FAILURE_REASON="测试被信号中断"
    RESULT="FAIL"
    exit 130
}

trap on_exit EXIT
trap on_signal INT TERM HUP

json_log_field() {
    local file="$1"
    local field="$2"
    python3 - "${file}" "${field}" <<'PY'
import ast
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
field = sys.argv[2]
value = None
if path.exists():
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line == "---":
            continue
        if line.startswith("data:"):
            line = line.split(":", 1)[1].strip()
        candidates = [line]
        try:
            decoded = ast.literal_eval(line)
            if isinstance(decoded, str):
                candidates.insert(0, decoded)
        except (SyntaxError, ValueError):
            pass
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict) and field in payload:
                value = payload[field]
if value is not None:
    if isinstance(value, bool):
        print(str(value).lower())
    elif isinstance(value, (dict, list)):
        print(json.dumps(value, ensure_ascii=False))
    else:
        print(value)
PY
}

wait_for_node() {
    local node="$1"
    local limit="$2"
    local end=$((SECONDS + limit))
    while ((SECONDS < end)); do
        ros2 node list 2>/dev/null | grep -Fxq "${node}" && return 0
        sleep 1
    done
    return 1
}

wait_for_topic() {
    local topic="$1"
    local limit="$2"
    local end=$((SECONDS + limit))
    while ((SECONDS < end)); do
        ros2 topic list 2>/dev/null | grep -Fxq "${topic}" && return 0
        sleep 1
    done
    return 1
}

topic_has_data() {
    local topic="$1"
    local logfile="$2"
    timeout 8 ros2 topic hz "${topic}" --window 2 >"${logfile}" 2>&1 || true
    grep -q 'average rate:' "${logfile}"
}

while (($#)); do
    case "$1" in
        --event-id)
            (($# >= 2)) || { usage; fail "--event-id 缺少参数"; }
            EVENT_ID="$2"
            shift 2
            ;;
        --timeout)
            (($# >= 2)) || { usage; fail "--timeout 缺少参数"; }
            TOTAL_TIMEOUT="$2"
            shift 2
            ;;
        --start-camera)
            (($# >= 2)) || { usage; fail "--start-camera 缺少参数"; }
            START_CAMERA="$2"
            shift 2
            ;;
        --keep-camera)
            KEEP_CAMERA=true
            shift
            ;;
        --no-build)
            NO_BUILD=true
            shift
            ;;
        --capture-dir)
            (($# >= 2)) || { usage; fail "--capture-dir 缺少参数"; }
            CAPTURE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            RESULT="PASS"
            FAILURE_REASON=""
            exit 0
            ;;
        *)
            usage
            fail "未知参数：$1"
            ;;
    esac
done

[[ "${TOTAL_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || fail "--timeout 必须为正整数"
[[ "${START_CAMERA}" == "auto" ]] || fail "--start-camera 当前仅支持 auto"
[[ "${CAPTURE_DIR}" == /* ]] || fail "--capture-dir 必须为绝对路径"
if [[ -z "${EVENT_ID}" ]]; then
    EVENT_ID="person_board_$(date '+%Y%m%d_%H%M%S')"
fi
[[ "${EVENT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || fail "event_id 含非法字符"

START_EPOCH=$(date +%s)
DEADLINE=$((START_EPOCH + TOTAL_TIMEOUT))
RUN_STAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="${LOG_ROOT}/${RUN_STAMP}"
mkdir -p "${LOG_DIR}" || fail "无法创建日志目录：${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/test.log"
PID_FILE="${LOG_DIR}/test_processes.pid"
: >"${PID_FILE}" || fail "无法创建 PID 文件"
exec > >(tee -a "${MAIN_LOG}") 2>&1

printf '[1/10] 初始化环境与工具检查\n'
for command_name in git python3 colcon ros2 timeout setsid tee awk grep sha256sum; do
    require_command "${command_name}"
done
source_environment
printf '测试开始：%s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
printf '事件 ID：%s\n' "${EVENT_ID}"
printf '日志目录：%s\n' "${LOG_DIR}"
printf 'ROS_DOMAIN_ID：%s\n' "${ROS_DOMAIN_ID:-未设置}"

printf '[2/10] Git 基线检查\n'
cd "${SOURCE_ROOT}" || fail "无法进入源码目录：${SOURCE_ROOT}"
BRANCH=$(git branch --show-current 2>/dev/null || true)
HEAD_COMMIT=$(git rev-parse HEAD 2>/dev/null || true)
REMOTE_COMMIT=$(git rev-parse "origin/${EXPECTED_BRANCH}" 2>/dev/null || true)
{
    printf 'branch=%s\nHEAD=%s\nremote=%s\n' "${BRANCH}" "${HEAD_COMMIT}" "${REMOTE_COMMIT}"
    printf 'git status:\n'
    git status --short
} | tee "${LOG_DIR}/git_info.txt"
[[ "${BRANCH}" == "${EXPECTED_BRANCH}" ]] || fail "当前分支不是 ${EXPECTED_BRANCH}"
[[ "${HEAD_COMMIT}" == "${REMOTE_COMMIT}" ]] || fail "本地 HEAD 与远端功能分支不一致"
[[ -z "$(git status --short)" ]] || fail "工作区不干净"

printf '[3/10] 静态检查与构建\n'
if [[ "${NO_BUILD}" == true ]]; then
    BUILD_OK="跳过（--no-build）"
    printf '已按参数跳过 py_compile 与 colcon build。\n'
else
    python3 -m py_compile \
        person_board_detection/person_board_detection/*.py \
        person_board_detection/launch/*.launch.py \
        2>&1 | tee "${LOG_DIR}/py_compile.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || fail "py_compile 失败"
    cd "${WORKSPACE}" || fail "无法进入工作空间：${WORKSPACE}"
    colcon build --symlink-install --packages-select person_board_detection \
        --event-handlers console_direct+ 2>&1 | tee "${LOG_DIR}/build.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || fail "colcon build 失败"
    BUILD_OK="是"
    source_environment
    cd "${SOURCE_ROOT}" || fail "无法返回源码目录"
fi

printf '[4/10] ROS2 包发现检查\n'
ros2 pkg prefix person_board_detection | tee "${LOG_DIR}/package_prefix.txt" || fail "无法发现 person_board_detection 包"
ros2 pkg executables person_board_detection | tee "${LOG_DIR}/package_executables.txt" || fail "无法查询包可执行项"
grep -q 'person_board_detector' "${LOG_DIR}/package_executables.txt" || fail "未发现 person_board_detector 可执行项"
PACKAGE_OK="是"

printf '[5/10] Aurora930 图像检查\n'
if wait_for_topic /aurora/rgb/image_raw 2 && topic_has_data /aurora/rgb/image_raw "${LOG_DIR}/camera_probe.log"; then
    AURORA_OK="是（已运行）"
    printf 'Aurora 图像话题已存在且有数据，不重复启动。\n'
else
    printf '未检测到有效 Aurora 图像，自动启动相机驱动。\n'
    start_group camera "${LOG_DIR}/camera.log" \
        ros2 launch deptrum-ros-driver-aurora930 aurora930_launch.py
    STARTED_CAMERA=true
    camera_wait=$(remaining_seconds)
    ((camera_wait > 25)) && camera_wait=25
    ((camera_wait > 0)) || fail "等待相机前已达到总超时"
    wait_for_topic /aurora/rgb/image_raw "${camera_wait}" || fail "相机启动后未出现 /aurora/rgb/image_raw"
    topic_has_data /aurora/rgb/image_raw "${LOG_DIR}/camera_probe.log" || fail "相机话题存在但没有图像数据"
    AURORA_OK="是（脚本启动）"
fi

printf '[6/10] 启动门控检测并检查空闲状态\n'
start_group detector "${LOG_DIR}/detector.log" \
    ros2 launch person_board_detection person_board_capture.launch.py
node_wait=$(remaining_seconds)
((node_wait > 30)) && node_wait=30
wait_for_node /person_board_detector "${node_wait}" || fail "检测节点未在规定时间出现"
ros2 node info /person_board_detector >"${LOG_DIR}/node_info_idle.txt" 2>&1 || fail "无法读取空闲节点信息"
grep -Fq '/person_board/control' "${LOG_DIR}/node_info_idle.txt" || fail "空闲节点缺少 control 订阅"
if grep -Fq '/aurora/rgb/image_raw' "${LOG_DIR}/node_info_idle.txt"; then
    fail "空闲阶段错误订阅了 Aurora 图像"
fi
IDLE_NO_IMAGE_SUB="是"
timeout 8 ros2 topic echo --once /person_board/capture_status std_msgs/msg/String --field data \
    >"${LOG_DIR}/idle_status.txt" 2>&1 || fail "未收到空闲 capture_status"
idle_state=$(json_log_field "${LOG_DIR}/idle_status.txt" state)
idle_subscription=$(json_log_field "${LOG_DIR}/idle_status.txt" image_subscription_active)
[[ "${idle_state}" == "PRELOADED_IDLE" ]] || fail "空闲状态不是 PRELOADED_IDLE：${idle_state:-无}"
[[ "${idle_subscription}" == "false" ]] || fail "空闲状态 image_subscription_active 不是 false"
IDLE_OK="是"

printf '[7/10] 启动话题记录并发送 start\n'
start_group status_echo "${LOG_DIR}/capture_status.log" \
    ros2 topic echo /person_board/capture_status std_msgs/msg/String --field data
start_group batch_echo "${LOG_DIR}/capture_batch.log" \
    ros2 topic echo /person_board/capture_batch std_msgs/msg/String --field data
start_group done_echo "${LOG_DIR}/capture_done.log" \
    ros2 topic echo /person_board/capture_done std_msgs/msg/String --field data
start_group detected_echo "${LOG_DIR}/detected.log" \
    ros2 topic echo /person_board/detected std_msgs/msg/Bool
start_group score_echo "${LOG_DIR}/score.log" \
    ros2 topic echo /person_board/score std_msgs/msg/Float32
start_group box_echo "${LOG_DIR}/box.log" \
    ros2 topic echo /person_board/box std_msgs/msg/Int32MultiArray
sleep 1
control_payload=$(printf '{data: '\''{"command":"start","event_id":"%s"}'\''}' "${EVENT_ID}")
ros2 topic pub --once /person_board/control std_msgs/msg/String "${control_payload}" \
    | tee "${LOG_DIR}/control_start.txt" || fail "发送 start 命令失败"
active_end=$((SECONDS + 10))
while ((SECONDS < active_end)); do
    ros2 node info /person_board_detector >"${LOG_DIR}/node_info_active.txt" 2>&1 || true
    grep -Fq '/aurora/rgb/image_raw' "${LOG_DIR}/node_info_active.txt" && break
    sleep 0.5
done
grep -Fq '/aurora/rgb/image_raw' "${LOG_DIR}/node_info_active.txt" || fail "start 后未建立 Aurora 图像订阅"
ACTIVE_IMAGE_SUB="是"

printf '[8/10] 记录相机与推理频率\n'
start_group camera_hz "${LOG_DIR}/camera_hz.log" \
    timeout 15 ros2 topic hz /aurora/rgb/image_raw
start_group inference_hz "${LOG_DIR}/inference_hz.log" \
    timeout 30 ros2 topic hz /person_board/inference_ms
printf '\n请将人形立牌放入 Aurora930 视野，并逐渐靠近相机。\n'
printf '脚本正在等待三帧裁剪完成，无需打开其他终端。\n\n'

printf '[9/10] 等待捕获完成、失败、节点退出或总超时\n'
while true; do
    done_success=$(json_log_field "${LOG_DIR}/capture_done.log" success)
    LAST_STATE=$(json_log_field "${LOG_DIR}/capture_status.log" state)
    [[ -n "${LAST_STATE}" ]] || LAST_STATE="未知"
    if [[ "${done_success}" == "true" ]]; then
        CAPTURE_DONE_OK="是"
        break
    fi
    if [[ "${LAST_STATE}" == "CAPTURE_FAILED" ]]; then
        reason=$(json_log_field "${LOG_DIR}/capture_status.log" error_reason)
        fail "检测状态进入 CAPTURE_FAILED：${reason:-未知原因}"
    fi
    if ! same_process detector; then
        sleep 1
        done_success=$(json_log_field "${LOG_DIR}/capture_done.log" success)
        if [[ "${done_success}" == "true" ]]; then
            CAPTURE_DONE_OK="是"
            break
        fi
        fail "检测节点退出前未收到 capture_done success=true"
    fi
    (( $(remaining_seconds) > 0 )) || fail "达到总超时 ${TOTAL_TIMEOUT} 秒"
    sleep 1
done

printf '[10/10] 验证固定目录、JPEG、manifest 和退出状态\n'
batch_request=$(json_log_field "${LOG_DIR}/capture_batch.log" request_id)
python3 - "${CAPTURE_DIR}" "${batch_request}" "${LOG_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import cv2

capture_dir = Path(sys.argv[1])
batch_request = sys.argv[2]
log_dir = Path(sys.argv[3])
expected = ["crop_01.jpg", "crop_02.jpg", "crop_03.jpg", "manifest.json"]
if not capture_dir.is_dir():
    raise SystemExit(f"固定目录不存在：{capture_dir}")
visible = sorted(path.name for path in capture_dir.iterdir() if not path.name.startswith("."))
if visible != expected:
    raise SystemExit(f"固定目录内容不符合要求：{visible}")
if list(capture_dir.glob("*.tmp")):
    raise SystemExit("固定目录残留 .tmp 文件")
manifest_path = capture_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
frames = manifest.get("frames")
if not isinstance(frames, list) or len(frames) != 3:
    raise SystemExit("manifest.frames 长度不是 3")
request_id = manifest.get("request_id")
if not request_id:
    raise SystemExit("manifest.request_id 为空")
if not batch_request or batch_request != request_id:
    raise SystemExit("capture_batch.request_id 与 manifest.request_id 不一致")
paths = manifest.get("image_paths")
if not isinstance(paths, list) or [Path(path).name for path in paths] != expected[:3]:
    raise SystemExit("manifest.image_paths 不是三个固定文件名")
if any(Path(path).parent != capture_dir for path in paths):
    raise SystemExit("manifest.image_paths 未指向指定固定目录")
inventory = []
hashes = []
dimensions = []
for name in expected:
    path = capture_dir / name
    size = path.stat().st_size
    if size <= 0:
        raise SystemExit(f"文件为空：{name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory.append(f"{name}\t{size} bytes")
    hashes.append(f"{digest}  {name}")
for name in expected[:3]:
    image = cv2.imread(str(capture_dir / name), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise SystemExit(f"JPEG 无法读取：{name}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise SystemExit(f"JPEG 尺寸非法：{name}")
    dimensions.append(f"{name}\t{width}x{height}")
(log_dir / "capture_inventory.txt").write_text("\n".join(inventory) + "\n", encoding="utf-8")
(log_dir / "capture_sha256.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
(log_dir / "capture_dimensions.txt").write_text("\n".join(dimensions) + "\n", encoding="utf-8")
(log_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("固定批次验证通过")
PY
[[ $? -eq 0 ]] || fail "固定目录或批次内容验证失败"
IMAGES_OK="是"
MANIFEST_OK="是"
TMP_REMAINS="否"
grep -Eq 'data:[[:space:]]*true|true' "${LOG_DIR}/detected.log" && TARGET_DETECTED="是"

detector_exit_end=$((SECONDS + 10))
while same_process detector && ((SECONDS < detector_exit_end)); do
    sleep 0.5
done
if same_process detector; then
    fail "捕获成功后检测节点未在宽限时间内退出"
fi
wait "${PROCESS_PIDS[detector]}" 2>/dev/null || true
DETECTOR_CLEAN_EXIT="是"
LAST_STATE="DONE"
RESULT="PASS"
FAILURE_REASON="无"
write_summary
printf '\n[PASS] 人形立牌自适应三帧裁剪实机测试通过。\n'
