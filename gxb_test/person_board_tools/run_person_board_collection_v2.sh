#!/usr/bin/env bash
set -Eeuo pipefail

# 人形立牌数据采集脚本 v2
#
# 主要改进：
# 1. source ROS 环境时临时关闭 nounset；
# 2. 自动把 SAVE_FPS=3 转为 3.0；
# 3. 禁止复用已存在的 RUN_NAME，避免 rosbag 目录冲突；
# 4. 启动 rosbag 后检查其是否存活；
# 5. 采集过程中持续监视 rosbag，异常退出时立即停止采集；
# 6. 采集结束后优雅关闭 rosbag 并输出 bag info。

WORKSPACE="${WORKSPACE:-/root/intelligent_car_ws}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/datasets/person_board}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"

SAVE_FPS="${SAVE_FPS:-3.0}"
MAX_IMAGES="${MAX_IMAGES:-100}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
WARMUP_SEC="${WARMUP_SEC:-2.0}"
RECORD_BAG="${RECORD_BAG:-1}"

CAMERA_PACKAGE="${CAMERA_PACKAGE:-deptrum-ros-driver-aurora930}"
CAMERA_LAUNCH="${CAMERA_LAUNCH:-aurora930_launch.py}"

RGB_TOPIC="${RGB_TOPIC:-}"
DEPTH_TOPIC="${DEPTH_TOPIC:-}"
RGB_INFO_TOPIC="${RGB_INFO_TOPIC:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR="${SCRIPT_DIR}/collect_rgb_dataset.py"

RAW_DIR="${DATA_ROOT}/raw/${RUN_NAME}"
BAG_DIR="${DATA_ROOT}/bags/${RUN_NAME}"
LOG_DIR="${DATA_ROOT}/logs"
CAMERA_LOG="${LOG_DIR}/${RUN_NAME}_camera.log"
BAG_LOG="${LOG_DIR}/${RUN_NAME}_bag.log"

CAMERA_PID=""
BAG_PID=""
COLLECTOR_PID=""
BAG_STOPPED=0

normalize_float() {
  local value="$1"
  if [[ "${value}" =~ ^[0-9]+$ ]]; then
    printf '%s.0' "${value}"
  else
    printf '%s' "${value}"
  fi
}

SAVE_FPS="$(normalize_float "${SAVE_FPS}")"
WARMUP_SEC="$(normalize_float "${WARMUP_SEC}")"

stop_process_gracefully() {
  local pid="$1"
  local name="$2"

  [[ -z "${pid}" ]] && return 0
  kill -0 "${pid}" 2>/dev/null || return 0

  echo "[信息] 正在停止 ${name}（PID=${pid}）……"
  kill -INT "${pid}" 2>/dev/null || true

  for _ in $(seq 1 50); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 0.2
  done

  echo "[警告] ${name} 未在 10 秒内退出，发送 TERM。"
  kill -TERM "${pid}" 2>/dev/null || true
  sleep 1

  if kill -0 "${pid}" 2>/dev/null; then
    echo "[警告] ${name} 仍未退出，发送 KILL。"
    kill -KILL "${pid}" 2>/dev/null || true
  fi

  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  set +e

  if [[ -n "${COLLECTOR_PID}" ]]; then
    stop_process_gracefully "${COLLECTOR_PID}" "RGB 采集节点"
  fi

  if [[ "${BAG_STOPPED}" != "1" && -n "${BAG_PID}" ]]; then
    stop_process_gracefully "${BAG_PID}" "rosbag"
  fi

  if [[ -n "${CAMERA_PID}" ]]; then
    stop_process_gracefully "${CAMERA_PID}" "Aurora930 相机"
  fi
}
trap cleanup EXIT INT TERM

# ROS/TROS setup.bash 可能读取未定义变量，source 期间临时关闭 nounset。
set +u
source /opt/tros/humble/setup.bash
if [[ -f "${WORKSPACE}/install/setup.bash" ]]; then
  source "${WORKSPACE}/install/setup.bash"
fi
set -u

if [[ ! -f "${COLLECTOR}" ]]; then
  echo "[错误] 找不到采集脚本：${COLLECTOR}"
  exit 1
fi

if [[ -e "${RAW_DIR}" || -e "${BAG_DIR}" ]]; then
  echo "[错误] RUN_NAME 已经存在，拒绝覆盖旧数据：${RUN_NAME}"
  [[ -e "${RAW_DIR}" ]] && echo "       图片目录：${RAW_DIR}"
  [[ -e "${BAG_DIR}" ]] && echo "       Bag 目录：${BAG_DIR}"
  echo
  echo "请换一个新的 RUN_NAME，例如："
  echo "RUN_NAME=run_002_speed_015_center \\"
  echo "SAVE_FPS=3.0 MAX_IMAGES=100 RECORD_BAG=1 \\"
  echo "./run_person_board_collection_v2.sh"
  exit 1
fi

mkdir -p "${RAW_DIR}" "${DATA_ROOT}/bags" "${LOG_DIR}"

topic_exists() {
  ros2 topic list 2>/dev/null | grep -Fxq "$1"
}

find_first_topic() {
  local pattern="$1"
  ros2 topic list 2>/dev/null | grep -Ei "${pattern}" | head -n 1 || true
}

if [[ -z "${RGB_TOPIC}" ]]; then
  for candidate in \
    /aurora/rgb/image_raw \
    /rgb/image_raw \
    /aurora930/rgb/image_raw \
    /camera/rgb/image_raw \
    /camera/color/image_raw
  do
    if topic_exists "${candidate}"; then
      RGB_TOPIC="${candidate}"
      break
    fi
  done
fi

if [[ -z "${RGB_TOPIC}" ]]; then
  echo "[信息] 当前未发现 RGB 话题，启动深度相机："
  echo "       ros2 launch ${CAMERA_PACKAGE} ${CAMERA_LAUNCH}"

  ros2 launch "${CAMERA_PACKAGE}" "${CAMERA_LAUNCH}" \
    >"${CAMERA_LOG}" 2>&1 &
  CAMERA_PID=$!

  echo "[信息] 等待相机话题，最长 25 秒……"
  for _ in $(seq 1 50); do
    sleep 0.5

    RGB_TOPIC="$(
      find_first_topic \
        '(/aurora)?/(rgb|color)/.*image.*raw|/(rgb|color)/image_raw'
    )"

    [[ -n "${RGB_TOPIC}" ]] && break

    if ! kill -0 "${CAMERA_PID}" 2>/dev/null; then
      echo "[错误] 相机 launch 已退出。日志：${CAMERA_LOG}"
      tail -n 100 "${CAMERA_LOG}" || true
      exit 1
    fi
  done
fi

if [[ -z "${RGB_TOPIC}" ]]; then
  echo "[错误] 未找到深度相机 RGB 图像话题。"
  ros2 topic list | grep -Ei 'rgb|color|depth|image|camera' || true
  exit 1
fi

if [[ -z "${DEPTH_TOPIC}" ]]; then
  for candidate in \
    /aurora/depth/image_raw \
    /depth/image_raw \
    /aurora930/depth/image_raw \
    /camera/depth/image_raw
  do
    if topic_exists "${candidate}"; then
      DEPTH_TOPIC="${candidate}"
      break
    fi
  done

  if [[ -z "${DEPTH_TOPIC}" ]]; then
    DEPTH_TOPIC="$(
      find_first_topic \
        '(/aurora)?/depth/.*image.*raw|/depth/image_raw'
    )"
  fi
fi

if [[ -z "${RGB_INFO_TOPIC}" ]]; then
  for candidate in \
    /aurora/rgb/camera_info \
    /rgb/camera_info \
    /aurora930/rgb/camera_info \
    /camera/rgb/camera_info \
    /camera/color/camera_info
  do
    if topic_exists "${candidate}"; then
      RGB_INFO_TOPIC="${candidate}"
      break
    fi
  done
fi

echo
echo "================ 本轮采集配置 ================"
echo "运行名称：${RUN_NAME}"
echo "RGB 话题：${RGB_TOPIC}"
echo "Depth 话题：${DEPTH_TOPIC:-未发现}"
echo "CameraInfo：${RGB_INFO_TOPIC:-未发现}"
echo "图片目录：${RAW_DIR}"
echo "采样频率：${SAVE_FPS} FPS"
echo "最大图片：${MAX_IMAGES}"
echo "记录 rosbag：${RECORD_BAG}"
echo "=============================================="
echo

echo "[信息] RGB 话题信息："
ros2 topic info "${RGB_TOPIC}"

if [[ "${RECORD_BAG}" == "1" ]]; then
  BAG_TOPICS=("${RGB_TOPIC}")
  [[ -n "${DEPTH_TOPIC}" ]] && BAG_TOPICS+=("${DEPTH_TOPIC}")
  [[ -n "${RGB_INFO_TOPIC}" ]] && BAG_TOPICS+=("${RGB_INFO_TOPIC}")

  echo "[信息] 启动 rosbag：${BAG_DIR}"
  ros2 bag record -o "${BAG_DIR}" "${BAG_TOPICS[@]}" \
    >"${BAG_LOG}" 2>&1 &
  BAG_PID=$!

  sleep 2

  if ! kill -0 "${BAG_PID}" 2>/dev/null; then
    echo "[错误] rosbag 启动后立即退出。日志如下："
    tail -n 120 "${BAG_LOG}" || true
    exit 1
  fi

  echo "[正常] rosbag 正在运行，PID=${BAG_PID}"
fi

echo "[信息] 启动 RGB 图片采集节点。"
python3 "${COLLECTOR}" --ros-args \
  -p image_topic:="${RGB_TOPIC}" \
  -p output_dir:="${RAW_DIR}" \
  -p save_fps:="${SAVE_FPS}" \
  -p max_images:="${MAX_IMAGES}" \
  -p jpeg_quality:="${JPEG_QUALITY}" \
  -p warmup_sec:="${WARMUP_SEC}" &
COLLECTOR_PID=$!

while kill -0 "${COLLECTOR_PID}" 2>/dev/null; do
  if [[ "${RECORD_BAG}" == "1" ]] &&
     ! kill -0 "${BAG_PID}" 2>/dev/null; then
    echo
    echo "[错误] 采集过程中 rosbag 异常退出，立即停止图片采集。"
    echo "===== rosbag 日志 ====="
    tail -n 120 "${BAG_LOG}" || true

    stop_process_gracefully "${COLLECTOR_PID}" "RGB 采集节点"
    COLLECTOR_PID=""
    exit 1
  fi

  sleep 1
done

set +e
wait "${COLLECTOR_PID}"
COLLECTOR_RC=$?
set -e
COLLECTOR_PID=""

if [[ "${COLLECTOR_RC}" -ne 0 ]]; then
  echo "[错误] RGB 图片采集节点退出码：${COLLECTOR_RC}"
  exit "${COLLECTOR_RC}"
fi

if [[ "${RECORD_BAG}" == "1" ]]; then
  stop_process_gracefully "${BAG_PID}" "rosbag"
  BAG_STOPPED=1
  BAG_PID=""
fi

IMAGE_COUNT="$(
  find "${RAW_DIR}" -maxdepth 1 -type f -name '*.jpg' | wc -l
)"

echo
echo "================ 采集完成 ================"
echo "图片目录：${RAW_DIR}"
echo "图片数量：${IMAGE_COUNT}"

if [[ "${RECORD_BAG}" == "1" ]]; then
  echo "Bag 目录：${BAG_DIR}"
  echo
  echo "===== ROS bag 信息 ====="
  ros2 bag info "${BAG_DIR}" || {
    echo "[错误] ros2 bag info 读取失败。"
    exit 1
  }
fi

echo "=========================================="
