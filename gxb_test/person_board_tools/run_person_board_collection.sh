#!/usr/bin/env bash
set -Eeuo pipefail

# ===================== 可通过环境变量覆盖的参数 =====================
WORKSPACE="${WORKSPACE:-/root/intelligent_car_ws}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/datasets/person_board}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"

SAVE_FPS="${SAVE_FPS:-2.0}"
MAX_IMAGES="${MAX_IMAGES:-200}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
WARMUP_SEC="${WARMUP_SEC:-2.0}"
RECORD_BAG="${RECORD_BAG:-1}"

# rclpy 参数类型严格区分 INTEGER 和 DOUBLE。
# 用户输入 SAVE_FPS=3 时，自动转换成 3.0。
if [[ "${SAVE_FPS}" =~ ^[0-9]+$ ]]; then
  SAVE_FPS="${SAVE_FPS}.0"
fi

if [[ "${WARMUP_SEC}" =~ ^[0-9]+$ ]]; then
  WARMUP_SEC="${WARMUP_SEC}.0"
fi

# 待验证：如实际包名或 launch 文件不同，只改这两个变量。
CAMERA_PACKAGE="${CAMERA_PACKAGE:-deptrum-ros-driver-aurora930}"
CAMERA_LAUNCH="${CAMERA_LAUNCH:-aurora930_launch.py}"

# 已知实际话题时可直接传入；留空则自动探测。
RGB_TOPIC="${RGB_TOPIC:-}"
DEPTH_TOPIC="${DEPTH_TOPIC:-}"
RGB_INFO_TOPIC="${RGB_INFO_TOPIC:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR="${SCRIPT_DIR}/collect_rgb_dataset.py"

RAW_DIR="${DATA_ROOT}/raw/${RUN_NAME}"
BAG_DIR="${DATA_ROOT}/bags/${RUN_NAME}"
LOG_DIR="${DATA_ROOT}/logs"
CAMERA_LOG="${LOG_DIR}/${RUN_NAME}_camera.log"

CAMERA_PID=""
BAG_PID=""

mkdir -p "${RAW_DIR}" "${DATA_ROOT}/bags" "${LOG_DIR}"

cleanup() {
  set +e
  if [[ -n "${BAG_PID}" ]] && kill -0 "${BAG_PID}" 2>/dev/null; then
    kill -INT "${BAG_PID}" 2>/dev/null
    wait "${BAG_PID}" 2>/dev/null
  fi
  if [[ -n "${CAMERA_PID}" ]] && kill -0 "${CAMERA_PID}" 2>/dev/null; then
    kill -INT "${CAMERA_PID}" 2>/dev/null
    wait "${CAMERA_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

# ROS/TROS 的 setup.bash 可能读取尚未定义的环境变量，
# 因此 source 期间临时关闭 nounset，加载完成后再恢复。
set +u
# ROS/TROS 的 setup.bash 可能访问未定义变量，
# source 期间临时关闭 nounset。
set +u
source /opt/tros/humble/setup.bash
if [[ -f "${WORKSPACE}/install/setup.bash" ]]; then
  source "${WORKSPACE}/install/setup.bash"
fi
set -u
set -u

if [[ ! -f "${COLLECTOR}" ]]; then
  echo "[错误] 找不到采集脚本：${COLLECTOR}"
  exit 1
fi

topic_exists() {
  ros2 topic list 2>/dev/null | grep -Fxq "$1"
}

find_first_topic() {
  local pattern="$1"
  ros2 topic list 2>/dev/null | grep -Ei "${pattern}" | head -n 1 || true
}

# 如果没有指定 RGB_TOPIC，且当前没有常见 RGB 话题，则自动启动相机。
if [[ -z "${RGB_TOPIC}" ]]; then
  for candidate in \
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
    RGB_TOPIC="$(find_first_topic '(/rgb/.*/?image|/rgb/image|/color/.*/?image|/color/image).*raw')"
    [[ -n "${RGB_TOPIC}" ]] && break
    if ! kill -0 "${CAMERA_PID}" 2>/dev/null; then
      echo "[错误] 相机 launch 已退出。日志：${CAMERA_LOG}"
      tail -n 80 "${CAMERA_LOG}" || true
      exit 1
    fi
  done
fi

if [[ -z "${RGB_TOPIC}" ]]; then
  echo "[错误] 未找到深度相机 RGB 图像话题。"
  echo "当前相关话题："
  ros2 topic list | grep -Ei 'rgb|color|depth|image|camera' || true
  echo "相机日志：${CAMERA_LOG}"
  exit 1
fi

if [[ -z "${DEPTH_TOPIC}" ]]; then
  for candidate in \
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
    DEPTH_TOPIC="$(find_first_topic '/depth/.*/?image.*raw|/depth/image_raw')"
  fi
fi

if [[ -z "${RGB_INFO_TOPIC}" ]]; then
  for candidate in \
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
ros2 topic info "${RGB_TOPIC}" || true

if [[ "${RECORD_BAG}" == "1" ]]; then
  BAG_TOPICS=("${RGB_TOPIC}")
  [[ -n "${DEPTH_TOPIC}" ]] && BAG_TOPICS+=("${DEPTH_TOPIC}")
  [[ -n "${RGB_INFO_TOPIC}" ]] && BAG_TOPICS+=("${RGB_INFO_TOPIC}")

  echo "[信息] 启动 rosbag 录制：${BAG_DIR}"
  ros2 bag record -o "${BAG_DIR}" "${BAG_TOPICS[@]}" \
    >"${LOG_DIR}/${RUN_NAME}_bag.log" 2>&1 &
  BAG_PID=$!
fi

echo "[信息] 开始保存 RGB 数据。达到最大图片数后自动结束；也可按 Ctrl+C。"
python3 "${COLLECTOR}" --ros-args \
  -p image_topic:="${RGB_TOPIC}" \
  -p output_dir:="${RAW_DIR}" \
  -p save_fps:="${SAVE_FPS}" \
  -p max_images:="${MAX_IMAGES}" \
  -p jpeg_quality:="${JPEG_QUALITY}" \
  -p warmup_sec:="${WARMUP_SEC}"

echo
echo "[完成] 本轮数据已保存："
echo "图片：${RAW_DIR}"
if [[ "${RECORD_BAG}" == "1" ]]; then
  echo "Bag：${BAG_DIR}"
fi
echo
echo "图片数量：$(find "${RAW_DIR}" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
