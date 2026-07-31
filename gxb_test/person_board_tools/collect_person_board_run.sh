#!/usr/bin/env bash
set -Eeuo pipefail

# 单轮人形立牌采集包装脚本
#
# 用法：
#   ./collect_person_board_run.sh board_2 1 center
#
# 前提：
#   1. 底盘节点已在另一个终端启动：
#      ros2 launch origincar_base base_serial.launch.py akmcar:=false
#   2. 前方至少 3.5 m 安全空间，旁边有人可断电/急停。
#
# 可覆盖参数：
#   SPEED=0.15 DURATION=19.5 SAVE_FPS=3.0 MAX_IMAGES=62 RECORD_BAG=0 \
#   ./collect_person_board_run.sh board_2 1 center

if [[ $# -ne 3 ]]; then
  echo "用法：$0 <board_id> <run_no> <position>"
  echo "示例：$0 board_2 1 center"
  exit 2
fi

BOARD_ID="$1"
RUN_NO="$2"
POSITION="$3"

WORKSPACE="${WORKSPACE:-/root/intelligent_car_ws}"
TOOLS_DIR="${TOOLS_DIR:-${WORKSPACE}/src/gxb_test/person_board_tools}"
COLLECT_SCRIPT="${COLLECT_SCRIPT:-${TOOLS_DIR}/run_person_board_collection_v2.sh}"
DRIVE_SCRIPT="${DRIVE_SCRIPT:-${WORKSPACE}/src/gxb_test/safe_constant_speed_drive.py}"

SPEED="${SPEED:-0.15}"
DURATION="${DURATION:-19.5}"
DRIVE_DELAY="${DRIVE_DELAY:-1.0}"
RAMP="${RAMP:-0.5}"
STOP_HOLD="${STOP_HOLD:-5}"

SAVE_FPS="${SAVE_FPS:-3.0}"
MAX_IMAGES="${MAX_IMAGES:-100}"
WARMUP_SEC="${WARMUP_SEC:-0.5}"
RECORD_BAG="${RECORD_BAG:-0}"

RUN_NO_PADDED="$(printf "%03d" "${RUN_NO}")"
RUN_NAME="${BOARD_ID}_run_${RUN_NO_PADDED}_${POSITION}"

DATA_ROOT="${WORKSPACE}/datasets/person_board"
RAW_DIR="${DATA_ROOT}/raw/${RUN_NAME}"
BAG_DIR="${DATA_ROOT}/bags/${RUN_NAME}"
LOG_DIR="${DATA_ROOT}/logs"
WRAPPER_LOG="${LOG_DIR}/${RUN_NAME}_wrapper.log"

mkdir -p "${LOG_DIR}"

set +u
source /opt/tros/humble/setup.bash
source "${WORKSPACE}/install/setup.bash"
set -u

if [[ ! -x "${COLLECT_SCRIPT}" ]]; then
  echo "[错误] 采集脚本不存在或不可执行：${COLLECT_SCRIPT}"
  exit 1
fi

if [[ ! -f "${DRIVE_SCRIPT}" ]]; then
  echo "[错误] 直行脚本不存在：${DRIVE_SCRIPT}"
  exit 1
fi

if [[ -e "${RAW_DIR}" || -e "${BAG_DIR}" ]]; then
  echo "[错误] 本轮名称已存在：${RUN_NAME}"
  echo "请增加 run_no，例如："
  echo "  $0 ${BOARD_ID} $((10#${RUN_NO} + 1)) ${POSITION}"
  exit 1
fi

if ! ros2 node list 2>/dev/null | grep -Fxq "/origincar_base"; then
  echo "[错误] 未发现 /origincar_base。"
  echo "请先在另一个终端启动："
  echo "  ros2 launch origincar_base base_serial.launch.py akmcar:=false"
  exit 1
fi

SUB_COUNT="$(
  ros2 topic info /cmd_vel 2>/dev/null \
  | awk -F': ' '/Subscription count/ {print $2}' \
  | tail -n 1
)"
SUB_COUNT="${SUB_COUNT:-0}"

if [[ "${SUB_COUNT}" -lt 1 ]]; then
  echo "[错误] /cmd_vel 没有订阅者，拒绝动车。"
  exit 1
fi

echo
echo "================ 单轮采集确认 ================"
echo "立牌编号：${BOARD_ID}"
echo "运行名称：${RUN_NAME}"
echo "位置标签：${POSITION}"
echo "速度：${SPEED} m/s"
echo "行驶时间：${DURATION} s"
echo "图片频率：${SAVE_FPS} FPS"
echo "图片上限：${MAX_IMAGES}"
echo "录制 rosbag：${RECORD_BAG}"
echo "输出目录：${RAW_DIR}"
echo "=============================================="
echo
echo "说明：本脚本会先启动相机采集，再自动让小车直行。"
echo "由于启动边界，可能包含极少量静止帧，PC 端复核时直接 SKIP。"
echo
read -r -p "确认前方安全并准备好急停后，输入 YES： " CONFIRM

if [[ "${CONFIRM}" != "YES" ]]; then
  echo "已取消。"
  exit 0
fi

COLLECT_PID=""
cleanup() {
  set +e
  if [[ -n "${COLLECT_PID}" ]] && kill -0 "${COLLECT_PID}" 2>/dev/null; then
    kill -INT "${COLLECT_PID}" 2>/dev/null || true
    wait "${COLLECT_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[信息] 启动采集：${RUN_NAME}"
(
  cd "${TOOLS_DIR}"
  RUN_NAME="${RUN_NAME}" \
  SAVE_FPS="${SAVE_FPS}" \
  MAX_IMAGES="${MAX_IMAGES}" \
  WARMUP_SEC="${WARMUP_SEC}" \
  RECORD_BAG="${RECORD_BAG}" \
  "${COLLECT_SCRIPT}"
) > >(tee -a "${WRAPPER_LOG}") 2>&1 &
COLLECT_PID=$!

# 等待采集进程创建图片目录并存下第一张图。
echo "[信息] 等待相机与采集节点就绪……"
READY=0
for _ in $(seq 1 60); do
  if ! kill -0 "${COLLECT_PID}" 2>/dev/null; then
    echo "[错误] 采集程序提前退出，请查看：${WRAPPER_LOG}"
    exit 1
  fi

  COUNT=0

  if [[ -d "${RAW_DIR}" ]]; then
    COUNT="$(
      find "${RAW_DIR}"         -maxdepth 1         -type f         -iname '*.jpg'         2>/dev/null       | wc -l       || true
    )"
  fi

  COUNT="${COUNT//[[:space:]]/}"
  COUNT="${COUNT:-0}"

  if [[ "${COUNT}" -ge 1 ]]; then
    READY=1
    break
  fi

  sleep 0.5
done

if [[ "${READY}" -ne 1 ]]; then
  echo "[错误] 30 秒内没有保存首张图片，停止本轮。"
  exit 1
fi

echo "[正常] 图像采集已开始，即将直行。"

python3 "${DRIVE_SCRIPT}" \
  --topic /cmd_vel \
  --speed "${SPEED}" \
  --duration "${DURATION}" \
  --delay "${DRIVE_DELAY}" \
  --ramp "${RAMP}" \
  --stop-hold "${STOP_HOLD}"

echo "[信息] 小车已停止，等待采集程序收尾……"

for _ in $(seq 1 30); do
  if ! kill -0 "${COLLECT_PID}" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if kill -0 "${COLLECT_PID}" 2>/dev/null; then
  echo "[信息] 采集尚未自动结束，发送 INT。"
  kill -INT "${COLLECT_PID}" 2>/dev/null || true
fi

set +e
wait "${COLLECT_PID}"
COLLECT_RC=$?
set -e
COLLECT_PID=""

IMAGE_COUNT="$(
  find "${RAW_DIR}" -maxdepth 1 -type f -iname '*.jpg' 2>/dev/null | wc -l
)"

echo
echo "================ 本轮完成 ================"
echo "运行名称：${RUN_NAME}"
echo "图片数量：${IMAGE_COUNT}"
echo "图片目录：${RAW_DIR}"
echo "日志文件：${WRAPPER_LOG}"
echo "采集退出码：${COLLECT_RC}"
echo "=========================================="

if [[ "${IMAGE_COUNT}" -lt 40 ]]; then
  echo "[警告] 图片少于 40 张，建议检查日志后补采。"
  exit 1
fi
