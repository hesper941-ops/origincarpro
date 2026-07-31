#!/usr/bin/env bash
set -Eeuo pipefail

# 训练服务器上使用。下面变量可通过环境变量覆盖。
DATASET_YAML="${DATASET_YAML:-/datas/gxb/person_board_dataset/person_board.yaml}"
PROJECT_DIR="${PROJECT_DIR:-/datas/gxb/person_board_runs}"
RUN_NAME="${RUN_NAME:-person_board_yolov8n_v1}"
MODEL="${MODEL:-yolov8n.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
PATIENCE="${PATIENCE:-25}"

if [[ ! -f "${DATASET_YAML}" ]]; then
  echo "[错误] 找不到数据集配置：${DATASET_YAML}"
  exit 1
fi

python3 - <<'PY'
import torch
import ultralytics
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("ultralytics:", ultralytics.__version__)
PY

mkdir -p "${PROJECT_DIR}"

echo
echo "================ 训练配置 ================"
echo "模型：${MODEL}"
echo "数据：${DATASET_YAML}"
echo "输出：${PROJECT_DIR}/${RUN_NAME}"
echo "epochs：${EPOCHS}"
echo "imgsz：${IMGSZ}"
echo "batch：${BATCH}"
echo "device：${DEVICE}"
echo "=========================================="
echo

# 先进行 3 epoch 冒烟测试，避免正式训练后才发现路径或标签错误。
yolo detect train \
  model="${MODEL}" \
  data="${DATASET_YAML}" \
  epochs=3 \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  workers="${WORKERS}" \
  project="${PROJECT_DIR}" \
  name="${RUN_NAME}_smoke" \
  exist_ok=True

echo
echo "[信息] 冒烟测试通过，开始正式训练。"
yolo detect train \
  model="${MODEL}" \
  data="${DATASET_YAML}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  workers="${WORKERS}" \
  patience="${PATIENCE}" \
  cache=True \
  close_mosaic=10 \
  project="${PROJECT_DIR}" \
  name="${RUN_NAME}" \
  exist_ok=True

BEST_PT="${PROJECT_DIR}/${RUN_NAME}/weights/best.pt"
if [[ ! -f "${BEST_PT}" ]]; then
  echo "[错误] 训练结束但未找到：${BEST_PT}"
  exit 1
fi

echo
echo "[信息] 在 test 集上验证 best.pt"
yolo detect val \
  model="${BEST_PT}" \
  data="${DATASET_YAML}" \
  split=test \
  imgsz="${IMGSZ}" \
  device="${DEVICE}" \
  project="${PROJECT_DIR}" \
  name="${RUN_NAME}_test" \
  exist_ok=True

echo
echo "[完成] 最佳模型：${BEST_PT}"
