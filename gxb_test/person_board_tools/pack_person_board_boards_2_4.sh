#!/usr/bin/env bash
set -Eeuo pipefail

RAW_ROOT="${RAW_ROOT:-/root/intelligent_car_ws/datasets/person_board/raw}"
EXPORT_ROOT="${EXPORT_ROOT:-/root/intelligent_car_ws/datasets/person_board/export}"
STAMP="$(date +%Y%m%d_%H%M%S)"
PACKAGE_NAME="person_board_boards_2_4_${STAMP}"
PACKAGE_DIR="${EXPORT_ROOT}/${PACKAGE_NAME}"
ZIP_PATH="${EXPORT_ROOT}/${PACKAGE_NAME}.zip"
MANIFEST="${PACKAGE_DIR}/manifest.csv"
README="${PACKAGE_DIR}/README.txt"

RUNS=(
  "board_2_run_002_center"
  "board_2_run_003_left"
  "board_2_run_004_right"
  "board_3_run_001_center"
  "board_3_run_002_left"
  "board_3_run_003_right"
  "board_4_run_002_center"
  "board_4_run_004_left"
  "board_4_run_005_right"
)

echo "========== 预检查 =========="
echo "原始目录：${RAW_ROOT}"
echo "导出目录：${EXPORT_ROOT}"
echo

missing=0
for run in "${RUNS[@]}"; do
  src="${RAW_ROOT}/${run}"
  if [[ ! -d "${src}" ]]; then
    echo "[缺失] ${src}"
    missing=1
    continue
  fi

  count="$(
    find "${src}" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      2>/dev/null | wc -l
  )"
  count="${count//[[:space:]]/}"
  count="${count:-0}"

  if [[ "${count}" -eq 0 ]]; then
    echo "[异常] ${run}: 0 张图片"
    missing=1
  else
    printf "[正常] %-32s %4d 张\n" "${run}" "${count}"
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo
  echo "[停止] 存在缺失目录或空目录，没有创建压缩包。"
  exit 1
fi

mkdir -p "${PACKAGE_DIR}"

echo "board_id,run_name,image_count,source_path" > "${MANIFEST}"

total=0

echo
echo "========== 整理图片 =========="

for run in "${RUNS[@]}"; do
  src="${RAW_ROOT}/${run}"
  dst="${PACKAGE_DIR}/${run}"
  mkdir -p "${dst}"

  # 只复制本地标注所需的图片，不带 rosbag 或失败日志。
  while IFS= read -r -d '' image; do
    cp -a "${image}" "${dst}/"
  done < <(
    find "${src}" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      -print0 | sort -z
  )

  count="$(
    find "${dst}" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      | wc -l
  )"
  count="${count//[[:space:]]/}"
  count="${count:-0}"

  board_id="${run%%_run_*}"
  echo "${board_id},${run},${count},${src}" >> "${MANIFEST}"
  total=$((total + count))

  printf "[完成] %-32s %4d 张\n" "${run}" "${count}"
done

cat > "${README}" <<EOF
人形立牌 YOLO 原始图片包
生成时间：$(date '+%F %T %z')
类别定义：0 = person_board
运行数量：${#RUNS[@]}
图片总数：${total}

目录说明：
- 每个 board_*_run_* 目录代表一整次独立采集；
- 不要随机打乱后再划分训练集；
- 最终 train / val / test 应按完整 run 划分；
- 四种立牌内部内容不同，但检测类别统一为 person_board；
- 标注时框住完整立牌外框，不只框内部人物图案。

本包包含：
- board_2：3 轮
- board_3：3 轮
- board_4：3 轮
EOF

echo
echo "========== 创建 ZIP =========="

python3 - "${PACKAGE_DIR}" "${ZIP_PATH}" <<'PY'
from pathlib import Path
import sys
import zipfile

package_dir = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()

with zipfile.ZipFile(
    zip_path,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
    allowZip64=True,
) as archive:
    for path in sorted(package_dir.rglob("*")):
        if path.is_file():
            archive.write(
                path,
                arcname=(Path(package_dir.name) / path.relative_to(package_dir)).as_posix(),
            )

print(zip_path)
PY

sha256sum "${ZIP_PATH}" > "${ZIP_PATH}.sha256"

echo
echo "========== 导出完成 =========="
echo "运行数量：${#RUNS[@]}"
echo "图片总数：${total}"
echo "整理目录：${PACKAGE_DIR}"
echo "ZIP 文件：${ZIP_PATH}"
echo "校验文件：${ZIP_PATH}.sha256"
echo
ls -lh "${ZIP_PATH}" "${ZIP_PATH}.sha256"
echo
echo "可用以下命令再次查看："
echo "  cat '${MANIFEST}'"
echo "  unzip -l '${ZIP_PATH}' | tail"
