#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
cd /root/intelligent_car_ws

if [ ! -f install/setup.bash ]; then
  echo "[FAIL] /root/intelligent_car_ws/install/setup.bash 不存在，请先完成 colcon build。"
  exit 1
fi

source install/setup.bash

SCRIPT="/root/intelligent_car_ws/src/roadblock_localization/tools/roadblock_acceptance_test.py"

if [ ! -f "$SCRIPT" ]; then
  echo "[FAIL] 找不到 $SCRIPT"
  exit 1
fi

exec python3 "$SCRIPT" "$@"
