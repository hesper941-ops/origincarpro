#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
cd /root/intelligent_car_ws
source install/setup.bash
python3 /root/intelligent_car_ws/src/roadblock_localization/tools/cone_9point_validate.py "$@"
