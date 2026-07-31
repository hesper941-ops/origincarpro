人形立牌数据采集与 YOLOv8n 训练工具包
=======================================

一、将工具包放到 RDK X5
----------------------

建议目录：

/root/intelligent_car_ws/src/gxb_test/person_board_tools/

包含：
- collect_rgb_dataset.py
- run_person_board_collection.sh
- train_person_board_yolov8n.sh

在 X5 上执行：

chmod +x collect_rgb_dataset.py
chmod +x run_person_board_collection.sh

二、最省事的数据采集命令
------------------------

进入工具目录后，一轮采集只需要输入：

RUN_NAME=run_001_slow_center \
SAVE_FPS=2 \
MAX_IMAGES=200 \
RECORD_BAG=1 \
./run_person_board_collection.sh

脚本会自动：
1. source TROS 和工作空间；
2. 检查深度相机 RGB 话题；
3. 必要时启动 Aurora930；
4. 自动探测 RGB/Depth/CameraInfo 话题；
5. 同时保存 JPEG 和 rosbag；
6. 达到最大图片数后自动退出并清理进程。

如实际相机包名或 launch 名不同：

CAMERA_PACKAGE=实际包名 \
CAMERA_LAUNCH=实际launch.py \
RUN_NAME=run_001 \
./run_person_board_collection.sh

如已经知道真实话题：

RGB_TOPIC=/实际/rgb/image_raw \
DEPTH_TOPIC=/实际/depth/image_raw \
RUN_NAME=run_001 \
./run_person_board_collection.sh

三、建议采集轮次
----------------

正样本：
run_001_slow_center
run_002_slow_left
run_003_slow_right
run_004_normal_center
run_005_normal_left
run_006_normal_right
run_007_bumpy
run_008_low_light
run_009_partial_occlusion
run_010_near_turn

负样本：
run_011_negative_straight
run_012_negative_turn
run_013_negative_background
run_014_negative_people

每轮建议保存 80～200 张；相邻帧不要太密，2～3 FPS 通常够用。

四、标注
--------

使用 CVAT 创建 Detection 任务，类别仅设：

person_board

标注要求：
- 框住整块矩形立牌；
- 不要只框护士、患者、轮椅或人物；
- 严重模糊到肉眼无法确认的图片删除；
- 没有立牌的图片保留为负样本，不画框；
- 导出为 Ultralytics YOLO Detection。

五、数据集划分
--------------

必须按“运行轮次”划分，不能把同一段运行的相邻帧随机拆到 train/val/test。

示例：
run_001～run_010 -> train
run_011～run_012 -> val
run_013～run_014 -> test

最终结构：

person_board_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── person_board.yaml

person_board.yaml 示例：

path: /datas/gxb/person_board_dataset
train: images/train
val: images/val
test: images/test

names:
  0: person_board

六、训练
--------

训练服务器中先安装环境：

conda create -n person_board_yolo python=3.10 -y
conda activate person_board_yolo
python -m pip install --upgrade pip
pip install ultralytics

然后执行：

chmod +x train_person_board_yolov8n.sh

DATASET_YAML=/datas/gxb/person_board_dataset/person_board.yaml \
PROJECT_DIR=/datas/gxb/person_board_runs \
RUN_NAME=person_board_yolov8n_v1 \
BATCH=16 \
./train_person_board_yolov8n.sh

脚本会自动：
1. 输出 PyTorch、CUDA、Ultralytics 版本；
2. 先训练 3 epoch 做数据冒烟测试；
3. 冒烟测试成功后正式训练 100 epoch；
4. 用 test 集验证；
5. 输出 best.pt 路径。

七、当前待验证项
----------------

- Aurora930 ROS2 包名是否正是 deptrum-ros-driver-aurora930；
- launch 文件是否正是 aurora930_launch.py；
- 实际 RGB/Depth 话题名称；
- RGB 分辨率、编码和实际帧率；
- cv_bridge 是否已安装；
- 深度相机是否已被其他进程启动。

如果自动探测失败，运行：

source /opt/tros/humble/setup.bash
source /root/intelligent_car_ws/install/setup.bash
ros2 topic list | grep -Ei "rgb|color|depth|image|camera"

把输出用于设置 RGB_TOPIC 和 DEPTH_TOPIC。
