# OriginCar ROS2 视觉链路说明

## 编译

```bash
colcon build --symlink-install
```

## 底盘控制

底盘启动保持原流程不变：

```bash
ros2 launch origincar_base origincar_bringup.launch.py
```

键盘控制：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## 单相机入口

视觉链路统一采用“单相机入口”原则，只有下面这个 launch 允许启动相机和编解码链路：

```bash
ros2 launch vision_camera camera_bridge.launch.py device:=/dev/video0
```

这个入口会启动：

- `hobot_usb_cam`
- `hobot_codec_decode`
- `vision_camera/hbm_image_bridge`

桥接后提供的话题：

- `/hbmem_img`
- `/image_out`
- `/image_out/compressed`

## 启动顺序

```bash
ros2 launch vision_camera camera_bridge.launch.py device:=/dev/video0
ros2 launch yolov8_test_mplus0 yolov8_detect.launch.py
ros2 launch vision_birdview bird_view.launch.py
```

## 约束说明

以下模块都不允许再启动相机，也不允许直接在正式节点里打开本地摄像头设备：

- YOLO
- Bird View
- 二维码识别
- Web 显示

这些模块只能订阅已有图像话题：

- `/image_out`
- `/image_out/compressed`
- `/hbmem_img`
