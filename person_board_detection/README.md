# person_board_detection

阶段 1 门控检测和固定三帧裁剪可通过一个脚本完成实机验证，不需要手动打开多个终端。
测试脚本是源码目录中的普通测试资产，不注册为 ROS2 节点，也不安装到包的可执行目录。

## 一键测试

```bash
cd /root/intelligent_car_ws/src
bash person_board_detection/scripts/test_adaptive_capture.sh
```

默认事件 ID 为 `person_board_YYYYMMDD_HHMMSS`，总超时 90 秒；当 Aurora930 图像不存在时自动启动相机。固定裁剪目录为：

```text
/root/intelligent_car_ws/runtime/person_board/latest_capture
```

支持参数：

```text
--event-id person_board_001
--timeout 90
--start-camera auto
--keep-camera
--no-build
--capture-dir /root/intelligent_car_ws/runtime/person_board/latest_capture
```

每次测试日志写入：

```text
/root/intelligent_car_ws/test_logs/person_board/<时间戳>/
```

最终结论和失败原因见该目录的 `summary.txt`。目录同时包含 Git 信息、构建日志、节点信息、各 ROS 话题日志、图片尺寸、SHA256 和 manifest 副本。

脚本使用 trap 和进程组清理其自行启动的检测、话题记录及频率测量进程。相机仅在由脚本启动且未指定 `--keep-camera` 时停止，不会宽泛终止其他 ROS、道路感知或底盘进程。

如测试进程异常遗留，可按 PID 文件精确清理：

```bash
bash person_board_detection/scripts/stop_adaptive_capture_test.sh
```

也可指定某次测试的 PID 文件：

```bash
bash person_board_detection/scripts/stop_adaptive_capture_test.sh \
  --pid-file /root/intelligent_car_ws/test_logs/person_board/<时间戳>/test_processes.pid
```
