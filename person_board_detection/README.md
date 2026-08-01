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

## 阶段 2：Mock 识别与 HDMI 常驻显示

阶段 2 复用阶段 1 的固定三张裁剪图和 `capture_batch` 协议，不改变
YOLO、动态调度、稳定判断、三帧采集和原子替换逻辑。链路由三个独立进程组成：

```text
person_board_detector
  -> /person_board/capture_batch
person_board_mock_llm_worker
  -> /person_board/llm_status
  -> /person_board/llm_result
person_board_display
  -> HDMI/X11
  -> /person_board/display_status
```

Mock worker 使用容量为 1 的队列和单独后台线程。ROS 回调只解析、去重和入队；
后台线程校验 manifest、固定文件名、三张图片、路径归属和图片可解码性，再模拟
耗时并发布结果。状态和最终结果均使用 reliable + transient local QoS。当前阶段
不发送 HTTP 请求、不读取 API key，也没有接入任何真实大模型。

HDMI 节点的 Tk `mainloop` 固定运行在主线程，ROS executor 运行在后台线程；ROS
回调仅向线程安全队列写入事件，Tk 通过 `after()` 更新控件。默认全屏显示，Esc
退出全屏，F11 切换全屏，允许键盘退出时 q 关闭节点。显示内容变化会发布
transient local 的 `/person_board/display_status`。

识别成功后，HDMI 会一直保持当前结果；后续 `IDLE` 心跳不会自动清屏。新任务
开始时会显示“正在识别……”，成功后由新结果覆盖旧结果，失败时显示“识别失败，
请重试”。显示节点按 `request_id` 跟踪任务，已被新任务取代的旧请求不能用迟到
结果覆盖当前画面。

手动清屏使用 `/person_board/display_control`（`std_msgs/msg/String`）：

```bash
ros2 topic pub --once \
  /person_board/display_control \
  std_msgs/msg/String \
  "{data: '{\"command\":\"clear\"}'}"
```

`clear` 只把画面恢复为“等待识别”并清除当前 `event_id/request_id`，不会关闭
display 节点、Tk 窗口或全屏模式。下一条有效 LLM 结果会正常显示并再次持续保持。

### 单独启动

Mock worker：

```bash
ros2 launch person_board_detection person_board_mock_llm.launch.py
```

HDMI 显示：

```bash
cd /root/intelligent_car_ws/src
bash person_board_detection/scripts/run_person_board_display.sh
```

显示脚本优先复用环境中的 `DISPLAY` 和可读的 `XAUTHORITY`，否则从 sunrise 的
桌面会话、`~/.Xauthority`、GDM 授权文件和图形会话进程环境中发现。它不会执行
`xhost +`，找不到有效授权文件时会输出诊断后退出。

### 一键验证

复用已有三张图片，不重新运行 YOLO：

```bash
bash person_board_detection/scripts/test_mock_llm_display.sh --mode replay
```

完整实机链路：

```bash
bash person_board_detection/scripts/test_mock_llm_display.sh \
  --mode full \
  --timeout 120 \
  --start-camera auto
```

支持参数：

```text
--mode replay|full
--timeout 120
--event-id person_board_stage2_001
--start-camera auto
--keep-camera
--no-build
--display :0
--capture-dir /root/intelligent_car_ws/runtime/person_board/latest_capture
```

日志写入：

```text
/root/intelligent_car_ws/test_logs/person_board_stage2/<时间戳>/
```

脚本只清理自己记录的进程组，并通过 `/proc/<pid>/stat` 的启动时间防止 PID
复用误杀。异常遗留时可指定某次日志的 PID 文件精确清理：

```bash
bash person_board_detection/scripts/stop_mock_llm_display_test.sh \
  --pid-file /root/intelligent_car_ws/test_logs/person_board_stage2/<时间戳>/test_processes.pid
```

RDK X5 首次使用需确认已安装 `python3-tk` 和中文字体（推荐
`Noto Sans CJK SC`）。字体不可用时显示节点会自动回退并记录实际字体。
# Stage 3: Qwen VL backend

The optional real backend selects the best of the three fixed crops and sends
only that JPEG to an OpenAI-compatible Qwen VL endpoint.  Stage 2 Mock remains
available and must not run at the same time as the Qwen worker.

Configure the private file `/root/.config/person_board_llm/env` with exported
`DASHSCOPE_API_KEY`, `PERSON_BOARD_LLM_BASE_URL`, and
`PERSON_BOARD_LLM_MODEL`.  The key is never a ROS parameter or log field.

Run the real worker with `scripts/run_person_board_qwen_vl.sh`.  Validate the
whole chain without a real key using `scripts/test_qwen_vl.sh --mode dry-run`.
