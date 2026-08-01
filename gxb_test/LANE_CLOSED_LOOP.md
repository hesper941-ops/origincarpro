# 黄色通道视觉闭环巡线

闭环链路如下：

```text
lane_perception_pipeline.py
  -> /gxb_test/pipeline/centerline_path + /gxb_test/pipeline/status
5_lane_path_controller.py
  -> /gxb_test/lane_control_cmd (JSON String)
lane_control_gate.py
  -> /cmd_vel (Twist)
origincar_base
```

控制器不导入 `Twist`，也不发布 `/cmd_vel`。只有安全 Gate 可以发布非零
`/cmd_vel`。中间控制 JSON 的稳定接口版本为 `gxb_lane_control_v1`，包含
`linear_x`、`angular_z`、`ready`、`quality`、`mode`、`lateral_error`、
`heading_error` 和 `timestamp`，并附带路径 freshness、置信度和诊断字段。

## 安全策略

Gate 默认 `motion_enabled=false`。此时所有输入即使完全有效也只发布零速，适合
静态/零输出 smoke。只有显式使用 `start --allow-motion` 才具备放行可能；启动后
仍需 pipeline、controller、Path 和 `/odom` 全部 fresh，且连续收到 4 个安全的
新 controller 帧。

允许模式：

- `green_dual_inner_edge`：RUNNING；
- `green_yellow_hybrid`：DEGRADED；
- `single_green_width_offset`：DEGRADED/recovery。

`single_boundary_normal_offset`、`invalid` 和未知模式全部禁止。controller BLOCKED、
pipeline/path/controller/feedback timeout、非法 JSON、非有限命令、倒车请求、反馈
超速、stop、estop 或异常都会立即输出 `(0, 0)`，安全停车不经过 slew。安全状态
下角速度才使用 slew，避免单帧从正最大值跳到负最大值。

初始限幅为：

```text
normal linear <= 0.030 m/s
degraded/recovery linear <= 0.018 m/s
abs(angular) <= 0.150 rad/s
```

线速度还会根据横向误差、航向误差、角速度和 controller confidence 自动降低。

## 一键命令

```bash
# 默认安全零输出模式，不会放行非零速度
bash gxb_test/tools/lane_closed_loop.sh start

# 仅在首次完整实车测试现场准备完成后使用
bash gxb_test/tools/lane_closed_loop.sh start --allow-motion

bash gxb_test/tools/lane_closed_loop.sh status
bash gxb_test/tools/lane_closed_loop.sh tail
bash gxb_test/tools/lane_closed_loop.sh logs
bash gxb_test/tools/lane_closed_loop.sh report
bash gxb_test/tools/lane_closed_loop.sh pack
bash gxb_test/tools/lane_closed_loop.sh stop
bash gxb_test/tools/lane_closed_loop.sh estop
```

`stop` 先让 Gate 锁止，再以 20 Hz 连续发布至少 2 秒零速，随后依次停止 gate、
controller、perception，最后停止 base，并检查 `/cmd_vel` 发布者和
`/dev/ttyACM0`。`estop` 使用独立零速发布器持续至少 5 秒，同时锁止并关闭 Gate。

## 日志与报告

每次运行写入：

```text
/root/intelligent_car_ws/test_logs/lane_closed_loop_<timestamp>/
```

主要文件包括 `base.log`、`perception.log`、`controller.log`、`gate.log`、
`feedback.log`、`cleanup.log`、`summary.json` 和 `summary.csv`。
`feedback.log` 是逐行 JSON，包含 controller/gate 命令、模式、路径质量、DP、FPS、
odom、robotvel、watchdog age 和 Gate 原因。`report` 汇总运行时间、模式和状态分布、
停车原因、误差/速度/路径/DP/FPS/反馈频率统计以及 timeout/exception 数量。
