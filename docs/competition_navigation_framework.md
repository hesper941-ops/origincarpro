# Competition Navigation Framework

## Core Direction

The formal competition flow is no longer a fixed waypoint loop. Nav2 and old waypoint tools can still be kept for debugging, but the competition line is:

`semantic_map.yaml` -> `task_manager` -> `/current_goal` -> `target_tracker` -> `/track_cmd_vel` -> `cmd_vel_mux` -> `/cmd_vel`

The static map is only for visualization in RViz or CoStudio. It can show the car pose, current goal, tracks, and approximate obstacle positions, but it is not the main decision map.

## Semantic Map

`origincar_nav/config/semantic_map.yaml` is the rough task map. It stores the start point, task station, channel entry, channel points, map boundary, route order, and goal thresholds.

The coordinates are intentionally coarse defaults. They should be measured and tuned on the real field.

## Target Tracking

`target_tracker` subscribes to `/odom_combined`, `/current_goal`, and `/track_enable`. It computes distance and heading error in odometry space and publishes `/track_cmd_vel`.

It does not depend on AMCL or Nav2. If odometry or the current goal is missing, it publishes zero velocity and waits.

## Task State Machine

`task_manager` loads `semantic_map.yaml`, publishes mission state, publishes the active `/current_goal`, and enables or disables tracking. It also reads QR results and chooses the clockwise or anticlockwise channel route.

Main states:

- `TRACK_TO_TASK_STATION`
- `WAIT_QR_RESULT`
- `TRACK_TO_CHANNEL_ENTRY`
- `CHANNEL_NAV`
- `RETURN_PREPARE`
- `BIRDVIEW_RETURN`
- `FINISH`

## Obstacle Avoidance

`yolo_avoid_controller` reads YOLO detections and `/odom_combined`, checks the semantic map boundary, chooses left or right avoidance, and publishes a short local avoidance command.

It does not perform a hard-coded return-to-heading action. When avoidance ends, it publishes `/avoid_finished` once and sets `/avoid_active=false`. Then `cmd_vel_mux` returns control to `target_tracker`, which naturally reconnects to the original goal using the current odometry pose.

This matters when obstacles are close together or near the field boundary. A forced correction after one obstacle can point the car into the next obstacle or outside the map.

## Bird View Return Interface

`vision_birdview` currently provides the interface for local visual return. It publishes transformed bird-view images and placeholder return topics:

- `/bird_view/p_point_valid`
- `/bird_view/p_point_pose`
- `/bird_view/line_error`
- `/bird_view/heading_error`

Reliable P-point recognition is not implemented yet. By default, `p_point_valid=false`. Dummy output is only enabled when `enable_dummy_output=true`.

## Topic Table

Inputs:

- `/odom_combined`
- `/qrcode_detected/info_result`
- `/sign_switch`
- `/avoid_active`
- `/avoid_finished`
- `/bird_view/p_point_valid`

Outputs:

- `/current_goal`
- `/track_enable`
- `/track_cmd_vel`
- `/avoid_cmd_vel`
- `/cmd_vel`
- `/mission_state`
- `/perception_mode`

## Startup

Build:

```bash
colcon build --symlink-install --packages-select origincar_nav origincar_avoid qr_code_detection vision_birdview
```

Source:

```bash
source install/setup.bash
```

Camera bridge:

```bash
ros2 launch vision_camera camera_bridge.launch.py device:=/dev/video0
```

Competition navigation:

```bash
ros2 launch origincar_nav competition_nav.launch.py
```

YOLO avoidance and velocity mux:

```bash
ros2 launch origincar_avoid yolov8_detect.launch.py
```

Bird View:

```bash
ros2 launch vision_birdview bird_view.launch.py
```

QR detection:

```bash
ros2 launch qr_code_detection qr_code_detection.launch.py
```

## Debug Order

1. Confirm `/odom_combined` is publishing.
2. Start `competition_nav.launch.py`.
3. Echo `/mission_state`, `/current_goal`, and `/track_cmd_vel`.
4. Publish a fake QR result and check the state transition.
5. Start `origincar_avoid` and manually publish `/avoid_active` plus `/avoid_cmd_vel` to verify `cmd_vel_mux`.
6. Start camera, YOLO, QR, and Bird View after the core navigation loop is stable.
