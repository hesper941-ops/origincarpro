# Competition Navigation Framework

## Formal Control Line

The formal competition control line is:

`semantic_map.yaml` -> `task_manager` -> `/current_goal` -> `target_tracker` -> `/track_cmd_vel` -> `cmd_vel_mux` -> `/cmd_vel`

Nav2 and fixed waypoint demos can stay as debugging tools, but they are not the formal competition path.

The static map and visualization markers are only for RViz or CoStudio inspection. They do not participate in task decisions or control arbitration.

## Semantic Map

`origincar_nav/config/semantic_map.yaml` is the rough semantic map used by the task layer. It stores the start point, task station, channel entry, channel points, boundary, route order, task thresholds, and visualization options.

The current coordinates and boundary values are placeholders. They must be calibrated on the real car and competition field before a real run.

## Half-Route QR Scan

The car does not stop at the task station and wait for QR decoding.

During `TRACK_TO_TASK_STATION`, `task_manager` keeps `/track_enable=true` and keeps the active goal at `task_station`. Once `/odom_combined` shows that the car is within `qr_scan_start_dist_to_task_station`, `/perception_mode` changes to `qr_scan` while tracking continues.

Before `qr_scan` is active, `task_manager` ignores `/qrcode_detected/info_result`. When QR scan is active and the result parses as `ClockWise`, `AntiClockWise`, `clockwise`, `anticlockwise`, `顺时针`, or `逆时针`, `task_manager` immediately selects the route, switches to `TRACK_TO_CHANNEL_ENTRY`, and publishes `channel_entry` as `/current_goal`.

If the car reaches the task station but QR has not decoded yet, it remains in QR scan mode and does not automatically enter the channel.

QR decoding is constrained by the YOLO `QR_code` ROI. `qr_code_detection` uses `qr_labels`, defaulting to `["QR_code"]`, and only runs QR decoding inside those ROIs. `/qrcode_detected/info_result` contains the decoded QR text, not the `QR_code` YOLO label.

The formal YOLO label set is:

- `QR_code`: QR board ROI for QR decoding.
- `line`: black line label reserved for later Bird View or return visual correction.
- `end`: P-point or finish confirmation label reserved for later return-to-P judgment.
- `roadblock`: physical blocking-object label, and the only label that triggers avoidance.

Numeric odd/even direction parsing is a debug fallback only and is disabled by default. The formal route rule is direction text confirming clockwise or anticlockwise.

## Target Tracking

`target_tracker` subscribes to `/odom_combined`, `/current_goal`, and `/track_enable`. It computes distance and heading error in odometry space and publishes `/track_cmd_vel`.

It does not depend on AMCL or Nav2. If odometry or the current goal is missing, it publishes zero velocity and waits.

## Obstacle Avoidance

`yolo_avoid_controller` only uses detections whose label is in `obstacle_labels`, which defaults to `["roadblock"]`.

It chooses left or right avoidance from the `roadblock` image position and the semantic map boundary. A `roadblock` on the left means prefer right avoidance; a `roadblock` on the right means prefer left avoidance; centered `roadblock` detections use boundary safety to choose a side.

If both side candidates are outside the boundary, the node performs a protective stop and warns. That branch is for abnormal localization or boundary configuration problems, not a regular competition strategy.

Avoidance is intentionally short. When it ends, the node publishes `/avoid_finished` and does not hard-code a return-to-heading action. Re-centering is handled by `target_tracker` using the current `/odom_combined` pose and the original goal.

The optional `/avoid/obstacle_debug_point` is an approximate visualization point estimated from the current pose, yaw, and `roadblock` bounding-box offset. It is not a precise mapped obstacle position and is not used for avoidance decisions.

## Visualization Layer

`semantic_map_visualizer` is for CoStudio/RViz debugging only. It displays:

- Boundary rectangle
- Semantic points
- Clockwise and anticlockwise route lines
- Current goal marker
- Vehicle path accumulated from `/odom_combined`
- Current mission and perception mode text
- Optional approximate `roadblock` debug marker
- Current vehicle pose marker

It publishes:

- `/semantic_map/markers`
- `/current_goal_marker`
- `/vehicle_path`
- `/mission_text_marker`
- `/avoid_obstacle_marker`
- `/semantic_map/current_pose_marker`

The visualization frame is parameterized. If no `map -> odom_combined` transform is available, launch with `visualization_frame:=odom_combined`.

## Formal Topics

Inputs:

- `/odom_combined`
- `/current_goal`
- `/track_enable`
- `/goal_reached`
- `/avoid_active`
- `/avoid_cmd_vel`
- `/avoid_finished`
- `/qrcode_detected/info_result`
- `/bird_view/p_point_valid`

Outputs:

- `/current_goal`
- `/track_enable`
- `/track_cmd_vel`
- `/avoid_cmd_vel`
- `/cmd_vel`
- `/mission_state`
- `/perception_mode`

Bird View reserved interfaces:

- `/bird_view/p_point_valid`
- `/bird_view/p_point_pose`
- `/bird_view/line_error`
- `/bird_view/heading_error`

## Reserved Work

These pieces are interfaces or placeholders, not completed algorithms:

- Bird View P-point recognition is not complete.
- `birdview_local_nav` local return control is not complete.
- Channel-area visual correction is not complete.
- `line` and `end` are reserved labels only; Bird View return recognition and P-point confirmation are not completed closed-loop algorithms yet.
- The `QR_code` ROI quality and QR decode chain need field confirmation.
- Semantic map coordinates and boundaries need on-car calibration.

## Startup

Build:

```bash
colcon build --symlink-install --packages-select origincar_nav origincar_avoid qr_code_detection vision_birdview
```

Source:

```bash
source install/setup.bash
```

Competition navigation and visualization:

```bash
ros2 launch origincar_nav competition_nav.launch.py
```

Use odometry frame for visualization if there is no map transform:

```bash
ros2 launch origincar_nav competition_nav.launch.py visualization_frame:=odom_combined
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

## Debug Checks

```bash
ros2 topic list | grep -E "semantic_map|current_goal|vehicle_path|mission_state|perception_mode|track_cmd_vel|avoid|cmd_vel"
ros2 topic echo /mission_state
ros2 topic echo /perception_mode
ros2 topic echo /current_goal
ros2 topic echo /vehicle_path
```

RViz or CoStudio should display:

- `/semantic_map/markers`
- `/current_goal_marker`
- `/vehicle_path`
- `/mission_text_marker`
- `/avoid_obstacle_marker`
