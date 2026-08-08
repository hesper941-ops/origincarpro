# roadblock_localization

This ROS2 package reports cone positions only. It never publishes motion
commands and contains no avoidance or driving policy.

## Team interface

- Inputs: `/hobot_dnn_detection` (`ai_msgs/msg/PerceptionTargets`) and `/odom`
  (`nav_msgs/msg/Odometry`)
- Output: `/roadblock_ground_array`
  (`roadblock_interfaces/msg/RoadblockArray`)
- Frame: `base_link`; origin is the chassis rotation center
- Coordinates: `x` forward-positive, `y` left-positive, unit metre

Each obstacle has exactly three fields: `id`, `x`, and `y`. `id` is a stable
local track ID for the lifetime of the running node; it is not a distance rank.
No valid track is represented by `obstacles: []`.

Minimal Python subscriber:

```python
from roadblock_interfaces.msg import RoadblockArray

def callback(msg):
    for obs in msg.obstacles:
        print(f"id={obs.id} x={obs.x:.3f}m y={obs.y:.3f}m")

subscription = node.create_subscription(
    RoadblockArray, "/roadblock_ground_array", callback, 10
)
```

## Internal method

For each complete, reliable YOLO `roadblock` bbox, its bottom-centre pixel is
undistorted and projected with the installed point IPM. The resulting ground
ray supplies direction. Range is fused from:

```text
d_A = distance(camera_ground_projection, IPM_bottom_center) + 0.10 m
d_B = 99.488 / bbox_height_px + 0.22381
d   = 0.70 * d_A + 0.30 * d_B
```

This fixed scheme C was selected from nine real-car measurements:

| Method | Mean | Median | RMSE | Max |
|---|---:|---:|---:|---:|
| A: IPM + 10 cm | 3.68 cm | 4.02 cm | 3.82 cm | 5.05 cm |
| B: IPM direction + bbox-height range | 4.06 cm | 4.39 cm | 4.15 cm | 5.04 cm |
| C: 0.70/0.30 fusion | 3.61 cm | 3.93 cm | 3.72 cm | 4.96 cm |

The measured accuracy coverage was mainly `X=0.6..1.0 m` and lateral
`Y=+-0.20 m`. This documents the tested region; it is not an algorithmic range
limit. Wider lateral coverage still requires real-car validation.

The fused centre is stored in the odom frame. Existing tracks are transformed
back into the current `base_link` frame for every publication, so a temporarily
unreliable/cropped bbox cannot freeze or overwrite the last reliable position.
Reliable observations use one-to-one ground-XY nearest-neighbour association.
IDs increase monotonically and are not reused during a node run.

## Start

```bash
cd /root/intelligent_car_ws
source /opt/tros/humble/setup.bash
source install/setup.bash
ros2 launch roadblock_localization roadblock_localization.launch.py
ros2 topic echo /roadblock_ground_array
```

The launch starts only the localizer; camera, YOLO, odometry, lane, avoidance,
and control nodes remain outside this package.

## Parameters

The runtime values are in `config/roadblock_localization.yaml`. Fixed physical
and calibrated values are:

- camera ground projection `(0.05, 0.00)` m; camera height `0.28` m
- cone height `0.30` m; base width/length `0.20/0.20` m
- IPM centre offset `0.10` m
- height model `99.488 / h_px + 0.22381`
- fusion IPM weight `0.70`

The following are engineering initial values and **are pending real-car
validation**, not calibrated constants:

- `edge_margin_px: 2.0`
- `enable_fov_gate: false`
- `fov_boundary_slope_y_per_x: 0.713`
- `fov_footprint_radius_m: 0.1414213562`
- `association_max_distance_m: 0.30`
- `track_ttl_sec: 2.0`
- `track_min_x_m: -0.30`
- `track_max_distance_m: 3.00`

Any bbox touching the configured edge margin, extending outside `640x400`, or
having invalid geometry is unreliable. It cannot create or update an absolute
track position. Tracks survive such frames only within TTL using odom
propagation.

The optional ground FOV is modelled as the wedge `|Y| <= 0.713*X`. For a cone
base conservatively represented by radius `R=sqrt(0.10^2+0.10^2)=0.1414 m`, a
new visual update must satisfy, when `enable_fov_gate` is true:

```text
0.713*X - |Y| >= R*sqrt(1 + 0.713^2)
```

This uses perpendicular distance to each sloped boundary, not a rectangular
`max_abs_y` check. The FOV gate is a geometric initial model pending full-view
real-car edge validation and is therefore **disabled by default**. When false,
it never rejects an otherwise valid visual measurement. When true, it gates
only new visual measurements; it does not delete a reliable historical Track.
TTL, odom propagation, behind-vehicle threshold, and maximum radial range
govern Track maintenance.

No Track is created or updated before the first finite `/odom` pose arrives.
The node remains alive and continues publishing `obstacles: []` while waiting.

The verified IPM is installed at
`share/roadblock_localization/config/ipm_calibration.yaml`.
