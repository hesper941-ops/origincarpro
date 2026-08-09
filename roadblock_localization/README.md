# roadblock_localization

This ROS2 package reports cone positions only. It never publishes motion
commands and contains no avoidance or driving policy.

## Team interface

- Input: `/hobot_dnn_detection` (`ai_msgs/msg/PerceptionTargets`)
- Output: `/roadblock_ground_array`
  (`roadblock_interfaces/msg/RoadblockArray`)
- Frame: `base_link`; origin is the chassis rotation center
- Coordinates: `x` forward-positive, `y` left-positive, unit metre

Each obstacle has exactly three fields: `id`, `x`, and `y`. `id` is a
short-term, vision-local association ID; it is neither a distance rank nor a
map/global obstacle identity.

**`/roadblock_ground_array` only contains fresh, currently valid visual
measurements.** `obstacles` 数组只包含当前时刻通过完整性和几何合法性检查的视觉测量。
某个障碍物本帧未出现在数组中，不代表障碍物已从环境中消失，只表示当前帧没有新的可靠视觉测量。
历史维护、TF/odom/map 坐标转换、障碍物生命周期及最终规划由下游模块负责。
`id` 为视觉侧短期局部关联 ID，仅用于辅助连续帧关联，不作为 map/global obstacle ID。
车辆发生较大运动、目标长时间丢失或重新进入视野时，下游不得假设该 `id` 是永久或全局 ID。

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
ray supplies direction. The formal `adaptive_ipm` model corrects only its
radius using observable bbox geometry:

```text
u_n    = (bbox_center_u - image_width/2) / (image_width/2)
aspect = bbox_width / bbox_height
wr     = bbox_width * r_raw

c_adaptive = c0
           + cr * (r_raw - 1.0)
           + cw * ((wr - 100.0) / 50.0)
           + ca * ((aspect - 0.78) / 0.15)
           + cu * u_n^2

d_final = r_raw + clamp(c_adaptive, 0.005, 0.156)
```

The fitted coefficients are stored only in the package YAML. They were fitted
from 211 valid stop-by-cone means across 59 stops in the full-distance motion
and near-distance real-car datasets. Stop-grouped 5-fold out-of-fold validation
gave `MAE=1.7322 cm`, `RMSE=2.1180 cm`, and `MaxAbs=6.0758 cm`. The offset
limits add a 5 mm guard band around the fitted-data range
`0.01055..0.15031 m`.

`distance_model: legacy_fusion` retains the previous fixed IPM/height fusion
for rollback. Height range never contributes to formal `adaptive_ipm` output.

Reliable current-frame measurements use one-to-one nearest-neighbour
association in `base_link` XY. The tracker keeps a short memory only to recover
local IDs after brief visual gaps. Unobserved tracks are never placed in an
output message, even while they remain in that internal memory. IDs increase
monotonically and are not reused during a node run.

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
- formal distance model `adaptive_ipm`
- adaptive coefficients and data-derived offset limits in the YAML

The fixed IPM centre offset, height model, and fusion weight remain in the YAML
only for the `legacy_fusion` rollback mode.

The fit can be reproduced without copying raw logs into the repository:

```bash
python3 tools/fit_adaptive_ipm.py \
  --motion-dir /root/intelligent_car_ws/test_logs/roadblock_motion_calibration/20260809_110818 \
  --near-dir /root/intelligent_car_ws/test_logs/roadblock_near_calibration/20260809_124603 \
  --edge-margin 8
```

The tool excludes `motion:POSE45_019` and `near:POSE45_041`, applies the
confirmed motion label remaps, uses session-prefixed stop groups, performs
5-fold grouped validation, and reports 8/20/40 px edge sensitivity.

The following are engineering initial values and **are pending real-car
validation**, not calibrated constants:

- `edge_margin_px: 2.0`
- `enable_fov_gate: false`
- `fov_boundary_slope_y_per_x: 0.713`
- `fov_footprint_radius_m: 0.1414213562`
- `association_max_distance_m: 0.30`
- `track_ttl_sec: 2.0`

Any bbox touching the configured edge margin, extending outside `640x400`, or
having invalid geometry is unreliable. It cannot create or update an absolute
visual measurement. The current message omits it immediately; no historical
coordinate is substituted.

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
The `track_ttl_sec` parameter controls internal ID association memory only. It
never causes an unobserved obstacle to remain in `/roadblock_ground_array`.
The formal localizer has no `/odom` dependency; downstream modules own TF,
history, lifetime, map/global association, and planning.

The verified IPM is installed at
`share/roadblock_localization/config/ipm_calibration.yaml`.
