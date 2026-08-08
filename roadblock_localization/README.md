# roadblock_localization

Independent ROS2 package that converts YOLO `roadblock` bounding boxes into
metric vehicle-ground coordinates. It does not make avoidance decisions and
never publishes motion commands.

## Interface

- Input: `/hobot_dnn_detection` (`ai_msgs/msg/PerceptionTargets`)
- Output: `/roadblock_ground_array` (`roadblock_interfaces/msg/RoadblockArray`)
- Output frame: `base_link`
- Coordinate convention: origin at chassis rotation center, `+X` forward,
  `+Y` left, ground `Z=0`, unit meter

The X5 YOLO output was verified against `/image_out/compressed`: ROI values are
already in the original 640x400 image coordinate system. For each ROI, V1 uses
only its bottom-center pixel:

```text
u = x_offset + width / 2
v = y_offset + height
```

The pixel is undistorted with the installed calibration K/D and projected by
the installed `H_image_to_ground`. The node does not subscribe to an image.

Each output array represents only the current detection frame. Obstacles are
sorted by `sqrt(x^2 + y^2)` and then numbered from 1. Therefore `id=1` is the
nearest current-frame obstacle, `id=2` is the second-nearest, and so on. These
IDs are recomputed every frame and are **not persistent tracking IDs**.

When no valid roadblock is present, the node publishes `obstacles: []`; it does
not retain a previous frame.

## Start

```bash
cd /root/intelligent_car_ws
source /opt/tros/humble/setup.bash
source install/setup.bash
ros2 launch roadblock_localization roadblock_localization.launch.py
```

Or run the executable directly:

```bash
ros2 run roadblock_localization roadblock_ground_localizer
```

Inspect output:

```bash
ros2 topic echo /roadblock_ground_array
```

The launch file starts only `roadblock_ground_localizer`; Aurora, YOLO, base,
lane, avoidance, and control nodes are deliberately outside this package.

## Parameters

| Name | Default | Meaning |
|---|---:|---|
| `detection_topic` | `/hobot_dnn_detection` | YOLO detections |
| `output_topic` | `/roadblock_ground_array` | Ground-position array |
| `roadblock_label` | `roadblock` | Exact target label after case normalization |
| `calibration_file` | installed package calibration | Fixed IPM YAML |
| `frame_id` | `base_link` | Verified vehicle-ground frame |
| `min_confidence` | `0.50` | Reject lower-confidence detections |
| `min_x_m` | `0.20` | Reject X at or behind this limit |
| `max_x_m` | `2.00` | Maximum validated forward range |
| `max_abs_y_m` | `1.00` | Maximum absolute lateral range |
| `debug_log` | `false` | Throttled localization log, at most 1 Hz |

Formal runtime calibration:

```text
share/roadblock_localization/config/ipm_calibration.yaml
```

It is a direct copy of the verified development calibration. Debug JPG/CSV
artifacts are intentionally not installed with this package.
