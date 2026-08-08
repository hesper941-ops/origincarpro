"""ROS2 cone localization using fused range and stable odom-frame tracks."""

import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import rclpy
from ai_msgs.msg import PerceptionTargets
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from roadblock_interfaces.msg import Roadblock, RoadblockArray

from .cone_position_fusion import (
    bbox_is_reliable,
    fuse_cone_position,
    ground_measurement_is_valid,
)
from .ipm_ground_projector import IPMGroundProjector
from .roadblock_tracker import OdomPose, RoadblockTracker


def detection_to_ground_pixel(x_offset, y_offset, width, height) -> Tuple[float, float]:
    """Return the bbox bottom-center pixel after basic geometry validation."""
    values = tuple(float(value) for value in (x_offset, y_offset, width, height))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox values must be finite")
    x, y, w, h = values
    if w <= 0.0 or h <= 0.0:
        raise ValueError("bbox width and height must be positive")
    return x + w * 0.5, y + h


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    values = (x, y, z, w)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("odom quaternion must be finite")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_output_message(items: List[dict], stamp, frame_id: str) -> RoadblockArray:
    output = RoadblockArray()
    output.header.stamp = stamp
    output.header.frame_id = frame_id
    for item in sorted(items, key=lambda value: value["id"]):
        obstacle = Roadblock()
        obstacle.id = int(item["id"])
        obstacle.x = float(item["x"])
        obstacle.y = float(item["y"])
        output.obstacles.append(obstacle)
    return output


def update_tracker_if_odom_ready(
    tracker: RoadblockTracker,
    reliable_positions: List[Tuple[float, float]],
    odom_pose: Optional[OdomPose],
    now_sec: float,
) -> bool:
    """Update tracks only after a real, validated odom pose has arrived."""
    if odom_pose is None:
        return False
    tracker.update(reliable_positions, odom_pose, now_sec)
    return True


class RoadblockGroundLocalizer(Node):
    def __init__(self):
        super().__init__("roadblock_ground_localizer")

        defaults = {
            "detection_topic": "/hobot_dnn_detection",
            "odom_topic": "/odom",
            "output_topic": "/roadblock_ground_array",
            "roadblock_label": "roadblock",
            "calibration_file": "",
            "frame_id": "base_link",
            "min_confidence": 0.50,
            "image_width": 640,
            "image_height": 400,
            "edge_margin_px": 2.0,
            "camera_ground_x_m": 0.05,
            "camera_ground_y_m": 0.0,
            "camera_height_m": 0.28,
            "cone_height_m": 0.30,
            "cone_base_width_m": 0.20,
            "cone_base_length_m": 0.20,
            "ipm_center_offset_m": 0.10,
            "height_model_a": 99.488,
            "height_model_b": 0.22381,
            "fusion_alpha_ipm": 0.70,
            "min_ipm_radius_m": 0.01,
            "min_x_m": 0.20,
            "max_x_m": 2.00,
            "enable_fov_gate": False,
            "fov_boundary_slope_y_per_x": 0.713,
            "fov_footprint_radius_m": 0.1414213562373095,
            "association_max_distance_m": 0.30,
            "track_ttl_sec": 2.0,
            "track_min_x_m": -0.30,
            "track_max_distance_m": 3.00,
            "publish_rate_hz": 10.0,
            "debug_log": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.roadblock_label = str(self.get_parameter("roadblock_label").value).strip().casefold()
        calibration_file = str(self.get_parameter("calibration_file").value).strip()
        self.frame_id = str(self.get_parameter("frame_id").value).strip()
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.edge_margin_px = float(self.get_parameter("edge_margin_px").value)
        self.camera_ground_x_m = float(self.get_parameter("camera_ground_x_m").value)
        self.camera_ground_y_m = float(self.get_parameter("camera_ground_y_m").value)
        self.camera_height_m = float(self.get_parameter("camera_height_m").value)
        self.cone_height_m = float(self.get_parameter("cone_height_m").value)
        self.cone_base_width_m = float(self.get_parameter("cone_base_width_m").value)
        self.cone_base_length_m = float(self.get_parameter("cone_base_length_m").value)
        self.ipm_center_offset_m = float(self.get_parameter("ipm_center_offset_m").value)
        self.height_model_a = float(self.get_parameter("height_model_a").value)
        self.height_model_b = float(self.get_parameter("height_model_b").value)
        self.fusion_alpha_ipm = float(self.get_parameter("fusion_alpha_ipm").value)
        self.min_ipm_radius_m = float(self.get_parameter("min_ipm_radius_m").value)
        self.min_x_m = float(self.get_parameter("min_x_m").value)
        self.max_x_m = float(self.get_parameter("max_x_m").value)
        self.enable_fov_gate = bool(self.get_parameter("enable_fov_gate").value)
        self.fov_boundary_slope_y_per_x = float(
            self.get_parameter("fov_boundary_slope_y_per_x").value
        )
        self.fov_footprint_radius_m = float(
            self.get_parameter("fov_footprint_radius_m").value
        )
        association_max_distance_m = float(
            self.get_parameter("association_max_distance_m").value
        )
        track_ttl_sec = float(self.get_parameter("track_ttl_sec").value)
        track_min_x_m = float(self.get_parameter("track_min_x_m").value)
        track_max_distance_m = float(self.get_parameter("track_max_distance_m").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.debug_log = bool(self.get_parameter("debug_log").value)
        self._last_debug_time = 0.0
        self._latest_odom: Optional[OdomPose] = None

        if not calibration_file:
            calibration_file = str(
                Path(get_package_share_directory("roadblock_localization"))
                / "config"
                / "ipm_calibration.yaml"
            )
        if not self.roadblock_label or self.frame_id != "base_link":
            raise ValueError("roadblock_label must be set and frame_id must be base_link")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0,1]")
        if self.image_width <= 0 or self.image_height <= 0 or self.edge_margin_px < 0.0:
            raise ValueError("invalid image boundary parameters")
        if any(
            value <= 0.0
            for value in (
                self.camera_height_m,
                self.cone_height_m,
                self.cone_base_width_m,
                self.cone_base_length_m,
                publish_rate_hz,
            )
        ):
            raise ValueError("physical dimensions and publish_rate_hz must be positive")
        if self.min_x_m < 0.0 or self.max_x_m <= self.min_x_m:
            raise ValueError("invalid ground-coordinate filter limits")
        if self.fov_boundary_slope_y_per_x <= 0.0 or self.fov_footprint_radius_m < 0.0:
            raise ValueError("invalid ground FOV parameters")

        self.projector = IPMGroundProjector(calibration_file)
        if (
            self.projector.image_width != self.image_width
            or self.projector.image_height != self.image_height
        ):
            raise ValueError("configured image dimensions do not match IPM calibration")
        self.tracker = RoadblockTracker(
            association_max_distance_m,
            track_ttl_sec,
            track_min_x_m,
            track_max_distance_m,
        )
        self.publisher = self.create_publisher(RoadblockArray, self.output_topic, 10)
        self.detection_subscription = self.create_subscription(
            PerceptionTargets, self.detection_topic, self._detection_callback, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, 20
        )
        self.publish_timer = self.create_timer(1.0 / publish_rate_hz, self._publish_tracks)

        self.get_logger().info(
            f"roadblock localization ready: detections={self.detection_topic}, "
            f"odom={self.odom_topic}, output={self.output_topic}; frame={self.frame_id}; "
            f"calibration={self.projector.calibration_file}"
        )
        self.get_logger().info(
            f"fusion={self.fusion_alpha_ipm:.2f}*(IPM+{self.ipm_center_offset_m:.2f}m)"
            f"+{1.0 - self.fusion_alpha_ipm:.2f}*height-model; stable IDs in odom frame"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _odom_callback(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        try:
            yaw = quaternion_to_yaw(
                orientation.x, orientation.y, orientation.z, orientation.w
            )
        except ValueError:
            self.get_logger().warning("ignoring odom with non-finite quaternion")
            return
        values = (float(position.x), float(position.y), yaw)
        if not all(math.isfinite(value) for value in values):
            self.get_logger().warning("ignoring odom with non-finite pose")
            return
        self._latest_odom = OdomPose(*values)

    def _detection_callback(self, msg: PerceptionTargets) -> None:
        if self._latest_odom is None:
            self._debug_report([], 0, "waiting for /odom")
            return

        reliable_positions: List[Tuple[float, float]] = []
        rejected = 0
        for target in msg.targets:
            if str(target.type).strip().casefold() != self.roadblock_label:
                continue
            for roi in target.rois:
                confidence = float(roi.confidence)
                rect = roi.rect
                xmin = float(rect.x_offset)
                ymin = float(rect.y_offset)
                xmax = xmin + float(rect.width)
                ymax = ymin + float(rect.height)
                height = ymax - ymin
                if not math.isfinite(confidence) or confidence < self.min_confidence:
                    rejected += 1
                    continue
                if not bbox_is_reliable(
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    self.image_width,
                    self.image_height,
                    self.edge_margin_px,
                ):
                    rejected += 1
                    continue
                try:
                    u, v = detection_to_ground_pixel(xmin, ymin, xmax - xmin, height)
                    point_x, point_y = self.projector.pixel_to_ground(u, v)
                    fusion = fuse_cone_position(
                        point_x,
                        point_y,
                        height,
                        self.camera_ground_x_m,
                        self.camera_ground_y_m,
                        self.ipm_center_offset_m,
                        self.height_model_a,
                        self.height_model_b,
                        self.fusion_alpha_ipm,
                        self.min_ipm_radius_m,
                    )
                except ValueError:
                    rejected += 1
                    continue
                if not ground_measurement_is_valid(
                    fusion.x,
                    fusion.y,
                    self.min_x_m,
                    self.max_x_m,
                    self.enable_fov_gate,
                    self.fov_boundary_slope_y_per_x,
                    self.fov_footprint_radius_m,
                ):
                    rejected += 1
                    continue
                reliable_positions.append((fusion.x, fusion.y))

        now_sec = self._now_sec()
        update_tracker_if_odom_ready(
            self.tracker, reliable_positions, self._latest_odom, now_sec
        )
        self._debug_report(self.tracker.snapshot(self._latest_odom, now_sec), rejected, "")

    def _publish_tracks(self) -> None:
        items = []
        if self._latest_odom is not None:
            items = self.tracker.snapshot(self._latest_odom, self._now_sec())
        output = build_output_message(
            items, self.get_clock().now().to_msg(), self.frame_id
        )
        self.publisher.publish(output)

    def _debug_report(self, tracks: List[dict], rejected: int, note: str) -> None:
        if not self.debug_log:
            return
        now = time.monotonic()
        if now - self._last_debug_time < 1.0:
            return
        self._last_debug_time = now
        lines = [f"ROADBLOCKS tracks={len(tracks)} rejected={rejected} {note}".rstrip()]
        lines.extend(
            f"id={item['id']} x={item['x']:.3f} y={item['y']:.3f}" for item in tracks
        )
        self.get_logger().info("\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RoadblockGroundLocalizer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
