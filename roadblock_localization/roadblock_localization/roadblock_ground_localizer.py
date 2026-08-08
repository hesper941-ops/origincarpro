"""ROS2 node publishing per-frame metric ground positions of YOLO roadblocks."""

import math
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import rclpy
from ai_msgs.msg import PerceptionTargets
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from roadblock_interfaces.msg import Roadblock, RoadblockArray

from .ipm_ground_projector import IPMGroundProjector


def detection_to_ground_pixel(x_offset, y_offset, width, height) -> Tuple[float, float]:
    """Return YOLO bbox bottom-center; this is the only V1 ground-point policy."""
    values = tuple(float(value) for value in (x_offset, y_offset, width, height))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox values must be finite")
    x, y, w, h = values
    if w <= 0.0 or h <= 0.0:
        raise ValueError("bbox width and height must be positive")
    return x + w * 0.5, y + h


def rank_by_distance(obstacles: Iterable[dict]) -> List[dict]:
    """Sort and assign per-frame distance-rank IDs (not persistent track IDs)."""
    ranked = sorted(
        (dict(item) for item in obstacles),
        key=lambda item: (item["distance"], item["x"], item["y"], -item["confidence"]),
    )
    for index, item in enumerate(ranked, start=1):
        item["id"] = index
    return ranked


class RoadblockGroundLocalizer(Node):
    def __init__(self):
        super().__init__("roadblock_ground_localizer")

        self.declare_parameter("detection_topic", "/hobot_dnn_detection")
        self.declare_parameter("output_topic", "/roadblock_ground_array")
        self.declare_parameter("roadblock_label", "roadblock")
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("min_confidence", 0.50)
        self.declare_parameter("min_x_m", 0.20)
        self.declare_parameter("max_x_m", 2.00)
        self.declare_parameter("max_abs_y_m", 1.00)
        self.declare_parameter("debug_log", False)

        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.roadblock_label = str(self.get_parameter("roadblock_label").value).strip().casefold()
        calibration_file = str(self.get_parameter("calibration_file").value).strip()
        self.frame_id = str(self.get_parameter("frame_id").value).strip()
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.min_x_m = float(self.get_parameter("min_x_m").value)
        self.max_x_m = float(self.get_parameter("max_x_m").value)
        self.max_abs_y_m = float(self.get_parameter("max_abs_y_m").value)
        self.debug_log = bool(self.get_parameter("debug_log").value)
        self._last_debug_time = 0.0

        if not calibration_file:
            calibration_file = str(
                Path(get_package_share_directory("roadblock_localization"))
                / "config"
                / "ipm_calibration.yaml"
            )
        if not self.roadblock_label:
            raise ValueError("roadblock_label must not be empty")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0,1]")
        if self.min_x_m < 0.0 or self.max_x_m <= self.min_x_m or self.max_abs_y_m <= 0.0:
            raise ValueError("invalid ground-coordinate filter limits")

        self.projector = IPMGroundProjector(calibration_file)
        self.publisher = self.create_publisher(RoadblockArray, self.output_topic, 10)
        self.subscription = self.create_subscription(
            PerceptionTargets,
            self.detection_topic,
            self._detection_callback,
            10,
        )

        self.get_logger().info(
            f"roadblock localization ready: {self.detection_topic} -> {self.output_topic}; "
            f"frame={self.frame_id}; calibration={self.projector.calibration_file}"
        )
        self.get_logger().info("V1 ground point=bbox bottom-center; id=per-frame distance rank")

    def _detection_callback(self, msg: PerceptionTargets) -> None:
        localized = []
        rejected = 0

        for target in msg.targets:
            if str(target.type).strip().casefold() != self.roadblock_label:
                continue
            for roi in target.rois:
                confidence = float(roi.confidence)
                if not math.isfinite(confidence) or confidence < self.min_confidence:
                    rejected += 1
                    continue
                rect = roi.rect
                x0 = float(rect.x_offset)
                y0 = float(rect.y_offset)
                width = float(rect.width)
                height = float(rect.height)
                try:
                    u, v = detection_to_ground_pixel(x0, y0, width, height)
                except ValueError:
                    rejected += 1
                    continue
                if (
                    x0 < 0.0
                    or y0 < 0.0
                    or x0 + width > self.projector.image_width
                    or y0 + height > self.projector.image_height
                ):
                    rejected += 1
                    continue
                try:
                    x_m, y_m = self.projector.pixel_to_ground(u, v)
                except ValueError:
                    rejected += 1
                    continue
                if (
                    not math.isfinite(x_m)
                    or not math.isfinite(y_m)
                    or x_m <= self.min_x_m
                    or x_m > self.max_x_m
                    or abs(y_m) > self.max_abs_y_m
                ):
                    rejected += 1
                    continue
                localized.append(
                    {
                        "x": x_m,
                        "y": y_m,
                        "distance": math.hypot(x_m, y_m),
                        "confidence": confidence,
                    }
                )

        ranked = rank_by_distance(localized)
        output = RoadblockArray()
        # PerceptionTargets has a Header; preserve its frame timestamp but use
        # the verified vehicle-ground frame for the projected coordinates.
        output.header.stamp = msg.header.stamp
        output.header.frame_id = self.frame_id
        for item in ranked:
            obstacle = Roadblock()
            obstacle.id = int(item["id"])
            obstacle.x = float(item["x"])
            obstacle.y = float(item["y"])
            obstacle.distance = float(item["distance"])
            obstacle.confidence = float(item["confidence"])
            output.obstacles.append(obstacle)
        self.publisher.publish(output)
        self._debug_report(ranked, rejected)

    def _debug_report(self, ranked: List[dict], rejected: int) -> None:
        if not self.debug_log:
            return
        now = time.monotonic()
        if now - self._last_debug_time < 1.0:
            return
        self._last_debug_time = now
        lines = [f"ROADBLOCKS count={len(ranked)} rejected={rejected}"]
        lines.extend(
            f"id={item['id']} x={item['x']:.3f} y={item['y']:.3f} "
            f"distance={item['distance']:.3f} confidence={item['confidence']:.3f}"
            for item in ranked
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
