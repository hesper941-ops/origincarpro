#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黄色区域视觉寻路独立测试脚本。

本脚本不接入正式 task_manager，只订阅相机并向 cmd_vel_gate 的输入话题发布
速度。默认发布到 /cmd_vel_raw，仍需使用比赛启动按钮放行。

推荐测试流程
============

1. 启动底盘和相机，不启动正式导航：

    cd /root/intelligent_car_ws
    source /opt/tros/humble/setup.bash
    source install/setup.bash

    ros2 launch origincar_bringup competition_bringup.launch.py \
      enable_base:=true \
      enable_camera:=true \
      enable_yolo_avoid:=false \
      enable_qr:=false \
      enable_birdview:=false \
      enable_nav:=false \
      enable_cobridge:=false \
      button_backend:=topic

如果当前 launch 不支持 enable_nav，则删除 enable_nav 参数。

2. 运行黄色区域寻路：

    python3 tools/yellow_area_follow.py \
      --ros-args \
      --params-file tools/yellow_area_params.yaml

3. 发车门锁：

    ros2 topic pub --once /competition/start_button \
      std_msgs/msg/Bool "{data: true}"

4. 急停：

    ros2 topic pub --once /competition/emergency_stop_cmd \
      std_msgs/msg/Bool "{data: true}"

5. 查看速度链路：

    timeout 3s ros2 topic echo /cmd_vel_raw || true
    timeout 3s ros2 topic echo /cmd_vel || true

Ctrl+C 退出时，本脚本会连续发布多次零速。首次测试请架空车轮或确保急停可用。
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


SCAN_BAND_RATIOS = (0.55, 0.68, 0.78, 0.88, 0.95)
MORPHOLOGY_KERNEL_SIZE = 5
STOP_PUBLISH_COUNT = 5
STOP_PUBLISH_INTERVAL_SEC = 0.05


@dataclass
class VisionResult:
    """Result of extracting the yellow road center from one image."""

    valid: bool
    road_center: Optional[float]
    yellow_pixels: int
    valid_bands: int
    mask: np.ndarray
    band_rows: List[Tuple[int, int, bool]]
    used_fallback: bool


def image_message_to_bgr(msg: Image) -> np.ndarray:
    """Convert bgr8/rgb8/mono8/nv12 sensor_msgs/Image data to packed BGR."""
    encoding = msg.encoding.strip().lower()
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("bgr8", "rgb8"):
        row_bytes = width * 3
        step = step or row_bytes
        required = step * height
        if step < row_bytes or raw.size < required:
            raise ValueError(
                f"invalid {encoding} buffer: step={step}, bytes={raw.size}"
            )
        packed = raw[:required].reshape(height, step)[:, :row_bytes]
        image = packed.reshape(height, width, 3)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(image)

    if encoding == "mono8":
        row_bytes = width
        step = step or row_bytes
        required = step * height
        if step < row_bytes or raw.size < required:
            raise ValueError(
                f"invalid mono8 buffer: step={step}, bytes={raw.size}"
            )
        mono = raw[:required].reshape(height, step)[:, :row_bytes]
        return cv2.cvtColor(np.ascontiguousarray(mono), cv2.COLOR_GRAY2BGR)

    if encoding == "nv12":
        if width % 2 or height % 2:
            raise ValueError(f"NV12 image size must be even: {width}x{height}")
        row_bytes = width
        step = step or row_bytes
        yuv_rows = height + height // 2
        required = step * yuv_rows
        if step < row_bytes or raw.size < required:
            raise ValueError(
                f"invalid nv12 buffer: step={step}, expected at least "
                f"{required} bytes, got {raw.size}"
            )
        yuv = raw[:required].reshape(yuv_rows, step)[:, :row_bytes]
        return cv2.cvtColor(
            np.ascontiguousarray(yuv), cv2.COLOR_YUV2BGR_NV12
        )

    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


def extract_road_center(
    bgr: np.ndarray,
    h_min: int,
    h_max: int,
    s_min: int,
    s_max: int,
    v_min: int,
    v_max: int,
    roi_y_start_ratio: float,
    scan_band_half_height: int,
    min_yellow_pixels: int,
    min_band_pixels: int,
    min_valid_bands: int,
) -> VisionResult:
    """Segment yellow and estimate its center using weighted scan bands."""
    image_height, image_width = bgr.shape[:2]
    roi_y_start = int(round(image_height * roi_y_start_ratio))
    roi_y_start = min(max(roi_y_start, 0), image_height - 1)
    roi = bgr[roi_y_start:, :]
    roi_height = roi.shape[0]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    yellow_pixels = int(cv2.countNonZero(mask))

    band_centers: List[float] = []
    band_weights: List[float] = []
    band_rows: List[Tuple[int, int, bool]] = []
    half_height = max(1, int(scan_band_half_height))

    for ratio in SCAN_BAND_RATIOS:
        center_y = int(round((roi_height - 1) * ratio))
        y0 = max(0, center_y - half_height)
        y1 = min(roi_height, center_y + half_height + 1)
        _, x_coordinates = np.nonzero(mask[y0:y1, :])
        is_valid = x_coordinates.size >= min_band_pixels
        band_rows.append((y0, y1, is_valid))
        if not is_valid:
            continue

        left = float(np.percentile(x_coordinates, 5))
        right = float(np.percentile(x_coordinates, 95))
        band_centers.append((left + right) * 0.5)
        # Squaring the relative height gives near-field observations more weight.
        band_weights.append(float(ratio * ratio))

    valid_bands = len(band_centers)
    if (
        yellow_pixels >= min_yellow_pixels
        and valid_bands >= min_valid_bands
    ):
        road_center = float(
            np.average(
                np.asarray(band_centers),
                weights=np.asarray(band_weights),
            )
        )
        return VisionResult(
            True,
            road_center,
            yellow_pixels,
            valid_bands,
            mask,
            band_rows,
            False,
        )

    # If scan bands do not describe the road well enough, use the largest
    # connected yellow component's centroid. Tiny overall masks remain invalid.
    if yellow_pixels >= min_yellow_pixels:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if component_count > 1:
            largest_label = 1 + int(
                np.argmax(stats[1:, cv2.CC_STAT_AREA])
            )
            component_mask = np.where(
                labels == largest_label, 255, 0
            ).astype(np.uint8)
            moments = cv2.moments(component_mask, binaryImage=True)
            if moments["m00"] > 0.0:
                road_center = float(moments["m10"] / moments["m00"])
                road_center = float(
                    np.clip(road_center, 0.0, image_width - 1.0)
                )
                return VisionResult(
                    True,
                    road_center,
                    yellow_pixels,
                    valid_bands,
                    mask,
                    band_rows,
                    True,
                )

    return VisionResult(
        False,
        None,
        yellow_pixels,
        valid_bands,
        mask,
        band_rows,
        False,
    )


def compute_control(
    raw_error: float,
    previous_error: Optional[float],
    smoothing_alpha: float,
    linear_speed: float,
    min_linear_speed: float,
    kp: float,
    kd: float,
    max_angular: float,
) -> Tuple[float, float, float]:
    """Filter road error and calculate adaptive linear/angular velocity."""
    if previous_error is None:
        filtered_error = raw_error
        derivative = 0.0
    else:
        filtered_error = (
            smoothing_alpha * raw_error
            + (1.0 - smoothing_alpha) * previous_error
        )
        derivative = filtered_error - previous_error

    angular = -kp * filtered_error - kd * derivative
    angular = float(np.clip(angular, -max_angular, max_angular))
    error_magnitude = min(abs(filtered_error), 1.0)
    linear = (
        linear_speed
        - (linear_speed - min_linear_speed) * error_magnitude
    )
    linear = float(
        np.clip(linear, min_linear_speed, linear_speed)
    )
    return filtered_error, linear, angular


def numpy_to_image_message(
    image: np.ndarray,
    encoding: str,
    source_msg: Image,
) -> Image:
    """Pack a contiguous uint8 numpy image without using cv_bridge."""
    packed = np.ascontiguousarray(image, dtype=np.uint8)
    output = Image()
    output.header = source_msg.header
    output.height = int(packed.shape[0])
    output.width = int(packed.shape[1])
    output.encoding = encoding
    output.is_bigendian = 0
    channels = 1 if packed.ndim == 2 else int(packed.shape[2])
    output.step = output.width * channels
    output.data = packed.tobytes()
    return output


class YellowAreaFollower(Node):
    """ROS2 node for standalone yellow-area following tests."""

    def __init__(self) -> None:
        super().__init__("yellow_area_follower")
        defaults = {
            "image_topic": "/image_out",
            "cmd_topic": "/cmd_vel_raw",
            "h_min": 15,
            "h_max": 40,
            "s_min": 45,
            "s_max": 255,
            "v_min": 80,
            "v_max": 255,
            "roi_y_start_ratio": 0.45,
            "scan_band_half_height": 8,
            "min_yellow_pixels": 800,
            "min_band_pixels": 40,
            "min_valid_bands": 2,
            "linear_speed": 0.08,
            "min_linear_speed": 0.05,
            "kp": 0.55,
            "kd": 0.08,
            "max_angular": 0.50,
            "smoothing_alpha": 0.35,
            "lost_stop_frames": 5,
            "search_on_lost": False,
            "search_angular": 0.25,
            "publish_debug_image": False,
            "debug_image_topic": "/yellow_area/debug_image",
            "debug_mask_topic": "/yellow_area/debug_mask",
            "log_interval_sec": 0.5,
            "image_timeout_sec": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.params = {
            name: self.get_parameter(name).value for name in defaults
        }
        self._validate_parameters()

        self.image_topic = str(self.params["image_topic"])
        self.cmd_topic = str(self.params["cmd_topic"])
        self.cmd_publisher = self.create_publisher(
            Twist, self.cmd_topic, 10
        )
        self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

        self.debug_image_publisher = None
        self.debug_mask_publisher = None
        if bool(self.params["publish_debug_image"]):
            self.debug_image_publisher = self.create_publisher(
                Image, str(self.params["debug_image_topic"]), 1
            )
            self.debug_mask_publisher = self.create_publisher(
                Image, str(self.params["debug_mask_topic"]), 1
            )

        self.previous_error: Optional[float] = None
        self.last_turn_direction = 1.0
        self.lost_frames = 0
        self.last_image_time = time.monotonic()
        self.last_log_time = 0.0
        self.last_error_log_time = 0.0
        self.watchdog_stopped = False
        self.create_timer(0.1, self._watchdog_callback)
        self._log_configuration()

    def _validate_parameters(self) -> None:
        p = self.params
        if not str(p["image_topic"]) or not str(p["cmd_topic"]):
            raise ValueError("image_topic and cmd_topic must not be empty")
        if not (
            0 <= int(p["h_min"]) <= int(p["h_max"]) <= 179
            and 0 <= int(p["s_min"]) <= int(p["s_max"]) <= 255
            and 0 <= int(p["v_min"]) <= int(p["v_max"]) <= 255
        ):
            raise ValueError("invalid HSV threshold ranges")
        if not 0.0 <= float(p["roi_y_start_ratio"]) < 1.0:
            raise ValueError("roi_y_start_ratio must be in [0.0, 1.0)")
        for name in (
            "scan_band_half_height",
            "min_yellow_pixels",
            "min_band_pixels",
            "min_valid_bands",
            "lost_stop_frames",
        ):
            if int(p[name]) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        linear_speed = float(p["linear_speed"])
        min_linear_speed = float(p["min_linear_speed"])
        if linear_speed < 0.0 or not 0.0 <= min_linear_speed <= linear_speed:
            raise ValueError(
                "speeds must satisfy 0 <= min_linear_speed <= linear_speed"
            )
        for name in ("kp", "kd", "max_angular", "search_angular"):
            if float(p[name]) < 0.0:
                raise ValueError(f"{name} must not be negative")
        if not 0.0 < float(p["smoothing_alpha"]) <= 1.0:
            raise ValueError("smoothing_alpha must be in (0.0, 1.0]")
        if float(p["log_interval_sec"]) <= 0.0:
            raise ValueError("log_interval_sec must be greater than zero")
        if float(p["image_timeout_sec"]) <= 0.0:
            raise ValueError("image_timeout_sec must be greater than zero")

    def _log_configuration(self) -> None:
        p = self.params
        self.get_logger().info(
            f"Yellow area follower started: image_topic={self.image_topic}, "
            f"cmd_topic={self.cmd_topic}"
        )
        self.get_logger().info(
            "HSV threshold: "
            f"H={p['h_min']}..{p['h_max']}, "
            f"S={p['s_min']}..{p['s_max']}, "
            f"V={p['v_min']}..{p['v_max']}; "
            f"roi_y_start_ratio={p['roi_y_start_ratio']}"
        )
        self.get_logger().info(
            "Control: "
            f"linear={p['linear_speed']}, min_linear={p['min_linear_speed']}, "
            f"kp={p['kp']}, kd={p['kd']}, max_angular={p['max_angular']}, "
            f"smoothing_alpha={p['smoothing_alpha']}, "
            f"lost_stop_frames={p['lost_stop_frames']}, "
            f"search_on_lost={p['search_on_lost']}"
        )

    def _image_callback(self, msg: Image) -> None:
        self.last_image_time = time.monotonic()
        self.watchdog_stopped = False
        try:
            bgr = image_message_to_bgr(msg)
            p = self.params
            result = extract_road_center(
                bgr,
                int(p["h_min"]),
                int(p["h_max"]),
                int(p["s_min"]),
                int(p["s_max"]),
                int(p["v_min"]),
                int(p["v_max"]),
                float(p["roi_y_start_ratio"]),
                int(p["scan_band_half_height"]),
                int(p["min_yellow_pixels"]),
                int(p["min_band_pixels"]),
                int(p["min_valid_bands"]),
            )
            if result.valid and result.road_center is not None:
                self._handle_valid_result(bgr, msg, result)
            else:
                self._handle_lost_result(bgr, msg, result)
        except Exception as exc:
            self.lost_frames += 1
            self.previous_error = None
            self._publish_stop()
            self._throttled_error(
                f"Image processing failed; stopping: {exc}"
            )

    def _handle_valid_result(
        self,
        bgr: np.ndarray,
        source_msg: Image,
        result: VisionResult,
    ) -> None:
        image_center = bgr.shape[1] * 0.5
        raw_error = (result.road_center - image_center) / image_center
        p = self.params
        filtered_error, linear, angular = compute_control(
            raw_error,
            self.previous_error,
            float(p["smoothing_alpha"]),
            float(p["linear_speed"]),
            float(p["min_linear_speed"]),
            float(p["kp"]),
            float(p["kd"]),
            float(p["max_angular"]),
        )
        self.previous_error = filtered_error
        if abs(filtered_error) > 1e-6:
            self.last_turn_direction = -1.0 if filtered_error > 0.0 else 1.0
        self.lost_frames = 0
        self._publish_velocity(linear, angular)
        self._throttled_status_log(
            result,
            filtered_error,
            linear,
            angular,
        )
        self._publish_debug(
            bgr, source_msg, result, filtered_error
        )

    def _handle_lost_result(
        self,
        bgr: np.ndarray,
        source_msg: Image,
        result: VisionResult,
    ) -> None:
        self.lost_frames += 1
        self.previous_error = None
        p = self.params
        angular = 0.0
        if (
            bool(p["search_on_lost"])
            and self.lost_frames <= int(p["lost_stop_frames"])
        ):
            angular = (
                self.last_turn_direction * float(p["search_angular"])
            )
            self._publish_velocity(0.0, angular)
        else:
            self._publish_stop()
        self._throttled_status_log(result, 0.0, 0.0, angular)
        self._publish_debug(bgr, source_msg, result, None)

    def _publish_velocity(self, linear: float, angular: float) -> None:
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)
        self.cmd_publisher.publish(command)

    def _publish_stop(self) -> None:
        self._publish_velocity(0.0, 0.0)

    def publish_stop_repeated(self) -> None:
        """Publish several stop messages before shutdown."""
        for _ in range(STOP_PUBLISH_COUNT):
            self._publish_stop()
            time.sleep(STOP_PUBLISH_INTERVAL_SEC)

    def _watchdog_callback(self) -> None:
        elapsed = time.monotonic() - self.last_image_time
        if elapsed < float(self.params["image_timeout_sec"]):
            return
        self._publish_stop()
        if not self.watchdog_stopped:
            self.previous_error = None
            self.watchdog_stopped = True
            self.get_logger().error(
                f"No image received for {elapsed:.2f}s; stopping."
            )

    def _throttled_status_log(
        self,
        result: VisionResult,
        error: float,
        linear: float,
        angular: float,
    ) -> None:
        now = time.monotonic()
        if now - self.last_log_time < float(
            self.params["log_interval_sec"]
        ):
            return
        self.last_log_time = now
        center_text = (
            f"{result.road_center:.1f}"
            if result.road_center is not None
            else "none"
        )
        self.get_logger().info(
            f"yellow_pixels={result.yellow_pixels}, "
            f"valid_bands={result.valid_bands}, "
            f"road_center={center_text}, error={error:.3f}, "
            f"linear.x={linear:.3f}, angular.z={angular:.3f}, "
            f"lost_frames={self.lost_frames}, "
            f"fallback={result.used_fallback}"
        )

    def _throttled_error(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_error_log_time < float(
            self.params["log_interval_sec"]
        ):
            return
        self.last_error_log_time = now
        self.get_logger().error(message)

    def _publish_debug(
        self,
        bgr: np.ndarray,
        source_msg: Image,
        result: VisionResult,
        error: Optional[float],
    ) -> None:
        if (
            self.debug_image_publisher is None
            or self.debug_mask_publisher is None
        ):
            return

        debug = bgr.copy()
        image_height, image_width = debug.shape[:2]
        roi_y_start = int(
            round(
                image_height
                * float(self.params["roi_y_start_ratio"])
            )
        )
        roi_y_start = min(max(roi_y_start, 0), image_height - 1)
        cv2.line(
            debug,
            (0, roi_y_start),
            (image_width - 1, roi_y_start),
            (255, 0, 0),
            1,
        )
        for y0, y1, valid in result.band_rows:
            color = (0, 255, 0) if valid else (0, 0, 255)
            cv2.rectangle(
                debug,
                (0, roi_y_start + y0),
                (image_width - 1, roi_y_start + y1 - 1),
                color,
                1,
            )
        cv2.line(
            debug,
            (image_width // 2, roi_y_start),
            (image_width // 2, image_height - 1),
            (255, 255, 0),
            2,
        )
        if result.road_center is not None:
            road_x = int(round(result.road_center))
            cv2.line(
                debug,
                (road_x, roi_y_start),
                (road_x, image_height - 1),
                (0, 255, 255),
                2,
            )
        error_text = "lost" if error is None else f"error={error:.3f}"
        cv2.putText(
            debug,
            error_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        full_mask[roi_y_start:, :] = result.mask
        self.debug_image_publisher.publish(
            numpy_to_image_message(debug, "bgr8", source_msg)
        )
        self.debug_mask_publisher.publish(
            numpy_to_image_message(full_mask, "mono8", source_msg)
        )


def main(args=None) -> int:
    rclpy.init(args=args)
    node: Optional[YellowAreaFollower] = None
    exit_code = 1
    try:
        node = YellowAreaFollower()
        rclpy.spin(node)
        exit_code = 0
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Ctrl+C received; publishing repeated stop.")
            node.publish_stop_repeated()
            exit_code = 0
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Fatal error; stopping: {exc}")
            node.publish_stop_repeated()
        else:
            print(f"Failed to start yellow area follower: {exc}", flush=True)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
