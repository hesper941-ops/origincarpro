#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黄色通道 HSV 阈值标定工具（只采集图像，不发布速度）。

推荐使用流程
============

第一步，启动相机：

    cd /root/intelligent_car_ws
    source /opt/tros/humble/setup.bash
    source install/setup.bash

    ros2 launch origincar_bringup competition_bringup.launch.py \
      enable_base:=false \
      enable_camera:=true \
      enable_yolo_avoid:=false \
      enable_qr:=false \
      enable_birdview:=false \
      enable_nav:=false \
      enable_cobridge:=false \
      button_backend:=topic

如果 launch 不支持 enable_nav，请删除 enable_nav 参数。

第二步，把小车相机对准黄色通道，运行标定：

    python3 tools/yellow_threshold_calib.py \
      --ros-args \
      -p image_topic:=/image_out \
      -p output_yaml:=tools/yellow_area_params.yaml \
      -p sample_frames:=60

第三步，运行黄色区域寻路：

    python3 tools/yellow_area_follow.py \
      --ros-args \
      --params-file tools/yellow_area_params.yaml

第四步，发车门锁：

    ros2 topic pub --once /competition/start_button \
      std_msgs/msg/Bool "{data: true}"

标定时应让黄色通道尽量占据画面下半部分。程序在每帧 ROI 中用宽松 HSV
范围寻找黄色，并且只收集最大连通区域的像素。收集完成后会打印可直接复制的
ROS 参数，并生成 yellow_area_follower 节点可加载的 YAML 文件。
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def image_message_to_bgr(msg: Image) -> np.ndarray:
    """Convert a supported sensor_msgs/Image to a packed BGR numpy array."""
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
        # Remove any per-row padding before passing the NV12 plane to OpenCV.
        yuv = raw[:required].reshape(yuv_rows, step)[:, :row_bytes]
        return cv2.cvtColor(
            np.ascontiguousarray(yuv), cv2.COLOR_YUV2BGR_NV12
        )

    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


def largest_candidate_pixels(
    bgr: np.ndarray,
    roi_y_start_ratio: float,
    morphology_kernel_size: int,
) -> Tuple[np.ndarray, int]:
    """Return HSV pixels and area from the largest loose-yellow ROI component."""
    height = bgr.shape[0]
    y_start = int(round(height * roi_y_start_ratio))
    y_start = min(max(y_start, 0), height - 1)
    roi = bgr[y_start:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    loose_lower = np.array([8, 30, 50], dtype=np.uint8)
    loose_upper = np.array([50, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, loose_lower, loose_upper)

    kernel_size = max(1, int(morphology_kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if component_count <= 1:
        return np.empty((0, 3), dtype=np.uint8), 0

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest_label, cv2.CC_STAT_AREA])
    return hsv[labels == largest_label], area


class YellowThresholdCalibrator(Node):
    """Collect images, estimate robust yellow HSV limits, and write ROS YAML."""

    def __init__(self) -> None:
        super().__init__("yellow_threshold_calibrator")

        self.declare_parameter("image_topic", "/image_out")
        self.declare_parameter("output_yaml", "tools/yellow_area_params.yaml")
        self.declare_parameter("sample_frames", 60)
        self.declare_parameter("roi_y_start_ratio", 0.45)
        self.declare_parameter("h_margin", 4)
        self.declare_parameter("s_margin", 20)
        self.declare_parameter("v_margin", 20)
        self.declare_parameter("morphology_kernel_size", 5)
        self.declare_parameter("min_candidate_area", 800)
        self.declare_parameter("image_timeout_sec", 10.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.output_yaml = str(self.get_parameter("output_yaml").value)
        self.sample_frames = int(self.get_parameter("sample_frames").value)
        self.roi_y_start_ratio = float(
            self.get_parameter("roi_y_start_ratio").value
        )
        self.h_margin = int(self.get_parameter("h_margin").value)
        self.s_margin = int(self.get_parameter("s_margin").value)
        self.v_margin = int(self.get_parameter("v_margin").value)
        self.kernel_size = int(
            self.get_parameter("morphology_kernel_size").value
        )
        self.min_candidate_area = int(
            self.get_parameter("min_candidate_area").value
        )
        self.image_timeout_sec = float(
            self.get_parameter("image_timeout_sec").value
        )

        self._validate_parameters()
        self._received_frames = 0
        self._candidate_frames = 0
        self._candidate_pixel_count = 0
        self._largest_candidate_area = 0
        self._samples = []
        self._start_time = time.monotonic()
        self._last_usable_image_time: Optional[float] = None
        self._done = False
        self.succeeded = False

        self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(0.25, self._check_timeout)

        self.get_logger().info(
            "Yellow HSV calibration started: "
            f"image_topic={self.image_topic}, sample_frames={self.sample_frames}, "
            f"roi_y_start_ratio={self.roi_y_start_ratio:.3f}, "
            f"output_yaml={self.output_yaml}"
        )

    def _validate_parameters(self) -> None:
        if not self.image_topic:
            raise ValueError("image_topic must not be empty")
        if not self.output_yaml:
            raise ValueError("output_yaml must not be empty")
        if self.sample_frames <= 0:
            raise ValueError("sample_frames must be greater than zero")
        if not 0.0 <= self.roi_y_start_ratio < 1.0:
            raise ValueError("roi_y_start_ratio must be in [0.0, 1.0)")
        if min(self.h_margin, self.s_margin, self.v_margin) < 0:
            raise ValueError("HSV margins must not be negative")
        if self.kernel_size <= 0:
            raise ValueError("morphology_kernel_size must be greater than zero")
        if self.min_candidate_area <= 0:
            raise ValueError("min_candidate_area must be greater than zero")
        if self.image_timeout_sec <= 0.0:
            raise ValueError("image_timeout_sec must be greater than zero")

    def _image_callback(self, msg: Image) -> None:
        if self._done:
            return

        try:
            bgr = image_message_to_bgr(msg)
        except (ValueError, cv2.error) as exc:
            self.get_logger().warning(f"Image conversion failed: {exc}")
            return

        self._last_usable_image_time = time.monotonic()
        self._received_frames += 1

        try:
            pixels, area = largest_candidate_pixels(
                bgr, self.roi_y_start_ratio, self.kernel_size
            )
        except cv2.error as exc:
            self.get_logger().warning(f"Yellow candidate extraction failed: {exc}")
            return

        self._candidate_pixel_count += area
        self._largest_candidate_area = max(self._largest_candidate_area, area)
        if area >= self.min_candidate_area:
            self._samples.append(pixels)
            self._candidate_frames += 1

        if (
            self._received_frames == 1
            or self._received_frames % 10 == 0
            or self._received_frames == self.sample_frames
        ):
            self.get_logger().info(
                f"collected frames: {self._received_frames}/{self.sample_frames}; "
                f"candidate pixel count: {self._candidate_pixel_count}"
            )

        if self._received_frames >= self.sample_frames:
            self._finish_calibration()

    def _check_timeout(self) -> None:
        if self._done:
            return
        now = time.monotonic()
        reference = self._last_usable_image_time or self._start_time
        if now - reference >= self.image_timeout_sec:
            self._fail(
                f"Timed out after {self.image_timeout_sec:.1f}s waiting for a "
                f"usable image on {self.image_topic}. Check the topic, camera, "
                "and supported encoding (bgr8/rgb8/mono8/nv12)."
            )

    def _finish_calibration(self) -> None:
        self.get_logger().info(
            f"collected frames: {self._received_frames}; "
            f"candidate frames: {self._candidate_frames}; "
            f"candidate pixel count: {self._candidate_pixel_count}"
        )

        if not self._samples:
            self._fail(
                "yellow candidate too small; YAML was not written "
                f"(largest area={self._largest_candidate_area}, required="
                f"{self.min_candidate_area}). Check the camera, lighting, ROI, "
                "and whether the camera is aimed at the yellow channel."
            )
            return

        all_pixels = np.concatenate(self._samples, axis=0)
        h_values = all_pixels[:, 0]
        s_values = all_pixels[:, 1]
        v_values = all_pixels[:, 2]

        h_min = int(np.clip(np.floor(np.percentile(h_values, 5)) - self.h_margin, 0, 179))
        h_max = int(np.clip(np.ceil(np.percentile(h_values, 95)) + self.h_margin, 0, 179))
        s_min = int(np.clip(np.floor(np.percentile(s_values, 5)) - self.s_margin, 0, 255))
        v_min = int(np.clip(np.floor(np.percentile(v_values, 5)) - self.v_margin, 0, 255))
        thresholds = (h_min, h_max, s_min, 255, v_min, 255)

        self.get_logger().info(
            "final HSV threshold: "
            f"H={h_min}..{h_max}, S={s_min}..255, V={v_min}..255"
        )
        print(
            f"-p h_min:={h_min} -p h_max:={h_max} "
            f"-p s_min:={s_min} -p s_max:=255 "
            f"-p v_min:={v_min} -p v_max:=255",
            flush=True,
        )

        try:
            output_path = self._write_yaml(thresholds)
        except OSError as exc:
            self._fail(f"Failed to write output YAML {self.output_yaml!r}: {exc}")
            return

        self.get_logger().info(f"output yaml path: {output_path}")
        self.succeeded = True
        self._stop()

    def _write_yaml(
        self, thresholds: Tuple[int, int, int, int, int, int]
    ) -> Path:
        h_min, h_max, s_min, s_max, v_min, v_max = thresholds
        output_path = Path(self.output_yaml).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        yaml_text = f"""yellow_area_follower:
  ros__parameters:
    image_topic: {self.image_topic}
    cmd_topic: /cmd_vel_raw
    h_min: {h_min}
    h_max: {h_max}
    s_min: {s_min}
    s_max: {s_max}
    v_min: {v_min}
    v_max: {v_max}
    roi_y_start_ratio: {self.roi_y_start_ratio}
    scan_band_half_height: 8
    min_yellow_pixels: 800
    min_band_pixels: 40
    min_valid_bands: 2
    linear_speed: 0.08
    min_linear_speed: 0.05
    kp: 0.55
    kd: 0.08
    max_angular: 0.50
    smoothing_alpha: 0.35
    lost_stop_frames: 5
    search_on_lost: false
"""
        temp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=str(output_path.parent),
                delete=False,
            ) as temp_file:
                temp_name = temp_file.name
                temp_file.write(yaml_text)
            os.replace(temp_name, output_path)
        except Exception:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
            raise
        return output_path

    def _fail(self, message: str) -> None:
        self.get_logger().error(message)
        self.succeeded = False
        self._stop()

    def _stop(self) -> None:
        self._done = True
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None) -> int:
    rclpy.init(args=args)
    node: Optional[YellowThresholdCalibrator] = None
    exit_code = 1
    try:
        node = YellowThresholdCalibrator()
        rclpy.spin(node)
        exit_code = 0 if node.succeeded else 1
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Calibration interrupted; no velocity was published.")
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Calibration failed: {exc}")
        else:
            print(f"Calibration failed: {exc}", flush=True)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
