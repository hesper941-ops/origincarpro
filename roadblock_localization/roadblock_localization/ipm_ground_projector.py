"""Pure OpenCV/Numpy image-pixel to vehicle-ground projection."""

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import yaml


class IPMGroundProjector:
    """Load a fixed IPM calibration and project distorted image pixels to X/Y.

    Ground coordinates use the calibration convention: origin at the chassis
    rotation center, +X forward, +Y left, and Z=0.  This class has no ROS or
    detection-message dependencies.
    """

    _PINHOLE_MODELS = ("plumb_bob", "rational_polynomial", "")
    _FISHEYE_MODELS = ("equidistant", "fisheye")

    def __init__(self, calibration_file):
        self.calibration_file = Path(calibration_file).expanduser().resolve()
        if not self.calibration_file.is_file():
            raise FileNotFoundError(f"IPM calibration not found: {self.calibration_file}")

        with self.calibration_file.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("IPM calibration YAML root must be a mapping")

        self.camera_matrix = self._matrix(data, "camera_matrix", (3, 3))
        self.distortion_coefficients = self._vector(data, "distortion_coefficients")
        self.h_image_to_ground = self._matrix(data, "H_image_to_ground", (3, 3))
        self.distortion_model = str(data.get("distortion_model", "")).strip().lower()
        self.image_width = self._positive_int(data, "image_width")
        self.image_height = self._positive_int(data, "image_height")

        if self.distortion_model not in self._PINHOLE_MODELS + self._FISHEYE_MODELS:
            raise ValueError(f"unsupported distortion_model: {self.distortion_model!r}")
        if self.distortion_model in self._FISHEYE_MODELS and self.distortion_coefficients.size != 4:
            raise ValueError("equidistant/fisheye calibration requires exactly 4 distortion coefficients")
        if self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0:
            raise ValueError("camera_matrix focal lengths must be positive")
        if abs(float(np.linalg.det(self.h_image_to_ground))) < 1e-15:
            raise ValueError("H_image_to_ground is singular")

    @staticmethod
    def _matrix(data, key: str, shape):
        if key not in data:
            raise KeyError(f"missing calibration field: {key}")
        value = np.asarray(data[key], dtype=np.float64)
        if value.size != int(np.prod(shape)):
            raise ValueError(f"{key} must contain {int(np.prod(shape))} values")
        value = value.reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN/Inf")
        return value

    @staticmethod
    def _vector(data, key: str):
        if key not in data:
            raise KeyError(f"missing calibration field: {key}")
        value = np.asarray(data[key], dtype=np.float64).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"{key} is empty or contains NaN/Inf")
        return value

    @staticmethod
    def _positive_int(data, key: str) -> int:
        if key not in data:
            raise KeyError(f"missing calibration field: {key}")
        value = int(data[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        return value

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        result = self.pixel_to_ground_many(((u, v),))
        return float(result[0, 0]), float(result[0, 1])

    def pixel_to_ground_many(self, points: Iterable[Tuple[float, float]]) -> np.ndarray:
        pixels = np.asarray(tuple(points), dtype=np.float64)
        if pixels.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("points must have shape Nx2")
        if not np.isfinite(pixels).all():
            raise ValueError("pixel coordinates must be finite")
        if (
            np.any(pixels[:, 0] < 0.0)
            or np.any(pixels[:, 0] > self.image_width)
            or np.any(pixels[:, 1] < 0.0)
            or np.any(pixels[:, 1] > self.image_height)
        ):
            raise ValueError(
                f"pixel outside calibrated image boundary [0,{self.image_width}] x [0,{self.image_height}]"
            )

        shaped = pixels.reshape(-1, 1, 2)
        if self.distortion_model in self._FISHEYE_MODELS:
            undistorted = cv2.fisheye.undistortPoints(
                shaped,
                self.camera_matrix,
                self.distortion_coefficients.reshape(-1, 1),
                P=self.camera_matrix,
            )
        else:
            undistorted = cv2.undistortPoints(
                shaped,
                self.camera_matrix,
                self.distortion_coefficients,
                P=self.camera_matrix,
            )

        undistorted = undistorted.reshape(-1, 2)
        homogeneous = np.column_stack((undistorted, np.ones(len(undistorted), dtype=np.float64)))
        projected = (self.h_image_to_ground @ homogeneous.T).T
        scales = projected[:, 2]
        if not np.isfinite(projected).all() or np.any(np.abs(scales) < 1e-12):
            raise ValueError("homography produced invalid or near-zero homogeneous scale")
        ground = projected[:, :2] / scales[:, None]
        if not np.isfinite(ground).all():
            raise ValueError("ground projection produced NaN/Inf")
        return ground
