"""Quality checks for unannotated crops from original BGR frames."""

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .bbox_utils import edge_touch_ratio


@dataclass(frozen=True)
class CropQualityResult:
    accepted: bool
    sharpness: float
    reason: str


def laplacian_sharpness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_crop(
    crop: np.ndarray,
    source_bbox: Sequence[float],
    image_width: int,
    image_height: int,
    minimum_width: int,
    minimum_height: int,
    minimum_sharpness: float,
    maximum_edge_touch_ratio: float,
) -> CropQualityResult:
    if crop.size == 0:
        return CropQualityResult(False, 0.0, "empty_crop")
    crop_height, crop_width = crop.shape[:2]
    if crop_width < minimum_width or crop_height < minimum_height:
        return CropQualityResult(False, 0.0, "crop_too_small")
    if (
        edge_touch_ratio(source_bbox, image_width, image_height)
        > maximum_edge_touch_ratio
    ):
        return CropQualityResult(False, 0.0, "target_severely_touches_edge")
    sharpness = laplacian_sharpness(crop)
    if sharpness < minimum_sharpness:
        return CropQualityResult(False, sharpness, "crop_not_sharp")
    return CropQualityResult(True, sharpness, "")
