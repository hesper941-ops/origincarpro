"""Three-frame capture and fixed-directory atomic batch publication."""

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CapturedFrame:
    crop: np.ndarray
    source_timestamp_sec: int
    source_timestamp_nanosec: int
    bbox: Tuple[int, int, int, int]
    padded_bbox: Tuple[int, int, int, int]
    yolo_confidence: float
    sharpness: float


class CaptureManager:
    """Keeps one latest batch; manifest replacement is the commit marker."""

    DEFAULT_SUBDIRECTORY = "latest_capture"
    _SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

    @classmethod
    def is_valid_event_id(cls, event_id: str) -> bool:
        return bool(cls._SAFE_EVENT_ID.fullmatch(event_id))

    def __init__(
        self,
        runtime_directory: str,
        fixed_capture_subdirectory: str,
        required_count: int,
        max_attempts: int,
        interval_ms: int,
        timeout_sec: float,
        jpeg_quality: int,
        source_topic: str,
    ) -> None:
        subdirectory = Path(fixed_capture_subdirectory)
        if (
            subdirectory.is_absolute()
            or ".." in subdirectory.parts
            or len(subdirectory.parts) != 1
        ):
            raise ValueError(
                "fixed_capture_subdirectory must be one safe relative directory name"
            )
        self.capture_directory = Path(runtime_directory).expanduser() / subdirectory
        self.required_count = required_count
        self.max_attempts = max_attempts
        self.interval_sec = interval_ms / 1000.0
        self.timeout_sec = timeout_sec
        self.jpeg_quality = jpeg_quality
        self.source_topic = source_topic
        self.reset_batch()

    def reset_batch(self) -> None:
        self.event_id = ""
        self.request_id = ""
        self.frames: List[CapturedFrame] = []
        self.attempts = 0
        self.started_at = 0.0
        self.last_attempt_at = 0.0
        self.last_rejection_reason = ""

    def begin(self, event_id: str, now: float) -> str:
        if not self.is_valid_event_id(event_id):
            raise ValueError(
                "event_id must contain only letters, digits, dot, underscore, or hyphen"
            )
        self.reset_batch()
        self.event_id = event_id
        self.request_id = f"req_{uuid.uuid4().hex}"
        self.started_at = now
        return self.request_id

    def may_attempt(self, now: float) -> bool:
        return (
            not self.last_attempt_at or now - self.last_attempt_at >= self.interval_sec
        )

    def timed_out(self, now: float) -> bool:
        return bool(self.started_at and now - self.started_at > self.timeout_sec)

    def record_rejection(self, reason: str, now: float) -> None:
        self.attempts += 1
        self.last_attempt_at = now
        self.last_rejection_reason = reason

    def add_frame(self, frame: CapturedFrame, now: float) -> None:
        self.attempts += 1
        self.last_attempt_at = now
        self.frames.append(frame)
        self.last_rejection_reason = ""

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts and not self.complete

    @property
    def complete(self) -> bool:
        return len(self.frames) >= self.required_count

    def paths(self) -> Tuple[List[Path], Path]:
        images = [
            self.capture_directory / f"crop_{index:02d}.jpg"
            for index in range(1, self.required_count + 1)
        ]
        return images, self.capture_directory / "manifest.json"

    def commit(self) -> Dict[str, object]:
        if len(self.frames) != self.required_count:
            raise RuntimeError(
                f"expected {self.required_count} frames, got {len(self.frames)}"
            )
        self.capture_directory.mkdir(parents=True, exist_ok=True)
        image_paths, manifest_path = self.paths()
        image_temps = [Path(str(path) + ".tmp") for path in image_paths]
        manifest_temp = Path(str(manifest_path) + ".tmp")
        temporary_paths = image_temps + [manifest_temp]
        created_at = datetime.now(timezone.utc).isoformat()
        previous_images = [
            path.read_bytes() if path.exists() else None for path in image_paths
        ]
        try:
            for captured, temp_path in zip(self.frames, image_temps):
                success, encoded = cv2.imencode(
                    ".jpg", captured.crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not success or encoded.size == 0:
                    raise OSError(f"JPEG encoding failed for {temp_path.name}")
                self._write_bytes(temp_path, encoded.tobytes())
            manifest = self._build_manifest(image_paths, created_at)
            self._write_bytes(
                manifest_temp,
                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            for temp_path, final_path in zip(image_temps, image_paths):
                os.replace(temp_path, final_path)
            os.replace(manifest_temp, manifest_path)
        except Exception:
            for temp_path in temporary_paths:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._restore_previous_images(image_paths, image_temps, previous_images)
            raise
        return {
            "event_id": self.event_id,
            "request_id": self.request_id,
            "manifest_path": str(manifest_path),
            "image_paths": [str(path) for path in image_paths],
        }

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        with path.open("wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())

    def _restore_previous_images(
        self,
        image_paths: Sequence[Path],
        image_temps: Sequence[Path],
        previous_images: Sequence[Optional[bytes]],
    ) -> None:
        """Best-effort rollback if replacement fails before manifest commit."""
        for final_path, temp_path, previous_data in zip(
            image_paths, image_temps, previous_images
        ):
            try:
                if previous_data is None:
                    final_path.unlink(missing_ok=True)
                else:
                    self._write_bytes(temp_path, previous_data)
                    os.replace(temp_path, final_path)
            except OSError:
                # The old manifest remains authoritative and no message is sent.
                pass

    def _build_manifest(
        self, image_paths: Sequence[Path], created_at: str
    ) -> Dict[str, object]:
        frame_entries = []
        for captured in self.frames:
            crop_height, crop_width = captured.crop.shape[:2]
            frame_entries.append(
                {
                    "event_id": self.event_id,
                    "request_id": self.request_id,
                    "source_timestamp_sec": captured.source_timestamp_sec,
                    "source_timestamp_nanosec": captured.source_timestamp_nanosec,
                    "bbox": list(captured.bbox),
                    "padded_bbox": list(captured.padded_bbox),
                    "yolo_confidence": captured.yolo_confidence,
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                    "sharpness": captured.sharpness,
                }
            )
        return {
            "schema_version": 1,
            "event_id": self.event_id,
            "request_id": self.request_id,
            "source_topic": self.source_topic,
            "image_paths": [str(path) for path in image_paths],
            "frames": frame_entries,
            "created_at": created_at,
            "status": "ready",
        }
