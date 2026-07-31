"""Asynchronous mock multimodal worker for fixed person-board crop batches."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .result_protocol import (
    CaptureBatch,
    ProtocolError,
    dumps_message,
    make_llm_result,
    make_llm_status,
    parse_capture_batch,
    parse_json_object,
    timestamp_now,
)


EXPECTED_FILENAMES = ["crop_01.jpg", "crop_02.jpg", "crop_03.jpg"]


class PersonBoardMockLlmWorker(Node):
    def __init__(self) -> None:
        super().__init__("person_board_mock_llm_worker")
        self._declare_parameters()
        self.capture_batch_topic = str(
            self.get_parameter("capture_batch_topic").value
        )
        self.llm_status_topic = str(
            self.get_parameter("llm_status_topic").value
        )
        self.llm_result_topic = str(
            self.get_parameter("llm_result_topic").value
        )
        self.backend_name = str(
            self.get_parameter("backend_name").value
        ).strip()
        self.mock_result_text = str(
            self.get_parameter("mock_result_text").value
        )
        self.processing_delay = float(
            self.get_parameter("mock_processing_delay_sec").value
        )
        self.validate_images = bool(
            self.get_parameter("validate_images").value
        )
        self.allowed_directory = (
            Path(str(self.get_parameter("allowed_capture_directory").value))
            .expanduser()
            .resolve()
        )
        self.max_pending_jobs = int(
            self.get_parameter("max_pending_jobs").value
        )
        self.remember_count = int(
            self.get_parameter("remember_processed_request_count").value
        )
        self.heartbeat_sec = float(
            self.get_parameter("status_heartbeat_sec").value
        )
        self._validate_parameters()

        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_publisher = self.create_publisher(
            String, self.llm_status_topic, latched_qos
        )
        self.result_publisher = self.create_publisher(
            String, self.llm_result_topic, latched_qos
        )
        self.subscription = self.create_subscription(
            String, self.capture_batch_topic, self._capture_callback, 10
        )
        self.jobs: "queue.Queue[CaptureBatch]" = queue.Queue(
            maxsize=self.max_pending_jobs
        )
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.active_request_id = ""
        self.queued_request_ids: Set[str] = set()
        self.processed_request_ids: Set[str] = set()
        self.finalized_request_ids: Set[str] = set()
        self.processed_order: Deque[str] = deque()
        self.last_status = make_llm_status(
            "IDLE",
            backend=self.backend_name,
            busy=False,
            progress_text="等待识别任务",
        )
        self._publish_json(self.status_publisher, self.last_status)
        if self.heartbeat_sec > 0.0:
            self.create_timer(self.heartbeat_sec, self._heartbeat)
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="person-board-mock-llm",
            daemon=True,
        )
        self.worker_thread.start()
        self.get_logger().info(
            f"mock backend ready; capture_batch={self.capture_batch_topic}, "
            f"allowed_directory={self.allowed_directory}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "capture_batch_topic": "/person_board/capture_batch",
            "llm_status_topic": "/person_board/llm_status",
            "llm_result_topic": "/person_board/llm_result",
            "backend_name": "mock",
            "mock_result_text": "检测到人形立牌",
            "mock_processing_delay_sec": 1.5,
            "validate_images": True,
            "allowed_capture_directory": (
                "/root/intelligent_car_ws/runtime/person_board/latest_capture"
            ),
            "max_pending_jobs": 1,
            "remember_processed_request_count": 100,
            "status_heartbeat_sec": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _validate_parameters(self) -> None:
        for name, value in (
            ("capture_batch_topic", self.capture_batch_topic),
            ("llm_status_topic", self.llm_status_topic),
            ("llm_result_topic", self.llm_result_topic),
            ("backend_name", self.backend_name),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        if self.processing_delay < 0.0:
            raise ValueError("mock_processing_delay_sec must be non-negative")
        if self.max_pending_jobs != 1:
            raise ValueError("max_pending_jobs must be exactly 1")
        if self.remember_count < 1:
            raise ValueError(
                "remember_processed_request_count must be positive"
            )
        if self.heartbeat_sec < 0.0:
            raise ValueError("status_heartbeat_sec must be non-negative")

    @staticmethod
    def _publish_json(publisher: Any, payload: Dict[str, Any]) -> None:
        message = String()
        message.data = dumps_message(payload)
        publisher.publish(message)

    def _publish_status(
        self,
        state: str,
        *,
        busy: bool,
        batch: Optional[CaptureBatch] = None,
        progress_text: str = "",
        error_reason: str = "",
    ) -> None:
        payload = make_llm_status(
            state,
            backend=self.backend_name,
            busy=busy,
            event_id=batch.event_id if batch else "",
            request_id=batch.request_id if batch else "",
            progress_text=progress_text,
            error_reason=error_reason,
        )
        with self.lock:
            self.last_status = payload
        self._publish_json(self.status_publisher, payload)

    def _heartbeat(self) -> None:
        with self.lock:
            payload = dict(self.last_status)
            payload["updated_at"] = timestamp_now()
            self.last_status = payload
        self._publish_json(self.status_publisher, payload)

    def _capture_callback(self, message: String) -> None:
        try:
            batch = parse_capture_batch(message.data)
        except ProtocolError as exc:
            self.get_logger().error(f"capture_batch rejected: {exc.reason}")
            self._publish_status("FAILED", busy=False, error_reason=exc.reason)
            return
        with self.lock:
            duplicate = (
                batch.request_id in self.processed_request_ids
                or batch.request_id in self.queued_request_ids
                or batch.request_id == self.active_request_id
            )
            busy = bool(self.active_request_id) or self.jobs.full()
            if not duplicate and not busy:
                self.queued_request_ids.add(batch.request_id)
        if duplicate:
            self._publish_status(
                "REJECTED_DUPLICATE",
                busy=bool(self.active_request_id),
                batch=batch,
                progress_text="重复任务已拒绝",
                error_reason="duplicate_request_id",
            )
            return
        if busy:
            self._publish_status(
                "REJECTED_BUSY",
                busy=True,
                batch=batch,
                progress_text="工作节点繁忙",
                error_reason="worker_busy",
            )
            return
        try:
            self.jobs.put_nowait(batch)
        except queue.Full:
            with self.lock:
                self.queued_request_ids.discard(batch.request_id)
            self._publish_status(
                "REJECTED_BUSY",
                busy=True,
                batch=batch,
                error_reason="worker_busy",
            )
            return
        self._publish_status(
            "RECEIVED",
            busy=True,
            batch=batch,
            progress_text="已收到三张裁剪图",
        )

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                batch = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.lock:
                self.queued_request_ids.discard(batch.request_id)
                self.active_request_id = batch.request_id
            try:
                self._process_batch(batch)
            except (
                Exception
            ) as exc:  # converted into an explicit final failure
                self.get_logger().exception(f"worker failure: {exc}")
                self._publish_failure(
                    batch, "worker_exception", timestamp_now(), 0.0
                )
            finally:
                self._remember_processed(batch.request_id)
                with self.lock:
                    self.active_request_id = ""
                self.jobs.task_done()

    def _process_batch(self, batch: CaptureBatch) -> None:
        started_perf = time.perf_counter()
        started_at = timestamp_now()
        self._publish_status(
            "VALIDATING",
            busy=True,
            batch=batch,
            progress_text="正在校验裁剪图",
        )
        try:
            image_paths, manifest_path = self._validate_batch(batch)
        except ProtocolError as exc:
            elapsed_ms = (time.perf_counter() - started_perf) * 1000.0
            self._publish_failure(batch, exc.reason, started_at, elapsed_ms)
            return
        self._publish_status(
            "ANALYZING",
            busy=True,
            batch=batch,
            progress_text="正在分析人形立牌",
        )
        if self.stop_event.wait(self.processing_delay):
            return
        elapsed_ms = (time.perf_counter() - started_perf) * 1000.0
        result = make_llm_result(
            success=True,
            backend=self.backend_name,
            event_id=batch.event_id,
            request_id=batch.request_id,
            result_text=self.mock_result_text,
            manifest_path=str(manifest_path),
            image_paths=[str(path) for path in image_paths],
            started_at=started_at,
            elapsed_ms=elapsed_ms,
        )
        self._publish_final_result(batch, result)
        self._publish_status(
            "SUCCEEDED",
            busy=False,
            batch=batch,
            progress_text=self.mock_result_text,
        )

    def _publish_failure(
        self,
        batch: CaptureBatch,
        reason: str,
        started_at: str,
        elapsed_ms: float,
    ) -> None:
        result = make_llm_result(
            success=False,
            backend=self.backend_name,
            event_id=batch.event_id,
            request_id=batch.request_id,
            result_text="",
            manifest_path=batch.manifest_path,
            image_paths=[],
            started_at=started_at,
            elapsed_ms=elapsed_ms,
            error_reason=reason,
        )
        if not self._publish_final_result(batch, result):
            return
        self._publish_status(
            "FAILED", busy=False, batch=batch, error_reason=reason
        )

    def _publish_final_result(
        self, batch: CaptureBatch, payload: Dict[str, Any]
    ) -> bool:
        with self.lock:
            if batch.request_id in self.finalized_request_ids:
                self.get_logger().error(
                    f"final result already published: {batch.request_id}"
                )
                return False
            self.finalized_request_ids.add(batch.request_id)
        self._publish_json(self.result_publisher, payload)
        return True

    def _validate_batch(self, batch: CaptureBatch) -> Tuple[List[Path], Path]:
        if len(batch.image_paths) != 3:
            raise ProtocolError("capture_batch_image_count_invalid")
        manifest_path = self._allowed_path(
            batch.manifest_path, "manifest_path"
        )
        if manifest_path.name != "manifest.json":
            raise ProtocolError("manifest_filename_invalid")
        if manifest_path.name.endswith(".tmp"):
            raise ProtocolError("temporary_file_not_allowed")
        if not manifest_path.is_file():
            raise ProtocolError("manifest_not_found")
        try:
            manifest = parse_json_object(
                manifest_path.read_text(encoding="utf-8"),
                "manifest_invalid_json",
            )
        except OSError as exc:
            raise ProtocolError("manifest_read_failed") from exc
        if manifest.get("request_id") != batch.request_id:
            raise ProtocolError("manifest_request_id_mismatch")
        if manifest.get("event_id") != batch.event_id:
            raise ProtocolError("manifest_event_id_mismatch")
        manifest_images = manifest.get("image_paths")
        if not isinstance(manifest_images, list) or len(manifest_images) != 3:
            raise ProtocolError("manifest_image_count_invalid")
        if manifest_images != batch.image_paths:
            raise ProtocolError("manifest_image_paths_mismatch")
        frames = manifest.get("frames")
        if not isinstance(frames, list) or len(frames) != 3:
            raise ProtocolError("manifest_frame_count_invalid")
        image_paths = [
            self._allowed_path(path, "image_path")
            for path in batch.image_paths
        ]
        if [path.name for path in image_paths] != EXPECTED_FILENAMES:
            raise ProtocolError("image_filename_invalid")
        for path in image_paths:
            if path.name.endswith(".tmp") or ".tmp" in path.suffixes:
                raise ProtocolError("temporary_file_not_allowed")
            if not path.is_file():
                raise ProtocolError("image_not_found")
            try:
                if path.stat().st_size <= 0:
                    raise ProtocolError("image_empty")
            except OSError as exc:
                raise ProtocolError("image_stat_failed") from exc
            if self.validate_images:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ProtocolError("image_decode_failed")
                if image.shape[0] <= 0 or image.shape[1] <= 0:
                    raise ProtocolError("image_dimensions_invalid")
        return image_paths, manifest_path

    def _allowed_path(self, raw_path: str, field: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ProtocolError(f"{field}_invalid")
        path = Path(raw_path).expanduser().resolve()
        try:
            path.relative_to(self.allowed_directory)
        except ValueError as exc:
            raise ProtocolError("path_outside_allowed_directory") from exc
        return path

    def _remember_processed(self, request_id: str) -> None:
        with self.lock:
            if request_id in self.processed_request_ids:
                return
            self.processed_request_ids.add(request_id)
            self.processed_order.append(request_id)
            while len(self.processed_order) > self.remember_count:
                expired = self.processed_order.popleft()
                self.processed_request_ids.discard(expired)
                self.finalized_request_ids.discard(expired)

    def close(self) -> None:
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(
                timeout=max(2.0, self.processing_delay + 0.5)
            )

    def destroy_node(self) -> bool:
        self.close()
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[PersonBoardMockLlmWorker] = None
    try:
        node = PersonBoardMockLlmWorker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
