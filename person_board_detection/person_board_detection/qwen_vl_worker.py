"""Asynchronous real Qwen VL worker for fixed person-board crop batches."""

import base64
import hashlib
import json
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .image_quality_selector import ImageSelectionError, select_best_image
from .qwen_vl_client import QwenClientError, QwenVlClient, build_payload
from .result_protocol import (
    CaptureBatch, ProtocolError, dumps_message, make_llm_status,
    parse_capture_batch, timestamp_now,
)

EXPECTED_FILENAMES = ["crop_01.jpg", "crop_02.jpg", "crop_03.jpg"]


class PersonBoardQwenVlWorker(Node):
    def __init__(self):
        super().__init__("person_board_qwen_vl_worker")
        self._declare_parameters()
        p = lambda name: self.get_parameter(name).value
        self.capture_topic = str(p("capture_batch_topic"))
        self.status_topic = str(p("llm_status_topic"))
        self.result_topic = str(p("llm_result_topic"))
        self.backend = str(p("backend_name"))
        self.default_model = str(p("default_model"))
        self.prompt_file = str(p("prompt_file"))
        self.allowed_dir = Path(str(p("allowed_capture_directory"))).resolve()
        self.validate_images = bool(p("validate_images"))
        self.weights = (float(p("sharpness_weight")),
                        float(p("confidence_weight")), float(p("area_weight")))
        self.sharpness_saturation = float(p("sharpness_saturation"))
        self.minimum_sharpness = float(p("minimum_sharpness"))
        self.minimum_width = int(p("minimum_crop_width"))
        self.minimum_height = int(p("minimum_crop_height"))
        self.maximum_bytes = int(p("maximum_image_bytes"))
        self.request_timeout = float(p("request_timeout_sec"))
        self.deadline = float(p("overall_deadline_sec"))
        self.max_retries = int(p("max_retries"))
        self.retry_delay = float(p("retry_delay_sec"))
        self.enable_thinking = bool(p("enable_thinking"))
        self.temperature = float(p("temperature"))
        self.max_tokens = int(p("max_completion_tokens"))
        self.max_chars = int(p("max_description_chars"))
        self.max_pending = int(p("max_pending_jobs"))
        self.remember_count = int(p("remember_processed_request_count"))
        self.heartbeat = float(p("status_heartbeat_sec"))
        self.api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        self.base_url = os.environ.get("PERSON_BOARD_LLM_BASE_URL", "").strip()
        self.model = os.environ.get("PERSON_BOARD_LLM_MODEL", "").strip() or self.default_model
        self._validate_parameters()
        self.prompt = self._load_prompt()

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_pub = self.create_publisher(String, self.status_topic, qos)
        self.result_pub = self.create_publisher(String, self.result_topic, qos)
        self.sub = self.create_subscription(String, self.capture_topic,
                                            self._capture_callback, 10)
        self.jobs = queue.Queue(maxsize=self.max_pending)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.active_request_id = ""
        self.queued = set()
        self.processed = set()
        self.processed_order = deque()
        self.last_status = make_llm_status(
            "IDLE", backend=self.backend, busy=False,
            progress_text="\u7b49\u5f85\u8bc6\u522b\u4efb\u52a1")
        self._publish(self.status_pub, self.last_status)
        if self.heartbeat > 0:
            self.create_timer(self.heartbeat, self._heartbeat)
        self.thread = threading.Thread(target=self._worker_loop,
                                       name="person-board-qwen-vl", daemon=True)
        self.thread.start()
        self.get_logger().info(
            "qwen backend ready; api_key_configured=%s model=%s capture_batch=%s" %
            (bool(self.api_key), self.model, self.capture_topic))

    def _declare_parameters(self):
        defaults = {
            "capture_batch_topic": "/person_board/capture_batch",
            "llm_status_topic": "/person_board/llm_status",
            "llm_result_topic": "/person_board/llm_result",
            "backend_name": "qwen_vl", "default_model": "qwen3-vl-flash",
            "prompt_file": "",
            "allowed_capture_directory":
                "/root/intelligent_car_ws/runtime/person_board/latest_capture",
            "validate_images": True,
            "sharpness_weight": 0.55, "confidence_weight": 0.30,
            "area_weight": 0.15, "sharpness_saturation": 100.0,
            "minimum_sharpness": 30.0, "minimum_crop_width": 80,
            "minimum_crop_height": 80, "maximum_image_bytes": 5242880,
            "request_timeout_sec": 12.0, "overall_deadline_sec": 20.0,
            "max_retries": 1, "retry_delay_sec": 1.0,
            "enable_thinking": False, "temperature": 0.1,
            "max_completion_tokens": 120, "max_description_chars": 50,
            "max_pending_jobs": 1, "remember_processed_request_count": 100,
            "status_heartbeat_sec": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _validate_parameters(self):
        for name, value in (("capture_batch_topic", self.capture_topic),
                            ("llm_status_topic", self.status_topic),
                            ("llm_result_topic", self.result_topic),
                            ("backend_name", self.backend),
                            ("default_model", self.default_model)):
            if not value:
                raise ValueError(name + " must not be empty")
        if abs(sum(self.weights) - 1.0) > 1e-6 or min(self.weights) < 0:
            raise ValueError("quality weights must be non-negative and sum to 1")
        if self.sharpness_saturation <= 0 or self.minimum_sharpness < 0:
            raise ValueError("sharpness parameters invalid")
        if min(self.minimum_width, self.minimum_height, self.maximum_bytes) <= 0:
            raise ValueError("image limits must be positive")
        if self.request_timeout <= 0 or self.deadline <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_retries < 0 or self.retry_delay < 0:
            raise ValueError("retry parameters invalid")
        if self.max_pending != 1 or self.remember_count < 1:
            raise ValueError("queue parameters invalid")
        if not self.validate_images:
            raise ValueError("validate_images must remain true for this backend")
        if not 0 <= self.temperature <= 2 or self.max_tokens <= 0 or self.max_chars <= 0:
            raise ValueError("model parameters invalid")

    def _load_prompt(self):
        path = Path(self.prompt_file) if self.prompt_file else (
            Path(get_package_share_directory("person_board_detection")) /
            "prompts/hospital_board_description_zh.txt")
        text = path.read_text(encoding="utf-8").strip()
        if "JSON" not in text:
            raise ValueError("prompt_missing_json_instruction")
        return text

    @staticmethod
    def _publish(pub, payload):
        msg = String()
        msg.data = dumps_message(payload)
        pub.publish(msg)

    def _status(self, state, busy, batch=None, progress="", error=""):
        payload = make_llm_status(
            state, backend=self.backend, busy=busy,
            event_id=batch.event_id if batch else "",
            request_id=batch.request_id if batch else "",
            progress_text=progress, error_reason=error)
        with self.lock:
            self.last_status = payload
        self._publish(self.status_pub, payload)

    def _heartbeat(self):
        with self.lock:
            value = dict(self.last_status)
            value["updated_at"] = timestamp_now()
            self.last_status = value
        self._publish(self.status_pub, value)

    def _capture_callback(self, msg):
        try:
            batch = parse_capture_batch(msg.data)
        except ProtocolError as exc:
            self._status("FAILED", False, error=exc.reason)
            return
        with self.lock:
            duplicate = (batch.request_id in self.processed or
                         batch.request_id in self.queued or
                         batch.request_id == self.active_request_id)
            busy = self.jobs.full()
            if not duplicate and not busy:
                self.queued.add(batch.request_id)
        if duplicate:
            self._status("REJECTED_DUPLICATE", busy, batch,
                         "\u91cd\u590d\u4efb\u52a1\u5df2\u62d2\u7edd",
                         "duplicate_request_id")
            return
        if busy:
            self._status("REJECTED_BUSY", True, batch,
                         "\u5de5\u4f5c\u8282\u70b9\u7e41\u5fd9", "worker_busy")
            return
        self.jobs.put_nowait(batch)
        self._status("RECEIVED", True, batch, "\u5df2\u63a5\u6536\u8bc6\u522b\u4efb\u52a1")

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                batch = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.lock:
                self.queued.discard(batch.request_id)
                self.active_request_id = batch.request_id
            try:
                self._process(batch)
            except Exception as exc:
                self.get_logger().error("unhandled worker error: %s" % type(exc).__name__)
                self._failure(batch, "worker_internal_error", timestamp_now(), time.monotonic(), 0)
            finally:
                self._remember(batch.request_id)
                with self.lock:
                    self.active_request_id = ""
                self.jobs.task_done()

    def _validate_batch(self, batch):
        allowed = self.allowed_dir
        manifest_path = Path(batch.manifest_path).resolve()
        if manifest_path.parent != allowed or manifest_path.name != "manifest.json":
            raise ProtocolError("manifest_path_not_allowed")
        if not manifest_path.is_file():
            raise ProtocolError("manifest_not_found")
        if list(allowed.glob("*.tmp")):
            raise ProtocolError("capture_contains_tmp_file")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ProtocolError("manifest_invalid_json")
        if manifest.get("event_id") != batch.event_id:
            raise ProtocolError("manifest_event_id_mismatch")
        if manifest.get("request_id") != batch.request_id:
            raise ProtocolError("manifest_request_id_mismatch")
        paths = manifest.get("image_paths")
        frames = manifest.get("frames")
        if not isinstance(paths, list) or len(paths) != 3:
            raise ProtocolError("manifest_image_count_invalid")
        if not isinstance(frames, list) or len(frames) != 3:
            raise ProtocolError("manifest_frame_count_invalid")
        if len(batch.image_paths) != 3:
            raise ProtocolError("capture_batch_image_count_invalid")
        resolved = [str(Path(value).resolve()) for value in batch.image_paths]
        manifest_resolved = [str(Path(value).resolve()) for value in paths]
        if resolved != manifest_resolved:
            raise ProtocolError("image_paths_mismatch")
        if [Path(value).name for value in resolved] != EXPECTED_FILENAMES:
            raise ProtocolError("image_filenames_invalid")
        if any(Path(value).parent != allowed for value in map(Path, resolved)):
            raise ProtocolError("image_path_not_allowed")
        return manifest, resolved

    def _process(self, batch):
        started_wall, started_mono = timestamp_now(), time.monotonic()
        attempts = 0
        try:
            self._status("VALIDATING", True, batch, "\u6b63\u5728\u6821\u9a8c\u56fe\u7247\u2026")
            manifest, paths = self._validate_batch(batch)
            self._status("SELECTING_IMAGE", True, batch, "\u6b63\u5728\u9009\u62e9\u6700\u4f73\u56fe\u7247\u2026")
            selected = select_best_image(
                manifest["frames"], [Path(value) for value in paths],
                sharpness_weight=self.weights[0],
                confidence_weight=self.weights[1], area_weight=self.weights[2],
                sharpness_saturation=self.sharpness_saturation,
                minimum_sharpness=self.minimum_sharpness,
                minimum_crop_width=self.minimum_width,
                minimum_crop_height=self.minimum_height)
            for score in selected.scores:
                self.get_logger().info(
                    "quality filename=%s valid=%s rejection=%s sharpness=%.6f "
                    "confidence=%.6f area=%d score=%.9f" %
                    (score.filename, score.valid, score.rejection_reason,
                     score.sharpness, score.yolo_confidence, score.crop_area,
                     score.final_quality_score))
            image_path = Path(selected.selected_image_path)
            self._status("ENCODING_IMAGE", True, batch, "\u6b63\u5728\u51c6\u5907\u56fe\u7247\u2026")
            data = image_path.read_bytes()
            if len(data) > self.maximum_bytes:
                raise ProtocolError("image_too_large")
            decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ProtocolError("selected_image_decode_failed")
            digest = hashlib.sha256(data).hexdigest()
            encoded = base64.b64encode(data).decode("ascii")
            self.get_logger().info(
                "selected=%s bytes=%d dimensions=%dx%d base64_length=%d sha256=%s" %
                (image_path, len(data), decoded.shape[1], decoded.shape[0],
                 len(encoded), digest))
            client = QwenVlClient(
                self.api_key, self.base_url, self.model, self.request_timeout,
                self.deadline, self.max_retries, self.retry_delay)
            payload = build_payload(
                self.model, "data:image/jpeg;base64," + encoded, self.prompt,
                self.enable_thinking, self.temperature, self.max_tokens)
            self._status("ANALYZING", True, batch, "\u6b63\u5728\u5206\u6790\u2026")
            try:
                response = client.complete(payload, self.max_chars)
            except QwenClientError:
                self._log_attempts(client.attempt_log)
                raise
            self._log_attempts(client.attempt_log)
            attempts = response.attempts
            self._status("PARSING", True, batch, "\u6b63\u5728\u6574\u7406\u7ed3\u679c\u2026")
            result = {
                "schema_version": 1, "success": True, "backend": self.backend,
                "model": self.model, "event_id": batch.event_id,
                "request_id": batch.request_id, "result_text": response.description,
                "manifest_path": batch.manifest_path, "image_paths": paths,
                "selected_image_path": str(image_path),
                "selected_image_index": selected.selected_image_index,
                "selected_quality_score": selected.selected_quality_score,
                "recognizable": response.recognizable,
                "started_at": started_wall, "completed_at": timestamp_now(),
                "elapsed_ms": round((time.monotonic() - started_mono) * 1000, 3),
                "api_attempts": attempts, "error_reason": "",
            }
            self._publish(self.result_pub, result)
            self._status("SUCCEEDED", False, batch, response.description)
        except (ProtocolError, ImageSelectionError, QwenClientError, OSError) as exc:
            reason = getattr(exc, "reason", type(exc).__name__)
            attempts = getattr(exc, "attempts", attempts)
            self._failure(batch, reason, started_wall, started_mono, attempts)

    def _failure(self, batch, reason, started_wall, started_mono, attempts):
        result = {
            "schema_version": 1, "success": False, "backend": self.backend,
            "model": self.model, "event_id": batch.event_id,
            "request_id": batch.request_id, "result_text": "",
            "manifest_path": batch.manifest_path, "image_paths": [],
            "selected_image_path": "", "selected_image_index": 0,
            "selected_quality_score": 0.0, "recognizable": False,
            "started_at": started_wall, "completed_at": timestamp_now(),
            "elapsed_ms": round(max(0.0, (time.monotonic() - started_mono) * 1000), 3),
            "api_attempts": attempts, "error_reason": reason,
        }
        self._publish(self.result_pub, result)
        self._status("FAILED", False, batch, "\u8bc6\u522b\u5931\u8d25", reason)

    def _remember(self, request_id):
        with self.lock:
            if request_id in self.processed:
                return
            self.processed.add(request_id)
            self.processed_order.append(request_id)
            while len(self.processed_order) > self.remember_count:
                self.processed.discard(self.processed_order.popleft())

    def _log_attempts(self, entries):
        for entry in entries:
            self.get_logger().info(
                "api attempt=%d http_status=%d elapsed_ms=%.3f "
                "retry_reason=%s retried=%s" %
                (entry["attempt"], entry["http_status"], entry["elapsed_ms"],
                 entry["retry_reason"], entry["retried"]))

    def destroy_node(self):
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=max(2.0, self.request_timeout + 1.0))
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PersonBoardQwenVlWorker()
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
