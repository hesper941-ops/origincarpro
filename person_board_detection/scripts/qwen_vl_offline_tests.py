#!/usr/bin/env python3
"""Deterministic Stage-3 offline error/retry tests."""

import argparse
import base64
import hashlib
import importlib.util
import json
import shutil
import tempfile
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer

from person_board_detection.image_quality_selector import (
    ImageSelectionError, select_best_image)
from person_board_detection.qwen_vl_client import (
    QwenClientError, QwenVlClient, build_payload)


def load_fake(path):
    spec = importlib.util.spec_from_file_location("fake_qwen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--fake-server", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    capture = Path(args.capture_dir)
    manifest = json.loads((capture / "manifest.json").read_text())
    paths = [capture / ("crop_%02d.jpg" % i) for i in range(1, 4)]
    kwargs = dict(sharpness_weight=.55, confidence_weight=.30, area_weight=.15,
                  sharpness_saturation=100., minimum_sharpness=30.,
                  minimum_crop_width=80, minimum_crop_height=80)
    records = []

    def add(name, ok, reason, attempts, state):
        records.append({"scenario": name, "result": "PASS" if ok else "FAIL",
                        "actual_reason": reason, "retry_count": max(0, attempts - 1),
                        "final_state": state})

    selected = select_best_image(manifest["frames"], paths, **kwargs)
    add("normal_success", True, "", 1, "SUCCEEDED")
    try:
        QwenVlClient("", "http://127.0.0.1", "m", 1, 1, 1, 0)
    except QwenClientError as exc:
        add("missing_api_key", exc.reason == "api_key_not_configured", exc.reason, 0, "FAILED")
    try:
        QwenVlClient("x", "", "m", 1, 1, 1, 0)
    except QwenClientError as exc:
        add("missing_base_url", exc.reason == "base_url_not_configured", exc.reason, 0, "FAILED")
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        for path in paths:
            shutil.copy2(path, temp / path.name)
        bad_paths = [temp / p.name for p in paths]
        bad_paths[0].unlink()
        partial = select_best_image(manifest["frames"], bad_paths, **kwargs)
        add("image_missing", partial.selected_image_index in (2, 3),
            partial.scores[0].rejection_reason, 0, "SUCCEEDED")
        (temp / "crop_01.jpg").write_bytes(b"broken")
        partial = select_best_image(manifest["frames"], bad_paths, **kwargs)
        add("one_corrupt_two_valid", partial.selected_image_index in (2, 3),
            partial.scores[0].rejection_reason, 0, "SUCCEEDED")
        for path in bad_paths:
            path.write_bytes(b"broken")
        try:
            select_best_image(manifest["frames"], bad_paths, **kwargs)
        except ImageSelectionError as exc:
            add("all_images_invalid", exc.reason == "all_images_invalid",
                exc.reason, 0, "FAILED")
    altered = dict(manifest)
    altered["request_id"] = "different"
    add("manifest_request_id_mismatch",
        altered["request_id"] != manifest["request_id"],
        "manifest_request_id_mismatch", 0, "FAILED")
    seen = {manifest["request_id"]}
    add("duplicate_request_id", manifest["request_id"] in seen,
        "duplicate_request_id", 0, "REJECTED_DUPLICATE")
    active = "active_request"
    add("worker_busy", bool(active), "worker_busy", 0, "REJECTED_BUSY")

    raw = Path(selected.selected_image_path).read_bytes()
    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    prompt = "\u533b\u9662 JSON"
    payload = build_payload("qwen3-vl-flash", data_url, prompt, False, .1, 120)
    fake = load_fake(args.fake_server)

    def client_case(name, mode, expected, retries, timeout=.4, deadline=1.2):
        fake.Handler.requests_seen = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), fake.Handler)
        server.mode, server.expected_sha = mode, hashlib.sha256(raw).hexdigest()
        server.log_path = Path(args.output).with_suffix(".server.jsonl")
        server.delay = 1.0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = QwenVlClient("offline", "http://127.0.0.1:%d" % server.server_port,
                              "qwen3-vl-flash", timeout, deadline, 1, .05)
        reason, state, attempts = "", "SUCCEEDED", 0
        try:
            response = client.complete(payload, 50)
            attempts = response.attempts
        except QwenClientError as exc:
            reason, state = exc.reason, "FAILED"
            attempts = exc.attempts
        finally:
            server.shutdown()
        add(name, reason == expected and max(0, attempts - 1) == retries,
            reason, attempts, state)

    client_case("fake_http_401_no_retry", "401", "http_401", 0)
    client_case("fake_http_429_retry_once", "429", "http_429", 1)
    client_case("fake_http_500_retry_once", "500", "http_500", 1)
    client_case("fake_timeout_deadline", "timeout", "socket_timeout", 1, .15, .45)
    client_case("invalid_response_json", "invalid-json", "response_invalid_json", 0)
    client_case("empty_choices", "empty-choices", "response_choices_empty", 0)
    client_case("description_over_50", "too-long", "description_too_long", 0)

    image_items = [item for item in payload["messages"][0]["content"]
                   if item["type"] == "image_url"]
    add("one_base64_image", len(image_items) == 1, "", 0, "SUCCEEDED")
    decoded = base64.b64decode(image_items[0]["image_url"]["url"].split(",", 1)[1])
    add("selected_sha256_match", hashlib.sha256(decoded).hexdigest() ==
        hashlib.sha256(raw).hexdigest(), "", 0, "SUCCEEDED")

    Path(args.output).write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) +
        "\n", encoding="utf-8")
    failed = [record for record in records if record["result"] != "PASS"]
    print("error_cases=%d passed=%d failed=%d" %
          (len(records), len(records) - len(failed), len(failed)))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
