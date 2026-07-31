#!/usr/bin/env python3
"""Local OpenAI-compatible fake server; never logs Authorization or Base64."""

import argparse
import base64
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    server_version = "QwenVlFake/1"
    requests_seen = 0

    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        type(self).requests_seen += 1
        mode = self.server.mode
        if mode == "timeout":
            time.sleep(self.server.delay)
        if mode in ("401", "429", "500"):
            self.send_error(int(mode))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            summary = validate(payload, self.path, bool(self.headers.get("Authorization")),
                               self.server.expected_sha)
            summary["request_number"] = type(self).requests_seen
            with self.server.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except Exception as exc:
            self.send_error(400, str(exc))
            return
        if mode == "invalid-json":
            body = b"{invalid"
        else:
            choices = []
            if mode != "empty-choices":
                description = ("\u8fd9\u662f\u4e00\u6bb5\u8d85\u8fc7\u4e94\u5341\u4e2a\u5b57\u7b26\u9650\u5236"
                               * 8 if mode == "too-long" else
                               "\u4e00\u540d\u533b\u62a4\u4eba\u5458\u63a8\u7740\u5750\u8f6e\u6905\u7684\u60a3\u8005\u524d\u884c\u3002")
                choices = [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content":
                                json.dumps({"recognizable": True,
                                            "description": description},
                                           ensure_ascii=False)}}]
            body = json.dumps({
                "id": "chatcmpl-local-test", "object": "chat.completion",
                "model": payload["model"], "choices": choices,
                "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                          "total_tokens": 120}}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def validate(payload, path, authorization_present, expected_sha):
    if path != "/chat/completions":
        raise ValueError("path_invalid")
    if not authorization_present:
        raise ValueError("authorization_missing")
    if not payload.get("model"):
        raise ValueError("model_missing")
    content = payload["messages"][0]["content"]
    images = [x for x in content if x.get("type") == "image_url"]
    texts = [x for x in content if x.get("type") == "text"]
    if len(images) != 1:
        raise ValueError("image_count_not_one")
    url = images[0]["image_url"]["url"]
    prefix = "data:image/jpeg;base64,"
    if not url.startswith(prefix):
        raise ValueError("image_data_url_invalid")
    raw = base64.b64decode(url[len(prefix):], validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha and digest != expected_sha:
        raise ValueError("selected_sha_mismatch")
    prompt = texts[0]["text"]
    if "\u533b\u9662" not in prompt or "JSON" not in prompt:
        raise ValueError("prompt_invalid")
    if payload.get("response_format") != {"type": "json_object"}:
        raise ValueError("response_format_invalid")
    if payload.get("enable_thinking") is not False or payload.get("stream") is not False:
        raise ValueError("request_flags_invalid")
    return {
        "path": path, "authorization_present": True, "image_count": 1,
        "decoded_image_sha256": digest, "selected_sha256_match":
            not expected_sha or digest == expected_sha,
        "model": payload["model"], "prompt_contains_hospital": True,
        "prompt_contains_JSON": True, "response_format": "json_object",
        "enable_thinking": False, "stream": False,
        "image_bytes": len(raw), "base64_length": len(url) - len(prefix),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--mode", choices=("success", "401", "429", "500",
                        "timeout", "invalid-json", "too-long", "empty-choices"),
                        default="success")
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--log", default="/tmp/fake_qwen_vl_server.log")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.mode, server.expected_sha = args.mode, args.expected_sha256
    server.log_path, server.delay = Path(args.log), args.delay
    server.serve_forever()


if __name__ == "__main__":
    main()
