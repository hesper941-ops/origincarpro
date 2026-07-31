"""Standard-library OpenAI-compatible Qwen VL client."""

import json
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

FALLBACK_DESCRIPTION = "\u7acb\u724c\u753b\u9762\u6a21\u7cca\uff0c\u65e0\u6cd5\u53ef\u9760\u8bc6\u522b\u3002"
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


class QwenClientError(RuntimeError):
    def __init__(self, reason, retryable=False, status=0, attempts=0):
        super().__init__(reason)
        self.reason, self.retryable = reason, retryable
        self.status, self.attempts = status, attempts


@dataclass
class QwenResponse:
    recognizable: bool
    description: str
    attempts: int
    elapsed_ms: float
    http_status: int
    metadata: dict


def endpoint_url(base_url):
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise QwenClientError("base_url_not_configured")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise QwenClientError("base_url_invalid")
    return base + "/chat/completions"


def build_payload(model, data_url, prompt, enable_thinking, temperature,
                  max_completion_tokens):
    if not model:
        raise QwenClientError("model_not_configured")
    if "JSON" not in prompt:
        raise QwenClientError("prompt_missing_json_instruction")
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]}],
        "response_format": {"type": "json_object"},
        "enable_thinking": bool(enable_thinking),
        "temperature": float(temperature),
        "max_completion_tokens": int(max_completion_tokens),
        "stream": False,
    }


def parse_response(body, max_chars):
    try:
        outer = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenClientError("response_invalid_json") from exc
    choices = outer.get("choices")
    if not isinstance(choices, list) or not choices:
        raise QwenClientError("response_choices_empty")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise QwenClientError("response_content_empty")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise QwenClientError("model_content_invalid_json") from exc
    if not isinstance(result, dict) or set(result) != {"recognizable", "description"}:
        raise QwenClientError("model_content_fields_invalid")
    recognizable, description = result["recognizable"], result["description"]
    if not isinstance(recognizable, bool):
        raise QwenClientError("recognizable_invalid")
    if not isinstance(description, str) or not description.strip():
        raise QwenClientError("description_empty")
    description = description.strip()
    if description.count(chr(96)) >= 3 or "\n" in description or "\r" in description:
        raise QwenClientError("description_format_invalid")
    if len(description) > max_chars:
        raise QwenClientError("description_too_long")
    forbidden = ("\u6839\u636e\u56fe\u7247", "\u5206\u6790\uff1a", "\u63cf\u8ff0\uff1a")
    if description.startswith(forbidden):
        raise QwenClientError("description_analysis_prefix_not_allowed")
    if recognizable and description == FALLBACK_DESCRIPTION:
        raise QwenClientError("recognizable_description_conflict")
    if not recognizable and description != FALLBACK_DESCRIPTION:
        raise QwenClientError("unrecognizable_description_invalid")
    return recognizable, description, outer


class QwenVlClient:
    def __init__(self, api_key, base_url, model, request_timeout_sec,
                 overall_deadline_sec, max_retries, retry_delay_sec):
        if not api_key:
            raise QwenClientError("api_key_not_configured")
        self.url, self.model = endpoint_url(base_url), model
        if not model:
            raise QwenClientError("model_not_configured")
        self.api_key = api_key
        self.request_timeout = float(request_timeout_sec)
        self.overall_deadline = float(overall_deadline_sec)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay_sec)
        self.attempt_log = []

    def complete(self, payload, max_chars):
        started = time.monotonic()
        deadline = started + self.overall_deadline
        last = QwenClientError("api_request_failed")
        for attempt in range(1, self.max_retries + 2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QwenClientError("overall_deadline_exceeded", attempts=attempt - 1)
            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": "Bearer " + self.api_key,
                         "Content-Type": "application/json"},
                method="POST")
            began = time.monotonic()
            try:
                kwargs = {"timeout": min(self.request_timeout, remaining)}
                if self.url.startswith("https:"):
                    kwargs["context"] = ssl.create_default_context()
                with urllib.request.urlopen(req, **kwargs) as response:
                    body, status = response.read(), int(response.status)
                recognizable, description, outer = parse_response(body, max_chars)
                self.attempt_log.append(self._entry(attempt, status, began, "", False))
                return QwenResponse(
                    recognizable, description, attempt,
                    (time.monotonic() - started) * 1000.0, status,
                    {"id": outer.get("id", ""), "model": outer.get("model", ""),
                     "usage": outer.get("usage", {})})
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                last = QwenClientError("http_%d" % status,
                                       status in RETRYABLE_HTTP, status, attempt)
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                reason = "socket_timeout" if (
                    isinstance(exc, (socket.timeout, TimeoutError)) or
                    isinstance(getattr(exc, "reason", None), socket.timeout)
                ) else "connection_failed"
                last = QwenClientError(reason, True, attempts=attempt)
            except QwenClientError as exc:
                exc.attempts = attempt
                raise
            retry = (last.retryable and attempt <= self.max_retries and
                     time.monotonic() + self.retry_delay < deadline)
            self.attempt_log.append(self._entry(
                attempt, last.status, began, last.reason, retry))
            if not retry:
                raise last
            time.sleep(self.retry_delay)
        raise last

    @staticmethod
    def _entry(attempt, status, began, reason, retry):
        return {"attempt": attempt, "http_status": status,
                "elapsed_ms": round((time.monotonic() - began) * 1000.0, 3),
                "retry_reason": reason, "retried": retry}
