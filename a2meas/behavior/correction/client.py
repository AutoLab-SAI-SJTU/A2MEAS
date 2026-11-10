"""Bounded structured requests shared by local policy workers."""

from contextlib import contextmanager
from copy import copy
import hashlib
import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from filelock import FileLock, Timeout

from .. import RUNTIME_ROOT
from .grounding import CAMERAS, image_url


class ReviewError(ValueError):
    """A failed request or invalid response cannot authorize correction."""


class ReviewClient:
    def __init__(self, *, timeout=180):
        self.url = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8001/v1").rstrip(
            "/"
        )
        self.model = "qwen3.8-27b"
        self.timeout = timeout
        self.concurrency = int(os.environ.get("A2MEAS_REVIEW_CONCURRENCY", "28"))
        if self.concurrency < 1:
            raise ValueError("A2MEAS_REVIEW_CONCURRENCY must be positive")
        endpoint = urllib.parse.urlsplit(self.url)
        if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
            raise ValueError("QWEN_BASE_URL must be an HTTP(S) endpoint")
        self.opener = self._opener()
        key = hashlib.sha256(self.url.encode()).hexdigest()[:16]
        self.slots = RUNTIME_ROOT / "cache" / "review-slots" / key
        self.slots.mkdir(parents=True, exist_ok=True)

    def _opener(self):
        host = urllib.parse.urlsplit(self.url).hostname
        try:
            private = ipaddress.ip_address(host).is_private
        except ValueError:
            private = host == "localhost"
        return (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if private
            else urllib.request.build_opener()
        )

    def fork_for_background(self):
        client = copy(self)
        client.opener = client._opener()
        return client

    @contextmanager
    def slot(self, deadline):
        while True:
            for index in range(self.concurrency):
                lock = FileLock(self.slots / f"{index}.lock")
                try:
                    lock.acquire(timeout=0)
                except Timeout:
                    continue
                try:
                    yield
                finally:
                    lock.release()
                return
            if time.monotonic() >= deadline:
                raise ReviewError("Timed out waiting for a shared review slot")
            time.sleep(0.05)

    def request(self, payload):
        deadline = time.monotonic() + self.timeout
        request = urllib.request.Request(
            self.url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.slot(deadline):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ReviewError("Review deadline exceeded")
                with self.opener.open(request, timeout=remaining) as response:
                    result = json.load(response)
            choice = result["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ReviewError("Incomplete review response")
            value = json.loads(choice["message"]["content"])
            if not isinstance(value, dict):
                raise ReviewError("Expected a structured review object")
            return value
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise ReviewError(str(error)) from error

    def build_request(self, *, images, frame_steps, **kwargs):
        content = [{"type": "text", "text": ""}]
        for step, frame in zip(frame_steps, images, strict=True):
            for camera, rgb in zip(CAMERAS, frame, strict=True):
                content.extend(
                    [
                        {"type": "text", "text": f"step={step} camera={camera}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url(rgb, quality=85)},
                        },
                    ]
                )
        return {
            "model": self.model,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_schema", "json_schema": {}},
        }

    def infer(self, **kwargs):
        value = self.request(self.build_request(**kwargs))
        try:
            return self.validate_response(value)
        except (ValueError, TypeError) as error:
            raise ReviewError(str(error)) from error
