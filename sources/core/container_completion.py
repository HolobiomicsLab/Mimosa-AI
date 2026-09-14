"""Credential-free completion transport for an explicitly staged container."""

import json
import sys
import threading
import uuid

FRAME_PREFIX = b"MIMOSA_COMPLETION_V1 "
MAX_FRAME_BYTES = 1024 * 1024
_LOCK = threading.Lock()


def parse_request(raw):
    """Accept only a correlation identity and messages, never provider settings."""
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError("completion frame exceeds byte limit")
    request = json.loads(raw)
    if not isinstance(request, dict) or set(request) != {"id", "messages"}:
        raise ValueError("untrusted completion fields")
    identifier = request["id"]
    if (
        not isinstance(identifier, str)
        or len(identifier) != 32
        or any(value not in "0123456789abcdef" for value in identifier)
    ):
        raise ValueError("invalid completion identity")
    return request


def complete(request):
    """Exchange text messages with the host; its policy and budget are binding."""
    identifier = uuid.uuid4().hex
    raw = json.dumps(
        {"id": identifier, "messages": request["messages"]}, allow_nan=False
    ).encode()
    parse_request(raw)
    with _LOCK:
        sys.__stdout__.buffer.write(FRAME_PREFIX + raw + b"\n")
        sys.__stdout__.buffer.flush()
        line = sys.__stdin__.buffer.readline(MAX_FRAME_BYTES + 1)
    if len(line) > MAX_FRAME_BYTES or not line.endswith(b"\n"):
        raise ValueError("invalid completion response frame")
    response = json.loads(line)
    if (
        not isinstance(response, dict)
        or set(response) != {"id", "result"}
        or response["id"] != identifier
    ):
        raise ValueError("completion response identity mismatch")
    return response["result"]
