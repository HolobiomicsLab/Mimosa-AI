"""One trusted host completion with a private, authoritative native budget."""

import json
import sys

from .container_completion import MAX_FRAME_BYTES, parse_request
from .harness_model import _SAFE_ENVELOPE_FIELDS, HarnessCompletionModel


def main():
    """Read host-owned policy plus an untrusted frame and emit a safe response."""
    raw = sys.stdin.buffer.read(2 * MAX_FRAME_BYTES + 1)
    if len(raw) > 2 * MAX_FRAME_BYTES:
        raise ValueError("host completion input exceeds byte limit")
    packet = json.loads(raw)
    if set(packet) != {"model_id", "settings", "request"}:
        raise ValueError("invalid host completion packet")
    request = parse_request(json.dumps(packet["request"]).encode())
    model = HarnessCompletionModel(packet["model_id"], packet["settings"])
    message = model.generate(request["messages"])
    model.assert_healthy()
    envelope = {name: message.raw.get(name) for name in _SAFE_ENVELOPE_FIELDS}
    envelope.update(
        protocol_version=1,
        requested_model=model.model_id,
        backend="codex_cli",
        auth_mode="subscription",
        text=message.content,
        error=None,
    )
    response = (
        json.dumps({"id": request["id"], "result": envelope}, allow_nan=False).encode()
        + b"\n"
    )
    if len(response) > MAX_FRAME_BYTES:
        raise ValueError("host completion response exceeds byte limit")
    sys.stdout.buffer.write(response)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
