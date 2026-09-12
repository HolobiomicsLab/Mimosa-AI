"""Thin in-process client for the explicitly configured completion bridge."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Any


BRIDGE_ENV = "HARNESS_COMPLETION_BRIDGE"
CLI_BACKENDS = {"codex-cli": "codex_cli", "claude-cli": "claude_cli"}
RESULT_STATUSES = {"completed", "failed", "timeout", "malformed", "unsupported"}
RESULT_FIELDS = {
    "protocol_version",
    "status",
    "text",
    "requested_model",
    "actual_model",
    "backend",
    "auth_mode",
    "usage_kind",
    "usage",
    "cost_usd",
    "cost_kind",
    "cli_version",
    "diagnostic_count",
    "error",
}


class CompletionBackendError(RuntimeError):
    """Report a bridge load, protocol, or completion failure."""


def is_cli_completion_provider(provider: str) -> bool:
    """Return whether a provider selects an explicit harness CLI backend."""
    return provider.lower() in CLI_BACKENDS


def load_completion_bridge(bridge_path: str | None = None) -> ModuleType:
    """Load the trusted bridge module from its explicit absolute path."""
    configured_path = bridge_path if bridge_path is not None else os.getenv(BRIDGE_ENV, "")
    if not configured_path:
        raise CompletionBackendError(f"{BRIDGE_ENV} is required for CLI completion models")
    bridge_path = Path(configured_path)
    if not bridge_path.is_absolute():
        raise CompletionBackendError(f"{BRIDGE_ENV} must be an absolute file path")
    if not bridge_path.is_file():
        raise CompletionBackendError(f"completion bridge file does not exist: {bridge_path}")
    suffix = hashlib.sha256(str(bridge_path).encode()).hexdigest()[:12]
    module_name = f"_mimosa_harness_completion_{suffix}"
    spec = importlib.util.spec_from_file_location(module_name, bridge_path)
    if spec is None or spec.loader is None:
        raise CompletionBackendError(f"cannot load completion bridge: {bridge_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "complete", None)):
        raise CompletionBackendError("completion bridge must define complete(request)")
    return module


def _validate_completed_result(
    result: dict[str, Any], request: dict[str, Any]
) -> None:
    """Reject a completed envelope with malformed route or provenance data."""
    expected_route = {
        "backend": request.get("backend"),
        "requested_model": request.get("model"),
        "auth_mode": request.get("auth_mode"),
    }
    mismatches = {
        field: (expected, result.get(field))
        for field, expected in expected_route.items()
        if result.get(field) != expected
    }
    if mismatches:
        raise CompletionBackendError(
            f"completion bridge returned mismatched route metadata: {mismatches}"
        )
    if not isinstance(result.get("text"), str) or not result["text"].strip():
        raise CompletionBackendError(
            "completed bridge result must contain non-empty text"
        )
    actual_model = result.get("actual_model")
    if actual_model is not None and (
        not isinstance(actual_model, str) or not actual_model.strip()
    ):
        raise CompletionBackendError(
            "completion bridge returned malformed model identity metadata"
        )
    for field in ("cli_version", "usage_kind", "cost_kind"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise CompletionBackendError(
                f"completion bridge returned malformed {field} metadata"
            )
    usage = result.get("usage")
    if usage is not None and (
        not isinstance(usage, dict)
        or any(
            value is not None and (type(value) is not int or value < 0)
            for value in usage.values()
        )
    ):
        raise CompletionBackendError("completion bridge returned malformed token usage")
    cost = result.get("cost_usd")
    if cost is not None and (
        type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
    ):
        raise CompletionBackendError("completion bridge returned malformed cost")


def _identity_model(entry: Any, source: str) -> str | None:
    """Read one sourced model claim without inferring missing telemetry."""
    if entry is None:
        return None
    if (
        not isinstance(entry, dict)
        or set(entry) != {"model", "source"}
        or entry.get("source") != source
        or not isinstance(entry.get("model"), str)
        or not entry["model"].strip()
    ):
        raise CompletionBackendError("completion bridge returned malformed model identity")
    return entry["model"]


def _validate_model_identity(result: dict[str, Any], request: dict[str, Any]) -> None:
    """Check optional v1 provenance while accepting older bridge envelopes."""
    if "model_identity" not in result:
        return
    identity = result["model_identity"]
    if not isinstance(identity, dict) or set(identity) != {"requested", "configured", "reported"}:
        raise CompletionBackendError("completion bridge returned malformed model identity")
    requested = _identity_model(identity["requested"], "request")
    configured = _identity_model(identity["configured"], "explicit_cli_argument")
    source = "codex.turn.completed.model" if request["backend"] == "codex_cli" else "claude.modelUsage"
    reported = _identity_model(identity["reported"], source)
    if (
        requested != request.get("model")
        or configured not in (None, requested)
        or (result["status"] == "completed" and configured is None)
        or reported != result.get("actual_model")
        or (reported is not None and configured is None)
    ):
        raise CompletionBackendError("completion bridge returned inconsistent model identity")


def call_completion_bridge(
    request: dict[str, Any], bridge_path: str | None = None
) -> dict[str, Any]:
    """Call the configured bridge once and validate its result envelope."""
    bridge = load_completion_bridge() if bridge_path is None else load_completion_bridge(bridge_path)
    result = bridge.complete(request)
    if not isinstance(result, dict):
        raise CompletionBackendError("completion bridge returned a non-dictionary result")
    if not RESULT_FIELDS.issubset(result):
        raise CompletionBackendError("completion bridge returned an invalid result shape")
    if result.get("protocol_version") != 1:
        raise CompletionBackendError("completion bridge returned an unsupported protocol version")
    if result.get("status") not in RESULT_STATUSES:
        raise CompletionBackendError("completion bridge returned an unknown status")
    diagnostic_count = result.get("diagnostic_count")
    if type(diagnostic_count) is not int or diagnostic_count < 0:
        raise CompletionBackendError(
            "completion bridge returned malformed diagnostic metadata"
        )
    if result["status"] == "completed":
        _validate_completed_result(result, request)
    _validate_model_identity(result, request)
    return result
