"""Durable call budgets for the opt-in native completion adapter.

The JSONL ledger is an operational guardrail shared by model instances and
processes on one host.  It deliberately stores no prompts, model output, or
exception text.  A reservation without one terminal completion is treated as
an unknown outcome and prevents another dispatch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import psutil
import re
import time
from types import MappingProxyType
from typing import Any, Iterator
import uuid


LEDGER_VERSION = 1
LOCK_POLL_SECONDS = 0.01
MAX_LEDGER_LINE_BYTES = 64 * 1024
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_REQUIRED_SETTINGS = {
    "bridge_path",
    "bridge_sha256",
    "ledger_path",
    "max_calls",
    "total_timeout_seconds",
    "call_timeout_seconds",
    "reasoning_effort",
}
_OPTIONAL_SETTINGS = {"max_observed_tokens"}
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_CALL_FIELDS = {
    "backend",
    "auth_mode",
    "model",
    "reasoning_effort",
    "agent_name_sha256",
    "transcript_sha256",
    "transcript_bytes",
    "message_count",
}
_COMPLETION_STATUSES = {
    "completed",
    "failed",
    "timeout",
    "malformed",
    "unsupported",
    "unknown",
}
_SAFE_COMPLETION_FIELDS = {
    "status",
    "model_identity",
    "actual_model",
    "observed_models",
    "usage",
    "observed_total_tokens",
    "usage_kind",
    "cost_usd",
    "cost_kind",
    "cli_version",
    "diagnostic_count",
    "elapsed_seconds",
    "response_sha256",
    "response_bytes",
    "stop_applied",
    "stop_sequence_sha256",
}
_BUDGET_STATUSES = {
    "within_limit",
    "missing_observed_tokens",
    "observed_token_limit_exceeded",
    "total_deadline_exceeded",
}
_USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
}


class HarnessBudgetError(RuntimeError):
    """Report a durable budget, lock, or poisoned-ledger failure."""


@dataclass(frozen=True)
class CallReservation:
    """One fsynced dispatch reservation and its remaining deadline."""

    reservation_id: str
    timeout_seconds: int
    started_monotonic: float
    deadline_monotonic: float


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def validate_native_harness_settings(settings: object) -> dict[str, Any]:
    """Return a normalized copy of one exact, side-effect-free configuration."""
    if not isinstance(settings, dict):
        raise ValueError("native harness settings must be a dictionary")
    fields = set(settings)
    missing = _REQUIRED_SETTINGS - fields
    unsupported = fields - _REQUIRED_SETTINGS - _OPTIONAL_SETTINGS
    if missing:
        raise ValueError(f"native harness settings are missing: {sorted(missing)}")
    if unsupported:
        raise ValueError(f"native harness settings contain unsupported keys: {sorted(unsupported)}")

    normalized = dict(settings)
    normalized.setdefault("max_observed_tokens", None)
    for field in ("bridge_path", "ledger_path"):
        value = normalized[field]
        if not isinstance(value, str) or not value or "\x00" in value or not os.path.isabs(value):
            raise ValueError(f"{field} must be an absolute path")
    if normalized["bridge_path"] == normalized["ledger_path"]:
        raise ValueError("bridge_path and ledger_path must be different")
    if not isinstance(normalized["bridge_sha256"], str) or not _DIGEST_RE.fullmatch(
        normalized["bridge_sha256"]
    ):
        raise ValueError("bridge_sha256 must be a lowercase SHA-256 digest")
    if not _positive_int(normalized["max_calls"]):
        raise ValueError("max_calls must be a positive integer")
    total_timeout = normalized["total_timeout_seconds"]
    if (
        type(total_timeout) not in (int, float)
        or not math.isfinite(total_timeout)
        or total_timeout <= 0
        or total_timeout > 1800
    ):
        raise ValueError("total_timeout_seconds must be finite, positive, and at most 1800")
    call_timeout = normalized["call_timeout_seconds"]
    if type(call_timeout) is not int or not 1 <= call_timeout <= 300:
        raise ValueError("call_timeout_seconds must be an integer from 1 through 300")
    if normalized["reasoning_effort"] not in _EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {sorted(_EFFORTS)}")
    observed_limit = normalized["max_observed_tokens"]
    if observed_limit is not None and not _positive_int(observed_limit):
        raise ValueError("max_observed_tokens must be a positive integer or null")

    normalized["total_timeout_seconds"] = float(total_timeout)
    return normalized


class HarnessCallBudget:
    """Reserve and complete calls in one append-only, process-shared ledger."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = MappingProxyType(validate_native_harness_settings(settings))
        self.ledger_path = Path(self.settings["ledger_path"])
        self.lock_path = Path(f"{self.ledger_path}.lock")

    def reserve(self, call: dict[str, Any]) -> CallReservation:
        """Fsync one reservation before dispatch and return its finite timeout."""
        sanitized_call = self._validate_call(call)
        lock_deadline = self._lock_deadline()
        with self._locked(lock_deadline):
            records = self._read_records()
            now = time.monotonic()
            now_wall = time.time()
            if not records:
                policy = self._new_policy(now, now_wall)
                records = [policy]
                records_to_append = [policy]
            else:
                self._validate_policy(records[0])
                records_to_append = []
            state = self._state(records)
            self._raise_if_poisoned(state)
            if len(state["reservations"]) >= self.settings["max_calls"]:
                raise HarnessBudgetError("native harness call budget is exhausted")

            deadline = records[0]["started_monotonic"] + self.settings["total_timeout_seconds"]
            wall_deadline = records[0]["started_wall_time"] + self.settings["total_timeout_seconds"]
            remaining = min(deadline - now, wall_deadline - now_wall)
            timeout_seconds = min(self.settings["call_timeout_seconds"], math.floor(remaining))
            if timeout_seconds < 1:
                raise HarnessBudgetError("native harness total deadline is exhausted")
            reservation_id = str(uuid.uuid4())
            reservation_record = {
                "event": "reservation",
                "version": LEDGER_VERSION,
                "reservation_id": reservation_id,
                "reserved_monotonic": now,
                "call_index": len(state["reservations"]) + 1,
                "timeout_seconds": timeout_seconds,
                **sanitized_call,
            }
            records_to_append.append(reservation_record)
            self._append_records(records_to_append)
            return CallReservation(reservation_id, timeout_seconds, now, deadline)

    def complete(
        self, reservation: CallReservation, receipt: dict[str, Any]
    ) -> dict[str, Any]:
        """Fsync a sanitized terminal record for an existing reservation."""
        safe_receipt = self._sanitize_completion(receipt)
        with self._locked(reservation.deadline_monotonic):
            records = self._read_records()
            if not records:
                raise HarnessBudgetError("native harness ledger has no policy")
            self._validate_policy(records[0])
            state = self._state(records)
            if reservation.reservation_id not in state["reservations"]:
                raise HarnessBudgetError("native harness reservation is missing")
            if reservation.reservation_id in state["completions"]:
                raise HarnessBudgetError("native harness reservation is already complete")

            observed = safe_receipt.get("observed_total_tokens")
            prior_observed = [
                record.get("observed_total_tokens")
                for record in state["completions"].values()
                if record.get("status") == "completed"
            ]
            observed_limit = self.settings["max_observed_tokens"]
            completed_monotonic = time.monotonic()
            completed_wall = time.time()
            deadline = records[0]["started_monotonic"] + self.settings["total_timeout_seconds"]
            wall_deadline = records[0]["started_wall_time"] + self.settings["total_timeout_seconds"]
            if completed_monotonic > deadline or completed_wall > wall_deadline:
                budget_status = "total_deadline_exceeded"
            elif observed_limit is None:
                budget_status = "within_limit"
            elif observed is None:
                budget_status = "missing_observed_tokens"
            elif any(value is None for value in prior_observed):
                budget_status = "missing_observed_tokens"
            elif sum(prior_observed) + observed > observed_limit:
                budget_status = "observed_token_limit_exceeded"
            else:
                budget_status = "within_limit"

            completion = {
                "event": "completion",
                "version": LEDGER_VERSION,
                "reservation_id": reservation.reservation_id,
                "completed_monotonic": completed_monotonic,
                "completed_wall_time": completed_wall,
                "budget_status": budget_status,
                **safe_receipt,
            }
            self._append_records([completion])
            return dict(completion)

    def assert_healthy(self) -> None:
        """Raise if an attempted call has an unknown or unusable outcome."""
        if not self.ledger_path.exists():
            return
        with self._locked(self._lock_deadline()):
            records = self._read_records()
            if not records:
                return
            self._validate_policy(records[0])
            self._raise_if_poisoned(self._state(records))

    def _new_policy(self, now: float, now_wall: float) -> dict[str, Any]:
        return {
            "event": "policy",
            "version": LEDGER_VERSION,
            "budget_id": str(uuid.uuid4()),
            "started_monotonic": now,
            "started_wall_time": now_wall,
            "boot_time_epoch": int(psutil.boot_time()),
            "bridge_sha256": self.settings["bridge_sha256"],
            "max_calls": self.settings["max_calls"],
            "total_timeout_seconds": self.settings["total_timeout_seconds"],
            "call_timeout_seconds": self.settings["call_timeout_seconds"],
            "max_observed_tokens": self.settings["max_observed_tokens"],
        }

    def _validate_policy(self, policy: object) -> None:
        expected = {
            "event": "policy",
            "version": LEDGER_VERSION,
            "bridge_sha256": self.settings["bridge_sha256"],
            "max_calls": self.settings["max_calls"],
            "total_timeout_seconds": self.settings["total_timeout_seconds"],
            "call_timeout_seconds": self.settings["call_timeout_seconds"],
            "max_observed_tokens": self.settings["max_observed_tokens"],
        }
        if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items()):
            raise HarnessBudgetError("native harness ledger policy does not match settings")
        if (
            set(policy) != set(expected) | {
                "budget_id",
                "started_monotonic",
                "started_wall_time",
                "boot_time_epoch",
            }
            or type(policy.get("version")) is not int
            or not isinstance(policy.get("budget_id"), str)
            or type(policy.get("started_monotonic")) not in (int, float)
            or not math.isfinite(policy["started_monotonic"])
            or policy["started_monotonic"] < 0
            or type(policy.get("started_wall_time")) not in (int, float)
            or not math.isfinite(policy["started_wall_time"])
            or policy["started_wall_time"] < 0
            or type(policy.get("boot_time_epoch")) is not int
            or policy["boot_time_epoch"] != int(psutil.boot_time())
            or time.monotonic() < policy["started_monotonic"]
        ):
            raise HarnessBudgetError("native harness ledger policy is malformed")

    def _state(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        reservations: dict[str, dict[str, Any]] = {}
        completions: dict[str, dict[str, Any]] = {}
        policy = records[0]
        for record in records[1:]:
            event = record.get("event")
            reservation_id = record.get("reservation_id")
            if not isinstance(reservation_id, str) or not reservation_id:
                raise HarnessBudgetError("native harness ledger record is malformed")
            if event == "reservation":
                if reservation_id in reservations or reservation_id in completions:
                    raise HarnessBudgetError("native harness ledger has a duplicate reservation")
                self._validate_reservation_record(record, policy, len(reservations) + 1)
                reservations[reservation_id] = record
            elif event == "completion":
                if reservation_id not in reservations or reservation_id in completions:
                    raise HarnessBudgetError("native harness ledger has an unmatched completion")
                self._validate_completion_record(
                    record,
                    reservations[reservation_id],
                    policy,
                    list(completions.values()),
                )
                completions[reservation_id] = record
            else:
                raise HarnessBudgetError("native harness ledger has an unknown event")
        return {"reservations": reservations, "completions": completions}

    def _raise_if_poisoned(self, state: dict[str, Any]) -> None:
        pending = set(state["reservations"]) - set(state["completions"])
        if pending:
            raise HarnessBudgetError("native harness has an unfinished call with unknown outcome")
        for completion in state["completions"].values():
            status = completion.get("status")
            if status != "completed":
                raise HarnessBudgetError(f"native harness is poisoned by a {status or 'unknown'} call")
            budget_status = completion.get("budget_status")
            if budget_status == "missing_observed_tokens":
                raise HarnessBudgetError("native harness observed token usage is missing")
            if budget_status == "observed_token_limit_exceeded":
                raise HarnessBudgetError("native harness observed token limit was exceeded")
            if budget_status == "total_deadline_exceeded":
                raise HarnessBudgetError("native harness total deadline was exceeded")
            if budget_status != "within_limit":
                raise HarnessBudgetError("native harness token budget status is malformed")

    def _validate_call(self, call: object) -> dict[str, Any]:
        if not isinstance(call, dict) or set(call) != _CALL_FIELDS:
            raise HarnessBudgetError("native harness call metadata is malformed")
        if call.get("backend") != "codex_cli" or call.get("auth_mode") != "subscription":
            raise HarnessBudgetError("native harness call route is unsupported")
        if not isinstance(call.get("model"), str) or not _MODEL_RE.fullmatch(call["model"]):
            raise HarnessBudgetError("native harness call model is malformed")
        if not isinstance(call.get("reasoning_effort"), str) or not call["reasoning_effort"]:
            raise HarnessBudgetError("native harness call metadata is malformed")
        if call["reasoning_effort"] != self.settings["reasoning_effort"]:
            raise HarnessBudgetError("native harness reasoning effort is inconsistent")
        for field in ("agent_name_sha256", "transcript_sha256"):
            if not isinstance(call.get(field), str) or not _DIGEST_RE.fullmatch(call[field]):
                raise HarnessBudgetError("native harness call digest is malformed")
        if type(call.get("transcript_bytes")) is not int or call["transcript_bytes"] < 0:
            raise HarnessBudgetError("native harness transcript size is malformed")
        if not _positive_int(call.get("message_count")):
            raise HarnessBudgetError("native harness message count is malformed")
        return {field: call[field] for field in sorted(_CALL_FIELDS)}

    def _validate_reservation_record(
        self,
        record: dict[str, Any],
        policy: dict[str, Any],
        expected_index: int,
    ) -> None:
        expected_fields = _CALL_FIELDS | {
            "event",
            "version",
            "reservation_id",
            "reserved_monotonic",
            "call_index",
            "timeout_seconds",
        }
        reserved = record.get("reserved_monotonic")
        if (
            set(record) != expected_fields
            or record.get("version") != LEDGER_VERSION
            or type(record.get("version")) is not int
            or type(record.get("call_index")) is not int
            or record["call_index"] != expected_index
            or type(record.get("timeout_seconds")) is not int
            or not 1 <= record["timeout_seconds"] <= self.settings["call_timeout_seconds"]
            or type(reserved) not in (int, float)
            or not math.isfinite(reserved)
            or reserved < policy["started_monotonic"]
            or reserved > policy["started_monotonic"] + self.settings["total_timeout_seconds"]
        ):
            raise HarnessBudgetError("native harness reservation record is malformed")
        try:
            parsed_id = uuid.UUID(record["reservation_id"])
        except (ValueError, AttributeError):
            raise HarnessBudgetError("native harness reservation ID is malformed") from None
        if str(parsed_id) != record["reservation_id"]:
            raise HarnessBudgetError("native harness reservation ID is malformed")
        self._validate_call({field: record[field] for field in _CALL_FIELDS})

    def _validate_completion_record(
        self,
        record: dict[str, Any],
        reservation: dict[str, Any],
        policy: dict[str, Any],
        prior_completions: list[dict[str, Any]],
    ) -> None:
        expected_fields = _SAFE_COMPLETION_FIELDS | {
            "event",
            "version",
            "reservation_id",
            "completed_monotonic",
            "completed_wall_time",
            "budget_status",
        }
        completed = record.get("completed_monotonic")
        completed_wall = record.get("completed_wall_time")
        if (
            set(record) != expected_fields
            or record.get("version") != LEDGER_VERSION
            or type(record.get("version")) is not int
            or record.get("budget_status") not in _BUDGET_STATUSES
            or type(completed) not in (int, float)
            or not math.isfinite(completed)
            or completed < reservation["reserved_monotonic"]
            or type(completed_wall) not in (int, float)
            or not math.isfinite(completed_wall)
            or completed_wall < policy["started_wall_time"]
        ):
            raise HarnessBudgetError("native harness completion record is malformed")
        safe = self._sanitize_completion(record)
        requested = safe["model_identity"]["requested"]["model"]
        if requested != reservation["model"]:
            raise HarnessBudgetError("native harness completion model is inconsistent")
        observed = safe["observed_total_tokens"]
        observed_limit = self.settings["max_observed_tokens"]
        mono_deadline = policy["started_monotonic"] + self.settings["total_timeout_seconds"]
        wall_deadline = policy["started_wall_time"] + self.settings["total_timeout_seconds"]
        prior_observed = [
            item.get("observed_total_tokens")
            for item in prior_completions
            if item.get("status") == "completed"
        ]
        if completed > mono_deadline or completed_wall > wall_deadline:
            expected_budget_status = "total_deadline_exceeded"
        elif observed_limit is None:
            expected_budget_status = "within_limit"
        elif observed is None or any(value is None for value in prior_observed):
            expected_budget_status = "missing_observed_tokens"
        elif sum(prior_observed) + observed > observed_limit:
            expected_budget_status = "observed_token_limit_exceeded"
        else:
            expected_budget_status = "within_limit"
        if record["budget_status"] != expected_budget_status:
            raise HarnessBudgetError("native harness completion budget status is inconsistent")

    def _sanitize_completion(self, receipt: object) -> dict[str, Any]:
        if not isinstance(receipt, dict):
            raise HarnessBudgetError("native harness completion metadata is malformed")
        status = receipt.get("status")
        if status not in _COMPLETION_STATUSES:
            raise HarnessBudgetError("native harness completion status is malformed")
        safe = {field: receipt.get(field) for field in sorted(_SAFE_COMPLETION_FIELDS)}
        safe["status"] = status
        self._validate_completion_metadata(safe)
        try:
            encoded = json.dumps(safe, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            raise HarnessBudgetError("native harness completion metadata is not JSON safe") from None
        if len(encoded.encode("utf-8")) > MAX_LEDGER_LINE_BYTES // 2:
            raise HarnessBudgetError("native harness completion metadata is too large")
        return safe

    def _validate_completion_metadata(self, safe: dict[str, Any]) -> None:
        observed = safe.get("observed_total_tokens")
        if observed is not None and (type(observed) is not int or observed < 0):
            raise HarnessBudgetError("native harness observed token metadata is malformed")
        usage = safe.get("usage")
        if usage is not None:
            if not isinstance(usage, dict) or set(usage) != _USAGE_FIELDS:
                raise HarnessBudgetError("native harness usage metadata is malformed")
            if any(
                value is not None and (type(value) is not int or value < 0)
                for value in usage.values()
            ):
                raise HarnessBudgetError("native harness usage metadata is malformed")
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]
            total_tokens = usage["total_tokens"]
            derived = (
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            )
            if derived is not None and total_tokens is not None and total_tokens != derived:
                raise HarnessBudgetError("native harness usage total is inconsistent")
            expected_observed = total_tokens if total_tokens is not None else derived
            if observed != expected_observed:
                raise HarnessBudgetError("native harness observed token metadata is inconsistent")
        elif observed is not None:
            raise HarnessBudgetError("native harness observed tokens require usage metadata")

        elapsed = safe.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise HarnessBudgetError("native harness elapsed-time metadata is malformed")
        if type(safe.get("response_bytes")) is not int or safe["response_bytes"] < 0:
            raise HarnessBudgetError("native harness response-size metadata is malformed")
        for field in ("response_sha256", "stop_sequence_sha256"):
            value = safe.get(field)
            if value is not None and (not isinstance(value, str) or not _DIGEST_RE.fullmatch(value)):
                raise HarnessBudgetError("native harness completion digest is malformed")
        if type(safe.get("stop_applied")) is not bool:
            raise HarnessBudgetError("native harness stop metadata is malformed")
        if safe["stop_applied"] != (safe.get("stop_sequence_sha256") is not None):
            raise HarnessBudgetError("native harness stop metadata is inconsistent")
        diagnostic_count = safe.get("diagnostic_count")
        if type(diagnostic_count) is not int or diagnostic_count < 0:
            raise HarnessBudgetError("native harness diagnostic metadata is malformed")
        cost = safe.get("cost_usd")
        if cost is not None and (
            type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
        ):
            raise HarnessBudgetError("native harness cost metadata is malformed")
        if safe.get("usage_kind") != "chatgpt_subscription":
            raise HarnessBudgetError("native harness usage route is malformed")
        if safe.get("cost_kind") not in {"unavailable", "client_estimate"}:
            raise HarnessBudgetError("native harness cost kind is malformed")
        cli_version = safe.get("cli_version")
        if cli_version is not None and (
            not isinstance(cli_version, str)
            or not cli_version
            or len(cli_version) > 200
            or any(ord(character) < 32 or ord(character) == 127 for character in cli_version)
        ):
            raise HarnessBudgetError("native harness CLI version is malformed")
        actual_model = safe.get("actual_model")
        if actual_model is not None and (
            not isinstance(actual_model, str) or not _MODEL_RE.fullmatch(actual_model)
        ):
            raise HarnessBudgetError("native harness actual model is malformed")
        observed_models = safe.get("observed_models")
        if not isinstance(observed_models, list) or any(
            not isinstance(model, str) or not _MODEL_RE.fullmatch(model)
            for model in observed_models
        ):
            raise HarnessBudgetError("native harness observed models are malformed")
        identity_models = self._validate_identity_metadata(
            safe.get("model_identity"), actual_model
        )
        if safe["status"] == "completed" and identity_models["configured"] is None:
            raise HarnessBudgetError("native completed call lacks configured model identity")
        if safe["status"] == "completed" and (
            not isinstance(cli_version, str) or not cli_version.strip()
        ):
            raise HarnessBudgetError("native completed call lacks CLI version metadata")
        if safe["status"] == "completed" and safe.get("response_sha256") is None:
            raise HarnessBudgetError("native completed call lacks response digest")

    @staticmethod
    def _validate_identity_metadata(
        identity: object, actual_model: str | None
    ) -> dict[str, str | None]:
        if not isinstance(identity, dict) or set(identity) != {
            "requested",
            "configured",
            "reported",
        }:
            raise HarnessBudgetError("native harness model identity is malformed")
        models: dict[str, str | None] = {}
        expected_sources = {
            "requested": "request",
            "configured": "explicit_cli_argument",
            "reported": "codex.turn.completed.model",
        }
        for name, source in expected_sources.items():
            claim = identity[name]
            if claim is None and name != "requested":
                models[name] = None
                continue
            if (
                not isinstance(claim, dict)
                or set(claim) != {"model", "source"}
                or claim.get("source") != source
                or not isinstance(claim.get("model"), str)
                or not _MODEL_RE.fullmatch(claim["model"])
            ):
                raise HarnessBudgetError("native harness model identity claim is malformed")
            models[name] = claim["model"]
        if (
            models["configured"] not in (None, models["requested"])
            or models["reported"] != actual_model
            or (models["reported"] is not None and models["configured"] is None)
        ):
            raise HarnessBudgetError("native harness model identity is inconsistent")
        return models

    def _lock_deadline(self) -> float:
        now = time.monotonic()
        configured_deadline = now + self.settings["total_timeout_seconds"]
        if not self.ledger_path.exists():
            return configured_deadline
        try:
            with self.ledger_path.open("rb") as handle:
                first = handle.readline(MAX_LEDGER_LINE_BYTES + 1)
            if len(first) > MAX_LEDGER_LINE_BYTES:
                return configured_deadline
            policy = json.loads(first)
            started = policy.get("started_monotonic") if isinstance(policy, dict) else None
            started_wall = policy.get("started_wall_time") if isinstance(policy, dict) else None
            boot_time = policy.get("boot_time_epoch") if isinstance(policy, dict) else None
            if boot_time != int(psutil.boot_time()):
                return now
            if (
                type(started) in (int, float)
                and math.isfinite(started)
                and type(started_wall) in (int, float)
                and math.isfinite(started_wall)
            ):
                wall_remaining = started_wall + self.settings["total_timeout_seconds"] - time.time()
                return min(
                    configured_deadline,
                    started + self.settings["total_timeout_seconds"],
                    now + max(0.0, wall_remaining),
                )
        except (OSError, ValueError, TypeError):
            pass
        return configured_deadline

    @contextmanager
    def _locked(self, deadline: float) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise HarnessBudgetError("native harness ledger lock deadline expired")
                    time.sleep(min(LOCK_POLL_SECONDS, remaining))
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            with self.ledger_path.open("rb") as handle:
                while True:
                    line = handle.readline(MAX_LEDGER_LINE_BYTES + 1)
                    if not line:
                        break
                    if len(line) > MAX_LEDGER_LINE_BYTES or not line.endswith(b"\n"):
                        raise HarnessBudgetError("native harness ledger line is malformed")
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        raise HarnessBudgetError("native harness ledger JSON is malformed") from None
                    if not isinstance(record, dict):
                        raise HarnessBudgetError("native harness ledger record is malformed")
                    records.append(record)
                    if len(records) > 2 * self.settings["max_calls"] + 1:
                        raise HarnessBudgetError("native harness ledger has too many records")
        except HarnessBudgetError:
            raise
        except OSError:
            raise HarnessBudgetError("native harness ledger could not be read") from None
        return records

    def _append_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(
            (json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            for record in records
        )
        descriptor = os.open(
            self.ledger_path,
            os.O_CREAT | os.O_WRONLY | os.O_APPEND,
            0o600,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short ledger write")
                view = view[written:]
            os.fsync(descriptor)
        except OSError:
            raise HarnessBudgetError("native harness ledger could not be persisted") from None
        finally:
            os.close(descriptor)
