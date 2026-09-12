"""A strict smolagents Model adapter for the local Codex completion bridge."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any
import uuid

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from .completion_backends import call_completion_bridge
from .harness_budget import (
    CallReservation,
    HarnessBudgetError,
    HarnessCallBudget,
    validate_native_harness_settings,
)


PROTOCOL_VERSION = 1
MAX_MESSAGES = 100
MAX_TRANSCRIPT_BYTES = 1024 * 1024
MAX_STOP_SEQUENCES = 32
MAX_STOP_BYTES = 1024
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_ROLES = {
    MessageRole.SYSTEM.value: MessageRole.SYSTEM.value,
    MessageRole.USER.value: MessageRole.USER.value,
    MessageRole.ASSISTANT.value: MessageRole.ASSISTANT.value,
    MessageRole.TOOL_CALL.value: MessageRole.ASSISTANT.value,
    MessageRole.TOOL_RESPONSE.value: MessageRole.USER.value,
}
_USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
}
_RESULT_STATUSES = {"completed", "failed", "timeout", "malformed", "unsupported"}
_SAFE_ENVELOPE_FIELDS = (
    "protocol_version",
    "status",
    "requested_model",
    "actual_model",
    "observed_models",
    "backend",
    "auth_mode",
    "usage_kind",
    "usage",
    "cost_usd",
    "cost_kind",
    "cli_version",
    "diagnostic_count",
    "model_identity",
)


class HarnessModelError(RuntimeError):
    """Report a strict adapter, completion, or receipt failure."""


class HarnessCompletionModel(Model):
    """Generate text through one pinned Codex CLI subscription bridge.

    Native execution requires the additive ``model_identity`` provenance object
    on every result.  That stricter rule does not change the older text-only
    consumer, which continues to accept legacy envelopes without the object.
    """

    def __init__(
        self,
        model_id: str,
        settings: dict[str, Any],
        agent_name: str = "agent",
        receipt_dir: str | None = None,
    ):
        self._validate_model_id(model_id)
        if not isinstance(agent_name, str) or not agent_name or len(agent_name) > 200:
            raise ValueError("agent_name must be a non-empty string of at most 200 characters")
        if receipt_dir is not None and (
            not isinstance(receipt_dir, str)
            or not receipt_dir
            or "\x00" in receipt_dir
            or not os.path.isabs(receipt_dir)
        ):
            raise ValueError("receipt_dir must be an absolute path or null")

        normalized = validate_native_harness_settings(settings)
        super().__init__(model_id=model_id)
        self._settings = MappingProxyType(dict(normalized))
        self.agent_name = agent_name
        self.receipt_dir = Path(receipt_dir) if receipt_dir is not None else None
        self.receipts: list[dict[str, Any]] = []
        self._budget = HarnessCallBudget(normalized)
        self._generation_failed = False

    @property
    def model_id(self) -> str:
        """Return the model identity sealed during construction."""
        return self._model_id

    @model_id.setter
    def model_id(self, value: str) -> None:
        if hasattr(self, "_model_id") and value != self._model_id:
            raise AttributeError("native harness model_id is immutable")
        self._model_id = value

    @property
    def settings(self) -> MappingProxyType:
        """Expose immutable normalized settings for orchestration metadata."""
        return self._settings

    def generate(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Run exactly one reserved completion and apply stop strings locally."""
        if self._generation_failed:
            raise HarnessModelError("native model generation already failed during this run")
        try:
            return self._generate(
                messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )
        except Exception:
            self._generation_failed = True
            raise

    def _generate(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Implement one generation while the public method latches failures."""
        self._validate_generation_options(
            stop_sequences, response_format, tools_to_call_from, kwargs
        )
        canonical_messages, transcript = self._canonical_messages(messages)
        self._verify_bridge_digest()
        call = {
            "backend": "codex_cli",
            "auth_mode": "subscription",
            "model": self.model_id,
            "reasoning_effort": self.settings["reasoning_effort"],
            "agent_name_sha256": _sha256_text(self.agent_name),
            "transcript_sha256": hashlib.sha256(transcript).hexdigest(),
            "transcript_bytes": len(transcript),
            "message_count": len(canonical_messages),
        }
        reservation = self._budget.reserve(call)
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "backend": "codex_cli",
            "model": self.model_id,
            "messages": canonical_messages,
            "response_format": "text",
            "auth_mode": "subscription",
            "effort": self.settings["reasoning_effort"],
            "timeout_seconds": reservation.timeout_seconds,
        }

        started = time.monotonic()
        try:
            result = call_completion_bridge(
                request, bridge_path=self.settings["bridge_path"]
            )
        except Exception:
            elapsed = max(0.0, time.monotonic() - started)
            receipt = self._unresolved_receipt(reservation, "unknown", elapsed)
            self._finish_attempt(reservation, receipt)
            raise HarnessModelError("native completion outcome is unknown") from None
        elapsed = max(0.0, time.monotonic() - started)

        try:
            receipt, text, observed_total = self._validate_result(
                result, reservation, elapsed
            )
        except HarnessModelError:
            malformed = self._unresolved_receipt(reservation, "malformed", elapsed)
            self._finish_attempt(reservation, malformed)
            raise

        matched_stop, returned_text = _apply_earliest_stop(text, stop_sequences or [])
        receipt.update(
            {
                "response_sha256": _sha256_text(text),
                "response_bytes": len(text.encode("utf-8")),
                "stop_applied": matched_stop is not None,
                "stop_sequence_sha256": (
                    _sha256_text(matched_stop) if matched_stop is not None else None
                ),
                "observed_total_tokens": observed_total,
            }
        )
        self._finish_attempt(reservation, receipt)
        if receipt["status"] != "completed":
            raise HarnessModelError(
                f"native completion ended with {receipt['status']} status"
            )

        usage = receipt["usage"]
        token_usage = None
        if usage is not None:
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]
            if input_tokens is not None and output_tokens is not None:
                token_usage = TokenUsage(
                    input_tokens=input_tokens, output_tokens=output_tokens
                )
        raw = copy.deepcopy(receipt)
        raw.update(
            {
                "stop_sequence": matched_stop,
                "provider_stop_enforced": False,
            }
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=returned_text,
            raw=raw,
            token_usage=token_usage,
        )

    def assert_healthy(self) -> None:
        """Reject a final agent result after any unknown or poisoned call."""
        self._budget.assert_healthy()
        if self._generation_failed:
            raise HarnessModelError("native model generation failed during this run")

    @staticmethod
    def _validate_model_id(model_id: object) -> None:
        if (
            not isinstance(model_id, str)
            or not _MODEL_RE.fullmatch(model_id)
            or model_id.lower().startswith("claude")
        ):
            raise ValueError("native harness requires a bare Codex model identifier")

    def _validate_generation_options(
        self,
        stop_sequences: object,
        response_format: object,
        tools_to_call_from: object,
        kwargs: dict[str, Any],
    ) -> None:
        if response_format is not None:
            raise HarnessModelError("native response schemas are unsupported")
        if tools_to_call_from:
            raise HarnessModelError("native model tool schemas are unsupported")
        if tools_to_call_from is not None and not isinstance(tools_to_call_from, list):
            raise HarnessModelError("native model tool schemas are unsupported")
        if kwargs:
            raise HarnessModelError(
                f"native generation options are unsupported: {sorted(kwargs)}"
            )
        if stop_sequences is None:
            return
        if not isinstance(stop_sequences, list) or len(stop_sequences) > MAX_STOP_SEQUENCES:
            raise HarnessModelError("native stop sequences are unsupported or too numerous")
        for sequence in stop_sequences:
            if (
                not isinstance(sequence, str)
                or not sequence
                or len(sequence.encode("utf-8")) > MAX_STOP_BYTES
            ):
                raise HarnessModelError("native stop sequence is malformed")

    def _canonical_messages(
        self, messages: object
    ) -> tuple[list[dict[str, str]], bytes]:
        if not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES:
            raise HarnessModelError("native messages must be a non-empty bounded list")
        canonical: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, ChatMessage):
                if message.tool_calls:
                    raise HarnessModelError("native message tool calls are unsupported")
                role = message.role
                content = message.content
            elif isinstance(message, dict):
                if set(message) != {"role", "content"}:
                    raise HarnessModelError("native message fields are malformed")
                role = message.get("role")
                content = message.get("content")
            else:
                raise HarnessModelError("native message must be a ChatMessage or dictionary")

            role_value = role.value if isinstance(role, MessageRole) else role
            if not isinstance(role_value, str) or role_value not in _ROLES:
                raise HarnessModelError("native message role is unsupported")
            text = self._flatten_content(content)
            canonical.append({"role": _ROLES[role_value], "content": text})
        if not any(message["content"] for message in canonical):
            raise HarnessModelError("native messages contain no text")
        transcript = json.dumps(
            canonical, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(transcript) > MAX_TRANSCRIPT_BYTES:
            raise HarnessModelError("native message transcript exceeds the bridge limit")
        return canonical, transcript

    @staticmethod
    def _flatten_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise HarnessModelError("native message content must be text")
        pieces: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                if isinstance(block, dict) and block.get("type") == "image":
                    raise HarnessModelError("native message images are unsupported")
                raise HarnessModelError("native message content block is unsupported")
            if set(block) != {"type", "text"} or not isinstance(block.get("text"), str):
                raise HarnessModelError("native text content block is malformed")
            pieces.append(block["text"])
        return "".join(pieces)

    def _verify_bridge_digest(self) -> None:
        path = Path(self.settings["bridge_path"])
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(128 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            raise HarnessModelError("pinned completion bridge could not be read") from None
        if digest.hexdigest() != self.settings["bridge_sha256"]:
            raise HarnessModelError("pinned completion bridge digest does not match")

    def _validate_result(
        self,
        result: object,
        reservation: CallReservation,
        elapsed: float,
    ) -> tuple[dict[str, Any], str, int | None]:
        if not isinstance(result, dict):
            raise HarnessModelError("native completion result is malformed")
        status = result.get("status")
        if status not in _RESULT_STATUSES:
            raise HarnessModelError("native completion result status is malformed")
        if (
            result.get("protocol_version") != PROTOCOL_VERSION
            or result.get("requested_model") != self.model_id
            or result.get("backend") != "codex_cli"
            or result.get("auth_mode") != "subscription"
        ):
            raise HarnessModelError("native completion result route is malformed")
        text = result.get("text")
        if not isinstance(text, str) or (status == "completed" and not text.strip()):
            raise HarnessModelError("native completion result text is malformed")
        if status != "completed" and text:
            raise HarnessModelError("native failed completion unexpectedly contains text")

        usage, observed_total = self._validate_usage(result.get("usage"))
        diagnostic_count = result.get("diagnostic_count")
        if type(diagnostic_count) is not int or diagnostic_count < 0:
            raise HarnessModelError("native diagnostic metadata is malformed")
        cost = result.get("cost_usd")
        if cost is not None and (
            type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
        ):
            raise HarnessModelError("native cost metadata is malformed")
        for field in ("usage_kind", "cost_kind"):
            if not isinstance(result.get(field), str) or not result[field]:
                raise HarnessModelError(f"native {field} metadata is malformed")
        if result["usage_kind"] != "chatgpt_subscription":
            raise HarnessModelError("native usage_kind does not match the subscription route")
        if result["cost_kind"] not in {"unavailable", "client_estimate"}:
            raise HarnessModelError("native cost_kind metadata is malformed")
        cli_version = result.get("cli_version")
        if status == "completed" and (
            not isinstance(cli_version, str) or not cli_version.strip()
        ):
            raise HarnessModelError("native CLI version metadata is malformed")
        if cli_version is not None and (
            not isinstance(cli_version, str)
            or len(cli_version) > 200
            or any(ord(character) < 32 or ord(character) == 127 for character in cli_version)
        ):
            raise HarnessModelError("native CLI version metadata is malformed")

        actual_model = result.get("actual_model")
        if actual_model is not None and (
            not isinstance(actual_model, str) or not _MODEL_RE.fullmatch(actual_model)
        ):
            raise HarnessModelError("native actual model metadata is malformed")
        observed_models = result.get("observed_models", [])
        if not isinstance(observed_models, list) or any(
            not isinstance(model, str) or not _MODEL_RE.fullmatch(model)
            for model in observed_models
        ):
            raise HarnessModelError("native observed model metadata is malformed")
        identity = self._validate_identity(result.get("model_identity"), status, actual_model)

        receipt = {
            field: copy.deepcopy(result.get(field)) for field in _SAFE_ENVELOPE_FIELDS
        }
        receipt.update(
            {
                "reservation_id": reservation.reservation_id,
                "usage": usage,
                "model_identity": identity,
                "observed_models": list(observed_models),
                "elapsed_seconds": elapsed,
            }
        )
        return receipt, text, observed_total

    def _validate_usage(
        self, usage: object
    ) -> tuple[dict[str, int | None] | None, int | None]:
        if usage is None:
            return None, None
        if not isinstance(usage, dict) or set(usage) != _USAGE_FIELDS:
            raise HarnessModelError("native token usage metadata is malformed")
        normalized: dict[str, int | None] = {}
        for field in _USAGE_FIELDS:
            value = usage[field]
            if value is not None and (type(value) is not int or value < 0):
                raise HarnessModelError("native token usage metadata is malformed")
            normalized[field] = value
        input_tokens = normalized["input_tokens"]
        output_tokens = normalized["output_tokens"]
        total_tokens = normalized["total_tokens"]
        if input_tokens is not None and output_tokens is not None:
            derived_total = input_tokens + output_tokens
            if total_tokens is not None and total_tokens != derived_total:
                raise HarnessModelError("native token usage total is inconsistent")
            observed_total = total_tokens if total_tokens is not None else derived_total
        else:
            observed_total = total_tokens
        return normalized, observed_total

    def _validate_identity(
        self, identity: object, status: str, actual_model: str | None
    ) -> dict[str, Any]:
        if not isinstance(identity, dict) or set(identity) != {
            "requested",
            "configured",
            "reported",
        }:
            raise HarnessModelError("native model identity metadata is malformed")
        requested = self._identity_claim(identity["requested"], "request", required=True)
        configured = self._identity_claim(
            identity["configured"], "explicit_cli_argument", required=False
        )
        reported = self._identity_claim(
            identity["reported"], "codex.turn.completed.model", required=False
        )
        if (
            requested != self.model_id
            or configured not in (None, self.model_id)
            or (status == "completed" and configured is None)
            or reported != actual_model
            or (reported is not None and configured is None)
        ):
            raise HarnessModelError("native model identity metadata is inconsistent")
        return copy.deepcopy(identity)

    @staticmethod
    def _identity_claim(entry: object, source: str, required: bool) -> str | None:
        if entry is None and not required:
            return None
        if (
            not isinstance(entry, dict)
            or set(entry) != {"model", "source"}
            or entry.get("source") != source
            or not isinstance(entry.get("model"), str)
            or not _MODEL_RE.fullmatch(entry["model"])
        ):
            raise HarnessModelError("native model identity claim is malformed")
        return entry["model"]

    def _unresolved_receipt(
        self, reservation: CallReservation, status: str, elapsed: float
    ) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "status": status,
            "requested_model": self.model_id,
            "actual_model": None,
            "observed_models": [],
            "backend": "codex_cli",
            "auth_mode": "subscription",
            "usage_kind": "chatgpt_subscription",
            "usage": None,
            "cost_usd": None,
            "cost_kind": "unavailable",
            "cli_version": None,
            "diagnostic_count": 0,
            "model_identity": {
                "requested": {"model": self.model_id, "source": "request"},
                "configured": None,
                "reported": None,
            },
            "reservation_id": reservation.reservation_id,
            "elapsed_seconds": elapsed,
            "response_sha256": None,
            "response_bytes": 0,
            "stop_applied": False,
            "stop_sequence_sha256": None,
            "observed_total_tokens": None,
        }

    def _finish_attempt(
        self, reservation: CallReservation, receipt: dict[str, Any]
    ) -> None:
        safe_receipt = copy.deepcopy(receipt)
        self.receipts.append(safe_receipt)
        try:
            self._write_receipt(safe_receipt)
        except OSError:
            poisoned = dict(safe_receipt)
            poisoned["status"] = "unknown"
            self._budget.complete(reservation, poisoned)
            raise HarnessModelError("native completion receipt could not be persisted") from None
        try:
            self._budget.complete(reservation, safe_receipt)
        except HarnessBudgetError:
            raise HarnessModelError("native completion ledger could not be finalized") from None

    def _write_receipt(self, receipt: dict[str, Any]) -> None:
        if self.receipt_dir is None:
            return
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        destination = self.receipt_dir / f"native_completion_{receipt['reservation_id']}.json"
        temporary = self.receipt_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        payload = json.dumps(
            {"completion_metadata": receipt},
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short receipt write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, destination)
            directory = os.open(self.receipt_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _apply_earliest_stop(text: str, stop_sequences: list[str]) -> tuple[str | None, str]:
    matches = [
        (position, index, sequence)
        for index, sequence in enumerate(stop_sequences)
        if (position := text.find(sequence)) >= 0
    ]
    if not matches:
        return None, text
    position, _, sequence = min(matches)
    return sequence, text[:position]


__all__ = [
    "HarnessBudgetError",
    "HarnessCompletionModel",
    "HarnessModelError",
    "validate_native_harness_settings",
]
