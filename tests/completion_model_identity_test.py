"""Sourced model identity must survive Mimosa without becoming an inference."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sources.core.completion_backends import CompletionBackendError, call_completion_bridge
from sources.core.llm_provider import LLMConfig, LLMProvider


def envelope(backend="codex_cli", reported=None):
    """Build independently sourced configured and reported identities."""
    requested = "gpt-5.6-luna" if backend == "codex_cli" else "claude-opus-5"
    identity = {
        "requested": {"model": requested, "source": "request"},
        "configured": {"model": requested, "source": "explicit_cli_argument"},
        "reported": None,
    }
    if reported:
        source = "codex.turn.completed.model" if backend == "codex_cli" else "claude.modelUsage"
        identity["reported"] = {"model": reported, "source": source}
    return {
        "protocol_version": 1, "status": "completed", "text": "observation",
        "requested_model": requested, "actual_model": reported,
        "observed_models": [reported] if reported else [],
        "backend": backend, "auth_mode": "subscription", "usage_kind": "subscription",
        "usage": None, "cost_usd": None, "cost_kind": "unavailable",
        "cli_version": "fixture", "diagnostic_count": 0, "error": None,
        "model_identity": identity,
    }


def consume(result):
    """Exercise the real bridge envelope validator without a subprocess."""
    request = {
        "backend": result["backend"], "model": result["requested_model"],
        "auth_mode": "subscription",
    }
    with patch("sources.core.completion_backends.load_completion_bridge",
               return_value=SimpleNamespace(complete=lambda _: result)):
        return call_completion_bridge(request)


@pytest.mark.parametrize("backend", ["codex_cli", "claude_cli"])
@pytest.mark.parametrize("reported", [None, "different-reported-model"])
def test_preserves_unknown_or_differing_reported_identity(backend, reported):
    result = envelope(backend, reported)
    original = copy.deepcopy(result)
    assert consume(result) == original
    assert result == original


def test_older_bridge_remains_compatible_without_identity_object():
    result = envelope(reported="older-model")
    del result["model_identity"]
    assert consume(result) is result


@pytest.mark.parametrize("identity", [None, [], {}, {
    "requested": {"model": "gpt-5.6-luna", "source": "request"},
    "configured": None, "reported": None,
}])
def test_completed_identity_requires_configured_launch_and_full_shape(identity):
    result = envelope()
    result["model_identity"] = identity
    with pytest.raises(CompletionBackendError, match="model identity"):
        consume(result)


@pytest.mark.parametrize("field,change", [
    ("requested", {"model": "other", "source": "request"}),
    ("configured", {"model": "other", "source": "explicit_cli_argument"}),
    ("configured", {"model": "gpt-5.6-luna", "source": "model_self_report"}),
    ("reported", {"model": "actual", "source": "claude.modelUsage"}),
    ("reported", {"model": "actual", "source": "catalogue"}),
    ("reported", {"model": " ", "source": "codex.turn.completed.model"}),
    ("reported", {"model": 42, "source": "codex.turn.completed.model"}),
    ("reported", {"model": "actual", "source": "codex.turn.completed.model", "verified": True}),
])
def test_rejects_inconsistent_identity_or_false_provenance(field, change):
    result = envelope(reported="actual")
    result["model_identity"][field] = change
    with pytest.raises(CompletionBackendError, match="model identity"):
        consume(result)


@pytest.mark.parametrize("actual", ["requested-copied-as-actual", None])
def test_actual_alias_must_equal_reported_identity(actual):
    result = envelope(reported="reported-model" if actual is None else None)
    result["actual_model"] = actual
    with pytest.raises(CompletionBackendError, match="model identity"):
        consume(result)


def test_failed_before_launch_can_report_requested_only():
    result = envelope()
    result.update(status="failed", text="", cli_version=None, error="unavailable")
    result["model_identity"]["configured"] = None
    assert consume(result)["model_identity"]["configured"] is None


def test_failed_result_cannot_launder_a_false_model_identity():
    result = envelope(reported="invented")
    result.update(status="failed", text="", error="failed")
    result["model_identity"]["reported"]["source"] = "request"
    with pytest.raises(CompletionBackendError, match="model identity"):
        consume(result)


def test_saved_completion_receipt_retains_sourced_identity(tmp_path):
    result = envelope()
    provider = LLMProvider(
        agent_name="identity-test", memory_path=str(tmp_path),
        config=LLMConfig(provider="codex-cli", model="gpt-5.6-luna"),
    )
    with patch("sources.core.completion_backends.load_completion_bridge",
               return_value=SimpleNamespace(complete=lambda _: result)):
        assert provider("Describe a measurement.", use_cache=False) == "observation"
    assert provider.last_completion_metadata["model_identity"] == result["model_identity"]
    receipts = [json.loads(path.read_text()) for path in tmp_path.rglob("*.json")]
    assert any(receipt.get("completion_metadata", {}).get("model_identity") == result["model_identity"]
               and receipt.get("cache_eligible") is False for receipt in receipts)
