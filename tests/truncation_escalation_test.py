"""A response cut off at max_tokens is not a success.

Reasoning models spend their output budget on reasoning *before* emitting
content, so a budget that would comfortably fit the answer can still return a
truncated one — with ``finish_reason="length"`` and no exception.

Observed against ``stealth/ox-alpha`` on the ``p_iimn`` re-run: the planner's
call returned ``Completion: 8192`` against a ``max_tokens`` of exactly 8192,
four truncation warnings in one run, and the plan JSON ended mid-string. The
parser then failed six times with ``Unterminated string`` and the task was
abandoned. No amount of JSON repair recovers a document that was never
finished — the budget has to grow.
"""

from types import SimpleNamespace

import pytest

from sources.core.llm_provider import (
    _MAX_OUTPUT_TOKENS,
    _MAX_TRUNCATION_RETRIES,
    LLMConfig,
    LLMProvider,
)


def _response(content, finish_reason):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
        stop_reason=None,
    )
    return SimpleNamespace(
        choices=[choice],
        usage=None,
        json=lambda: {},
    )


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reason", ["length", "max_tokens"])
def test_budget_exhaustion_is_detected_for_both_provider_spellings(reason):
    assert LLMProvider._is_truncated(_response("partial", reason)) is True


@pytest.mark.parametrize("reason", ["stop", "tool_calls", "content_filter", None])
def test_other_stop_reasons_are_not_truncation(reason):
    assert LLMProvider._is_truncated(_response("done", reason)) is False


def test_malformed_response_is_not_reported_as_truncated():
    assert LLMProvider._is_truncated(SimpleNamespace(choices=[])) is False
    assert LLMProvider._is_truncated(object()) is False


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

def _provider(max_tokens=8192):
    cfg = LLMConfig(model="ox-alpha", provider="openrouter", key="k",
                    max_tokens=max_tokens, openrouter_provider=None)
    return LLMProvider(agent_name=None, memory_path=None, system_msg=None, config=cfg)


def _patch_completion(monkeypatch, responses):
    """Record the max_tokens of each call and return queued responses."""
    seen = []

    def fake_completion(**params):
        seen.append(params["max_tokens"])
        return responses[min(len(seen) - 1, len(responses) - 1)]

    import sources.core.llm_provider as mod
    monkeypatch.setattr(mod.litellm, "completion", fake_completion)
    return seen


def test_a_complete_response_is_returned_without_escalation(monkeypatch):
    seen = _patch_completion(monkeypatch, [_response("all good", "stop")])
    assert _provider()("prompt") == "all good"
    assert seen == [8192]


def test_truncation_doubles_the_budget_and_retries(monkeypatch):
    seen = _patch_completion(monkeypatch, [
        _response("cut off", "length"),
        _response("complete now", "stop"),
    ])
    assert _provider()("prompt") == "complete now"
    assert seen == [8192, 16384]


def test_escalation_is_bounded(monkeypatch):
    """Persistent truncation must not retry forever."""
    seen = _patch_completion(monkeypatch, [_response("cut off", "length")])
    result = _provider()("prompt")

    assert result == "cut off"  # the caller still gets what there is
    assert len(seen) == _MAX_TRUNCATION_RETRIES + 1
    assert seen == [8192, 16384, 32768]


def test_escalation_never_exceeds_the_output_ceiling(monkeypatch):
    seen = _patch_completion(monkeypatch, [_response("cut off", "length")])
    _provider(max_tokens=_MAX_OUTPUT_TOKENS)("prompt")
    # Already at the ceiling: nothing to escalate to, so exactly one call.
    assert seen == [_MAX_OUTPUT_TOKENS]


def test_the_configured_budget_is_not_mutated(monkeypatch):
    """Escalation is per-call; a later call starts from the configured value."""
    provider = _provider()
    _patch_completion(monkeypatch, [
        _response("cut off", "length"),
        _response("ok", "stop"),
    ])
    provider("prompt")
    assert provider.config.max_tokens == 8192
