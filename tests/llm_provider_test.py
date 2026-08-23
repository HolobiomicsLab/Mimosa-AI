"""Tests for the LLMProvider transient-error retry loop.

Focus: the retry loop in ``LLMProvider.__call__`` is bounded by
``self.max_retries`` on both retryable paths (timeout and generic retryable
errors) and can no longer loop forever on a persistently failing provider.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.llm_provider import LLMConfig, LLMProvider


QUANT_404 = Exception(
    'litellm.NotFoundError: OpenrouterException - {"error":{"message":"No '
    'endpoints found for the request with quantization: bf16,fp16,fp8. To learn '
    'more about provider routing, visit: '
    'https://openrouter.ai/docs/guides/routing/provider-selection","code":404}}'
)


class _FakeResponse:
    """Minimal stand-in for a litellm completion response."""

    def __init__(self, content: str = "ok"):
        self.choices = [SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop",
        )]
        self.usage = None

    def json(self):
        return {}


def _make_provider() -> LLMProvider:
    """Build a provider with no network dependencies and no on-disk cache."""
    return LLMProvider(agent_name="test", config=LLMConfig(model="anthropic/claude-sonnet-4-5"))


def _make_openrouter_provider(quantizations=("bf16", "fp16", "fp8")) -> LLMProvider:
    """OpenRouter provider pinned to one endpoint with a quantization filter."""
    return LLMProvider(
        agent_name="test",
        config=LLMConfig(
            model="deepseek/deepseek-v3.2",
            provider="openrouter",
            openrouter_provider=["novita"],
            openrouter_quantizations=quantizations,
        ),
    )


def test_retry_loop_raises_after_max_retries():
    """A persistently retryable error terminates instead of looping forever."""
    provider = _make_provider()
    retryable = Exception("provider overloaded, please retry")

    with patch("litellm.completion", side_effect=retryable) as completion, patch(
        "sources.core.llm_provider.time.sleep"
    ):
        with pytest.raises(RuntimeError, match="retries"):
            provider("hello", use_cache=False)

    # attempts 0..max_retries are tried, then the ceiling raises.
    assert completion.call_count == provider.max_retries + 1


def test_timeout_retry_is_bounded():
    """Repeated timeouts terminate at the max_retries ceiling."""
    provider = _make_provider()

    with patch("litellm.completion", side_effect=TimeoutError("slow")) as completion, patch(
        "sources.core.llm_provider.time.sleep"
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            provider("hello", use_cache=False)

    assert completion.call_count == provider.max_retries + 1


def test_non_retryable_error_raises_immediately():
    """A non-retryable error is raised on the first attempt, with no retry."""
    provider = _make_provider()

    with patch("litellm.completion", side_effect=ValueError("bad request")) as completion, patch(
        "sources.core.llm_provider.time.sleep"
    ):
        with pytest.raises(RuntimeError):
            provider("hello", use_cache=False)

    assert completion.call_count == 1


def test_quantization_404_falls_back_to_unfiltered():
    """A 'no endpoints for quantization' 404 drops the filter and retries."""
    provider = _make_openrouter_provider()

    with patch(
        "litellm.completion", side_effect=[QUANT_404, _FakeResponse()]
    ) as completion:
        result = provider("hello", use_cache=False)

    assert result == "ok"
    assert completion.call_count == 2
    # First attempt carried the quantization filter, the retry did not.
    first_routing = completion.call_args_list[0].kwargs["extra_body"]["provider"]
    second_routing = completion.call_args_list[1].kwargs["extra_body"]["provider"]
    assert first_routing["quantizations"] == ["bf16", "fp16", "fp8"]
    assert "quantizations" not in second_routing
    # The fallback persists on the config so later calls skip the doomed filter.
    assert provider.config.openrouter_quantizations is None


def test_quantization_404_without_filter_raises_immediately():
    """With no quantization filter configured the same 404 is not retried."""
    provider = _make_openrouter_provider(quantizations=None)

    with patch("litellm.completion", side_effect=QUANT_404) as completion, patch(
        "sources.core.llm_provider.time.sleep"
    ):
        with pytest.raises(RuntimeError):
            provider("hello", use_cache=False)

    assert completion.call_count == 1


if __name__ == "__main__":
    test_retry_loop_raises_after_max_retries()
    test_timeout_retry_is_bounded()
    test_non_retryable_error_raises_immediately()
    print("llm_provider retry tests passed")
