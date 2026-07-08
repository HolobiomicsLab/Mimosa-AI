"""Tests for the LLMProvider transient-error retry loop.

Focus: the retry loop in ``LLMProvider.__call__`` is bounded by
``self.max_retries`` on both retryable paths (timeout and generic retryable
errors) and can no longer loop forever on a persistently failing provider.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.llm_provider import LLMConfig, LLMProvider


def _make_provider() -> LLMProvider:
    """Build a provider with no network dependencies and no on-disk cache."""
    return LLMProvider(agent_name="test", config=LLMConfig(model="anthropic/claude-sonnet-4-5"))


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


if __name__ == "__main__":
    test_retry_loop_raises_after_max_retries()
    test_timeout_retry_is_bounded()
    test_non_retryable_error_raises_immediately()
    print("llm_provider retry tests passed")
