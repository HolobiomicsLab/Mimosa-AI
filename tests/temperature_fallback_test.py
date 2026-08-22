"""Temperature-rejection detection for gateways that do not tag the field.

The retry path in ``LLMProvider.__call__`` already knows how to recover from a
provider refusing ``temperature``: drop to 1.0 and retry. It only ever fired
for OpenAI-style errors that name the field in ``error.param``.

OpenRouter forwards an upstream refusal as a bare 400 whose body carries no
``param`` (``metadata.raw`` is literally ``"ERROR"``). Observed against
``stealth/ox-alpha``: temperature 1.0 succeeds, 1.3 returns that 400. Because
``workflow_factory`` samples ``random.uniform(0.7, 1.3)`` for every workflow
generation, roughly half of all generations aborted the whole task instead of
retrying one step lower.
"""

from types import SimpleNamespace

import pytest

from sources.core.llm_provider import _SAFE_MAX_TEMPERATURE, LLMProvider


class _BadRequest(Exception):
    """Stands in for litellm.BadRequestError: 400, no ``param`` attribute."""

    def __init__(self, message="OpenrouterException - Provider returned error"):
        super().__init__(message)
        self.status_code = 400


def test_param_tagged_error_still_detected():
    err = SimpleNamespace(param="temperature")
    assert LLMProvider._is_temperature_error(err) is True


def test_untagged_400_above_ceiling_is_treated_as_temperature_rejection():
    assert LLMProvider._is_temperature_error(_BadRequest(), 1.3) is True


def test_untagged_400_at_or_below_ceiling_is_not_attributed_to_temperature():
    """A 400 at a safe temperature is some other problem; do not mask it."""
    assert LLMProvider._is_temperature_error(_BadRequest(), 1.0) is False
    assert LLMProvider._is_temperature_error(_BadRequest(), 0.7) is False


def test_missing_temperature_argument_keeps_legacy_behaviour():
    """Callers that pass no temperature get the original param-only check."""
    assert LLMProvider._is_temperature_error(_BadRequest()) is False


@pytest.mark.parametrize("temperature", [1.01, 1.3, 2.0])
def test_ceiling_is_the_documented_safe_value(temperature):
    assert temperature > _SAFE_MAX_TEMPERATURE
    assert LLMProvider._is_temperature_error(_BadRequest(), temperature) is True


def test_non_400_error_is_not_a_temperature_rejection():
    """A timeout at high temperature must stay retryable-as-timeout."""

    class _Timeout(Exception):
        pass

    assert LLMProvider._is_temperature_error(_Timeout("read timed out"), 1.3) is False
