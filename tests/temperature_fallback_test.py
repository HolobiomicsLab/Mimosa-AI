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


# ---------------------------------------------------------------------------
# Upstream provider failures surfaced as 400
# ---------------------------------------------------------------------------
# After the temperature fix landed, every temp>1.0 generation recovered and
# succeeded — the log shows 1.20/1.10/1.24/1.08 each falling back to 1.0 and
# then "generated in Ns". The generation failures that remained were all at
# temperature <= 1.0, i.e. a different cause: OpenRouter reporting an upstream
# fault as HTTP 400 "Provider returned error" (metadata.raw "ERROR").
#
# Six identical calls at temperature 0.85 with a full-size prompt succeeded in
# isolation while the same shape failed intermittently under four concurrent
# lanes, so the request is fine and the fault is transient. _is_retryable_error
# did not match that wording, so it raised at once instead of backing off, and
# each occurrence cost a whole workflow generation.


class _UpstreamFailure(Exception):
    def __init__(self):
        super().__init__(
            'litellm.BadRequestError: OpenrouterException - {"error":'
            '{"message":"Provider returned error","code":400,'
            '"metadata":{"raw":"ERROR","provider_name":"Stealth"}}}'
        )
        self.status_code = 400


def _provider_for_retry_check():
    from sources.core.llm_provider import LLMConfig

    cfg = LLMConfig(model="m", provider="openrouter", key="k")
    return LLMProvider(agent_name=None, memory_path=None, system_msg=None, config=cfg)


def test_upstream_provider_failure_is_recognised():
    assert LLMProvider._is_upstream_provider_failure(_UpstreamFailure()) is True


def test_upstream_provider_failure_is_retryable():
    assert _provider_for_retry_check()._is_retryable_error(_UpstreamFailure()) is True


def test_an_ordinary_client_error_is_still_not_retryable():
    """Narrow by design — a real bad request must not loop."""
    class _BadInput(Exception):
        status_code = 400

    err = _BadInput("Invalid value for 'messages': expected a list")
    assert LLMProvider._is_upstream_provider_failure(err) is False
    assert _provider_for_retry_check()._is_retryable_error(err) is False


def test_matching_is_case_insensitive():
    class _E(Exception):
        pass

    assert LLMProvider._is_upstream_provider_failure(_E("PROVIDER RETURNED ERROR")) is True
