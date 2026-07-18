"""Tests for the PreCheck OpenRouter probe quantization fallback.

When OpenRouter 404s with "No endpoints found for the request with
quantization: ...", the probe must drop its pinned quantization filter and
keep probing unfiltered instead of hard-failing the endpoint.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.append(str(Path(__file__).parent.parent))

from sources.utils.precheck import PreCheck

QUANT_404 = Exception(
    'litellm.NotFoundError: OpenrouterException - {"error":{"message":"No '
    'endpoints found for the request with quantization: fp8.","code":404}}'
)

VALID_PAYLOAD = '{"code": "print(1)"}'


class _FakeResponse:
    def __init__(self, content: str = VALID_PAYLOAD):
        self.choices = [SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop",
        )]

    def json(self):
        return {}


def _make_precheck() -> PreCheck:
    return PreCheck(config=SimpleNamespace(save_logprobs=False))


def _recording_completion(seen: list[dict], fail_first: bool):
    """Fake litellm.completion that snapshots the provider routing per call.

    Mock's call_args_list keeps references, and the probe mutates its params
    dict in place when dropping the filter — snapshot copies are required to
    tell the first (filtered) call apart from the later (unfiltered) ones.
    """
    def _completion(**kwargs):
        seen.append(dict(kwargs["extra_body"]["provider"]))
        if fail_first and len(seen) == 1:
            raise QUANT_404
        return _FakeResponse()
    return _completion


def test_probe_drops_quantization_filter_on_404():
    """A quantization 404 mid-probe drops the filter and retries unfiltered."""
    pc = _make_precheck()
    seen: list[dict] = []

    with patch("litellm.completion", side_effect=_recording_completion(seen, fail_first=True)) as completion:
        r = pc._probe("openrouter/deepseek/deepseek-v3.2", "novita", ["fp8"])

    assert completion.call_count == 3
    # First call pinned fp8; after the 404 the filter is gone.
    assert seen[0]["quantizations"] == ["fp8"]
    assert "quantizations" not in seen[-1]
    # The fallback is reflected in the result: 2 valid unfiltered responses.
    assert r["quantizations"] is None
    assert r["n_calls"] == 2
    assert r["valid_count"] == 2
    assert r["deterministic"]


def test_probe_provider_marks_unknown_after_filter_fallback():
    """An endpoint that only passes unfiltered is recorded as untagged."""
    pc = _make_precheck()
    ok = _FakeResponse()

    with patch("litellm.completion", side_effect=[QUANT_404, ok, ok]):
        r = pc._probe_provider("openrouter/deepseek/deepseek-v3.2", "novita", "fp8")

    assert r["discovered_quant"] == "unknown"
    assert r["tier"] == 1  # strict pass once unfiltered


def test_probe_keeps_filter_when_endpoint_serves_it():
    """No 404 -> the pinned quantization filter is used for all calls."""
    pc = _make_precheck()
    seen: list[dict] = []

    with patch("litellm.completion", side_effect=_recording_completion(seen, fail_first=False)) as completion:
        r = pc._probe("openrouter/deepseek/deepseek-v3.2", "novita", ["fp8"])

    assert completion.call_count == 3
    for routing in seen:
        assert routing["quantizations"] == ["fp8"]
    assert r["quantizations"] == ["fp8"]
    assert r["n_calls"] == 3


if __name__ == "__main__":
    test_probe_drops_quantization_filter_on_404()
    test_probe_provider_marks_unknown_after_filter_fallback()
    test_probe_keeps_filter_when_endpoint_serves_it()
    print("precheck quantization fallback tests passed")
