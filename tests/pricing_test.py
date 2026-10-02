#!/usr/bin/env python3
"""
Test script for pricing functionality
Tests the OpenRouterPricingClient and Config integration
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from sources.utils.pricing import OpenRouterPricingClient


def test_openrouter_pricing_client():
    """Test OpenRouterPricingClient functionality."""
    print("🧪 Testing OpenRouterPricingClient...")

    # Test 1: Fallback pricing
    print("\n1️⃣ Testing fallback pricing...")
    client = OpenRouterPricingClient()
    fallback = client.get_fallback_pricing()

    assert "default" in fallback, "Missing 'default' model in fallback pricing"
    assert "deepseek/deepseek-chat" in fallback, "Missing DeepSeek model in fallback"
    assert "input" in fallback["default"], "Missing 'input' key in pricing"
    assert "output" in fallback["default"], "Missing 'output' key in pricing"
    print("✅ Fallback pricing structure is correct")

    # Test 2: Cache directory creation
    print("\n2️⃣ Testing cache directory creation...")
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "cache", "test_pricing.json")
        test_client = OpenRouterPricingClient()
        test_client.cache_file = cache_file
        test_client._ensure_cache_dir()

        assert os.path.exists(os.path.dirname(cache_file)), (
            "Cache directory not created"
        )
        print("✅ Cache directory creation works")

    # Test 3: Cache save/load functionality
    print("\n3️⃣ Testing cache functionality...")
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "test_cache.json")
        test_client = OpenRouterPricingClient()
        test_client.cache_file = cache_file

        test_data = {"test-model": {"input": 1.0, "output": 2.0}}

        test_client._save_cache(test_data)
        assert os.path.exists(cache_file), "Cache file not created"

        loaded = test_client._load_cache()
        assert loaded == test_data, "Cache data doesn't match saved data"
        print("✅ Cache save/load works correctly")


def test_config_pricing_integration():
    """Test Config class pricing integration."""
    print("\n🧪 Testing Config pricing integration...")

    # Mock the OpenRouterPricingClient to avoid API calls during tests
    with patch("config.OpenRouterPricingClient") as mock_client:
        # Set up mock to return test pricing data
        mock_instance = MagicMock()
        mock_instance.get_model_pricing_dict.return_value = {
            "test-model": {"input": 5.0, "output": 10.0}
        }
        mock_instance.get_fallback_pricing.return_value = {
            "default": {"input": 1.0, "output": 2.0}
        }
        mock_client.return_value = mock_instance

        # Test 1: Config initialization
        print("\n1️⃣ Testing Config initialization...")
        config = Config()
        assert hasattr(config, "_pricing_client"), "Config missing pricing client"
        assert config._model_pricing_cache is None, "Pricing cache should start as None"
        print("✅ Config initializes correctly")

        # Test 2: Model pricing property
        print("\n2️⃣ Testing model_pricing property...")
        pricing = config.model_pricing

        # Should call get_model_pricing_dict first
        mock_instance.get_model_pricing_dict.assert_called_once()
        assert "test-model" in pricing, "Test model not found in pricing"
        assert pricing["test-model"]["input"] == 5.0, "Incorrect input pricing"
        print("✅ Model pricing property returns correct data")

        # Test 3: Cache behavior
        print("\n3️⃣ Testing pricing cache behavior...")
        pricing2 = config.model_pricing  # Second call should use cache

        # Should not call API again (still only one call)
        assert mock_instance.get_model_pricing_dict.call_count == 1, (
            "API called multiple times"
        )
        assert pricing == pricing2, "Cached pricing differs from original"
        print("✅ Pricing cache works correctly")

        # Test 4: Refresh functionality
        # ``refresh_pricing()`` is lazy by design — it nulls the cache; the
        # next read of ``model_pricing`` is what triggers the fresh API call.
        # Reading proactively here verifies the documented contract.
        print("\n4️⃣ Testing pricing refresh...")
        config.refresh_pricing()
        assert config._model_pricing_cache is None, "Cache not cleared after refresh"

        _ = config.model_pricing  # next read should re-hit the API
        assert mock_instance.get_model_pricing_dict.call_count == 2, (
            "Read after refresh should have triggered a new API call"
        )
        print("✅ Pricing refresh works correctly")


def test_pricing_fallback_behavior():
    """Test fallback behavior when API fails."""
    print("\n🧪 Testing pricing fallback behavior...")

    with patch("config.OpenRouterPricingClient") as mock_client:
        mock_instance = MagicMock()

        # Simulate API failure by returning empty dict
        mock_instance.get_model_pricing_dict.return_value = {}
        mock_instance.get_fallback_pricing.return_value = {
            "default": {"input": 0.7, "output": 2.5},
            "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
        }
        mock_client.return_value = mock_instance

        config = Config()
        pricing = config.model_pricing

        # Should fall back to static pricing when API fails
        mock_instance.get_fallback_pricing.assert_called_once()
        assert "default" in pricing, "Default model missing from fallback"
        assert pricing["default"]["input"] == 0.7, "Incorrect fallback input pricing"
        print("✅ Fallback behavior works when API fails")


def test_pricing_data_format():
    """Test that pricing data format is compatible with judge.py expectations."""
    print("\n🧪 Testing pricing data format compatibility...")

    client = OpenRouterPricingClient()
    fallback = client.get_fallback_pricing()

    # Test format expected by judge.py
    for model_id, pricing in fallback.items():
        assert isinstance(pricing, dict), f"Pricing for {model_id} is not a dict"
        assert "input" in pricing, f"Missing 'input' key for {model_id}"
        assert "output" in pricing, f"Missing 'output' key for {model_id}"
        assert isinstance(pricing["input"], int | float), (
            f"Input pricing for {model_id} is not numeric"
        )
        assert isinstance(pricing["output"], int | float), (
            f"Output pricing for {model_id} is not numeric"
        )

    print("✅ Pricing data format is compatible with judge.py")


def test_calculate_cost_handles_scalar_and_list_model_id():
    """`calculate_cost` must price a run whether `state_result.json`'s
    ``model_id`` is a single string or a list of candidate models.

    A list `smolagent_model_id` gets persisted into ``model_id`` and reached the
    pricing lookup, which does ``model_id in self.model_pricing`` — raising
    ``unhashable type: 'list'``. Cost is attributed to one model, so a list must
    be priced against its first element (the default every agent falls back to).
    Uses ``MockDataGenerator`` for a realistic ``state_result`` payload.
    """
    print("\n🧪 Testing calculate_cost with mock scalar/list model_id...")

    from sources.utils.mock_data import MockDataGenerator
    from sources.utils.pricing import PricingCalculator

    gen = MockDataGenerator(seed=42)
    model_pricing = {
        "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
        "openrouter/z-ai/glm-5.2": {"input": 0.20, "output": 0.80},
    }
    # priced against deepseek/deepseek-chat, the scalar / first-of-list model
    expected = (1000 * 0.27 + 500 * 1.10) / 1_000_000

    def _cost_for(model_id_value):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = os.path.join(tmp, "memory")
            workflow_dir = os.path.join(tmp, "workflow")
            run_uuid = "run_mock"
            os.makedirs(os.path.join(memory_dir, run_uuid))
            os.makedirs(os.path.join(workflow_dir, run_uuid))

            state = gen.generate_state_result(workflow_uuid=run_uuid)
            state["model_id"] = model_id_value  # the shape under test
            with open(os.path.join(workflow_dir, run_uuid, "state_result.json"), "w") as fh:
                json.dump(state, fh)

            # one agent task memory file — the token usage calculate_cost sums
            steps = [{"token_usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}}]
            with open(os.path.join(memory_dir, run_uuid, "task_solver.json"), "w") as fh:
                json.dump(steps, fh)

            config = SimpleNamespace(
                memory_dir=memory_dir, workflow_dir=workflow_dir, model_pricing=model_pricing
            )
            return PricingCalculator(config).calculate_cost(run_uuid)

    scalar_cost = _cost_for("deepseek/deepseek-chat")
    assert abs(scalar_cost - expected) < 1e-9, f"scalar model_id: got {scalar_cost}, expected {expected}"
    print("✅ scalar model_id priced correctly")

    # The reported regression: a list must not raise and prices against the first.
    list_cost = _cost_for(["deepseek/deepseek-chat", "openrouter/z-ai/glm-5.2"])
    assert abs(list_cost - expected) < 1e-9, f"list model_id: got {list_cost}, expected {expected}"
    print("✅ list model_id priced against first element (no unhashable crash)")


def test_single_agent_memory_is_priced():
    """A single-agent run's memory must be summed into the run's cost.

    ``single_agent_factory`` saves its memory as ``task_single_agent.json``
    (``save_agent_memories(agent, MEMORY_PATH, "single_agent")``), so the
    ``task_`` prefix matches it. The run folder is named ``single_agent_*``,
    which is a directory rather than a memory file.
    """
    print("\n🧪 Testing single-agent memory is priced...")

    from sources.utils.pricing import PricingCalculator

    model_pricing = {"deepseek/deepseek-chat": {"input": 0.27, "output": 1.10}}
    expected = (1000 * 0.27 + 500 * 1.10) / 1_000_000

    with tempfile.TemporaryDirectory() as tmp:
        memory_dir = os.path.join(tmp, "memory")
        workflow_dir = os.path.join(tmp, "workflow")
        run_uuid = "single_agent_20260101_abc123"
        os.makedirs(os.path.join(memory_dir, run_uuid))
        os.makedirs(os.path.join(workflow_dir, run_uuid))

        with open(os.path.join(workflow_dir, run_uuid, "state_result.json"), "w") as fh:
            json.dump({"model_id": "deepseek/deepseek-chat"}, fh)

        steps = [{"token_usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}}]
        with open(os.path.join(memory_dir, run_uuid, "task_single_agent.json"), "w") as fh:
            json.dump(steps, fh)

        config = SimpleNamespace(
            memory_dir=memory_dir, workflow_dir=workflow_dir, model_pricing=model_pricing
        )
        cost = PricingCalculator(config).calculate_cost(run_uuid)

    assert abs(cost - expected) < 1e-9, f"single-agent memory: got {cost}, expected {expected}"
    print("✅ task_single_agent.json still priced")


def _calculator_with_one_priced_model():
    """A PricingCalculator whose table cannot match `some/unlisted-model`."""
    from sources.utils.pricing import PricingCalculator

    config = SimpleNamespace(
        memory_dir="/tmp",
        workflow_dir="/tmp",
        model_pricing={"deepseek/deepseek-chat": {"input": 0.27, "output": 1.10}},
    )
    return PricingCalculator(config)


def test_unpriced_model_does_not_prompt_when_stdin_is_not_a_tty():
    """An unmatched model id must never block an unattended run on stdin.

    `_get_model_pricing_with_fallback` runs on the hot path of every evolution
    iteration (`evolution_engine._evaluate_and_calculate_cost`). A bare
    `input()` there hangs a batch or cron run at 0% CPU until it is killed,
    discarding every paid call the run had already made.
    """
    print("\n🧪 Testing headless pricing fallback (non-TTY)...")

    from sources.utils.pricing import PricingCalculator

    calc = _calculator_with_one_priced_model()

    def _explode(*_args, **_kwargs):
        raise AssertionError("input() must not be called when stdin is not a TTY")

    with patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: False)), \
            patch("builtins.input", _explode):
        pricing = calc._get_model_pricing_with_fallback("some/unlisted-model")

    assert pricing == PricingCalculator.DEFAULT_PRICING, (
        f"Expected default pricing, got {pricing}"
    )
    print("✅ Non-TTY run falls back without prompting")


def test_unpriced_model_survives_eof_on_stdin():
    """Closed stdin must fall back, not raise. EOFError was not caught."""
    print("\n🧪 Testing pricing fallback on EOFError...")

    from sources.utils.pricing import PricingCalculator

    calc = _calculator_with_one_priced_model()

    with patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: True)), \
            patch("builtins.input", side_effect=EOFError):
        pricing = calc._get_model_pricing_with_fallback("some/unlisted-model")

    assert pricing == PricingCalculator.DEFAULT_PRICING, (
        f"Expected default pricing, got {pricing}"
    )
    print("✅ EOFError falls back instead of propagating")


def test_interactive_manual_pricing_entry_still_works():
    """The TTY path must keep accepting manual entry — guards over-correction."""
    print("\n🧪 Testing interactive manual pricing entry...")

    calc = _calculator_with_one_priced_model()

    with patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: True)), \
            patch("builtins.input", side_effect=["1.5", "6.0"]):
        pricing = calc._get_model_pricing_with_fallback("some/unlisted-model")

    assert pricing == {"input": 1.5, "output": 6.0}, (
        f"Manual entry not honoured, got {pricing}"
    )
    print("✅ Interactive manual entry preserved")


def test_default_pricing_constant_is_not_mutated_by_callers():
    """The fallback returns a copy — a caller must not corrupt the constant."""
    print("\n🧪 Testing DEFAULT_PRICING immutability...")

    from sources.utils.pricing import PricingCalculator

    calc = _calculator_with_one_priced_model()

    with patch.object(sys, "stdin", SimpleNamespace(isatty=lambda: False)):
        pricing = calc._get_model_pricing_with_fallback("some/unlisted-model")
    pricing["input"] = 999.0

    assert PricingCalculator.DEFAULT_PRICING["input"] == 3.0, (
        "DEFAULT_PRICING was mutated through the returned dict"
    )
    print("✅ DEFAULT_PRICING is returned as a copy")


def run_all_tests():
    """Run all pricing tests."""
    print("Starting pricing functionality tests...\n")

    try:
        test_openrouter_pricing_client()
        test_config_pricing_integration()
        test_pricing_fallback_behavior()
        test_pricing_data_format()
        test_calculate_cost_handles_scalar_and_list_model_id()
        test_single_agent_memory_is_priced()
        test_unpriced_model_does_not_prompt_when_stdin_is_not_a_tty()
        test_unpriced_model_survives_eof_on_stdin()
        test_interactive_manual_pricing_entry_still_works()
        test_default_pricing_constant_is_not_mutated_by_callers()

        print("\n🎉 All pricing tests passed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
