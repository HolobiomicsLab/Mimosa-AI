#!/usr/bin/env python3
"""
Test script for pricing functionality
Tests the OpenRouterPricingClient and Config integration
"""

import os
import sys
import tempfile
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
        print("\n4️⃣ Testing pricing refresh...")
        config.refresh_pricing()
        assert config._model_pricing_cache is None, "Cache not cleared after refresh"

        assert mock_instance.get_model_pricing_dict.call_count == 2, (
            "Refresh didn't trigger new API call"
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


def run_all_tests():
    """Run all pricing tests."""
    print("🚀 Starting pricing functionality tests...\n")

    try:
        test_openrouter_pricing_client()
        test_config_pricing_integration()
        test_pricing_fallback_behavior()
        test_pricing_data_format()

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
