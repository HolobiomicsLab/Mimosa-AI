#!/usr/bin/env python3
"""
Test script for OpenRouter pricing integration
"""

import os
import sys

from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from config import Config
from sources.utils.pricing import OpenRouterPricingClient


def test_openrouter_client():
    """Test OpenRouter pricing client directly."""
    print("🔍 Testing OpenRouter pricing client...")
    
    client = OpenRouterPricingClient(cache_duration_hours=0)  # Force fresh data
    
    # Test fetching pricing data
    pricing_data = client.get_model_pricing_dict()
    
    if pricing_data:
        print(f"✅ Retrieved pricing for {len(pricing_data)} models")
        
        # Show a few examples
        sample_models = list(pricing_data.keys())[:5]
        print("\n📊 Sample pricing data:")
        for model in sample_models:
            pricing = pricing_data[model]
            print(f"  {model}: ${pricing['input']:.2f}M input, ${pricing['output']:.2f}M output")
    else:
        print("❌ Failed to retrieve pricing data")
    
    return pricing_data


def test_config_integration():
    """Test pricing integration in Config class."""
    print("\n🔍 Testing Config pricing integration...")
    
    # Test with OpenRouter enabled
    os.environ["ENABLE_OPENROUTER_PRICING"] = "true"
    config = Config()
    
    print(f"✅ Config initialized with {len(config.model_pricing)} models")
    
    # Show some pricing examples
    print("\n📊 Sample pricing from config:")
    sample_models = list(config.model_pricing.keys())[:5]
    for model in sample_models:
        pricing = config.model_pricing[model]
        print(f"  {model}: ${pricing['input']:.2f}M input, ${pricing['output']:.2f}M output")
    
    # Test refresh functionality
    print("\n🔄 Testing pricing refresh...")
    config.refresh_pricing()
    
    return config


def test_fallback_pricing():
    """Test fallback pricing when OpenRouter is disabled."""
    print("\n🔍 Testing fallback pricing...")
    
    # Test with OpenRouter disabled
    os.environ["ENABLE_OPENROUTER_PRICING"] = "false"
    config = Config()
    
    print(f"✅ Fallback config initialized with {len(config.model_pricing)} models")
    
    # Show fallback pricing
    print("\n📊 Fallback pricing:")
    for model, pricing in config.model_pricing.items():
        print(f"  {model}: ${pricing['input']:.2f}M input, ${pricing['output']:.2f}M output")
    
    return config


if __name__ == "__main__":
    print("🚀 Testing OpenRouter pricing integration\n")
    
    try:
        # Test OpenRouter client
        pricing_data = test_openrouter_client()
        
        # Test config integration
        config = test_config_integration()
        
        # Test fallback pricing
        fallback_config = test_fallback_pricing()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)