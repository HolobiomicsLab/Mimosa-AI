"""
OpenRouter API client for real-time model pricing
"""

import json
import os
from datetime import datetime, timedelta

import requests


class OpenRouterPricingClient:
    """Client for fetching real-time model pricing from OpenRouter API."""

    def __init__(self, cache_duration_hours: int = 24):
        self.base_url = "https://openrouter.ai/api/v1"
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_file = "sources/cache/openrouter_pricing.json"
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _load_cache(self) -> dict | None:
        """Load cached pricing data if still valid."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file) as f:
                    cache_data = json.load(f)

                cache_time = datetime.fromisoformat(cache_data.get("timestamp", ""))
                if datetime.now() - cache_time < self.cache_duration:
                    return cache_data.get("pricing", {})
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return None

    def _save_cache(self, pricing_data: dict):
        """Save pricing data to cache."""
        cache_data = {"timestamp": datetime.now().isoformat(), "pricing": pricing_data}
        with open(self.cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

    def fetch_model_pricing(self) -> dict[str, dict[str, float]]:
        """Fetch current model pricing from OpenRouter API."""
        cached_data = self._load_cache()
        if cached_data:
            return cached_data

        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()

            models_data = response.json()
            pricing_data = {}

            for model in models_data.get("data", []):
                model_id = model.get("id")
                pricing_info = model.get("pricing", {})

                if model_id and pricing_info:
                    # Convert OpenRouter pricing to per-million token pricing
                    input_per_million = float(pricing_info.get("prompt", 0)) * 1_000_000
                    output_per_million = (
                        float(pricing_info.get("completion", 0)) * 1_000_000
                    )

                    pricing_data[model_id] = {
                        "input": input_per_million,
                        "output": output_per_million,
                    }

            # Save to cache
            cache_data = pricing_data.copy()

            self._save_cache(cache_data)
            return pricing_data

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Failed to fetch OpenRouter pricing: {e}")
            return {}

    def get_model_pricing_dict(self) -> dict[str, dict[str, float]]:
        """Get model pricing in the format expected by existing code."""
        return self.fetch_model_pricing()

    def get_fallback_pricing(self) -> dict[str, dict[str, float]]:
        """Get fallback pricing for common models."""
        return {
            # OpenAI models
            "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40},
            "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40},
            "o3-2025-04-16": {"input": 2, "output": 8},
            # Deepseek models
            "deepseek/deepseek-reasoner": {"input": 0.55, "output": 2.19},
            "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
            # Default pricing for unknown models
            "default": {"input": 0.70, "output": 2.50},
        }  # Per 1M tokens
