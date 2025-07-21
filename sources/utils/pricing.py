"""
OpenRouter API client for real-time model pricing
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests


@dataclass
class ModelPricing:
    """Model pricing information from OpenRouter."""
    input_cost_per_token: float
    output_cost_per_token: float
    request_cost: float = 0.0
    image_cost: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary format compatible with existing code."""
        return {
            "input": self.input_cost_per_token * 1_000_000,  # Convert to per-million tokens
            "output": self.output_cost_per_token * 1_000_000,  # Convert to per-million tokens
            "request": self.request_cost,
            "image": self.image_cost
        }


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
                    
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
                if datetime.now() - cache_time < self.cache_duration:
                    return cache_data.get('pricing', {})
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return None
    
    def _save_cache(self, pricing_data: dict):
        """Save pricing data to cache."""
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'pricing': pricing_data
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def fetch_model_pricing(self) -> dict[str, ModelPricing]:
        """Fetch current model pricing from OpenRouter API."""
        cached_data = self._load_cache()
        if cached_data:
            return {model_id: ModelPricing(**pricing) for model_id, pricing in cached_data.items()}
        
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            pricing_data = {}
            
            for model in models_data.get('data', []):
                model_id = model.get('id')
                pricing_info = model.get('pricing', {})
                
                if model_id and pricing_info:
                    # Convert OpenRouter pricing to ModelPricing
                    pricing_data[model_id] = ModelPricing(
                        input_cost_per_token=float(pricing_info.get('prompt', 0)),
                        output_cost_per_token=float(pricing_info.get('completion', 0)),
                        request_cost=float(pricing_info.get('request', 0)),
                        image_cost=float(pricing_info.get('image', 0))
                    )
            
            # Save to cache
            cache_data = {model_id: {
                'input_cost_per_token': pricing.input_cost_per_token,
                'output_cost_per_token': pricing.output_cost_per_token,
                'request_cost': pricing.request_cost,
                'image_cost': pricing.image_cost
            } for model_id, pricing in pricing_data.items()}
            
            self._save_cache(cache_data)
            return pricing_data
            
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Failed to fetch OpenRouter pricing: {e}")
            return {}
    
    def get_model_pricing_dict(self) -> dict[str, dict[str, float]]:
        """Get model pricing in the format expected by existing code."""
        pricing_data = self.fetch_model_pricing()
        return {model_id: pricing.to_dict() for model_id, pricing in pricing_data.items()}
    
    def get_fallback_pricing(self) -> dict[str, dict[str, float]]:
        """Get fallback pricing for common models."""
        return {
            # OpenAI models
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "o1-preview": {"input": 15.0, "output": 60.0},
            "o1-mini": {"input": 3.0, "output": 12.0},
            
            # Anthropic models
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            
            # Deepseek models
            "deepseek-chat": {"input": 0.27, "output": 1.10},
            "deepseek-reasoner": {"input": 0.55, "output": 2.19},
            
            # Default pricing
            "default": {"input": 0.70, "output": 2.50}
        }