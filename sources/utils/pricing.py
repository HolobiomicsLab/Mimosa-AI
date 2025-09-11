"""
OpenRouter API client for real-time model pricing
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests


@dataclass
class TokenUsage:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class PricingCalculator:
    def __init__(self, config):
        self.memory_dir = Path(config.memory_dir)
        self.workflow_dir = Path(config.workflow_dir)
        self.model_pricing = config.model_pricing
    
    def _find_model_by_substring(self, target_model: str) -> str | None:
        """
        Find model using substring matching
        
        For anthropic/claude-opus-4-20250514:
        1. Check if any available model is a substring of the target
        2. Return the longest match (most specific)
        
        Examples:
        - anthropic/claude-opus-4-20250514 contains anthropic/claude-opus-4 ✅
        - openai/gpt-5-2025-08-07 contains openai/gpt-5 ✅
        """
        if '/' not in target_model:
            return None
        
        # Find all available models that are substrings of target
        matches = []
        for available_model in self.model_pricing:
            if available_model in target_model:
                matches.append(available_model)
        
        if not matches:
            return None
        
        # Return the longest match (most specific)
        # e.g., prefer "anthropic/claude-opus-4" over "anthropic/claude"
        return max(matches, key=len)
    
    def _get_model_pricing_with_fallback(self, model_name: str) -> dict:
        """Get model pricing with intelligent fallback."""
        
        # 1. Try exact match
        if model_name in self.model_pricing:
            return self.model_pricing[model_name]
        
        # 2. Try substring matching
        pattern_match = self._find_model_by_substring(model_name)
        if pattern_match:
            print(f"📊 Using pricing for {pattern_match} (pattern matched from {model_name})")
            return self.model_pricing[pattern_match]
        
        # 3. Default fallback
        print(f"⚠️  No match found for {model_name}, using default pricing")
        return self.model_pricing.get("default", {"input": 0.70, "output": 2.50})

    def calculate_cost(self, uuid: str) -> float:
        """Calculate the cost of a workflow run based on token usage.

        Args:
            config: The configuration object
            uuid: Optional UUID of the workflow run to calculate cost for.
                If not provided, will try to find the most recent run.

        Returns:
            float: The total cost in USD
        """

        print("\n📊 Calculating final cost...")

        memory_path = Path(self.memory_dir) / uuid

        if not memory_path.exists():
            print(f"❌ Memory directory not found: {memory_path}")
            return 0.0

        llm_calls: list[TokenUsage] = []

        # Orchestrator and Judge LLM calls

        for call in ["workflow_creator", "judge"]:
            memory_file = memory_path / f"{call}.json"
            if not memory_file.exists():
                continue

            with open(memory_file) as f:
                json_call = json.load(f)
                llm_calls.append(
                    TokenUsage(
                        call,
                        json_call["model"],
                        json_call["usage"]["prompt_tokens"],
                        json_call["usage"]["completion_tokens"],
                        json_call["usage"]["total_tokens"],
                    )
                )

        workflow_path = Path(self.workflow_dir) / uuid

        if not workflow_path.exists():
            print(f"❌ Workflow directory not found: {workflow_path}")
            return 0.0

        try:
            with open(workflow_path / "state_result.json") as f:
                state_results = json.load(f)
                model_id = state_results.get("model_id", None)
        except FileNotFoundError:
            print(f"❌ State result file not found for UUID {uuid} in {workflow_path}.")
            return 0.0

        try:
            for file in os.listdir(memory_path):
                if file.startswith("task_") and file.endswith(".json"):
                    with open(memory_path / file) as f:
                        steps = json.load(f)
                        token_usage = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        }
                        for step in steps:
                            step_usage = step.get("token_usage", None)
                            if token_usage:
                                token_usage = {
                                    key: token_usage[key] + step_usage[key]
                                    for key in step_usage
                                }
                        llm_calls.append(
                            TokenUsage(
                                file.replace("task_", "").replace(".json", ""),
                                model_id,
                                *token_usage.values(),
                            )
                        )
        except Exception as e:
            print(f"❌ Error reading workflow steps: {str(e)}")
            return 0.0

        total_cost = 0.0
        print("\n💰 Cost Breakdown:")
        print("=" * 60)
        for call in llm_calls:
            pricing = self._get_model_pricing_with_fallback(call.model)
            cost = (
                call.input_tokens * pricing["input"]
                + call.output_tokens * pricing["output"]
            ) / 1_000_000
            print("Agent:", call.agent)
            print(f"  Model: {call.model}")
            print(f"  Tokens: {call.total_tokens:,}")
            print(f"  Cost: {cost:.3f} USD")
            print("-" * 40)
            total_cost += cost

        return total_cost


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
            # OpenAI models (OpenRouter format)
            "openai/gpt-5": {"input": 2.00, "output": 8.00},
            "openai/gpt-5-mini": {"input": 1.10, "output": 4.40},
            "openai/gpt-5-nano": {"input": 0.30, "output": 1.20},
            "openai/o3": {"input": 2.00, "output": 8.00},
            "openai/o3-mini": {"input": 1.10, "output": 4.40},
            "openai/o1": {"input": 15.00, "output": 60.00},
            "openai/o1-mini": {"input": 3.00, "output": 12.00},
            "openai/gpt-4o": {"input": 2.50, "output": 10.00},
            "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "openai/gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            # Anthropic models
            "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
            "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
            # Deepseek models
            "deepseek/deepseek-reasoner": {"input": 0.55, "output": 2.19},
            "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
            # Meta models
            "meta-llama/llama-3.1-8b-instruct": {"input": 0.07, "output": 0.07},
            # Mistral models
            "mistralai/mixtral-8x7b-instruct": {"input": 0.24, "output": 0.24},
            # Default pricing for unknown models
            "default": {"input": 0.70, "output": 2.50},
        }  # Per 1M tokens
