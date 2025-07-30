"""
OpenRouter API client for real-time model pricing
"""

from dataclasses import dataclass
import json
import os
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
            pricing = self.model_pricing.get(
                call.model,
                self.model_pricing.get("default", {"input": 0.70, "output": 2.50}),
            )
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
