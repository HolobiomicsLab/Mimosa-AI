"""
OpenRouter API client for real-time model pricing
"""

import json
import os
import re
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

    # Common routing prefixes that should be stripped for matching
    ROUTING_PREFIXES = ['openrouter/', 'litellm/', 'together/', 'anyscale/']

    def _strip_routing_prefix(self, model_name: str) -> str:
        """Strip common routing prefixes from model name.

        Handles cases like:
        - openrouter/mistralai/mistral-large-2407 -> mistralai/mistral-large-2407
        - litellm/anthropic/claude-3.5-sonnet -> anthropic/claude-3.5-sonnet
        """
        lower_name = model_name.lower()
        for prefix in self.ROUTING_PREFIXES:
            if lower_name.startswith(prefix):
                return model_name[len(prefix):]
        return model_name

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name for flexible matching.

        Handles variations like:
        - claude-haiku-4-5 vs claude-haiku-4.5 (hyphen vs dot in versions)
        - claude-haiku-4-5-20251001 (strips date suffixes)
        - claude-3.5-sonnet vs claude-3-5-sonnet (version separators)
        - openrouter/mistralai/... -> mistralai/... (routing prefixes)
        """
        # First strip any routing prefixes
        normalized = self._strip_routing_prefix(model_name)
        normalized = normalized.lower()

        # Remove common date/version suffixes (e.g., -20251001, -2024-11-20, -v1, -001)
        # Match patterns like: -YYYYMMDD, -YYYY-MM-DD, -MMDD, -vX, -XXX (3+ digits at end)
        normalized = re.sub(r'-\d{8}$', '', normalized)  # -20251001
        normalized = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', normalized)  # -2024-11-20
        normalized = re.sub(r'-\d{4}$', '', normalized)  # -2501 (YYMM format)
        normalized = re.sub(r'-\d{3}$', '', normalized)  # -001

        # Normalize version separators: replace dots and underscores with hyphens
        # This makes "4.5" become "4-5" and "4_5" become "4-5"
        normalized = normalized.replace('.', '-').replace('_', '-')

        # Collapse multiple consecutive hyphens into one
        normalized = re.sub(r'-+', '-', normalized)

        return normalized.strip('-')

    def _find_model_by_substring(self, target_model: str) -> str | None:
        """Find best matching model from pricing data using flexible matching.

        Uses a multi-strategy approach:
        1. Direct substring match (original behavior)
        2. Normalized name matching (handles version format variations)
        3. Provider + base model matching (fallback for similar models)
        """
        if '/' not in target_model:
            return None

        matches = []

        # Strategy 1: Direct substring match (original logic)
        for available_model in self.model_pricing:
            if available_model in target_model:
                idx = target_model.find(available_model)
                end_idx = idx + len(available_model)
                valid_start = idx == 0 or target_model[idx-1] in ['/', '-', ':']
                valid_end = end_idx == len(target_model) or target_model[end_idx] in ['/', '-', ':']
                if valid_start and valid_end:
                    matches.append((available_model, len(available_model), 1))  # priority 1 (best)

        # Strategy 2: Normalized name matching
        normalized_target = self._normalize_model_name(target_model)
        for available_model in self.model_pricing:
            normalized_available = self._normalize_model_name(available_model)

            # Check if normalized available model is contained in normalized target
            if normalized_available in normalized_target:
                idx = normalized_target.find(normalized_available)
                end_idx = idx + len(normalized_available)
                valid_start = idx == 0 or normalized_target[idx-1] in ['/', '-', ':']
                valid_end = end_idx == len(normalized_target) or normalized_target[end_idx] in ['/', '-', ':']
                if valid_start and valid_end:
                    # Avoid duplicates from strategy 1
                    if not any(m[0] == available_model for m in matches):
                        matches.append((available_model, len(normalized_available), 2))  # priority 2

        # Strategy 3: Provider + base model name matching (more lenient)
        # Extract provider (e.g., "anthropic") and base model name
        target_provider = target_model.split('/')[0]
        target_base = target_model.split('/')[-1] if '/' in target_model else target_model

        for available_model in self.model_pricing:
            if '/' not in available_model:
                continue
            available_provider = available_model.split('/')[0]
            available_base = available_model.split('/')[-1]

            # Must match provider
            if target_provider != available_provider:
                continue

            # Normalize base names and check for significant overlap
            norm_target_base = self._normalize_model_name(target_base)
            norm_avail_base = self._normalize_model_name(available_base)

            # Check if one contains the other (after normalization)
            if norm_avail_base in norm_target_base or norm_target_base in norm_avail_base:
                # Avoid duplicates
                if not any(m[0] == available_model for m in matches):
                    # Calculate similarity based on common prefix length
                    common_len = len(os.path.commonprefix([norm_target_base, norm_avail_base]))
                    if common_len >= 5:  # Require at least 5 chars of common prefix
                        matches.append((available_model, common_len, 3))  # priority 3 (lowest)

        if not matches:
            return None

        # Sort by: priority (ascending), then length (descending)
        # This prefers direct matches, then longer matches within same priority
        matches.sort(key=lambda x: (x[2], -x[1]))
        return matches[0][0]

    def _get_model_pricing_with_fallback(self, model_name: str) -> dict:
        """Get model pricing with intelligent fallback."""

        # 1. Try exact match
        if model_name in self.model_pricing:
            return self.model_pricing[model_name]

        # 2. Try exact match after stripping routing prefix
        stripped_name = self._strip_routing_prefix(model_name)
        if stripped_name != model_name and stripped_name in self.model_pricing:
            return self.model_pricing[stripped_name]

        # 3. Try substring matching (includes normalization and prefix stripping)
        pattern_match = self._find_model_by_substring(model_name)
        if pattern_match:
            return self.model_pricing[pattern_match]

        print(f"⚠️  No match found for {model_name}, please enter model cost manually:")
        try:
            input_str = input("Input cost per 1M tokens: ")
            output_str = input("Output cost per 1M tokens: ")
            input_cost = float(input_str)
            output_cost = float(output_str)
            self.model_pricing[model_name] = {
                "input": input_cost,
                "output": output_cost
            }
            return self.model_pricing[model_name]
        except (ValueError, TypeError) as e:
            print(f"❌ Invalid input: {e}. Using default pricing.")
            return {"input": 3.0, "output": 15.0}

    def calculate_cost(self, uuid: str) -> float:
        """Calculate the cost of a workflow run based on token usage.

        Args:
            config: The configuration object
            uuid: Optional UUID of the workflow run to calculate cost for.
                If not provided, will try to find the most recent run.

        Returns:
            float: The total cost in USD
        """

        memory_path = Path(self.memory_dir) / uuid

        if not memory_path.exists():
            print(f"❌ Memory directory not found: {memory_path}")
            return 0.0

        llm_calls: list[TokenUsage] = []

        # Orchestrator and Judge LLM calls (multi-agent mode)
        orchestrator_calls_found = False
        for call in ["workflow_creator", "judge"]:
            memory_file = memory_path / f"{call}.json"
            if not memory_file.exists():
                continue
            orchestrator_calls_found = True
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

        # Check for single agent mode (no orchestrator calls but has task files)
        if not orchestrator_calls_found:
            print("📊 Single agent mode detected - calculating agent execution costs only")

        workflow_path = Path(self.workflow_dir) / uuid

        if not workflow_path.exists():
            print(f"❌ Workflow directory not found: {workflow_path}")
            return 0.0

        model_id = None
        try:
            with open(workflow_path / "state_result.json") as f:
                state_results = json.load(f)
                model_id = state_results.get("model_id", None)
        except FileNotFoundError:
            print(f"⚠️  State result file not found for UUID {uuid} - workflow may have failed during execution.")
            print("📊 Will calculate costs for workflow generation and judge calls only.")

        # Only process SmolAgent costs if workflow execution succeeded and we have model_id
        if model_id:
            try:
                for file in os.listdir(memory_path):
                    if (file.startswith("task_") or file.startswith("single_agent")) and file.endswith(".json"):
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
                # Don't return 0.0 here - we can still calculate workflow generation costs
        else:
            print("📊 Skipping SmolAgent cost calculation (workflow execution failed)")

        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_all_tokens = 0

        # Pre-calculate costs
        call_costs = []
        for call in llm_calls:
            pricing = self._get_model_pricing_with_fallback(call.model)
            cost = (
                call.input_tokens * pricing["input"]
                + call.output_tokens * pricing["output"]
            ) / 1_000_000
            call_costs.append((call, cost))
            total_cost += cost
            total_input_tokens += call.input_tokens
            total_output_tokens += call.output_tokens
            total_all_tokens += call.total_tokens

        from sources.cli.pretty_print import (
            BOLD, CYAN, DIM, GREEN, MAGENTA, RESET, YELLOW,
        )

        W = 64

        # Header
        print(f"\n{CYAN}{'─' * W}{RESET}")
        print(f"{CYAN}{'💰  COST BREAKDOWN':^{W}}{RESET}")
        print(f"{CYAN}{'─' * W}{RESET}")

        for call, cost in call_costs:
            # Agent name row
            label = call.agent.replace("_", " ").title()
            print(f"  {BOLD}{MAGENTA}▸ {label}{RESET}")
            # Details as aligned key-value pairs
            print(f"    {DIM}Model{RESET}    {call.model}")
            print(
                f"    {DIM}Tokens{RESET}   "
                f"{call.input_tokens:>9,} in  │  "
                f"{call.output_tokens:>9,} out  │  "
                f"{BOLD}{call.total_tokens:>10,}{RESET} total"
            )
            cost_color = GREEN if cost < 0.01 else (YELLOW if cost < 0.10 else CYAN)
            print(f"    {DIM}Cost{RESET}     {cost_color}${cost:.4f}{RESET}")
            print(f"  {DIM}{'·' * (W - 4)}{RESET}")

        # Totals
        print(f"\n  {BOLD}{'Total Tokens':<14}{RESET}  {total_all_tokens:>10,}   "
              f"{DIM}({total_input_tokens:,} in / {total_output_tokens:,} out){RESET}")
        total_color = GREEN if total_cost < 0.05 else (YELLOW if total_cost < 0.50 else CYAN)
        print(f"  {BOLD}{'Total Cost':<14}{RESET}  {total_color}{BOLD}${total_cost:.4f} USD{RESET}")
        print(f"{CYAN}{'─' * W}{RESET}\n")

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
            "default": {"input": 3.00, "output": 15.00},
        }  # Per 1M tokens
