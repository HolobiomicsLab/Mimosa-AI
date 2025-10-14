import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import glob

import litellm

def extract_model_pattern(llm_model: str) -> tuple[str, str]:
    # Extract provider and model from OpenRouter format (provider/model)
    if "/" in llm_model:
        provider, model = llm_model.split("/", 1)
    else:
        # Fallback for backward compatibility
        provider = "openai"
        model = llm_model
    return provider, model


@dataclass
class LLMConfig:
    """Configuration for Large Language Model interactions."""

    model: str = "claude-3-7-sonnet-20250219"
    provider: str = "anthropic"
    temperature: float = 1.0
    key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    reasoning_effort: str = "medium"

    def __init__(self, model=model, provider=provider, temperature=1.0, key="", reasoning_effort="medium"):
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        self.key = key
        self.reasoning_effort = reasoning_effort
        self.__post_init__()

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set appropriate API key based on provider
        if self.provider == "anthropic" and not self.key:
            self.key = os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "openai" and not self.key:
            self.key = os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "deepseek" and not self.key:
            self.key = os.getenv("DEEPSEEK_API_KEY", "")

        if not self.key:
            if self.provider == "anthropic":
                env_var = "ANTHROPIC_API_KEY"
            elif self.provider == "openai":
                env_var = "OPENAI_API_KEY"
            elif self.provider == "deepseek":
                env_var = "DEEPSEEK_API_KEY"
            else:
                env_var = f"{self.provider.upper()}_API_KEY"

            raise ValueError(
                f"API key not provided and {env_var} environment variable not set"
            )
        self.temperature = float(self.temperature)  # Ensure numeric type

        # Validate reasoning effort
        valid_efforts = {"minimal", "low", "medium", "high"}
        if self.reasoning_effort not in valid_efforts:
            raise ValueError(
                f"reasoning_effort must be one of {valid_efforts}, got '{self.reasoning_effort}'"
            )

    @classmethod
    def from_dict(cls, config: dict = None) -> "LLMConfig":
        """Alternative constructor from dictionary (maintains backward compatibility)."""
        config = config or {}
        return cls(
            model=config.get("model", "o3-2025-04-16"),
            provider=config.get("provider", "openai"),
            temperature=config.get("temperature", 1.0),
            key=config.get("key", ""),
            reasoning_effort=config.get("reasoning_effort", "medium"),
        )


class LLMProvider:
    """Handles interactions with various LLM APIs including OpenAI, Anthropic, and DeepSeek.

    Supported providers:
    - anthropic: Claude models (claude-3-5-sonnet, claude-3-opus, etc.)
    - openai: GPT models (gpt-4, gpt-3.5-turbo, o1, o3, etc.)
    - deepseek: DeepSeek models (deepseek-chat, deepseek-coder, etc.)

    Uses litellm for unified API access across providers.
    """

    def __init__(
        self,
        agent_name: str = None,
        memory_path=None,
        system_msg: str = None,
        config: LLMConfig = None,
    ) -> None:
        """Initialize the LLM provider with API clients."""
        if not config:
            config = LLMConfig()

        self.config = config
        self.sys_msg = system_msg
        self.agent_name = agent_name
        self.memory_path = memory_path
        self.max_retries = 3
        self.logger = logging.getLogger(__name__)

    def _supports_reasoning_tokens(self) -> bool:
        """Check if the current model supports reasoning tokens."""
        model_name = self.config.model.lower()
        reasoning_models = ["o1", "o3", "gpt-5"]
        return any(reasoning_model in model_name for reasoning_model in reasoning_models)

    def _is_claude_model(self) -> bool:
        """Check if the current model is a Claude model."""
        return self.config.provider == "anthropic" or "claude" in self.config.model.lower()

    def save_call(self, call: dict) -> None:
        """
        Save the API call details to a JSON file.

        Args:
            call: Dictionary containing API call details
            uuid_str: Unique identifier for the request
        """
        path = os.path.join(self.memory_path, f"{self.agent_name}.json")
        with open(path, "w") as f:
            json.dump(call, f, indent=2)

    def _find_cache_match(self, prompt: str) -> str | None:
        if not self.agent_name:
            return None

        expected_messages = []
        if self.sys_msg is not None:
            expected_messages.append({"content": self.sys_msg, "role": "system"})
        expected_messages.append({"role": "user", "content": prompt})

        base_memory_dir = os.path.dirname(self.memory_path) if self.memory_path else "sources/memory"
        uuid_pattern = os.path.join(base_memory_dir, "*")
        uuid_folders = [d for d in glob.glob(uuid_pattern) if os.path.isdir(d)]

        for uuid_folder in uuid_folders:
            agent_file = os.path.join(uuid_folder, f"{self.agent_name}.json")
            if os.path.exists(agent_file):
                try:
                    with open(agent_file) as f:
                        cached_data = json.load(f)

                    cached_messages = cached_data.get('message', [])
                    if self._messages_match(expected_messages, cached_messages):
                        self.logger.info(f"Cache hit for agent '{self.agent_name}' with complete context match")
                        return cached_data.get('response', '')

                except OSError as e:
                    self.logger.warning(f"Error reading cache file {agent_file}: {e}")
                    continue

        self.logger.info(f"Cache miss for agent '{self.agent_name}'")
        return None

    def _messages_match(self, expected: list, cached: list) -> bool:
        """Compare two message arrays for exact match."""
        if len(expected) != len(cached):
            return False

        for exp_msg, cached_msg in zip(expected, cached):
            r_a = exp_msg.get('role')
            r_b = cached_msg.get('role')
            c_a = exp_msg.get('content')
            c_b = cached_msg.get('content')

            if (r_a != r_b or c_a != c_b):
                return False

        return True

    def __call__(self, prompt: str, timeout: int = 180):
        cached_response = self._find_cache_match(prompt)
        if cached_response:
            self.logger.info(f"Returning cached response for agent '{self.agent_name}'")
            return cached_response

        message = []
        if self.sys_msg is not None:
            message.append({"content": self.sys_msg, "role": "system"})

        message.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                completion_params = {
                    "model": f"{self.config.provider}/{self.config.model}",
                    "messages": message,
                    "temperature": self.config.temperature,
                    "timeout": timeout,
                }
                completion_params["api_key"] = self.config.key
                # Add reasoning effort if supported (not for Claude models)
                if self._supports_reasoning_tokens() and not self._is_claude_model():
                    completion_params["reasoning_effort"] = self.config.reasoning_effort
                    self.logger.info(f"Using reasoning_effort: {self.config.reasoning_effort}")

                response = litellm.completion(**completion_params)
                break
            except TimeoutError:
                print(f"⌛ Timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.1)  # Small delay before retry
                    continue
                raise RuntimeError(f"❌ LLM Tiemout {self.max_retries} times") from None
            except Exception as e:
                raise RuntimeError(f"❌ LLM API error: {str(e)}") from e

        res = response.choices[0].message.content

        json_res = {
            **response.json(),
            "response": res,
            "message": message,
            "temperature": self.config.temperature,
            "reasoning_effort": self.config.reasoning_effort if not self._is_claude_model() else None,
            "model": f"{self.config.provider}/{self.config.model}",  # Ensure consistent model format for pricing
        }
        if self.memory_path and self.agent_name:
            self.save_call(json_res)

        return res

if __name__ == "__main__":
    # DeepSeek example
    deepseek_config = LLMConfig(
        model="deepseek-chat",
        provider="deepseek",
        temperature=0.7,
        key=os.getenv("DEEPSEEK_API_KEY", ""),
        reasoning_effort="medium"
    )

    # Use DeepSeek provider
    llm_provider = LLMProvider(
        agent_name="test_agent",
        memory_path=None,
        system_msg="You are a helpful assistant.",
        config=deepseek_config
    )

    prompt = "Explain the theory of relativity in simple terms."
    try:
        response = llm_provider(prompt)
        print("DeepSeek Response:", response)
    except Exception as e:
        print(f"Error with DeepSeek: {e}")
        print("Make sure DEEPSEEK_API_KEY environment variable is set")
