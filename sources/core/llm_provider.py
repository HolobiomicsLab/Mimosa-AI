import json
import logging
import os
import time
from dataclasses import dataclass, field

import litellm


@dataclass
class LLMConfig:
    """Configuration for Large Language Model interactions."""

    model: str = "o3-2025-04-16"
    provider: str = "openai"
    temperature: float = 1.0
    key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    reasoning_effort: str = "medium"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.key:
            raise ValueError(
                "API key not provided and OPENAI_API_KEY environment variable not set"
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
            key=config.get("key", os.getenv("OPENAI_API_KEY", "")),
            reasoning_effort=config.get("reasoning_effort", "medium"),
        )


class LLMProvider:
    """Handles interactions with various LLM APIs.
    Attributes:
        deepseek_client (OpenAI): Client for Deepseek API
        openai_client (OpenAI): Client for OpenAI API
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

    def __call__(self, prompt: str, timeout: int = 120):
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
                
                # Add reasoning effort if supported
                if self._supports_reasoning_tokens():
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
            "reasoning_effort": self.config.reasoning_effort,
        }
        if self.memory_path and self.agent_name:
            self.save_call(json_res)

        return res
