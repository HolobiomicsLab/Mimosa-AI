import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any
import glob
import random

import litellm

def extract_model_pattern(llm_model: str) -> tuple[str, str]:
    """Split a model identifier into provider and model components.

    Supports the OpenRouter-style ``provider/model`` format and falls back to
    ``"anthropic"`` as the provider for bare model names.

    Args:
        llm_model: Model identifier, optionally prefixed by a provider name
            and a forward slash.

    Returns:
        Tuple of ``(provider, model)`` strings.
    """
    # Extract provider and model from OpenRouter format (provider/model)
    if "/" in llm_model:
        provider, model = llm_model.split("/", 1)
    else:
        # Fallback for backward compatibility
        provider = "anthropic"
        model = llm_model
    return provider, model


@dataclass
class LLMConfig:
    """Configuration for Large Language Model interactions."""

    model: str = "anthropic/claude-sonnet-4-5"
    provider: str = "anthropic"
    temperature: float = 1.0
    key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    reasoning_effort: str = "medium"
    max_tokens = 8192
    openrouter_provider: list[str] | None = None
    # OpenRouter `quantizations` exclusion filter. `None` means omit the
    # filter (required when routing to untagged first-party endpoints like
    # google-vertex). Empty/default list applies a safety filter at runtime.
    openrouter_quantizations: list[str] | None = field(
        default_factory=lambda: ["bf16", "fp16", "fp8"]
    )

    def __init__(self, model: str = model, provider: str = provider, temperature: float = 1.0, key: str = "", reasoning_effort: str = "medium", max_tokens: int = 8192, openrouter_provider: list[str] | str | None = None, openrouter_quantizations: list[str] | tuple[str, ...] | None = ("bf16", "fp16", "fp8")) -> None:
        """Initialize an LLMConfig from explicit arguments.

        Args:
            model: Model identifier (e.g. ``"claude-sonnet-4-5"`` or
                ``"anthropic/claude-sonnet-4-5"``).
            provider: Provider name (lower-cased internally). Examples:
                ``"anthropic"``, ``"openai"``, ``"deepseek"``, ``"openrouter"``.
            temperature: Sampling temperature; coerced to float by
                ``__post_init__``.
            key: API key. If empty, the relevant ``*_API_KEY`` environment
                variable is consulted.
            reasoning_effort: One of ``"minimal"``, ``"low"``, ``"medium"``,
                ``"high"``.
            max_tokens: Maximum number of tokens to request from the model.
            openrouter_provider: Provider routing for OpenRouter. A single
                string is wrapped into a list. ``None`` disables provider
                pinning.
            openrouter_quantizations: Quantization exclusion filter for
                OpenRouter. ``None`` disables the filter; an iterable is
                converted to a list of strings.
        """
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        self.key = key
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        if isinstance(openrouter_provider, str):
            openrouter_provider = [openrouter_provider]
        self.openrouter_provider = openrouter_provider
        # Tuple default keeps a non-None mutable-safe sentinel for "use safety
        # filter"; explicit `None` disables the filter entirely.
        if openrouter_quantizations is None:
            self.openrouter_quantizations = None
        else:
            self.openrouter_quantizations = list(openrouter_quantizations)
        self.__post_init__()

    def __post_init__(self) -> None:
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

            #raise ValueError(
            #    f"API key not provided and {env_var} environment variable not set"
            #)
        self.temperature = float(self.temperature)  # Ensure numeric type

        # Validate reasoning effort
        valid_efforts = {"minimal", "low", "medium", "high"}
        if self.reasoning_effort not in valid_efforts:
            raise ValueError(
                f"reasoning_effort must be one of {valid_efforts}, got '{self.reasoning_effort}'"
            )

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None = None) -> "LLMConfig":
        """Construct an :class:`LLMConfig` from a dictionary.

        Maintains backward compatibility with callers that pass configuration
        as a plain dict. Missing keys fall back to constructor defaults.
        ``openrouter_quantizations`` is only forwarded when explicitly present
        in ``config``.

        Args:
            config: Mapping of configuration values. ``None`` is treated as an
                empty mapping.

        Returns:
            A new :class:`LLMConfig` instance.
        """
        config = config or {}
        kwargs = dict(
            model=config.get("model", "anthropic/claude-sonnet-4-5"),
            provider=config.get("provider", "anthropic"),
            temperature=config.get("temperature", 1.0),
            key=config.get("key", ""),
            reasoning_effort=config.get("reasoning_effort", "medium"),
            max_tokens=config.get("max_tokens", 8192),
            openrouter_provider=config.get("openrouter_provider"),
        )
        # Only forward `openrouter_quantizations` if the caller set it;
        # otherwise inherit the constructor default.
        if "openrouter_quantizations" in config:
            kwargs["openrouter_quantizations"] = config["openrouter_quantizations"]
        return cls(**kwargs)


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
        agent_name: str | None = None,
        memory_path: str | None = None,
        system_msg: str | None = None,
        config: LLMConfig | None = None,
        use_flat_cache: bool = False,
    ) -> None:
        """Initialize the LLM provider with API clients.

        Args:
            agent_name: Name of the agent for cache identification.
            memory_path: Path to memory directory used for cache files.
            system_msg: System message prepended to every prompt.
            config: LLM configuration. A default :class:`LLMConfig` is created
                when None.
            use_flat_cache: If True, cache files are stored/searched directly
                in ``memory_path`` without UUID subfolders (useful for plan
                generation).
        """
        if not config:
            config = LLMConfig()

        self.config = config
        self.sys_msg = system_msg
        self.agent_name = agent_name
        self.memory_path = memory_path
        self.use_flat_cache = use_flat_cache
        # Hard cap on transient-error retries in __call__. The retry loop is
        # otherwise unbounded (`while True`), so a persistently overloaded or
        # rate-limited provider would retry forever, re-sending the full prompt
        # each time. Both retryable paths raise once this ceiling is reached.
        self.max_retries = 3
        self.logger = logging.getLogger(__name__)

    def _supports_reasoning_tokens(self) -> bool:
        """Check if the current model supports reasoning tokens.

        Returns:
            True if the model name contains a known reasoning-capable prefix
            (``o1``, ``o3``, ``gpt-5``), False otherwise.
        """
        model_name = self.config.model.lower()
        reasoning_models = ["o1", "o3", "gpt-5"]
        return any(reasoning_model in model_name for reasoning_model in reasoning_models)

    def _is_claude_model(self) -> bool:
        """Check if the current model is a Claude model.

        Returns:
            True when the provider is ``"anthropic"`` or the model name
            contains ``"claude"``, False otherwise.
        """
        return self.config.provider == "anthropic" or "claude" in self.config.model.lower()

    def _supports_prompt_caching(self) -> bool:
        """True for providers that honour Anthropic-style ``cache_control`` hints.

        Anthropic direct caches the marked prefix; OpenRouter forwards the
        hint to upstreams that support it and silently ignores it elsewhere.
        OpenAI caches long prompts automatically and needs no flag.
        """
        return self.config.provider in ("anthropic", "openrouter")

    def _apply_cache_control(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a copy of ``messages`` with an ephemeral breakpoint on the system block.

        Leaves the input list untouched so persistence and the exact-match
        disk cache keep their plain-string shape. Only the system message is
        marked — one of Anthropic's four allowed breakpoints.
        """
        if not self.sys_msg or not self._supports_prompt_caching():
            return messages

        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                out.append({
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    }],
                })
            else:
                out.append(msg)
        return out

    def save_call(self, call: dict[str, Any]) -> None:
        """Save the API call details to a JSON file.

        Args:
            call: Dictionary containing API call details to persist. Written
                to ``<memory_path>/<agent_name>.json``.
        """
        path = os.path.join(self.memory_path, f"{self.agent_name}.json")
        with open(path, "w") as f:
            json.dump(call, f, indent=2)

    def _find_cache_match(self, prompt: str) -> str | None:
        """Look up a cached response matching the current agent and prompt.

        Searches either flat cache (single ``<memory_path>/<agent>.json`` file)
        or UUID-subfolder cache depending on ``self.use_flat_cache``.

        Args:
            prompt: User prompt to match against cached messages.

        Returns:
            Cached response string when an exact message-list match is found,
            otherwise None.
        """
        if not self.agent_name:
            return None

        expected_messages = []
        if self.sys_msg is not None:
            expected_messages.append({"content": self.sys_msg, "role": "system"})
        expected_messages.append({"role": "user", "content": prompt})

        # For flat cache, search directly in memory_path
        if self.use_flat_cache:
            agent_file = os.path.join(self.memory_path, f"{self.agent_name}.json")
            if os.path.exists(agent_file):
                try:
                    with open(agent_file) as f:
                        cached_data = json.load(f)

                    cached_messages = cached_data.get('message', [])
                    if self._messages_match(expected_messages, cached_messages):
                        self.logger.info(f"Cache hit (flat) for agent '{self.agent_name}' with complete context match")
                        return cached_data.get('response', '')
                except OSError as e:
                    self.logger.warning(f"Error reading cache file {agent_file}: {e}")

            self.logger.info(f"Cache miss (flat) for agent '{self.agent_name}'")
            return None

        # For UUID-based cache, search in subfolders
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

    def _messages_match(self, expected: list[dict[str, Any]], cached: list[dict[str, Any]]) -> bool:
        """Compare two message arrays for exact role/content match.

        Args:
            expected: Message list assembled from current sys_msg and prompt.
            cached: Message list loaded from a cache file.

        Returns:
            True when both lists have identical length and identical
            ``role``/``content`` values per position, False otherwise.
        """
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

    @staticmethod
    def _is_temperature_error(error: Exception) -> bool:
        """True when the API rejected ``temperature``, read from ``error.param``."""
        return getattr(error, "param", None) == "temperature"

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable (temporary/transient).

        Args:
            error: The exception to check

        Returns:
            True if the error is retryable, False otherwise
        """
        error_type_name = type(error).__name__.lower()
        error_str = str(error).lower()

        # Check for specific error types (including context window exceeded)
        if "contextwindowexceeded" in error_type_name:
            return True

        # Check for specific retryable error patterns
        retryable_patterns = [
            "overload",  # Overloaded error
            "rate_limit",  # Rate limiting
            "timeout",  # Timeout errors
            "connection",  # Connection errors
            "temporarily unavailable",  # Service temporarily unavailable
            "internal server error",  # 500 errors
            "service unavailable",  # 503 errors
            "gateway",  # Gateway errors
            "too many requests",  # 429 errors
            "context",  # Context window errors
            "token limit",  # Token limit errors
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def _calculate_backoff_wait(self, attempt: int, max_wait: int = 500) -> float:
        """Calculate exponential backoff with jitter.

        Args:
            attempt: The attempt number (0-indexed)
            max_wait: Maximum wait time in seconds (default 500s)

        Returns:
            Number of seconds to wait before the next attempt
        """
        # Exponential backoff: 2^attempt with jitter
        base_wait = min(2 ** attempt, max_wait)
        # Add random jitter (±10% to avoid thundering herd)
        jitter = base_wait * (0.1 * random.random())
        wait_time = base_wait + jitter
        return min(wait_time, max_wait)

    def __call__(self, prompt: str, timeout: int = 180, use_cache: bool = True) -> str:
        """Send a prompt to the configured LLM and return its text response.

        Wraps the call in caching, retry-with-backoff, optional context-window
        shrinkage, and result persistence.

        Args:
            prompt: User prompt to send.
            timeout: Per-attempt timeout in seconds for the underlying
                ``litellm.completion`` call. Defaults to 180.
            use_cache: When True, attempt to return a cached response before
                making a network call. Defaults to True.

        Returns:
            Text content of the LLM's response (cached or freshly produced).

        Raises:
            RuntimeError: When a non-retryable error is raised by the
                underlying API call.
        """
        cached_response = self._find_cache_match(prompt) if use_cache else None
        if cached_response:
            self.logger.info(f"Returning cached response for agent '{self.agent_name}'")
            return cached_response

        message = []
        if self.sys_msg is not None:
            message.append({"content": self.sys_msg, "role": "system"})

        message.append({"role": "user", "content": prompt})

        attempt = 0
        max_wait = 500  # Maximum wait time in seconds
        context_window_retry_count = 0  # Track context window errors specifically
        effective_temperature = self.config.temperature

        while True:  # Infinite retry loop
            try:
                completion_params = {
                    "model": f"{self.config.provider}/{self.config.model}",
                    "messages": self._apply_cache_control(message),
                    "timeout": timeout,
                    "max_tokens": self.config.max_tokens,
                    "drop_params": True,
                }
                # Anthropic models reject (Opus 4.x) or ignore an explicit
                # temperature; omit it for all of them rather than version-gate.
                if not self._is_claude_model():
                    completion_params["temperature"] = effective_temperature
                completion_params["api_key"] = self.config.key
                # Add reasoning effort if supported (not for Claude models)
                if self._supports_reasoning_tokens() and not self._is_claude_model():
                    completion_params["reasoning_effort"] = self.config.reasoning_effort
                    self.logger.info(f"Using reasoning_effort: {self.config.reasoning_effort}")

                # Pin OpenRouter inference provider for reproducible benchmarks.
                # Avoids silent routing to alternative providers that may use different
                # quantizations or serving stacks and produce divergent outputs.
                if self.config.provider == "openrouter" and self.config.openrouter_provider:
                    provider_routing = {
                        "order": self.config.openrouter_provider,
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                    # OpenRouter's `quantizations` field is an exclusion filter:
                    # untagged endpoints (e.g. google-vertex, google-ai-studio)
                    # are dropped when it's set. Omit it when precheck selected
                    # such a provider (config records `None` in that case).
                    if self.config.openrouter_quantizations:
                        provider_routing["quantizations"] = self.config.openrouter_quantizations
                    completion_params["extra_body"] = {"provider": provider_routing}

                response = litellm.completion(**completion_params)

                # Success - break out of retry loop
                break

            except TimeoutError as e:
                # Timeout is retryable, up to the max_retries ceiling.
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"❌ LLM API error: timed out after {self.max_retries} retries"
                    ) from e
                wait_time = self._calculate_backoff_wait(attempt, max_wait)
                self.logger.warning(
                    f"⌛ Timeout on attempt {attempt + 1}. Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                attempt += 1

            except Exception as e:
                if self._is_temperature_error(e) and effective_temperature != 1.0:
                    self.logger.warning(
                        f"Provider rejected temperature={effective_temperature:.2f}; "
                        f"falling back to 1.0 and retrying."
                    )
                    effective_temperature = 1.0
                    continue

                # Check if this is a retryable error
                if self._is_retryable_error(e):
                    error_type = type(e).__name__.lower()
                    is_context_error = "contextwindowexceeded" in error_type or "context" in str(e).lower()

                    if is_context_error and context_window_retry_count < 3:
                        # For context window errors, reduce the prompt and retry
                        context_window_retry_count += 1
                        reduction_factor = 0.5 ** context_window_retry_count  # 0.5, 0.25, 0.125

                        # Reduce the user prompt content
                        if len(message) > 0 and message[-1].get("role") == "user":
                            original_length = len(message[-1]["content"])
                            message[-1]["content"] = message[-1]["content"][:int(original_length * reduction_factor)]
                            self.logger.warning(
                                f"Context window error on attempt {attempt + 1}. "
                                f"Reduced prompt to {len(message[-1]['content'])} chars (factor: {reduction_factor}). "
                                f"Retrying immediately..."
                            )
                        attempt += 1
                    else:
                        # Regular retry with backoff for other retryable errors,
                        # bounded by the max_retries ceiling so a persistently
                        # failing provider cannot loop forever.
                        if attempt >= self.max_retries:
                            raise RuntimeError(
                                f"❌ LLM API error after {self.max_retries} retries: {str(e)}"
                            ) from e
                        wait_time = self._calculate_backoff_wait(attempt, max_wait)
                        self.logger.warning(
                            f"⚠️  Retryable error on attempt {attempt + 1}: {str(e)[:512]}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                        attempt += 1
                else:
                    # Non-retryable error - raise immediately
                    raise RuntimeError(f"❌ LLM API error: {str(e)}") from e

        res = response.choices[0].message.content

        # Log token usage for debugging
        usage = getattr(response, 'usage', None)
        if usage:
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
            completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
            total_tokens = getattr(usage, 'total_tokens', 0) or 0
            # Anthropic surfaces cache hits as cache_{read,creation}_input_tokens via
            # litellm; OpenAI's automatic cache appears under prompt_tokens_details.
            cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
            cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0
            if not cache_read:
                details = getattr(usage, 'prompt_tokens_details', None)
                if details is not None:
                    cache_read = getattr(details, 'cached_tokens', 0) or 0
            cache_suffix = (
                f", Cache read: {cache_read}, Cache creation: {cache_creation}"
                if (cache_read or cache_creation) else ""
            )
            self.logger.info(
                f"📊 Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
                f"Total: {total_tokens}{cache_suffix} (max_tokens: {self.config.max_tokens})"
            )

        # Check for truncation due to max_tokens limit
        stop_reason = getattr(response.choices[0], 'stop_reason', None) or \
                      getattr(response.choices[0], 'finish_reason', None)
        if stop_reason == 'max_tokens' or stop_reason == 'length':
            self.logger.warning(
                f"⚠️  LLM response was truncated due to max_tokens limit ({self.config.max_tokens}). "
                f"Consider increasing max_tokens in config for longer outputs."
            )

        json_res = {
            **response.json(),
            "response": res,
            "message": message,
            "temperature": effective_temperature,
            "reasoning_effort": self.config.reasoning_effort if not self._is_claude_model() else None,
            "model": f"{self.config.provider}/{self.config.model}",  # Ensure consistent model format for pricing
        }
        if self.memory_path and self.agent_name:
            self.save_call(json_res)

        return res

if __name__ == "__main__":
    config = LLMConfig(
        model="claude-haiku-4-5-20251001",
        provider="anthropic"
    )

    llm_provider = LLMProvider(
        agent_name="test_agent",
        memory_path=None,
        system_msg="You are a helpful assistant.",
        config=config
    )

    prompt = """hello"""
    try:
        response = llm_provider(prompt)
        print("Response:", response)
    except Exception as e:
        print(f"Error : {e}")
        print("Make sure DEEPSEEK_API_KEY environment variable is set")
