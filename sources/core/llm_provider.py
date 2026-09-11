import glob
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import litellm

from .completion_backends import (
    CLI_BACKENDS,
    CompletionBackendError,
    call_completion_bridge,
    is_cli_completion_provider,
)

# Highest temperature every supported backend accepts. Some serving stacks
# refuse anything above this with an opaque 400 rather than a typed error, so
# it doubles as the value the retry path falls back to.
_SAFE_MAX_TEMPERATURE = 1.0

# Output-budget escalation. A reasoning model spends its budget on reasoning
# before emitting content, so a budget that fits the answer can still yield a
# truncated one. Doubling twice covers that without unbounded spend.
_MAX_TRUNCATION_RETRIES = 2
_MAX_OUTPUT_TOKENS = 65536


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
    api_base: str | None = None
    api_key_env: str | None = None
    harness_auth_mode: str = "subscription"
    openrouter_provider: list[str] | None = None
    # OpenRouter `quantizations` exclusion filter. `None` means omit the
    # filter (required when routing to untagged first-party endpoints like
    # google-vertex). Empty/default list applies a safety filter at runtime.
    openrouter_quantizations: list[str] | None = field(
        default_factory=lambda: ["bf16", "fp16", "fp8"]
    )

    def __init__(
        self,
        model: str = model,
        provider: str = provider,
        temperature: float = 1.0,
        key: str = "",
        reasoning_effort: str = "medium",
        max_tokens: int = 8192,
        openrouter_provider: list[str] | str | None = None,
        openrouter_quantizations: list[str] | tuple[str, ...] | None = (
            "bf16",
            "fp16",
            "fp8",
        ),
        api_base: str | None = None,
        api_key_env: str | None = None,
        harness_auth_mode: str = "subscription",
    ) -> None:
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
        self.api_base = api_base
        self.api_key_env = api_key_env
        self.harness_auth_mode = harness_auth_mode
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
        is_cli = is_cli_completion_provider(self.provider)
        if self.harness_auth_mode not in {"subscription", "api_key"}:
            raise ValueError("harness_auth_mode must be 'subscription' or 'api_key'")
        if self.provider == "codex-cli" and self.harness_auth_mode != "subscription":
            raise ValueError("codex-cli supports subscription authentication only")
        if self.api_base:
            parsed_base = urlsplit(self.api_base)
            if parsed_base.scheme != "https" or not parsed_base.netloc:
                raise ValueError("api_base must be an absolute HTTPS URL")
            if parsed_base.username is not None or parsed_base.password is not None:
                raise ValueError("api_base must not contain user information")
            if parsed_base.query or parsed_base.fragment:
                raise ValueError("api_base must not contain a query or fragment")
        if self.api_key_env and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if self.api_base and not self.api_key_env and not self.key:
            raise ValueError(
                "api_base requires api_key_env or an explicitly supplied API key"
            )
        if self.api_key_env and (not is_cli or self.harness_auth_mode == "api_key"):
            configured_key = os.getenv(self.api_key_env)
            if not configured_key:
                raise ValueError(
                    "configured API key environment variable is not set: "
                    f"{self.api_key_env}"
                )
            self.key = configured_key
        # Set appropriate API key based on provider
        if self.api_key_env or is_cli:
            pass
        elif self.provider == "anthropic" and not self.key:
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
        valid_efforts = (
            {"low", "medium", "high", "xhigh", "max"}
            if is_cli
            else {"minimal", "low", "medium", "high"}
        )
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
            api_base=config.get("api_base"),
            api_key_env=config.get("api_key_env"),
            harness_auth_mode=config.get("harness_auth_mode", "subscription"),
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
        self.max_retries = 100
        self.logger = logging.getLogger(__name__)
        self.last_completion_metadata: dict[str, Any] | None = None

    def _call_cli_completion(self, prompt: str, timeout: int) -> str:
        """Call one configured CLI backend without cache, retry, or API fallback."""
        if not 1 <= timeout <= 300:
            raise ValueError("CLI completion timeout must be between 1 and 300 seconds")
        messages = []
        if self.sys_msg is not None:
            messages.append({"role": "system", "content": self.sys_msg})
        messages.append({"role": "user", "content": prompt})
        request = {
            "protocol_version": 1,
            "backend": CLI_BACKENDS[self.config.provider],
            "model": self.config.model,
            "messages": messages,
            "response_format": "text",
            "auth_mode": self.config.harness_auth_mode,
            "effort": self.config.reasoning_effort,
            "timeout_seconds": timeout,
        }
        if self.config.provider == "claude-cli" and self.config.harness_auth_mode == "api_key":
            if self.config.api_base:
                request["api_base"] = self.config.api_base
            if self.config.api_key_env:
                request["api_key_env"] = self.config.api_key_env
        unsupported = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        self.logger.warning(
            "CLI text completion does not support temperature or max_tokens; "
            "both controls are omitted from the bridge request."
        )
        result = None
        try:
            result = call_completion_bridge(request)
            if result["status"] != "completed":
                detail = result.get("error") or "bridge did not complete the request"
                raise CompletionBackendError(
                    f"CLI completion {result['status']}: {detail}"
                )
            # Retain these checks at the dispatch seam for test doubles and
            # future alternate clients that bypass call_completion_bridge.
            route_fields = {
                "backend": request["backend"],
                "requested_model": request["model"],
                "auth_mode": request["auth_mode"],
            }
            mismatches = {
                field: (expected, result.get(field))
                for field, expected in route_fields.items()
                if result.get(field) != expected
            }
            if mismatches:
                raise CompletionBackendError(
                    "completion bridge returned mismatched route metadata: "
                    f"{mismatches}"
                )
            if not isinstance(result.get("text"), str) or not result["text"].strip():
                raise CompletionBackendError(
                    "completed bridge result must contain non-empty text"
                )
        except Exception as exc:
            result_metadata = (
                result
                if isinstance(result, dict) and result.get("status") != "completed"
                else {}
            )
            self.last_completion_metadata = {
                **result_metadata,
                "status": result_metadata.get("status", "malformed"),
                "backend": request["backend"],
                "requested_model": self.config.model,
                "unsupported_controls": unsupported,
                "error": f"{type(exc).__name__}: {exc}",
            }
            raise CompletionBackendError(str(exc)) from exc
        self.last_completion_metadata = {**result, "unsupported_controls": unsupported}
        if self.memory_path and self.agent_name:
            self.save_call(
                {
                    "response": result["text"],
                    "message": messages,
                    "completion_metadata": self.last_completion_metadata,
                    "cache_eligible": False,
                }
            )
        return result["text"]

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

                    if cached_data.get("cache_eligible") is False:
                        self.logger.info(
                            "Skipping cache-ineligible record for agent '%s'",
                            self.agent_name,
                        )
                        return None
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

                    if cached_data.get("cache_eligible") is False:
                        continue
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
    def _is_temperature_error(error: Exception, temperature: float | None = None) -> bool:
        """True when the API rejected ``temperature``.

        OpenAI-style backends name the offending field in ``error.param``, so
        that is checked first. Many gateways do not: OpenRouter forwards an
        upstream refusal as a bare 400 whose body carries no ``param`` and
        whose ``metadata.raw`` is often just ``"ERROR"``. For those, the only
        signal available is the pairing of a 400/bad-request with a request
        that asked for a temperature above the widely-supported 1.0 ceiling —
        so treat that combination as a temperature rejection and let the
        caller retry at 1.0 rather than abort the run.
        """
        if getattr(error, "param", None) == "temperature":
            return True
        if temperature is None or temperature <= _SAFE_MAX_TEMPERATURE:
            return False
        status = getattr(error, "status_code", None)
        error_str = str(error).lower()
        looks_bad_request = status == 400 or "badrequest" in type(error).__name__.lower() or (
            "400" in error_str and "error" in error_str
        )
        return bool(looks_bad_request)

    @staticmethod
    def _is_truncated(response: Any) -> bool:
        """True when the provider stopped because the output budget ran out.

        Providers spell it ``length`` (OpenAI-style) or ``max_tokens``
        (Anthropic-style); litellm surfaces whichever the upstream sent.
        """
        try:
            choice = response.choices[0]
        except (AttributeError, IndexError, TypeError):
            return False
        reason = getattr(choice, "stop_reason", None) or getattr(choice, "finish_reason", None)
        return reason in ("length", "max_tokens")

    @staticmethod
    def _is_quantization_routing_error(error: Exception) -> bool:
        """True when OpenRouter found no live endpoint for the requested quantizations.

        OpenRouter answers 404 ("No endpoints found for the request with
        quantization: ...") when the ``quantizations`` routing filter excludes
        every provider currently serving the model — e.g. stale precheck data
        or an endpoint that was requantized. Dropping the filter lets
        OpenRouter route to any available endpoint.
        """
        error_str = str(error).lower()
        return "no endpoints found" in error_str and "quantization" in error_str

    @staticmethod
    def _is_upstream_provider_failure(error: Exception) -> bool:
        """True for a gateway reporting that the *upstream* model failed.

        OpenRouter surfaces an upstream fault as HTTP 400 with
        ``message: "Provider returned error"`` and ``metadata.raw: "ERROR"``.
        The 400 makes it look like a malformed request, but the request is
        fine — the same payload succeeds on the next attempt. Measured against
        stealth/ox-alpha: 6/6 identical calls succeeded in isolation while the
        same prompt shape was failing intermittently under four concurrent
        lanes. Semantically this is a 502, so it is retryable; a genuinely
        malformed request keeps failing and still exhausts the retry ceiling.

        Deliberately narrow: matches the gateway's own wording, not 400s in
        general, so real client errors are not retried in a loop.
        """
        return "provider returned error" in str(error).lower()

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

        if self._is_upstream_provider_failure(error):
            return True

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
        if is_cli_completion_provider(self.config.provider):
            return self._call_cli_completion(prompt, timeout)

        cache_eligible = not (self.config.api_base or self.config.api_key_env)
        cached_response = (
            self._find_cache_match(prompt) if use_cache and cache_eligible else None
        )
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
        # None means "omit temperature from the request". mimosa_v2 generalised
        # the Opus-4-only check into _is_claude_model: Anthropic models either
        # reject an explicit temperature (Opus 4.x) or ignore it, so it is
        # omitted for all of them rather than version-gated.
        effective_temperature = None if self._is_claude_model() else self.config.temperature
        effective_max_tokens = self.config.max_tokens
        truncation_retry_count = 0  # Track output-budget escalations

        while True:  # Infinite retry loop
            try:
                completion_params = {
                    "model": f"{self.config.provider}/{self.config.model}",
                    "messages": self._apply_cache_control(message),
                    "timeout": timeout,
                    "max_tokens": effective_max_tokens,
                    "drop_params": True,
                }
                # Anthropic models reject (Opus 4.x) or ignore an explicit
                # temperature; omit it for all of them rather than version-gate.
                if not self._is_claude_model():
                    completion_params["temperature"] = effective_temperature
                completion_params["api_key"] = self.config.key
                if self.config.api_base:
                    completion_params["api_base"] = self.config.api_base
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

                # A response cut off at the budget is not a success: the
                # caller gets a truncated document (JSON ending mid-string,
                # code ending mid-function) and no exception. Reasoning models
                # make this common, because the budget is spent on reasoning
                # before any content is emitted. Escalate the budget and retry
                # rather than hand back something unparsable.
                if (
                    self._is_truncated(response)
                    and truncation_retry_count < _MAX_TRUNCATION_RETRIES
                    and effective_max_tokens < _MAX_OUTPUT_TOKENS
                ):
                    truncation_retry_count += 1
                    effective_max_tokens = min(effective_max_tokens * 2, _MAX_OUTPUT_TOKENS)
                    self.logger.warning(
                        f"⚠️  Response truncated at max_tokens; retrying with "
                        f"max_tokens={effective_max_tokens} "
                        f"(escalation {truncation_retry_count}/{_MAX_TRUNCATION_RETRIES})."
                    )
                    continue

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
                if (
                    self._is_temperature_error(e, effective_temperature)
                    and effective_temperature != _SAFE_MAX_TEMPERATURE
                ):
                    self.logger.warning(
                        f"Provider rejected temperature={effective_temperature:.2f}; "
                        f"falling back to {_SAFE_MAX_TEMPERATURE} and retrying."
                    )
                    effective_temperature = _SAFE_MAX_TEMPERATURE
                    continue

                # OpenRouter 404: the `quantizations` routing filter excluded
                # every live endpoint for this model. Drop the filter once and
                # retry — subsequent calls on this provider skip it from the
                # start (config is mutated), avoiding one doomed request per call.
                if self._is_quantization_routing_error(e) and self.config.openrouter_quantizations:
                    self.logger.warning(
                        f"⚠️  OpenRouter found no endpoint matching "
                        f"quantizations={self.config.openrouter_quantizations}; "
                        f"dropping the quantization filter and retrying."
                    )
                    self.config.openrouter_quantizations = None
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

        # Still truncated after every escalation: the caller is about to
        # receive an incomplete document, so say so loudly.
        if self._is_truncated(response):
            self.logger.warning(
                f"⚠️  LLM response still truncated at max_tokens={effective_max_tokens} "
                f"after {truncation_retry_count} escalation(s). The caller is "
                f"receiving an incomplete response; raise max_tokens in config."
            )

        json_res = {
            **response.json(),
            "response": res,
            "message": message,
            "temperature": effective_temperature,
            # Record what was actually sent, not what was configured. The
            # request only carries reasoning_effort when the model is one of
            # the reasoning families; persisting the configured value for
            # every other model puts a parameter in the run's provenance that
            # the provider never saw.
            "reasoning_effort": (
                self.config.reasoning_effort
                if self._supports_reasoning_tokens() and not self._is_claude_model()
                else None
            ),
            "model": f"{self.config.provider}/{self.config.model}",  # Ensure consistent model format for pricing
            "cache_eligible": cache_eligible,
        }
        if self.memory_path and self.agent_name:
            self.save_call(json_res)

        self.last_completion_metadata = {
            "status": "completed",
            "backend": "litellm",
            "requested_model": f"{self.config.provider}/{self.config.model}",
            "actual_model": getattr(response, "model", None),
            "usage": response.json().get("usage"),
        }

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
