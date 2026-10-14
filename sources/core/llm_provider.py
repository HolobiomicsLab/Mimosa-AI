"""Multi-provider LLM access for every Mimosa-AI layer that calls a model.
    This module wraps litellm behind two objects: LLMConfig, which holds model,
    provider, key, and sampling settings from the config file, and LLMProvider,
    which sends prompts, retries transient failures, and caches responses. The
    Layer 0 planner, agent prompts, and the Layer 2 evolution engine obtain
    their model clients through it. It runs on every LLM call and writes a
    JSON call record per agent under the memory directory.

"""

import glob
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import litellm


def extract_model_pattern(llm_model: str) -> tuple[str, str]:
    """Split a model identifier into its provider and model parts.
        Config parsing code uses this to turn a config-file model string such as
        ``"openrouter/deepseek-v3"`` into routing fields. It receives the raw
        model string and returns the part before the first slash and the part
        after it. A string with no slash maps to provider ``"anthropic"``.

        Args:
            llm_model (str): Model identifier from the config file, with or
                without a ``provider/`` prefix.

        Returns:
            tuple[str, str]: ``(provider, model)`` strings. The provider is
            ``"anthropic"`` when the input carries no slash.

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
    """Hold model, provider, and sampling settings for one LLM endpoint.
        Callers build it with explicit arguments or through ``from_dict`` from
        the config file, then pass it to LLMProvider. As an invariant, after
        construction the provider name is lower case, the temperature is a
        float, and an empty key is filled from the provider's environment
        variable.

    """

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
        """Build a config from explicit arguments and validate it.
            The config loader and default constructions call it with values read
            from the config file or with class defaults. It stores the fields,
            wraps a single-string provider preference into a list, copies the
            quantization filter into a list, and runs the post-init validation.

            Args:
                model (str): Model identifier from the config file, with or
                    without a ``provider/`` prefix.
                provider (str): Provider name from the config file, stored lower
                    case. Examples: ``"anthropic"``, ``"openai"``, ``"deepseek"``,
                    ``"openrouter"``.
                temperature (float): Sampling temperature from the config file.
                key (str): API key from the config file. When empty, the
                    provider's ``*_API_KEY`` environment variable fills it.
                reasoning_effort (str): Reasoning effort from the config file.
                    One of ``"minimal"``, ``"low"``, ``"medium"``, ``"high"``.
                max_tokens (int): Token budget per request from the config file.
                openrouter_provider (list[str] | str | None): OpenRouter provider
                    preference from the config file. A string becomes a one-item
                    list; ``None`` disables provider pinning.
                openrouter_quantizations (list[str] | tuple[str, ...] | None):
                    Quantization exclusion filter for OpenRouter from the config
                    file. ``None`` disables the filter; the tuple default selects
                    the runtime safety filter. Stored as a list.

            Raises:
                ValueError: When ``reasoning_effort`` is not an allowed value
                (raised by the post-init validation this constructor runs).

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
        """Fill provider defaults and validate the reasoning effort.
            The constructor runs it right after storing the fields. It reads the
            provider environment variables when the key is empty, coerces the
            temperature to float, and checks the reasoning effort against the
            allowed set.

            Raises:
                ValueError: When ``reasoning_effort`` is not one of
                ``"minimal"``, ``"low"``, ``"medium"``, or ``"high"``.

        """
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
            elif self.provider == "z-ai":
                env_var = "ZAI_API_KEY"
            else:
                env_var = f"{self.provider.upper()}_API_KEY"
            _ = env_var  # kept for the commented-out raise below

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
        """Build a config from a plain dictionary.
            Call sites that hold settings as dicts use this constructor to adapt
            them without mutating the dict. Missing keys fall back to the
            constructor defaults, and the quantization filter is forwarded only
            when the dict contains it.

            Args:
                config (dict[str, Any] | None): Settings from the config file.
                    ``None`` is treated as an empty mapping.

            Returns:
                LLMConfig: A new validated instance built from the mapping.

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
    """Send prompts to the configured provider and return text responses.
        The planner, agents, and workflow steps that need model output hold one
        instance built from an LLMConfig. On top of litellm it adds disk
        caching keyed by agent name, retry with backoff, temperature and
        quantization fallbacks, token-budget handling, and usage accounting.
        Invariant: the first exactly matching cached response wins, and each
        successful call is saved as a JSON record when a memory path and an
        agent name are set.

    """

    def __init__(
        self,
        agent_name: str | None = None,
        memory_path: str | None = None,
        system_msg: str | None = None,
        config: LLMConfig | None = None,
        use_flat_cache: bool = False,
    ) -> None:
        """Create a provider for one agent and its cache location.
            The workflow runner and agents construct it at start-up with the agent
            name, the memory directory, and an LLMConfig. It stores these values
            and creates a named logger; no network call happens here.

            Args:
                agent_name (str | None): Agent identifier from the caller; it
                    names the cache file. ``None`` disables cache lookups.
                memory_path (str | None): Memory directory from the caller;
                    cache records are written under it. ``None`` disables saving.
                system_msg (str | None): System message from the caller,
                    prepended to every prompt. ``None`` sends no system message.
                config (LLMConfig | None): Model and sampling settings from the
                    config file. A default LLMConfig is built when ``None``.
                use_flat_cache (bool): Cache mode flag from the caller. ``True``
                    reads and writes ``<memory_path>/<agent_name>.json`` directly;
                    ``False`` searches UUID subfolders of the memory directory's
                    parent (defaulting to ``sources/memory`` when unset).

            Returns:
                None.

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

    def _supports_reasoning_tokens(self) -> bool:
        """Report whether the configured model supports reasoning controls.
            ``__call__`` consults this before adding reasoning-effort parameters.
            It matches the lower-cased model name from the config against known
            reasoning-capable markers, so it needs no network access.

            Returns:
                bool: ``True`` when the model name contains a known marker, else
                ``False``.

        """
        model_name = self.config.model.lower()
        reasoning_models = [
            "o1", "o3", "gpt-5",
            # OpenRouter reasoning-capable families that share one
            # max_tokens budget across reasoning + content; without an
            # explicit effort they default to full-strength thinking and
            # can overrun the budget on long prompts (empty content).
            "glm-5.3", "deepseek-v4", "deepseek-v3",
            "qwen3", "kimi-k3",
        ]
        return any(reasoning_model in model_name for reasoning_model in reasoning_models)

    def _is_claude_model(self) -> bool:
        """Report whether the target model is a Claude model.
            ``__call__`` uses this to omit temperature and reasoning parameters
            that Anthropic endpoints reject or ignore. It checks the provider
            field and the model name from the config.

            Returns:
                bool: ``True`` when the provider is ``"anthropic"`` or the model
                name contains ``"claude"``, else ``False``.

        """
        return self.config.provider == "anthropic" or "claude" in self.config.model.lower()

    def _token_budget_param(self) -> str:
        """Return the token-budget parameter name the endpoint expects.
            ``__call__`` uses this key when building completion parameters,
            because OpenAI's newer models reject the legacy ``max_tokens`` name
            while other providers still expect it. It reads the provider from the
            config.

            Returns:
                str: ``"max_completion_tokens"`` for the ``"openai"`` provider,
                else ``"max_tokens"``.

        """
        return "max_completion_tokens" if self.config.provider == "openai" else "max_tokens"

    def _supports_prompt_caching(self) -> bool:
        """Report whether the provider honours explicit cache breakpoints.
            ``_apply_cache_control`` consults this before marking the system
            message. It reads the provider from the config and needs no network
            access.

            Returns:
                bool: ``True`` for the ``"anthropic"`` and ``"openrouter"``
                providers, else ``False``.

        """
        return self.config.provider in ("anthropic", "openrouter")

    def _apply_cache_control(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Copy the message list and mark the system block as cacheable.
            ``__call__`` passes each request's message list so repeated prefixes
            hit the provider prompt cache. The input list is left untouched; only
            a plain string system message gains an ephemeral cache-control
            breakpoint, and only when a system message exists and the provider
            supports caching.

            Args:
                messages (list[dict[str, Any]]): Chat messages built by
                    ``__call__`` from the system message and the user prompt.

            Returns:
                list[dict[str, Any]]: A copy of the messages with the system
                block marked. The unchanged input list when caching does not
                apply.

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
        """Persist one call record as JSON next to the agent's cache files.
            ``__call__`` invokes it after each successful completion so later
            runs and spend reports can audit the exchange. It writes the record
            dict to ``<memory_path>/<agent_name>.json``, overwriting any previous
            record.

            Args:
                call (dict[str, Any]): Call record built by ``__call__`` from the
                    API response, the message list, and the effective settings.

            Returns:
                None.

        """
        path = os.path.join(self.memory_path, f"{self.agent_name}.json")
        with open(path, "w") as f:
            json.dump(call, f, indent=2)

    def _find_cache_match(self, prompt: str) -> str | None:
        """Look up a cached response for this agent and exact prompt.
            ``__call__`` calls it first on every request so identical agent,
            system-message, and prompt triples reuse the stored answer without a
            network call. It compares the expected message list against each
            candidate cache record and returns the first exact match.

            Args:
                prompt (str): User prompt text from the caller.

            Returns:
                str | None: Cached response text from the matching record, else
                ``None`` when no record matches or caching is disabled.

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
        """Compare two message lists field by field.
            ``_find_cache_match`` uses it to decide whether a cache record was
            produced for exactly the current conversation. It checks equal length
            and equal ``role`` and ``content`` at every position.

            Args:
                expected (list[dict[str, Any]]): Messages built from the current
                    system message and prompt.
                cached (list[dict[str, Any]]): Messages loaded from a cache
                    record.

            Returns:
                bool: ``True`` when both lists match position for position, else
                ``False``.

        """
        if len(expected) != len(cached):
            return False

        for exp_msg, cached_msg in zip(expected, cached, strict=False):
            r_a = exp_msg.get('role')
            r_b = cached_msg.get('role')
            c_a = exp_msg.get('content')
            c_b = cached_msg.get('content')

            if (r_a != r_b or c_a != c_b):
                return False

        return True

    @staticmethod
    def _is_temperature_error(error: Exception) -> bool:
        """Decide whether an API error rejected the temperature parameter.
            The retry loop in ``__call__`` consults it to fall back to temperature
            1.0. It inspects the ``param`` attribute that provider SDK errors
            carry.

            Args:
                error (Exception): Exception raised by the API call.

            Returns:
                bool: ``True`` when the error names ``"temperature"`` as the
                rejected parameter, else ``False``.

        """
        return getattr(error, "param", None) == "temperature"

    @staticmethod
    def _is_quantization_routing_error(error: Exception) -> bool:
        """Decide whether OpenRouter rejected the quantization filter.
            The retry loop in ``__call__`` consults it to drop the filter and
            retry. It matches OpenRouter's "no endpoints found" failure text with
            a quantization mention, which appears when the filter excludes every
            live endpoint.

            Args:
                error (Exception): Exception raised by the API call.

            Returns:
                bool: ``True`` when the error text names both missing endpoints
                and quantization, else ``False``.

        """
        error_str = str(error).lower()
        return "no endpoints found" in error_str and "quantization" in error_str

    def _is_retryable_error(self, error: Exception) -> bool:
        """Decide whether a failed API call is worth retrying.
            The retry loop in ``__call__`` consults it before backing off. It
            matches the error type name and message text against transient
            failure patterns such as overload, rate limits, timeouts, and
            context-window errors.

            Args:
                error (Exception): Exception raised by the API call.

            Returns:
                bool: ``True`` for known transient or context-window errors, else
                ``False``.

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
        """Compute the wait time before the next retry attempt.
            The retry loop in ``__call__`` sleeps for this duration between
            attempts. It doubles the base wait with each attempt and adds up to
            10 percent jitter, capped at the maximum wait.

            Args:
                attempt (int): Zero-indexed retry attempt number from the loop.
                max_wait (int): Upper bound in seconds; the loop passes 500.

            Returns:
                float: Seconds to sleep before the next attempt, never above
                ``max_wait``.

        """
        # Exponential backoff: 2^attempt with jitter
        base_wait = min(2 ** attempt, max_wait)
        # Add random jitter (±10% to avoid thundering herd)
        jitter = base_wait * (0.1 * random.random())
        wait_time = base_wait + jitter
        return min(wait_time, max_wait)

    @staticmethod
    def _usage_dict(usage: Any) -> dict:
        """Convert a litellm usage object into a plain dict.
            ``_merge_usage`` and ``__call__`` use it for token accounting across
            retries. It copies dict input, unwraps objects exposing ``json()``,
            and yields an empty dict when usage is missing or unreadable.

            Args:
                usage (Any): Usage object from a litellm response, or ``None``.

            Returns:
                dict: Plain token-counter mapping; empty when nothing is usable.

        """
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return dict(usage)
        try:
            return dict(usage.json()) if hasattr(usage, "json") else {}
        except Exception:  # noqa: BLE001 — usage is best-effort accounting
            return {}

    @classmethod
    def _merge_usage(cls, acc: dict | None, usage: Any) -> dict:
        """Add one attempt's usage counters into the accumulated totals.
            ``__call__`` merges the usage of abandoned retry attempts so per-agent
            spend reflects the full cost of a request. Numeric values are summed;
            other values keep the latest reading; booleans are skipped.

            Args:
                acc (dict | None): Totals merged so far, or ``None`` on the
                    first call.
                usage (Any): Usage object from the attempt being merged, or
                    ``None``.

            Returns:
                dict: New mapping with summed numeric counters.

        """
        out = dict(acc or {})
        for k, v in cls._usage_dict(usage).items():
            if isinstance(v, bool):
                continue
            if isinstance(v, int | float):
                out[k] = out.get(k, 0) + v
            else:
                out[k] = v
        return out

    def __call__(self, prompt: str, timeout: int = 180, use_cache: bool = True) -> str:
        """Return the model's text answer for a prompt.
            Agents, the planner, and workflow steps call the provider instance
            directly with prompt text and read the returned string. It checks the
            cache first, then calls litellm with retries, backoff, and fallbacks:
            a rejected temperature falls back to 1.0, a rejected quantization
            filter is dropped, context-window errors shrink the user prompt up to
            three times, and a length-truncated answer is retried once with a
            doubled token budget capped at 32768. Non-retryable failures raise,
            as does retry exhaustion. On success it logs token usage, saves the
            call record when a memory path and an agent name are set, and returns
            the text.

            Args:
                prompt (str): User prompt text from the caller.
                timeout (int): Per-attempt request timeout in seconds from the
                    caller; defaults to 180.
                use_cache (bool): Cache flag from the caller. ``True`` returns a
                    cached answer when one matches; ``False`` always calls the
                    API.

            Returns:
                str: Response text, or ``"<LLM response error>"`` when the model
                returns null content.

            Raises:
                RuntimeError: On a non-retryable API error, or when retries are
                exhausted.

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
        # P5.3: usage of ABANDONED (truncated) attempts, summed into the
        # dumped usage so spend accounting reports the full call cost.
        retried_usage: dict | None = None
        while True:  # Infinite retry loop
            try:
                # OpenAI deprecated `max_tokens` in favour of
                # `max_completion_tokens` and its newer models (gpt-5+,
                # o-series, and unreleased ones litellm does not yet know,
                # e.g. gpt-6.1-sol) hard-reject `max_tokens` with a 400.
                # `max_completion_tokens` is accepted by every OpenAI chat
                # model, so use it for the whole openai provider; every
                # other provider keeps `max_tokens`.
                budget_key = self._token_budget_param()
                completion_params = {
                    "model": f"{self.config.provider}/{self.config.model}",
                    "messages": self._apply_cache_control(message),
                    "timeout": timeout,
                    budget_key: self.config.max_tokens,
                    "drop_params": True,
                }
                # Anthropic models reject (Opus 4.x) or ignore an explicit
                # temperature; omit it for all of them rather than version-gate.
                if not self._is_claude_model():
                    completion_params["temperature"] = effective_temperature
                completion_params["api_key"] = self.config.key
                # Add reasoning effort if supported (not for Claude models).
                # FIX 2026-10-07: for OpenRouter reasoning models (deepseek,
                # GLM, etc.), the effort must go through extra_body.reasoning
                # — a top-level reasoning_effort param is silently ignored,
                # causing the model to burn the ENTIRE max_tokens budget on
                # full-strength reasoning (observed: 24000 reasoning tokens,
                # 0 content, on every policy generation attempt).
                if self._supports_reasoning_tokens() and not self._is_claude_model():
                    if self.config.provider == "openrouter":
                        # OpenRouter: reasoning control via extra_body
                        extra = completion_params.get("extra_body") or {}
                        extra["reasoning"] = {
                            "effort": self.config.reasoning_effort,
                        }
                        completion_params["extra_body"] = extra
                        self.logger.info(
                            f"Using reasoning effort via extra_body: "
                            f"{self.config.reasoning_effort}"
                        )
                    else:
                        completion_params["reasoning_effort"] = (
                            self.config.reasoning_effort
                        )
                        self.logger.info(
                            f"Using reasoning_effort: "
                            f"{self.config.reasoning_effort}"
                        )

                # Pin OpenRouter inference provider for reproducible benchmarks.
                # Merged into extra_body (not replacing) so reasoning config
                # from above survives.
                if self.config.provider == "openrouter" and self.config.openrouter_provider:
                    provider_routing = {
                        "order": self.config.openrouter_provider,
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                    if self.config.openrouter_quantizations:
                        provider_routing["quantizations"] = self.config.openrouter_quantizations
                    extra = completion_params.get("extra_body") or {}
                    extra["provider"] = provider_routing
                    completion_params["extra_body"] = extra

                response = litellm.completion(**completion_params)

                # Reasoning models (GLM/deepseek class) draw reasoning tokens
                # from the SAME max_tokens budget as content: on long prompts
                # the reasoning can consume everything, returning finish_
                # reason "length" with empty or single-fragment content
                # (observed live: directives of "" / "∈"). Retry ONCE with a
                # doubled budget before accepting the response.
                content_now = response.choices[0].message.content or ""
                # FIX 2026-10-06: retry on ANY length truncation, not just
                # short content — a 500-char truncated Python file also needs
                # a retry (the code is cut mid-function even though the JSON
                # envelope may have completed). Only skip if we're already
                # at the ceiling.
                if (
                    response.choices[0].finish_reason == "length"
                    and completion_params[budget_key] < 32768
                ):
                    bumped = min(32768, completion_params[budget_key] * 2)
                    self.logger.warning(
                        f"⚠️  Response truncated by length (reasoning "
                        f"overrun; content={len(content_now)} chars); "
                        f"retrying with token budget "
                        f"{completion_params[budget_key]} → {bumped}."
                    )
                    retried_usage = self._merge_usage(
                        retried_usage, getattr(response, "usage", None)
                    )
                    completion_params[budget_key] = bumped
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
        if res is None:
            self.logger.warning(
                "LLM returned null content (finish_reason=%s); returning empty string.",
                getattr(response.choices[0], "finish_reason", None),
            )
            res = "<LLM response error>"

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
        if retried_usage is not None:
            # P5.3: report the FULL cost — the abandoned truncated attempt
            # was billed too.
            json_res["usage"] = self._merge_usage(
                retried_usage, getattr(response, "usage", None)
            )
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
