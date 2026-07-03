"""
Interactive onboarding CLI for Mimosa-AI.

Guides new users through setup step-by-step (Claude-code style),
checks that Toolomics is online, clarifies and refines the user's
objective using an LLM conversation loop, classifies it as Goal-mode
or Task-mode, then hands off to the appropriate execution entry-point
(planner.start_planner or evolve.start_workflow_evolution).
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from typing import Literal

from config import Config
from sources.core.llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from sources.core.tools_manager import ToolManager
from sources.utils import paths
from sources.utils.list_files import list_files
from sources.utils.transfer_toolomics import LocalTransfer


def _persisted_config_path() -> str:
    """Return where onboarding loads and saves persistent settings.

    Repo checkouts keep the historical ``config_default.json`` in the
    working directory; installed (uv tool / pip) runs persist to
    ``~/.config/mimosa/config.json`` so onboarding works from any path.
    """
    if os.path.isfile("config_default.json") or paths.is_repo_checkout():
        return "config_default.json"
    return str(paths.user_config_file())


def _parse_indices(text: str, max_index: int) -> tuple[set[int], bool]:
    """Parse a '1,3' / '2-5' selection string.

    Args:
        text: Raw user input.
        max_index: Highest valid 1-based index.

    Returns:
        A tuple of (valid selected indices, whether any token was invalid).
    """
    selected: set[int] = set()
    had_bad_token = False
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        bounds = part.split("-", 1) if "-" in part else [part, part]
        try:
            lo, hi = int(bounds[0]), int(bounds[1])
        except ValueError:
            had_bad_token = True
            continue
        selected.update(range(lo, hi + 1))
    return {i for i in selected if 1 <= i <= max_index}, had_bad_token


def _list_subdirectories(root: str) -> list[str]:
    """Return sorted, non-hidden subdirectory names of *root* ([] on error)."""
    try:
        return sorted(
            entry.name for entry in os.scandir(root)
            if entry.is_dir(follow_symlinks=False) and not entry.name.startswith(".")
        )
    except OSError:
        return []


def _upsert_env_file(env_file, values: dict[str, str]) -> None:
    """Create or update *env_file* (a Path), replacing entries listed in *values*."""
    env_file.parent.mkdir(parents=True, exist_ok=True)
    lines = env_file.read_text().splitlines() if env_file.is_file() else []
    kept = [ln for ln in lines if ln.split("=", 1)[0].strip() not in values]
    kept.extend(f"{key}={value}" for key, value in values.items())
    env_file.write_text("\n".join(kept) + "\n")
    env_file.chmod(0o600)


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

MIMOSA_BANNER = f"""
{CYAN}{BOLD}
  ███╗   ███╗██╗███╗   ███╗ ██████╗ ███████╗ █████╗
  ████╗ ████║██║████╗ ████║██╔═══██╗██╔════╝██╔══██╗
  ██╔████╔██║██║██╔████╔██║██║   ██║███████╗███████║
  ██║╚██╔╝██║██║██║╚██╔╝██║██║   ██║╚════██║██╔══██║
  ██║ ╚═╝ ██║██║██║ ╚═╝ ██║╚██████╔╝███████║██║  ██║
  ╚═╝     ╚═╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
{RESET}
{DIM}  Self-evolving AI Framework for Autonomous Scientific Research{RESET}
"""

MIMOSA_START_BANNER = f"""
{GREEN}{BOLD}
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║    🌱  M I M O S A   —   S T A R T I N G   U P  🌱           ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
{RESET}
"""

TOTAL_STEPS = 9

# ---------------------------------------------------------------------------
# Model presets — ordered by quality/preference
# (env_key, display_label, litellm_model_id)
# ---------------------------------------------------------------------------
_MODEL_PRESETS: list[tuple[str, str, str]] = [
    ("OPENROUTER_API_KEY", "GLM-5 via OpenRouter (z-ai)",     "openrouter/z-ai/glm-5"),
    ("ANTHROPIC_API_KEY",  "Claude Opus 4.7  (Anthropic)",  "anthropic/claude-opus-4-7"),
    ("DEEPSEEK_API_KEY",   "DeepSeek Chat      (DeepSeek)",   "deepseek/deepseek-chat"),
    ("OPENAI_API_KEY",     "GPT-4o             (OpenAI)",     "openai/gpt-4o"),
    ("MISTRAL_API_KEY",    "Mistral Large      (Mistral)",    "mistral/mistral-large-latest"),
]
# Config keys that all share the same "main" LLM selection
_MODEL_CFG_KEYS = [
    "planner_llm_model",
    "workflow_llm_model",
    "judge_model",
]


def _print_step(step: int, total: int, title: str, no_count: bool = False) -> None:
    bar = "─" * 60
    print(f"\n{CYAN}{bar}{RESET}")
    if not no_count:
        print(f"{CYAN}  Step {step}/{total}  ·  {title}{RESET}")
    else:
        print(f"{CYAN}  {title}{RESET}")
    print(f"{CYAN}{bar}{RESET}")


def _ok(msg: str) -> None:
    print(f"{GREEN}  ✅  {msg}{RESET}")


def _warn(msg: str) -> None:
    print(f"{YELLOW}  ⚠️   {msg}{RESET}")


def _err(msg: str) -> None:
    print(f"{RED}  ❌  {msg}{RESET}")


def _info(msg: str) -> None:
    print(f"{DIM}  ℹ️   {msg}{RESET}")


def _ask(prompt: str, default: str = "") -> str:
    """Print a prompt and return stripped user input.  Empty → *default*."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"\n{BOLD}  ➤  {prompt}{suffix}: {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return answer if answer else default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question and return a boolean."""
    hint = "Y/n" if default else "y/N"
    raw = _ask(f"{prompt} ({hint})", default="y" if default else "n").lower()
    return raw in ("y", "yes", "1", "true")


def _wrap(text: str, width: int = 72, indent: int = 4) -> str:
    return textwrap.fill(text, width=width, initial_indent=" " * indent,
                         subsequent_indent=" " * indent)


def _build_llm(config: Config, temperature: float = 0.0,
               max_tokens: int = 512) -> LLMProvider:
    """Build a lightweight LLMProvider from the planner model config."""
    provider, model = extract_model_pattern(config.planner_llm_model)
    llm_config = LLMConfig(
        model=model,
        provider=provider,
        temperature=temperature,
        reasoning_effort="low",
        max_tokens=max_tokens,
    )
    return LLMProvider(
        agent_name=None,
        memory_path=None,
        system_msg=None,   # system msg set per call below
        config=llm_config,
    )


def _call_llm(llm: LLMProvider, system: str, user: str) -> str:
    """Override the provider's system message and call it."""
    llm.sys_msg = system
    return llm(user, use_cache=False)


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_md_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    return text


def _repair_json(text: str) -> str:
    """Apply small lossless repairs to make malformed LLM JSON parseable.

    Currently: drop trailing commas before `}` or `]`. We deliberately do NOT
    try to escape interior quotes here — that requires real parsing and the
    regex fallback in `_parse_json_response` handles that case instead.
    """
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _extract_string_field(text: str, key: str) -> str | None:
    """Lenient extraction of a JSON string field, tolerating unescaped quotes.

    Uses a lazy match anchored on the next key (`"key": `) or the closing `}`,
    so interior `"` characters are preserved instead of prematurely closing
    the string.
    """
    pattern = (
        rf'"{re.escape(key)}"\s*:\s*"(.*?)"\s*'
        rf'(?=,\s*"[A-Za-z_][A-Za-z0-9_]*"\s*:|\}})'
    )
    m = re.search(pattern, text, flags=re.DOTALL)
    return m.group(1) if m else None


def _extract_bool_field(text: str, key: str) -> bool | None:
    m = re.search(
        rf'"{re.escape(key)}"\s*:\s*(true|false)\b',
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).lower() == "true"


def _extract_number_field(text: str, key: str) -> float | None:
    m = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    return float(m.group(1)) if m else None


def _extract_known_keys(text: str, expected_keys: dict[str, str]) -> dict | None:
    """Pull out a known set of top-level keys via regex.

    Last-ditch fallback for when ``json.loads`` cannot parse the LLM response
    (typically because of unescaped quotes deep inside a long string value).

    Args:
        text: Raw JSON-ish text.
        expected_keys: Mapping of ``key -> "str" | "bool" | "number"``.

    Returns:
        Dict with whatever keys were successfully extracted, or None if nothing
        could be pulled out.
    """
    out: dict = {}
    for key, kind in expected_keys.items():
        if kind == "str":
            val = _extract_string_field(text, key)
        elif kind == "bool":
            val = _extract_bool_field(text, key)
        elif kind == "number":
            val = _extract_number_field(text, key)
        else:
            val = None
        if val is not None:
            out[key] = val
    return out or None


def _parse_json_response(
    raw: str,
    expected_keys: dict[str, str] | None = None,
) -> dict:
    """Robustly parse JSON from an LLM response.

    Stages:
      1. Strip markdown fences.
      2. ``json.loads`` directly.
      3. Substring between the first '{' and last '}', then ``json.loads``.
      4. Light repair (trailing-comma removal), then ``json.loads``.
      5. Regex extraction of known keys (only when ``expected_keys`` provided).

    The regex stage is what saves us from unescaped quotes inside long string
    values — the classic OpenRouter-quantized-provider failure mode.
    """
    text = _strip_md_fences(raw)
    if not text:
        raise ValueError("Empty LLM response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    snippet = text[start : end + 1] if 0 <= start < end else text

    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        pass

    repaired = _repair_json(snippet)
    last_err: Exception
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        last_err = exc

    if expected_keys:
        extracted = _extract_known_keys(snippet, expected_keys)
        if extracted:
            return extracted

    raise ValueError(f"Could not parse JSON ({last_err}).")


def _call_llm_json(
    llm: LLMProvider,
    system: str,
    user: str,
    expected_keys: dict[str, str] | None = None,
    retries: int = 2,
) -> dict:
    """Call the LLM expecting JSON; retry with self-correction on parse failure.

    On each parse failure the user prompt is augmented with the parser error
    plus an explicit reminder to escape interior quotes, newlines and
    backslashes. This recovers the common case where the LLM emitted nearly
    valid JSON but missed an escape.
    """
    last_err: Exception | None = None
    current_user = user
    for _ in range(retries + 1):
        raw = _call_llm(llm, system, current_user)
        try:
            return _parse_json_response(raw, expected_keys=expected_keys)
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            current_user = (
                f"{user}\n\n"
                f"IMPORTANT: your previous response could not be parsed as JSON "
                f"(parser error: {exc}). Return ONLY a single valid JSON object "
                f"— no prose, no markdown fences. Escape every interior "
                f'double-quote as \\", every newline as \\n, and every '
                f"backslash as \\\\."
            )
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_CLARIFIER_SYSTEM = """\
You are an expert scientific research assistant helping users formulate \
their research objectives clearly for the Mimosa-AI autonomous research framework.

Given the user's current objective (and any additional context they provided), decide:
1. Is the objective sufficiently clear and actionable for an AI to execute autonomously?
2. If NOT clear: identify the single most important missing piece of information and \
   formulate one concise clarifying question.
3. If CLEAR: produce a polished, detailed, self-contained restatement that an AI agent \
   can act on directly (include dataset names, metrics, file paths, or any specifics \
   already mentioned).

Return ONLY valid JSON (no markdown fences, no prose before or after) in this exact shape:
{
  "is_clear": true | false,
  "question": "<single clarifying question, or empty string if clear>",
  "refined_prompt": "<actionable restatement of the full objective, or empty string if not yet clear>"
}

JSON formatting rules (CRITICAL — failure to follow these breaks downstream parsing):
- Escape EVERY interior double-quote as \\".  e.g. write  She said \\"hi\\"  not  She said "hi".
- Escape every newline inside a string as \\n.
- Escape every backslash as \\\\.
- No trailing commas, no comments, no markdown fences.
- The entire response must be a single JSON object — nothing else.

Content rules:
- Ask at most ONE question per turn.
- Only mark is_clear=true when you have enough detail to write a rich refined_prompt.
- The refined_prompt must incorporate ALL context provided so far.
- Do not ask for information that is not strictly necessary for execution.
"""

_CLASSIFIER_SYSTEM = """\
You are an expert assistant for the Mimosa-AI scientific research framework.
Your job is to classify a user's research objective into one of two execution modes.

MODES:
• "task" — A single, focused, self-contained operation that does NOT require multi-step
  planning.  Examples: training a model on one dataset, running a literature review,
  producing a specific figure, downloading/processing a file.

• "goal" — A high-level, multi-step scientific objective that benefits from autonomous
  decomposition into sub-tasks before execution.  Examples: reproducing a full paper,
  building an end-to-end ML pipeline from scratch, running a complete bioinformatics
  analysis across several steps.

Return ONLY valid JSON (no markdown fences, no prose) in this exact shape:
{
  "mode": "task" | "goal",
  "confidence": 0.0-1.0,
  "reasoning": "<one sentence>",
  "suggested_label": "<a short label of 8 words or fewer for the objective>"
}

JSON formatting rules:
- Escape interior quotes as \\", newlines as \\n, backslashes as \\\\.
- No trailing commas, no comments, no fences.
- The entire response must be a single JSON object — nothing else.
"""


# ---------------------------------------------------------------------------
# Main onboarding class
# ---------------------------------------------------------------------------

ModeType = Literal["task", "goal"]


class OnboardCLI:
    """Interactive setup wizard that guides the user through Mimosa-AI setup."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._objective: str = ""
        self._mode: ModeType = "task"
        self._learn: bool = False

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the full onboarding flow, then launch the selected mode."""
        print(MIMOSA_BANNER)
        print(_wrap(
            "Welcome to Mimosa-AI!"
            "Press Ctrl-C at any time to quit.",
            width=70, indent=2,
        ))

        # Step 1 – API keys
        _print_step(1, TOTAL_STEPS, "API Key Check")
        self._check_api_keys()

        # Step 2 – Config file
        _print_step(2, TOTAL_STEPS, "Configuration")
        self._load_config()

        # Step 3 – LLM model selection
        _print_step(3, TOTAL_STEPS, "LLM Model Selection")
        self._choose_models()

        # Step 4 – Toolomics / MCP connectivity (loops until online or skipped)
        _print_step(4, TOTAL_STEPS, "Toolomics MCP Connectivity")
        await self._check_toolomics()

        # Step 5 – Workspace file setup (select/clean files, optionally import)
        _print_step(5, TOTAL_STEPS, "Workspace Setup")
        self._setup_workspace_files()

        first_pass = True
        while True:
            # Infine loop for conversation to continue
            # Step 6 – Initial objective
            step_6_text = "Your Research Objective" if first_pass else "Keep working on the same objective"
            _print_step(6, TOTAL_STEPS, "Your Research Objective", no_count=not first_pass)
            self._collect_objective()

            # Step 7 – LLM clarification + prompt refinement loop
            _print_step(7, TOTAL_STEPS, "Objective Clarification & Refinement", no_count=not first_pass)
            self._clarify_and_refine()

            # Step 8 – Mode classification
            _print_step(8, TOTAL_STEPS, "Mode Selection (Goal vs Task)", no_count=not first_pass)
            self._classify_and_confirm()

            # Step 9 – Extra options then launch
            _print_step(9, TOTAL_STEPS, "Options & Launch", no_count=not first_pass)
            self._collect_options()

            await self._launch()
            first_pass = False

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    _KNOWN_API_KEYS = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "HF_TOKEN",
        "OPENROUTER_API_KEY",
    ]

    def _check_api_keys(self) -> None:
        """Check for known LLM API keys; ask which providers the user has."""
        found = [k for k in self._KNOWN_API_KEYS if os.getenv(k)]
        if found:
            for k in found:
                _ok(f"Found {k}")
            return

        _warn("No LLM API key found in environment.")
        selected = self._select_api_key_names()
        if not selected:
            _warn("Continuing without API keys — only locally served models will work.")
            return
        entered = self._prompt_key_values(selected)
        if entered:
            self._offer_env_file_save(entered)
        else:
            _warn("No key entered — only locally served models will work.")

    def _select_api_key_names(self) -> list[str]:
        """Show the supported providers and return the key names the user has."""
        print(_wrap(
            "Mimosa needs at least one API key to call a hosted LLM. "
            "Which of these do you have?",
            width=70, indent=2,
        ))
        print()
        for idx, key in enumerate(self._KNOWN_API_KEYS, start=1):
            print(f"    {CYAN}[{idx}]{RESET}  {key}")
        print()
        while True:
            choice = _ask("Your keys (e.g. 1,3 — or 'none' for local models only)")
            choice = choice.strip().lower()
            if choice in ("", "none") and _ask_yn(
                "Continue without any API key (local models only)?", default=False,
            ):
                return []
            indices, _ = _parse_indices(choice, len(self._KNOWN_API_KEYS))
            if indices:
                return [self._KNOWN_API_KEYS[i - 1] for i in sorted(indices)]
            _warn("No valid selection — type numbers like '1,3', or 'none'.")

    def _prompt_key_values(self, key_names: list[str]) -> dict[str, str]:
        """Prompt for each selected key's value and export it for this session."""
        entered: dict[str, str] = {}
        for key in key_names:
            value = _ask(f"Enter {key} (leave blank to skip)")
            if value:
                os.environ[key] = value
                entered[key] = value
                _ok(f"{key} set for this session.")
        return entered

    def _offer_env_file_save(self, entered: dict[str, str]) -> None:
        """Offer to persist entered keys to the user env file for future runs."""
        env_file = paths.user_env_file()
        if not _ask_yn(f"Save key(s) to {env_file} for future runs?", default=True):
            return
        try:
            _upsert_env_file(env_file, entered)
            _ok(f"Saved {len(entered)} key(s) to {env_file}")
        except OSError as exc:
            _warn(f"Could not save keys: {exc}")

    def _load_config(self) -> None:
        """Optionally load a JSON config file.

        When the user leaves the path blank, *config_default.json* is loaded
        automatically (if it exists) so that any previously saved settings —
        including the Toolomics workspace_dir — are picked up without asking.
        """
        _info(
            "A config file lets you override LLM models, workspace paths, "
            "port ranges, etc. (see config_default.json for reference)."
        )
        path = _ask(
            f"Path to config file (leave blank to auto-load {self._CONFIG_DEFAULT_PATH})"
        )
        loaded = False
        if path:
            if not os.path.isfile(path):
                _warn(f"File not found: {path}. Using default configuration.")
            else:
                try:
                    self.config.load(path)
                    _ok(f"Configuration loaded from {path}")
                    loaded = True
                except Exception as exc:
                    _warn(f"Failed to load config ({exc}). Using defaults.")
        else:
            # Auto-load config_default.json if it exists
            if os.path.isfile(self._CONFIG_DEFAULT_PATH):
                try:
                    self.config.load(self._CONFIG_DEFAULT_PATH)
                    _ok(f"Loaded {self._CONFIG_DEFAULT_PATH} (workspace: {self.config.workspace_dir})")
                    loaded = True
                except Exception as exc:
                    _warn(f"Failed to load {self._CONFIG_DEFAULT_PATH} ({exc}). Using built-in defaults.")
            else:
                _info("Using built-in default configuration.")

        if not loaded:
            _warn(
                f"{self._CONFIG_DEFAULT_PATH} could not be loaded. "
                "Using the built-in default configuration from config.py."
            )
            self._dump_full_config_default()

        # Ensure internal directories exist before later steps need them
        self.config.create_paths()

    def _dump_full_config_default(self) -> None:
        """Write the full current configuration to *config_default.json*."""
        try:
            self.config.dump(self._CONFIG_DEFAULT_PATH)
            _ok(f"Saved full default configuration to {self._CONFIG_DEFAULT_PATH}")
        except Exception as exc:
            _warn(f"Could not save default configuration to {self._CONFIG_DEFAULT_PATH}: {exc}")

    async def _check_toolomics(self) -> None:
        """Discover MCP servers; loop until at least one is found or user skips."""
        print(_wrap(
            "Mimosa requires Toolomics (the companion MCP server) to be running "
            "before execution. Scanning your configured discovery addresses …",
            width=70, indent=2,
        ))

        tool_manager = ToolManager(config=self.config)

        while True:
            mcps = await self._discover_once(tool_manager)

            if mcps:
                tool_manager.mcps = mcps
                for mcp in mcps:
                    _ok(f"MCP server online: {mcp}")
                bash_ok = await tool_manager.verify_tools()
                if not bash_ok:
                    _warn(
                        "No 'execute_command' tool found.\n"
                        "Make sure the shell MCP is deployed in Toolomics.\n"
                        "Retrying soon..."
                    )
                    time.sleep(15)
                    continue
                else:
                    _ok("Shell tool (execute_command) is available.")

                # ── Workspace directory check ──────────────────────────
                self._verify_workspace_dir()
                return   # ← success, exit loop

            # No MCPs found — ask the user what to do
            _err("No MCP/Toolomics servers found.")
            print(_wrap(
                "Please start Toolomics on the configured port range "
                f"({self.config.discovery_addresses}).",
                width=70, indent=2,
            ))
            print(f"\n  {BOLD}Options:{RESET}")
            print(f"    {CYAN}Enter{RESET}   – retry scan")
            print(f"    {CYAN}skip{RESET}    – continue without Toolomics "
                  f"(execution will fail later)")
            choice = _ask("Retry or skip?").lower()
            if choice == "skip":
                _warn("Skipping Toolomics check. Execution may fail at runtime.")
                return
            # Any other input (including blank/Enter) → retry

    async def _discover_once(self, tool_manager: ToolManager) -> list:
        """Run a single MCP discovery pass, returning the list (may be empty)."""
        try:
            return await tool_manager.discover_mcp_servers()
        except Exception as exc:
            _warn(f"Discovery error: {exc}")
            return []

    _CONFIG_DEFAULT_PATH = _persisted_config_path()

    def _verify_workspace_dir(self) -> None:
        """Check that config.workspace_dir exists; prompt the user until it does.

        When the user supplies a valid path it is written back to
        *config_default.json* so that subsequent runs don't ask again.
        """
        while True:
            workspace = self.config.workspace_dir
            if os.path.isdir(workspace):
                _ok(f"Workspace directory found: {workspace}")
                self._warn_if_workspace_not_in_toolomics(workspace)
                return

            _err(f"Workspace directory not found: {workspace}")
            print(_wrap(
                "This path must point to the Toolomics workspace folder — the shared "
                "directory where Mimosa reads and writes task artifacts. "
                "Please enter the correct absolute path, or press Enter to skip.",
                width=70, indent=2,
            ))
            new_path = _ask("Workspace directory path (Enter to skip)")
            if not new_path:
                _warn(
                    "Skipping workspace check. "
                    "Execution will fail unless workspace_dir is set correctly."
                )
                return
            new_path = os.path.expanduser(new_path.strip())
            if os.path.isdir(new_path):
                self.config.workspace_dir = new_path
                _ok(f"Workspace directory set to: {new_path}")
                self._persist_workspace_dir(new_path)
                self._warn_if_workspace_not_in_toolomics(new_path)
                return
            _err(f"Directory does not exist: {new_path}. Please try again.")

    def _warn_if_workspace_not_in_toolomics(self, path: str) -> None:
        """Warn when the workspace path is not inside a Toolomics directory."""
        if "toolomics" not in os.path.abspath(path).lower():
            _warn(
                "The workspace path does not contain a 'toolomics' directory. "
                "If you wish to use Toolomics, execution will fail; "
                "otherwise, if you are bringing your own MCP, ensure they are "
                "configured to the same path as the file mount."
            )

    def _persist_workspace_dir(self, path: str) -> None:
        """Write *path* as workspace_dir into config_default.json.

        The full configuration is written so that other settings are not lost.
        """
        try:
            self.config.dump(self._CONFIG_DEFAULT_PATH)
            _ok(f"Saved workspace_dir to {self._CONFIG_DEFAULT_PATH}")
        except Exception as exc:
            _warn(f"Could not persist workspace path to {self._CONFIG_DEFAULT_PATH}: {exc}")

    # ------------------------------------------------------------------
    # Workspace file setup
    # ------------------------------------------------------------------

    def _setup_workspace_files(self) -> None:
        """Step 5 – Let the user curate workspace contents before execution.

        • If the workspace contains files, list them and let the user choose
          which ones to keep (the rest are deleted), or delete everything.
        • If the workspace is empty (or became empty after cleanup), offer to
          copy files from a user-supplied source directory.
        """
        import shutil
        from pathlib import Path

        workspace = self.config.workspace_dir
        if not os.path.isdir(workspace):
            _warn(f"Workspace directory not found ({workspace}). Skipping setup.")
            return

        # ── List current workspace files ──────────────────────────────
        raw_listing = list_files(path=workspace, max_depth=2)
        file_list: list[str] = [
            f for f in raw_listing.splitlines() if f.strip()
        ]

        workspace_was_empty = len(file_list) == 0
        kept_files: list[str] = []

        if file_list:
            print(_wrap(
                f"The workspace ({workspace}) currently contains "
                f"{len(file_list)} file(s):",
                width=70, indent=2,
            ))
            print()

            # Show numbered file list
            for idx, fname in enumerate(file_list, start=1):
                print(f"    {CYAN}[{idx}]{RESET}  {fname}")

            print()
            print(f"  {BOLD}Options:{RESET}")
            print(f"    {CYAN}Enter numbers{RESET}  – comma-separated list of files to "
                  f"{GREEN}keep{RESET} (others will be deleted)")
            print(f"    {CYAN}all{RESET}           – keep all files")
            print(f"    {CYAN}none{RESET}          – {RED}delete all{RESET} files in the workspace")
            print()

            while True:
                choice = _ask("Files to keep").strip().lower()

                if choice == "all":
                    kept_files = list(file_list)
                    _ok(f"Keeping all {len(kept_files)} file(s).")
                    break
                if choice == "none":
                    # Delete everything (explicit user choice)
                    kept_files = []
                    if not _ask_yn(
                        "Confirm: delete ALL files in the workspace?",
                        default=False,
                    ):
                        _info("Aborted — please choose again.")
                        continue
                    break
                if choice == "":
                    _warn("Empty input — please type numbers, 'all', or 'none'.")
                    continue

                # Parse comma-separated indices
                selected_indices: set[int] = set()
                had_bad_token = False
                for part in choice.replace(" ", "").split(","):
                    if not part:
                        continue
                    # Support ranges like "1-5"
                    if "-" in part:
                        bounds = part.split("-", 1)
                        try:
                            lo, hi = int(bounds[0]), int(bounds[1])
                            selected_indices.update(range(lo, hi + 1))
                        except ValueError:
                            _warn(f"Invalid range: {part}")
                            had_bad_token = True
                    else:
                        try:
                            selected_indices.add(int(part))
                        except ValueError:
                            _warn(f"Invalid number: {part}")
                            had_bad_token = True

                kept_files = [
                    file_list[idx - 1]
                    for idx in sorted(selected_indices)
                    if 1 <= idx <= len(file_list)
                ]

                if kept_files:
                    _ok(f"Keeping {len(kept_files)} file(s).")
                    break
                # No valid indices — re-ask instead of silently deleting all.
                if had_bad_token:
                    _warn(
                        "No valid file numbers recognised. "
                        "Please type indices like '1,3,5' or '1-5', "
                        "or 'all' / 'none'."
                    )
                else:
                    _warn(
                        "No files selected. Type 'none' explicitly if you "
                        "want to delete everything."
                    )

            # ── Perform deletion of un-kept files ─────────────────────
            if kept_files and len(kept_files) < len(file_list):
                kept_set = set(kept_files)
                deleted = 0
                for fname in file_list:
                    if fname not in kept_set:
                        full_path = os.path.join(workspace, fname)
                        try:
                            if os.path.isfile(full_path):
                                os.remove(full_path)
                                deleted += 1
                            elif os.path.isdir(full_path):
                                shutil.rmtree(full_path)
                                deleted += 1
                        except OSError as exc:
                            _warn(f"Could not delete {fname}: {exc}")
                # Clean up empty parent directories left behind
                self._prune_empty_dirs(workspace)
                if deleted:
                    _ok(f"Deleted {deleted} file(s) from workspace.")
            elif not kept_files and file_list:
                # Delete everything
                for item in Path(workspace).iterdir():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    except OSError as exc:
                        _warn(f"Could not delete {item.name}: {exc}")
                _ok("All workspace files deleted.")

        # ── Offer to import files if workspace is (now) empty ─────────
        # Re-check after potential deletions
        fresh_listing = list_files(path=workspace, max_depth=2)
        remaining = [f for f in fresh_listing.splitlines() if f.strip()]
        workspace_is_empty = len(remaining) == 0

        if workspace_is_empty:
            if workspace_was_empty:
                _info("Workspace is empty.")
            print()
            want_import = _ask_yn(
                "Copy files from a source directory into the workspace?",
                default=False,
            )
            if want_import:
                self._import_files_to_workspace()
            else:
                _info("Workspace will remain empty — agents can create files at runtime.")

    def _import_files_to_workspace(self) -> None:
        """Copy one or more source directories into the workspace
        using ``LocalTransfer.transfer_files_to_workspace``.
        """
        sources = self._select_import_sources()
        if not sources:
            _info("No source selected — skipping import.")
            return
        transfer = LocalTransfer(
            config=self.config,
            workspace_path=self.config.workspace_dir,
            runs_capsule_dir=self.config.runs_capsule_dir,
        )
        for src in sources:
            try:
                copied = transfer.transfer_files_to_workspace(src)
                _ok(f"Copied {copied} file(s) from {src} into workspace.")
            except Exception as exc:
                _err(f"File transfer failed for {src}: {exc}")

    def _select_import_sources(self) -> list[str]:
        """List folders under the current directory and/or accept a typed path.

        Returns:
            Absolute source directory paths to import ([] to skip).
        """
        cwd = os.getcwd()
        folders = _list_subdirectories(cwd)
        if folders:
            print(_wrap(f"Folders in {cwd}:", width=70, indent=2))
            print()
            for idx, name in enumerate(folders, start=1):
                print(f"    {CYAN}[{idx}]{RESET}  {name}/")
            print()
            _info("Type numbers like '1,3', a directory path, or Enter to skip.")
        while True:
            choice = _ask("Folder(s) to import")
            if not choice:
                return []
            indices, _ = _parse_indices(choice.strip().lower(), len(folders))
            if indices:
                return [os.path.join(cwd, folders[i - 1]) for i in sorted(indices)]
            path = os.path.abspath(os.path.expanduser(choice.strip()))
            if os.path.isdir(path):
                return [path]
            _err(f"Directory not found: {choice}. Type numbers, a valid path, or Enter to skip.")

    @staticmethod
    def _prune_empty_dirs(root: str) -> None:
        """Remove empty sub-directories under *root* (bottom-up)."""
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            if dirpath == root:
                continue
            if not filenames and not dirnames:
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Model selection helpers
    # ------------------------------------------------------------------

    def _model_menu(
        self,
        prompt_desc: str,
        current_value: str,
        available: list[tuple[str, str]],
    ) -> str:
        """Generic numbered model-selection menu.

        Args:
            prompt_desc: One-line description shown to the user (what this model
                         controls).
            current_value: Value already in config (may be empty string).
            available: List of (display_label, litellm_model_id) for presets whose
                       API key is present.

        Returns:
            The chosen model ID (may equal *current_value* if the user just
            pressed Enter).
        """
        suggested = current_value or (available[0][1] if available else "")

        if current_value:
            _info(f"Current value (from config): {current_value}")

        print(_wrap(prompt_desc, width=70, indent=2))
        print()

        if available:
            print(f"  {BOLD}Available presets:{RESET}")
            for idx, (label, model_id) in enumerate(available, start=1):
                is_default = (model_id == suggested)
                tag = f"{GREEN}← default{RESET}" if is_default else ""
                num_color = GREEN if is_default else CYAN
                print(f"  {num_color}[{idx}]{RESET}  {label}  {tag}")
                print(f"         {DIM}{model_id}{RESET}")
            print(f"  {CYAN}[c]{RESET}  Enter a custom model ID")
        else:
            _warn("No matching API key found — enter a model ID manually.")

        print()
        while True:
            if suggested:
                choice = _ask(
                    "Select number, 'c' for custom, or Enter to keep current",
                    default="",
                )
            else:
                choice = _ask("Select number or 'c' for custom")

            if not choice and suggested:
                return suggested
            if choice.lower() == "c" or (not available):
                custom = _ask(
                    "Enter model ID  (e.g. openai/gpt-4o, "
                    "anthropic/claude-3-5-sonnet-20241022)"
                ).strip()
                if custom:
                    return custom
                if suggested:
                    _warn("No model ID entered — keeping current.")
                    return suggested
                _warn("No model ID entered — please try again.")
                continue
            # Numbered selection
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    return available[idx][1]
                _warn(
                    f"Number out of range: {choice}. "
                    f"Please pick 1-{len(available)} or 'c'."
                )
                continue
            except ValueError:
                _warn(
                    f"Unrecognised input '{choice}'. "
                    "Please pick a number, 'c', or press Enter to keep current."
                )
                continue

    def _choose_models(self) -> None:
        """Step 3 – model selection.

        Sub-step 3a: orchestration model (planner, prompts, workflow, judge).
        Sub-step 3b: agent execution model (smolagent_model_id).

        Both choices are persisted to *config_default.json*.
        """
        available: list[tuple[str, str]] = [
            (label, model_id)
            for env_key, label, model_id in _MODEL_PRESETS
            if os.getenv(env_key)
        ]

        # ── 3a · Orchestration model ──────────────────────────────────
        print(f"\n{BOLD}  3a · Orchestration model{RESET}")
        print(f"  {DIM}Used for planning, workflow generation, and evaluation.{RESET}")
        orch_model = self._model_menu(
            prompt_desc=(
                "Choose the main LLM Mimosa will use for orchestration "
                "(planning, workflow generation, and evaluation). Applied to "
                "planner, prompts, workflow, and judge roles."
            ),
            current_value=self.config.planner_llm_model or "",
            available=available,
        )

        if orch_model:
            for key in _MODEL_CFG_KEYS:
                setattr(self.config, key, orch_model)
            _ok(f"Orchestration model: {orch_model}")
        else:
            _warn("No orchestration model chosen — keeping existing config values.")

        # ── 3b · Agent execution model (smolagent_model_id) ──────────
        print(f"\n{BOLD}  3b · Agent execution model (SmolAgents){RESET}")
        print(f"  {DIM}Used by the code-executing agents inside each workflow.{RESET}")
        print(f"  {DIM}Can be the same as the orchestration model or a faster/cheaper one.{RESET}")
        agent_model = self._model_menu(
            prompt_desc=(
                "Choose the LLM for agent execution (SmolAgents tasks). "
                "A fast, cost-effective model works well here."
            ),
            current_value=self.config.smolagent_model_id or "",
            available=available,
        )

        if agent_model:
            self.config.smolagent_model_id = agent_model
            _ok(f"Agent execution model: {agent_model}")
        else:
            _warn("No agent model chosen — keeping existing config values.")

        # Persist both choices at once
        self._persist_models(orch_model or "", agent_model or "")

    def _persist_models(self, orch_model_id: str, agent_model_id: str) -> None:
        """Write both model choices to config_default.json.

        The full configuration is written so that other settings are not lost.
        """
        try:
            self.config.dump(self._CONFIG_DEFAULT_PATH)
            _ok(f"Saved model choices to {self._CONFIG_DEFAULT_PATH}")
        except Exception as exc:
            _warn(f"Could not persist model choices to {self._CONFIG_DEFAULT_PATH}: {exc}")

    def _collect_objective(self) -> None:
        """Prompt the user for their initial research objective."""
        print(_wrap(
            "Describe what you want Mimosa to do. This can be a high-level "
            "scientific goal (e.g. 'Reproduce Figure 3 from paper X') or a "
            "focused task (e.g. 'Train a toxicity model on the ClinTox dataset'). "
            "Don't worry about being too vague — we'll refine it together next.",
            width=70, indent=2,
        ))
        while True:
            objective = _ask("Your objective")
            if len(objective.strip()) >= 10:
                self._objective = objective.strip()
                break
            _warn("Please enter a more descriptive objective (at least 10 characters).")

    def _clarify_and_refine(self) -> None:
        """LLM conversation loop: clarify missing info, then refine the prompt."""
        print(_wrap(
            "The assistant will now check whether your objective is clear enough "
            "for Mimosa to execute and may ask one or more follow-up questions. "
            "Once complete, it will produce a refined, actionable prompt.",
            width=70, indent=2,
        ))

        # Lower temperature → more reliable JSON; bigger token budget so long
        # refined_prompts don't get truncated mid-string.
        llm = _build_llm(self.config, temperature=0.1, max_tokens=1024)
        expected_keys = {
            "is_clear": "bool",
            "question": "str",
            "refined_prompt": "str",
        }

        # Accumulate context: original objective + Q&A pairs
        context_lines: list[str] = [f"Objective: {self._objective}"]
        max_clarification_rounds = 5

        for round_num in range(max_clarification_rounds):
            full_context = "\n".join(context_lines)

            print(f"\n{DIM}  [Clarification round {round_num + 1}/{max_clarification_rounds}]{RESET}")

            try:
                result = _call_llm_json(
                    llm,
                    _CLARIFIER_SYSTEM,
                    full_context,
                    expected_keys=expected_keys,
                )
            except Exception as exc:
                _warn(
                    f"LLM clarification failed after retries ({exc}). "
                    "Falling back to manual refinement."
                )
                self._manual_refinement_fallback()
                return

            is_clear = bool(result.get("is_clear", False))
            question = str(result.get("question", "")).strip()
            refined_prompt = str(result.get("refined_prompt", "")).strip()

            if not is_clear and question:
                # Ask the clarifying question
                print()
                print(f"  {BOLD}Assistant:{RESET}  {question}")
                answer = _ask("Your answer (or 'skip' to stop clarifying)")
                if answer.lower() in ("skip", "stop", "done"):
                    _info("Stopping clarification — using current objective.")
                    return
                if answer:
                    context_lines.append(f"Q: {question}")
                    context_lines.append(f"A: {answer}")
                else:
                    _info("No answer provided — skipping this question.")
                continue   # loop for next round

            if is_clear and refined_prompt:
                # Show the refined prompt and ask for confirmation
                print()
                print(f"  {BOLD}Refined objective:{RESET}")
                print()
                # Print wrapped refined prompt with colour
                for line in textwrap.wrap(refined_prompt, width=64):
                    print(f"    {CYAN}{line}{RESET}")
                print()
                confirmed = _ask_yn("Accept this refined objective?", default=True)
                if confirmed:
                    self._objective = refined_prompt
                    _ok("Objective accepted.")
                    return
                else:
                    # Let the user correct it manually
                    correction = _ask(
                        "Edit the objective (or press Enter to keep the original)"
                    )
                    if correction:
                        self._objective = correction.strip()
                        context_lines = [f"Objective: {self._objective}"]
                    _ok(f"Continuing with: {self._objective[:80]}")
                    return

            # Neither a usable question nor a refined prompt — nudge the LLM
            # and retry within the same loop round budget.
            _info("LLM returned an incomplete response — retrying.")
            context_lines.append(
                "(Reminder: respond with valid JSON containing either a "
                "clarifying question OR a refined_prompt — never both empty.)"
            )

        # Exhausted rounds without clarity — keep whatever we have
        _warn(
            f"Clarification loop completed ({max_clarification_rounds} rounds). "
            "Using current objective as-is."
        )

    def _manual_refinement_fallback(self) -> None:
        """Offer the user a way to refine the objective by hand when the LLM
        clarification round cannot recover.
        """
        print()
        _info("You can refine your objective manually below.")
        edited = _ask(
            "Edit your objective (press Enter to keep it as-is)"
        )
        if edited:
            self._objective = edited.strip()
            _ok(f"Objective updated: {self._objective[:80]}")
        else:
            _info("Keeping objective as-is.")

    def _classify_and_confirm(self) -> None:
        """Use LLM to classify objective as goal or task, confirm with user."""
        print(_wrap(
            "Asking the LLM to classify your objective as Goal-mode "
            "(multi-step planning) or Task-mode (single focused operation) …",
            width=70, indent=2,
        ))

        classification: dict | None = None
        llm = _build_llm(self.config, temperature=0.0, max_tokens=384)
        expected_keys = {
            "mode": "str",
            "confidence": "number",
            "reasoning": "str",
            "suggested_label": "str",
        }

        try:
            classification = _call_llm_json(
                llm,
                _CLASSIFIER_SYSTEM,
                f"Classify this research objective:\n\n{self._objective}",
                expected_keys=expected_keys,
            )
        except Exception as exc:
            _warn(
                f"LLM classification failed after retries ({exc}). "
                "Falling back to manual selection."
            )

        if classification:
            mode_raw = str(classification.get("mode", "")).lower().strip()
            if mode_raw not in ("task", "goal"):
                _warn(
                    f"LLM returned an unexpected mode '{mode_raw}'. "
                    "Falling back to manual selection."
                )
                classification = None

        if classification:
            mode       = mode_raw  # validated above
            try:
                confidence = float(classification.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            reasoning  = str(classification.get("reasoning", ""))
            label      = str(
                classification.get("suggested_label", self._objective[:40])
            )

            print()
            print(f"  {BOLD}Suggested mode:{RESET}  {CYAN}{mode.upper()}{RESET}  "
                  f"(confidence: {confidence:.0%})")
            print(f"  {BOLD}Reasoning:{RESET}      {reasoning}")
            print(f"  {BOLD}Label:{RESET}          {label}")
            print()
            _info(
                "Goal mode  → Mimosa decomposes the objective into a plan of tasks "
                "and executes them sequentially (planner).\n"
                "  ℹ️    Task mode  → Mimosa directly synthesises and runs a single "
                "multi-agent workflow for the objective (evolution engine)."
            )

            confirmed = _ask_yn(f"Accept '{mode}' mode?", default=True)
            if confirmed:
                self._mode = mode  # type: ignore[assignment]
                return

        # Manual fallback / override — loop until the user picks a valid mode.
        print()
        print(f"  {BOLD}Available modes:{RESET}")
        print(f"    {CYAN}goal{RESET}  – high-level research objective (planner + evolution engine)")
        print(f"    {CYAN}task{RESET}  – single focused operation (evolution engine only)")
        while True:
            choice = _ask("Choose mode (goal/task)", default="task").lower().strip()
            if choice in ("goal", "g"):
                self._mode = "goal"
                break
            if choice in ("task", "t"):
                self._mode = "task"
                break
            _warn(f"Unrecognised choice '{choice}'. Please type 'goal' or 'task'.")
        _ok(f"Mode set to: {self._mode.upper()}")

    def _collect_options(self) -> None:
        """Ask about learning mode and other options."""
        print(_wrap(
            "Learning mode enables Mimosa to iteratively improve its workflow "
            "through Darwinian self-evolution until a quality threshold is met "
            "(recommended for first-time runs on a new objective).",
            width=70, indent=2,
        ))
        self._learn = _ask_yn("Enable learning mode?", default=False)
        if self._learn:
            _ok("Learning mode enabled.")
        else:
            _info("Learning mode disabled (single-pass execution).")

        # ASTRA export — opt-in, off by default. Adds an LLM decision-
        # extraction pass over the best run's memory trace at the end.
        print()
        print(_wrap(
            "ASTRA export writes a standards-compliant YAML "
            "(https://astra-spec.org) describing the scientific decisions "
            "the best run made — useful for audit and reproducibility.",
            width=70, indent=2,
        ))
        _warn("This adds an extra LLM pass after evolution and takes a bit more time.")
        self.config.export_astra = _ask_yn("Save the best run as ASTRA?", default=False)
        if self.config.export_astra:
            _ok("ASTRA export enabled.")
        else:
            _info("ASTRA export disabled.")

        # Summary
        print()
        print(f"  {BOLD}{'─'*54}{RESET}")
        print(f"  {BOLD}LAUNCH SUMMARY{RESET}")
        print(f"  {'─'*54}")
        print(f"  Mode:      {CYAN}{self._mode.upper()}{RESET}")
        print(f"  Learning:  {'Yes' if self._learn else 'No'}")
        print(f"  ASTRA:     {'Yes' if self.config.export_astra else 'No'}")
        print(f"  Objective: {self._objective[:60]}{'…' if len(self._objective) > 60 else ''}")
        print(f"  {'─'*54}")
        print()
        go = _ask_yn("Launch Mimosa now?", default=True)
        if not go:
            print("\n  Exiting without launching. Run again when ready.\n")
            sys.exit(0)

    async def _launch(self) -> None:
        """Validate config paths and start the selected execution mode."""
        try:
            self.config.validate_paths()
        except AssertionError as exc:
            _err(f"Configuration validation failed: {exc}")
            _info(
                "Check that your workspace_dir and other paths in config are correct. "
                "Make sure Toolomics is running and the workspace exists and try again."
            )
            sys.exit(1)

        print(MIMOSA_START_BANNER)

        if self._mode == "goal":
            await self._launch_goal()
        else:
            await self._launch_task()
        # Archive workspace after completion
        trs = LocalTransfer(
            config=self.config,
            workspace_path=self.config.workspace_dir,
            runs_capsule_dir=self.config.runs_capsule_dir,
        )
        capsule = trs.transfer_workspace_files_to_capsule(self._objective)
        print(f"\n{GREEN}{BOLD}  Workspace files archived to capsule: {capsule}{RESET}\n")

    async def _launch_goal(self) -> None:
        """Start planner mode (multi-step goal)."""
        from sources.core.planner import Planner

        print(f"\n{GREEN}{BOLD}  Launching in GOAL mode …{RESET}\n")
        planner = Planner(self.config)
        await planner.start_planner(
            goal=self._objective,
            judge=True,
        )

    async def _launch_task(self) -> None:
        """Start evolution engine task mode (single operation)."""
        from sources.core.evolution_engine import EvolutionEngine

        print(f"\n{GREEN}{BOLD}  Launching in TASK mode …{RESET}\n")
        evolve = EvolutionEngine(self.config)
        await evolve.start_workflow_evolution(
            goal=self._objective,
            judge=True,
            enable_evolution=self._learn,
        )
