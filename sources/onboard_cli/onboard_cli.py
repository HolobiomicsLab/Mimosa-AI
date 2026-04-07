"""
Interactive onboarding CLI for Mimosa-AI.

Guides new users through setup step-by-step (Claude-code style),
checks that Toolomics is online, clarifies and refines the user's
objective using an LLM conversation loop, classifies it as Goal-mode
or Task-mode, then hands off to the appropriate execution entry-point
(planner.start_planner or dgm.start_dgm).
"""

from __future__ import annotations

import asyncio
import json
import time
import os
import sys
import textwrap
from typing import Literal

from config import Config
from sources.core.llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from sources.core.tools_manager import ToolManager


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

TOTAL_STEPS = 8

# ---------------------------------------------------------------------------
# Model presets — ordered by quality/preference
# (env_key, display_label, litellm_model_id)
# ---------------------------------------------------------------------------
_MODEL_PRESETS: list[tuple[str, str, str]] = [
    ("ANTHROPIC_API_KEY",  "Claude Sonnet 4.5  (Anthropic)",  "anthropic/claude-sonnet-4-5"),
    ("DEEPSEEK_API_KEY",   "DeepSeek Chat      (DeepSeek)",   "deepseek/deepseek-chat"),
    ("OPENROUTER_API_KEY", "GLM-5 via OpenRouter (z-ai)",     "openrouter/z-ai/glm-5"),
    ("OPENAI_API_KEY",     "GPT-4o             (OpenAI)",     "openai/gpt-4o"),
    ("MISTRAL_API_KEY",    "Mistral Large      (Mistral)",    "mistral/mistral-large-latest"),
]
# Config keys that all share the same "main" LLM selection
_MODEL_CFG_KEYS = [
    "planner_llm_model",
    "prompts_llm_model",
    "workflow_llm_model",
    "judge_model",
]


def _print_step(step: int, total: int, title: str) -> None:
    bar = "─" * 60
    print(f"\n{CYAN}{bar}{RESET}")
    print(f"{CYAN}  Step {step}/{total}  ·  {title}{RESET}")
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


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    return json.loads(raw)


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

Return ONLY valid JSON (no markdown fences) in this exact shape:
{
  "is_clear": true | false,
  "question": "<single clarifying question, or empty string if clear>",
  "refined_prompt": "<actionable restatement of the full objective, or empty string if not yet clear>"
}

Rules:
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
  analysis across several tools.

Return ONLY valid JSON (no markdown fences) like:
{
  "mode": "task" | "goal",
  "confidence": 0.0-1.0,
  "reasoning": "<one sentence>",
  "suggested_label": "<a short ≤ 8-word label for the objective>"
}
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
            "Welcome! This interactive guide will walk you through Mimosa setup "
            "and launch the right execution mode for your scientific objective. "
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

        # Step 5 – Initial objective
        _print_step(5, TOTAL_STEPS, "Your Research Objective")
        self._collect_objective()

        # Step 6 – LLM clarification + prompt refinement loop
        _print_step(6, TOTAL_STEPS, "Objective Clarification & Refinement")
        self._clarify_and_refine()

        # Step 7 – Mode classification
        _print_step(7, TOTAL_STEPS, "Mode Selection (Goal vs Task)")
        self._classify_and_confirm()

        # Step 8 – Extra options then launch
        _print_step(8, TOTAL_STEPS, "Options & Launch")
        self._collect_options()
        await self._launch()

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _check_api_keys(self) -> None:
        """Check for at least one known LLM API key in the environment."""
        known_keys = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "MISTRAL_API_KEY",
            "HF_TOKEN",
            "OPENROUTER_API_KEY",
        ]
        found = [k for k in known_keys if os.getenv(k)]

        if found:
            for k in found:
                _ok(f"Found {k}")
            return

        _warn("No LLM API key found in environment.")
        print(_wrap(
            "Mimosa needs at least one API key to call an LLM. "
            "Supported variables: " + ", ".join(known_keys),
            width=70, indent=2,
        ))
        print()
        for k in known_keys:
            value = _ask(f"Enter {k} (leave blank to skip)")
            if value:
                os.environ[k] = value
                _ok(f"{k} set for this session.")
                break
        else:
            _err("No API key provided. Mimosa cannot run without one.")
            sys.exit(1)

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
        if path:
            if not os.path.isfile(path):
                _warn(f"File not found: {path}. Using default configuration.")
            else:
                try:
                    self.config.load(path)
                    _ok(f"Configuration loaded from {path}")
                except Exception as exc:
                    _warn(f"Failed to load config ({exc}). Using defaults.")
        else:
            # Auto-load config_default.json if it exists
            if os.path.isfile(self._CONFIG_DEFAULT_PATH):
                try:
                    self.config.load(self._CONFIG_DEFAULT_PATH)
                    _ok(f"Loaded {self._CONFIG_DEFAULT_PATH} (workspace: {self.config.workspace_dir})")
                except Exception as exc:
                    _warn(f"Failed to load {self._CONFIG_DEFAULT_PATH} ({exc}). Using built-in defaults.")
            else:
                _info("Using built-in default configuration.")

        # Ensure internal directories exist before later steps need them
        self.config.create_paths()

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

    _CONFIG_DEFAULT_PATH = "config_default.json"

    def _verify_workspace_dir(self) -> None:
        """Check that config.workspace_dir exists; prompt the user until it does.

        When the user supplies a valid path it is written back to
        *config_default.json* so that subsequent runs don't ask again.
        """
        while True:
            workspace = self.config.workspace_dir
            if os.path.isdir(workspace):
                _ok(f"Workspace directory found: {workspace}")
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
                return
            _err(f"Directory does not exist: {new_path}. Please try again.")

    def _persist_workspace_dir(self, path: str) -> None:
        """Write *path* as workspace_dir into config_default.json."""
        cfg_path = self._CONFIG_DEFAULT_PATH
        try:
            # Read existing config (or start from empty dict)
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as fh:
                    data = json.load(fh)
            else:
                data = {}

            data["workspace_dir"] = path

            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
                fh.write("\n")

            _ok(f"Saved workspace_dir to {cfg_path}")
        except Exception as exc:
            _warn(f"Could not persist workspace path to {cfg_path}: {exc}")

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
            )
            return custom.strip() if custom.strip() else suggested
        # Numbered selection
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                return available[idx][1]
            _warn(f"Invalid selection '{choice}'. Using default.")
        except ValueError:
            _warn(f"Unrecognised input '{choice}'. Using default.")
        return suggested or (available[0][1] if available else "")

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
        """Write both model choices to config_default.json."""
        cfg_path = self._CONFIG_DEFAULT_PATH
        try:
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as fh:
                    data = json.load(fh)
            else:
                data = {}

            if orch_model_id:
                for key in _MODEL_CFG_KEYS:
                    data[key] = orch_model_id

            if agent_model_id:
                data["smolagent_model_id"] = agent_model_id

            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
                fh.write("\n")

            _ok(f"Saved model choices to {cfg_path}")
        except Exception as exc:
            _warn(f"Could not persist model choices to {cfg_path}: {exc}")

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

        llm = _build_llm(self.config, temperature=0.3, max_tokens=768)

        # Accumulate context: original objective + Q&A pairs
        context_lines: list[str] = [f"Objective: {self._objective}"]
        max_clarification_rounds = 5

        for round_num in range(max_clarification_rounds):
            full_context = "\n".join(context_lines)

            print(f"\n{DIM}  [Clarification round {round_num + 1}/{max_clarification_rounds}]{RESET}")

            try:
                raw = _call_llm(llm, _CLARIFIER_SYSTEM, full_context)
                result = _parse_json_response(raw)
            except Exception as exc:
                _warn(f"LLM clarification failed ({exc}). Skipping refinement.")
                return

            is_clear = result.get("is_clear", False)
            question = result.get("question", "").strip()
            refined_prompt = result.get("refined_prompt", "").strip()

            if not is_clear and question:
                # Ask the clarifying question
                print()
                print(f"  {BOLD}Assistant:{RESET}  {question}")
                answer = _ask("Your answer")
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

        # Exhausted rounds without clarity — keep whatever we have
        _warn(
            f"Clarification loop completed ({max_clarification_rounds} rounds). "
            "Using current objective as-is."
        )

    def _classify_and_confirm(self) -> None:
        """Use LLM to classify objective as goal or task, confirm with user."""
        print(_wrap(
            "Asking the LLM to classify your objective as Goal-mode "
            "(multi-step planning) or Task-mode (single focused operation) …",
            width=70, indent=2,
        ))

        classification: dict | None = None
        llm = _build_llm(self.config, temperature=0.0, max_tokens=256)

        try:
            raw = _call_llm(
                llm,
                _CLASSIFIER_SYSTEM,
                f"Classify this research objective:\n\n{self._objective}",
            )
            classification = _parse_json_response(raw)
        except Exception as exc:
            _warn(f"LLM classification failed ({exc}). Falling back to manual selection.")

        if classification:
            mode       = classification.get("mode", "task")
            confidence = float(classification.get("confidence", 0.0))
            reasoning  = classification.get("reasoning", "")
            label      = classification.get("suggested_label", self._objective[:40])

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
                "multi-agent workflow for the objective (DGM)."
            )

            confirmed = _ask_yn(f"Accept '{mode}' mode?", default=True)
            if confirmed:
                self._mode = mode  # type: ignore[assignment]
                return

        # Manual fallback / override
        print()
        print(f"  {BOLD}Available modes:{RESET}")
        print(f"    {CYAN}goal{RESET}  – high-level research objective (planner + DGM)")
        print(f"    {CYAN}task{RESET}  – single focused operation (DGM only)")
        choice = _ask("Choose mode", default="task").lower()
        self._mode = "goal" if choice.startswith("g") else "task"
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

        # Summary
        print()
        print(f"  {BOLD}{'─'*54}{RESET}")
        print(f"  {BOLD}LAUNCH SUMMARY{RESET}")
        print(f"  {'─'*54}")
        print(f"  Mode:      {CYAN}{self._mode.upper()}{RESET}")
        print(f"  Learning:  {'Yes' if self._learn else 'No'}")
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
                "Make sure Toolomics is running and the workspace exists."
            )
            sys.exit(1)

        if self._mode == "goal":
            await self._launch_goal()
        else:
            await self._launch_task()

    async def _launch_goal(self) -> None:
        """Start planner mode (multi-step goal)."""
        from sources.core.planner import Planner
        from sources.utils.transfer_toolomics import LocalTransfer

        print(f"\n{GREEN}{BOLD}  🚀  Launching in GOAL mode …{RESET}\n")
        planner = Planner(self.config)
        await planner.start_planner(
            goal=self._objective,
            judge=True,
            max_evolve_iteration=self.config.max_learning_evolve_iterations if self._learn else 1,
        )
        # Archive workspace after completion
        trs = LocalTransfer(
            config=self.config,
            workspace_path=self.config.workspace_dir,
            runs_capsule_dir=self.config.runs_capsule_dir,
        )
        trs.transfer_workspace_files_to_capsule(self._objective)

    async def _launch_task(self) -> None:
        """Start DGM task mode (single operation)."""
        from sources.core.dgm import DarwinMachine

        print(f"\n{GREEN}{BOLD}  🚀  Launching in TASK mode …{RESET}\n")
        dgm = DarwinMachine(self.config)
        await dgm.start_dgm(
            goal=self._objective,
            judge=True,
            learning_mode=self._learn,
            max_iteration=self.config.max_learning_evolve_iterations if self._learn else 1,
        )
