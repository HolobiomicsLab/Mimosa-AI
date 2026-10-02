"""
Interactive evaluation CLI for Mimosa-AI.

Guides the user through evaluation setup: model selection (smolagent only),
port range, workspace folder, evaluation mode, then launches CsvEvaluationMode
on the ScienceAgentBench dataset.  Supports queuing multiple evaluation runs
with different configurations, validates unique workspaces, and executes them
sequentially in queue order.

Saves run metadata (including detected MCPs) to ``run_notes/evaluations/``
at start and appends final results at the end.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from config import AddressMCP, Config
from sources.cli.onboard_cli import _MODEL_PRESETS
from sources.cli.theme import (
    AMBER,
    EMBER,
    GREY,
    LOCKED,
    RESET,
    WHITE,
    banner,
    frame_bottom,
    frame_top,
    kv,
    section,
)
from sources.cli.theme import (
    ask as _ask,
)
from sources.cli.theme import (
    ask_yn as _ask_yn,
)
from sources.cli.theme import (
    fail as _err,
)
from sources.cli.theme import (
    info as _info,
)
from sources.cli.theme import (
    ok as _ok,
)
from sources.cli.theme import (
    step_header as _print_step,
)
from sources.cli.theme import (
    warn as _warn,
)
from sources.cli.theme import (
    wrap as _wrap,
)
from sources.core.tools_manager import ToolManager

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_EVAL_BANNER = banner("Evaluation console", "ScienceAgentBench")

TOTAL_STEPS = 7  # Config → Model → Connectivity → Mode → Tasks → Advanced → Queue/Launch
_CONFIG_DEFAULT_PATH = "config_default.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalRunSpec:
    """Specification for a single evaluation run in the queue."""
    run_id: int
    config: Config
    eval_mode: str  # one_shot | single_agent | iterative
    csv_runs_limit: int
    mcp_list: list[str] = field(default_factory=list)
    notes_path: Path | None = None
    # Advanced/ablation fields the user changed from defaults (name -> value)
    overrides: dict = field(default_factory=dict)
    # Recovery decisions resolved during the configuration phase, so the
    # queue can execute unattended — no stdin prompts after launch.
    start_row: int = 0          # 0-based first CSV row to process
    restore_cache: bool = True  # restore stats from previous run notes if found
    # Populated after execution
    status: str = "pending"


class EvaluationCLI:
    """Interactive setup wizard for running ScienceAgentBench evaluations."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._mcp_list: list[str] = []
        self._queue: list[EvalRunSpec] = []
        self._next_run_id: int = 1

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the full evaluation setup flow, then launch the queue."""
        print(_EVAL_BANNER)

        # Step 1 – Config file (auto-load)
        _print_step(1, TOTAL_STEPS, "Configuration")
        self._load_config()

        # Resolve the Toolomics workspace before anything is configured or
        # launched — the CLI refuses to start without a valid one.
        self._resolve_workspace()

        # Build queue: loop of configure-run → validate → "add another?"
        while True:
            run_spec = await self._configure_single_run()
            self._queue.append(run_spec)
            _ok(f"Run #{run_spec.run_id} added to queue.")

            add_more = _ask_yn("Add another evaluation run to the queue?", default=False)
            if not add_more:
                break

        # Final summary & launch
        _print_step(TOTAL_STEPS, TOTAL_STEPS, "Queue Summary & Launch")
        self._print_queue_summary()

        go = _ask_yn("Launch all queued evaluations now?", default=True)
        if not go:
            for spec in self._queue:
                self._save_start_notes(spec)
                self._update_notes(spec.notes_path, {"status": "cancelled"})
            print("\n  Exiting without launching. Run again when ready.\n")
            sys.exit(0)

        # Save start notes for all runs
        for spec in self._queue:
            self._save_start_notes(spec)

        # Execute runs one after another
        await self._launch_queue()

    # ------------------------------------------------------------------
    # Step 1 – Config
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        if os.path.isfile(_CONFIG_DEFAULT_PATH):
            try:
                self.config.load(_CONFIG_DEFAULT_PATH)
                _ok(f"Loaded {_CONFIG_DEFAULT_PATH}")
            except Exception as exc:
                _warn(f"Could not load {_CONFIG_DEFAULT_PATH} ({exc}). Using defaults.")
        else:
            _info("No config_default.json found – using built-in defaults.")
        self.config.create_paths()

    # ------------------------------------------------------------------
    # Toolomics workspace resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _is_toolomics_workspace(path: str) -> bool:
        """True if *path* is an existing directory with 'toolomics' in it (any case)."""
        try:
            resolved = os.path.realpath(os.path.expanduser(path))
        except (OSError, ValueError):
            return False
        return os.path.isdir(resolved) and "toolomics" in resolved.lower()

    def _resolve_workspace(self) -> None:
        """Resolve the Toolomics workspace, refusing to start without one.

        Resolution order:
        1. The configured ``workspace_dir`` (config file / built-in default),
           kept as-is when it already points at an existing 'toolomics' folder.
        2. The sibling checkout ``<repo>/../Toolomics/workspace`` — the layout
           ``auto-install.sh`` creates (Toolomics cloned next to Mimosa-AI),
           checked from the repo location and from the current directory.
        3. Interactive prompt. The user is re-prompted until the given path is
           an existing directory containing 'toolomics' (case-insensitive);
           empty, missing or non-matching answers are refused.

        A detected or typed path is persisted to the config file so later
        launches skip this step.
        """
        section("TOOLOMICS WORKSPACE")

        current = self.config.workspace_dir
        if self._is_toolomics_workspace(current):
            _ok(f"Workspace: {os.path.realpath(current)}")
            return

        # Auto-detect the sibling Toolomics checkout (auto-install layout).
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root.parent / "Toolomics" / "workspace",
            (Path.cwd() / ".." / "Toolomics" / "workspace").resolve(),
        ]
        for candidate in candidates:
            if self._is_toolomics_workspace(str(candidate)):
                self.config.workspace_dir = str(candidate)
                _ok(f"Auto-detected workspace: {os.path.realpath(str(candidate))}")
                self._persist_workspace_dir()
                return

        _warn(f"Configured workspace is not an existing 'toolomics' folder: {current}")
        print(_wrap(
            "Every artefact the agents produce is written to the Toolomics "
            "shared workspace, so a valid folder is required before any "
            "evaluation can run.",
        ))

        while True:
            raw = _ask("Workspace directory path (e.g. ../Toolomics/workspace)")
            if not raw or not raw.strip():
                _warn("No path given — a 'toolomics' workspace is required to continue.")
                continue
            path = os.path.expanduser(raw.strip())
            if self._is_toolomics_workspace(path):
                self.config.workspace_dir = os.path.realpath(path)
                _ok(f"Workspace: {self.config.workspace_dir}")
                self._persist_workspace_dir()
                return
            if os.path.isdir(path):
                _err(
                    f"'{path}' exists but 'toolomics' is not in its path — "
                    "refusing to use it as the workspace."
                )
            else:
                _err(f"'{path}' is not an existing directory — try again.")

    def _persist_workspace_dir(self) -> None:
        """Write the current workspace_dir into the persisted config file."""
        try:
            self.config.dump(_CONFIG_DEFAULT_PATH)
            _ok(f"Saved workspace_dir to {_CONFIG_DEFAULT_PATH}")
        except Exception as exc:
            _warn(f"Could not persist workspace path to {_CONFIG_DEFAULT_PATH}: {exc}")

    # ------------------------------------------------------------------
    # Configure a single run (Steps 2–5 for each queued run)
    # ------------------------------------------------------------------

    async def _configure_single_run(self) -> EvalRunSpec:
        """Walk the user through configuring one evaluation run."""
        run_id = self._next_run_id
        self._next_run_id += 1

        section(f"CONFIGURING RUN #{run_id}")
        print()

        # Deep-copy the base config so each run is independent
        run_config = copy.deepcopy(self.config)

        # Step 2 – Agent model
        _print_step(2, TOTAL_STEPS, f"Agent Model Selection (Run #{run_id})")
        self._choose_agent_model(run_config)

        # Step 3 – Port range & workspace
        _print_step(3, TOTAL_STEPS, f"Toolomics / Workspace (Run #{run_id})")
        mcp_list = await self._setup_connectivity(run_config, run_id)

        # Step 4 – Evaluation mode
        _print_step(4, TOTAL_STEPS, f"Evaluation Mode (Run #{run_id})")
        eval_mode = self._choose_eval_mode()

        # Step 5 – Number of tasks (csv_runs_limit)
        _print_step(5, TOTAL_STEPS, f"Task Limit (Run #{run_id})")
        csv_runs_limit = self._ask_csv_runs_limit()
        start_row, restore_cache = self._ask_recovery_options()

        # Step 6 – Advanced / ablation options (optional)
        _print_step(6, TOTAL_STEPS, f"Advanced / Ablation Options (Run #{run_id})")
        overrides = self._configure_advanced_options(run_config, eval_mode)

        return EvalRunSpec(
            run_id=run_id,
            config=run_config,
            eval_mode=eval_mode,
            csv_runs_limit=csv_runs_limit,
            mcp_list=mcp_list,
            overrides=overrides,
            start_row=start_row,
            restore_cache=restore_cache,
        )

    # ------------------------------------------------------------------
    # Step 2 – Agent model (smolagent_model_id only)
    # ------------------------------------------------------------------

    def _choose_agent_model(self, run_config: Config) -> None:
        available: list[tuple[str, str]] = [
            (label, model_id)
            for env_key, label, model_id in _MODEL_PRESETS
            if os.getenv(env_key)
        ]

        print(_wrap(
            "Choose the LLM for agent execution (SmolAgents). "
            "This is the only model you can change for evaluations.",
        ))

        suggested = run_config.smolagent_model_id or (
            available[0][1] if available else ""
        )

        if run_config.smolagent_model_id:
            _info(f"Current value: {run_config.smolagent_model_id}")

        if available:
            print(f"\n  {WHITE}Available presets:{RESET}")
            for idx, (label, model_id) in enumerate(available, start=1):
                is_default = model_id == suggested
                tag = f"  {LOCKED}" if is_default else ""
                print(f"  {AMBER}[{idx}]{RESET}  {WHITE}{label}{RESET}{tag}")
                print(f"         {GREY}{model_id}{RESET}")
            print(f"  {AMBER}[c]{RESET}  {GREY}Enter a custom model ID{RESET}")
        else:
            _warn("No matching API key found – enter a model ID manually.")

        while True:
            choice = _ask(
                "Select number, 'c' for custom, or Enter to keep current",
                default="",
            )

            if not choice and suggested:
                model = suggested
                break
            if choice.lower() == "c" or not available:
                custom = _ask(
                    "Enter model ID (e.g. anthropic/claude-sonnet-4-5)"
                ).strip()
                if custom:
                    model = custom
                    break
                if suggested:
                    _warn("No model ID entered — keeping current.")
                    model = suggested
                    break
                _warn("No model ID entered — please try again.")
                continue
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    model = available[idx][1]
                    break
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

        run_config.smolagent_model_id = model
        _ok(f"Agent model: {model}")

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def _queued_workspaces(self) -> list[tuple[int, str]]:
        """Return ``(run_id, resolved_workspace)`` for every queued run."""
        return [
            (spec.run_id, os.path.realpath(spec.config.workspace_dir))
            for spec in self._queue
        ]

    def _suggest_workspace(self, run_config: Config, run_id: int) -> str:
        """
        Suggest a workspace path that doesn't collide with any queued run.
        For run #1 the current config value is returned as-is; for later
        runs a ``_runN`` suffix is appended.
        """
        base_ws = os.path.realpath(run_config.workspace_dir)
        if not self._queue:
            return base_ws

        taken = {ws for _, ws in self._queued_workspaces()}
        if base_ws not in taken:
            return base_ws

        # Append _run<N> suffix; strip any existing _run<M> suffix first
        stripped = re.sub(r"_run\d+$", "", base_ws)
        candidate = f"{stripped}_run{run_id}"
        while candidate in taken:
            run_id += 1
            candidate = f"{stripped}_run{run_id}"
        return candidate

    # ------------------------------------------------------------------
    # Step 3 – Toolomics connectivity & workspace
    # ------------------------------------------------------------------

    async def _setup_connectivity(self, run_config: Config, run_id: int) -> list[str]:
        """Configure port range and discover MCPs. Returns MCP list.

        The workspace itself is resolved once in ``run()`` before any run is
        configured, so every queued run inherits the same validated path.
        """

        # ---- Port range ------------------------------------------------
        if self._queue:
            # Sequential runs may share ports — default to the previous run's range
            run_config.discovery_addresses = copy.deepcopy(
                self._queue[-1].config.discovery_addresses
            )

        current = run_config.discovery_addresses
        _info(
            f"Current discovery addresses: "
            f"{', '.join(f'{a.ip}:{a.port_min}-{a.port_max}' for a in current)}"
        )

        keep = _ask_yn("Use the same port range?", default=True)
        if not keep:
            while True:
                ip = _ask("IP address (empty to keep current range)")
                if not ip:
                    _info("Keeping current port range.")
                    break
                port_min = _ask("Port min")
                port_max = _ask("Port max")
                try:
                    run_config.discovery_addresses = [
                        AddressMCP(ip=ip, port_min=int(port_min), port_max=int(port_max))
                    ]
                    _ok(f"Discovery: {ip}:{port_min}-{port_max}")
                    break
                except Exception as exc:
                    _warn(f"Invalid address ({exc}). Please try again.")

        # ---- Discover MCPs ---------------------------------------------
        tool_manager = ToolManager(config=run_config)
        mcp_list: list[str] = []
        while True:
            try:
                mcps = await tool_manager.discover_mcp_servers()
            except Exception as exc:
                _warn(f"Discovery error: {exc}")
                mcps = []

            if mcps:
                tool_manager.mcps = mcps
                mcp_list = [str(m) for m in mcps]
                for m in mcps:
                    _ok(f"MCP online: {m}")
                break

            _err("No MCP servers found.")
            addrs = run_config.discovery_addresses
            addr_str = ", ".join(
                f"{a.ip}:{a.port_min}-{a.port_max}" for a in addrs
            )
            print(_wrap(
                f"Please start Toolomics on the configured port range ({addr_str}).",
            ))
            print(f"\n  {WHITE}Options:{RESET}")
            print(f"    {AMBER}Enter{RESET}   {GREY}– retry scan{RESET}")
            print(f"    {AMBER}skip{RESET}    {GREY}– queue this run anyway "
                  f"(will fail at launch){RESET}")
            choice = _ask("Retry or skip?").strip().lower()
            if choice == "skip":
                _warn(
                    "Queuing run with no MCPs detected. "
                    "Evaluation may fail at runtime."
                )
                break

        return mcp_list

    # ------------------------------------------------------------------
    # Step 4 – Evaluation mode
    # ------------------------------------------------------------------

    def _choose_eval_mode(self) -> str:
        print(_wrap(
            "Choose how Mimosa should run each benchmark task:",
        ))
        print()
        print(f"  {AMBER}[1]{RESET}  {WHITE}Single-agent{RESET}        "
              f"{GREY}– one agent per task (baseline comparison){RESET}")
        print(f"  {AMBER}[2]{RESET}  {WHITE}One-shot{RESET}            "
              f"{GREY}– multi-agent workflow, no learning{RESET}")
        print(f"  {AMBER}[3]{RESET}  {WHITE}Iterative learning{RESET}  "
              f"{GREY}– multi-agent with evolution loop{RESET}")
        print()

        while True:
            choice = _ask("Select mode (1/2/3)", default="2").strip().lower()
            if choice in ("1", "single", "single_agent", "single-agent"):
                _ok("Mode: Single-agent")
                return "single_agent"
            if choice in ("2", "one_shot", "one-shot", "oneshot"):
                _ok("Mode: One-shot (no learning)")
                return "one_shot"
            if choice in ("3", "iterative", "learning"):
                _ok("Mode: Iterative learning")
                return "iterative"
            _warn(
                f"Unrecognised choice '{choice}'. "
                "Please enter 1, 2, or 3 (or the mode name)."
            )

    # ------------------------------------------------------------------
    # Step 5 – csv_runs_limit
    # ------------------------------------------------------------------

    def _ask_csv_runs_limit(self) -> int:
        """Ask how many tasks to evaluate (csv_runs_limit)."""
        print(_wrap(
            "How many benchmark tasks should this run evaluate? "
            "This is equivalent to --csv_runs_limit. "
            "Enter a number (default: 200 = all tasks).",
        ))
        while True:
            raw = _ask("Number of tasks", default="200")
            try:
                val = int(raw.strip())
                if val < 1:
                    _warn("Must be at least 1.")
                    continue
                _ok(f"Task limit: {val}")
                return val
            except ValueError:
                _warn(f"Invalid number '{raw}'. Please enter a whole number.")

    def _ask_recovery_options(self) -> tuple[int, bool]:
        """Resolve start-row and cache-restore decisions at configuration time.

        csv_mode used to prompt for these when each queued run *started*;
        resolving them here keeps the queue fully unattended after launch.
        """
        print(_wrap(
            "Recovery options: resume from a given CSV row and/or restore "
            "statistics from a previous run's notes (same model)."
        ))
        while True:
            raw = _ask("Starting row (1 = first task)", default="1")
            try:
                start_row = max(0, int(raw.strip()) - 1)
                break
            except ValueError:
                _warn(f"Invalid value '{raw}'. Please enter a whole number.")
        restore_cache = _ask_yn(
            "Restore previous run statistics from cache if found?", default=True,
        )
        _ok(f"Start row: {start_row + 1}, restore cache: {'yes' if restore_cache else 'no'}")
        return start_row, restore_cache

    # ------------------------------------------------------------------
    # Step 6 – Advanced / ablation options (optional)
    # ------------------------------------------------------------------

    # Selection strategies supported by sources/core/selection.py
    _SELECTION_STRATEGIES = ("qd", "tournament", "greedy", "novelty")

    def _configure_advanced_options(self, run_config: Config, eval_mode: str) -> dict:
        """Optionally tweak ablation-level config knobs for this run.

        Returns a dict of ``{field_name: new_value}`` for every field the
        user changed (used for the queue summary and run notes).
        """
        print(_wrap(
            "Optional: override evolution / orchestration knobs for ablation "
            "runs (selection strategy, novelty weight, grounding, iteration "
            "budget, …). Press Enter to keep the defaults from your config.",
        ))
        customise = _ask_yn("Customise advanced / ablation options?", default=False)
        if not customise:
            _info("Keeping default advanced options.")
            return {}

        learning_only = eval_mode != "iterative"
        overrides: dict = {}

        # (key, label, current-value getter, editor, learning_only)
        options = [
            (
                "workflow_llm_model",
                "Orchestrator backbone (workflow_llm_model)",
                lambda: run_config.workflow_llm_model,
                lambda: self._edit_text(
                    "Orchestrator backbone model ID",
                    run_config.workflow_llm_model,
                ),
                False,
            ),
            (
                "orchestrator_choose_model",
                "Per-agent model assignment (orchestrator_choose_model)",
                lambda: run_config.orchestrator_choose_model,
                lambda: _ask_yn(
                    "Let the orchestrator assign models per agent?",
                    default=bool(run_config.orchestrator_choose_model),
                ),
                False,
            ),
            (
                "literrature_grounding",
                "Literature grounding (literrature_grounding)",
                lambda: run_config.literrature_grounding,
                lambda: _ask_yn(
                    "Enable literature grounding (Perspicacité)?",
                    default=bool(run_config.literrature_grounding),
                ),
                False,
            ),
            (
                "selection_strategy",
                "Selection strategy",
                lambda: run_config.selection_strategy,
                lambda: self._edit_selection_strategy(run_config.selection_strategy),
                True,
            ),
            (
                "novelty_weight",
                "Novelty weight (QD quality/novelty mix)",
                lambda: run_config.novelty_weight,
                lambda: self._edit_float(
                    "Novelty weight (0.0 = quality-only, 1.0 = novelty-only)",
                    run_config.novelty_weight, lo=0.0, hi=1.0,
                ),
                True,
            ),
            (
                "max_learning_evolve_iterations",
                "Max evolve iterations",
                lambda: run_config.max_learning_evolve_iterations,
                lambda: self._edit_int(
                    "Max evolve iterations (1 = evolution off / best-of-N)",
                    run_config.max_learning_evolve_iterations, lo=1,
                ),
                True,
            ),
            (
                "learned_score_threshold",
                "Early-stop score threshold",
                lambda: run_config.learned_score_threshold,
                lambda: self._edit_float(
                    "Early-stop score threshold",
                    run_config.learned_score_threshold, lo=0.0, hi=1.0,
                ),
                True,
            ),
            (
                "crossover_rate",
                "Crossover rate",
                lambda: run_config.crossover_rate,
                lambda: self._edit_float(
                    "Crossover rate (0.0 = off)",
                    run_config.crossover_rate, lo=0.0, hi=1.0,
                ),
                True,
            ),
            (
                "population_size",
                "Population / archive size",
                lambda: run_config.population_size,
                lambda: self._edit_int(
                    "Population / archive size",
                    run_config.population_size, lo=1,
                ),
                True,
            ),
        ]

        while True:
            print()
            for idx, (_, label, getter, _, learn_only) in enumerate(options, start=1):
                tag = f"  {GREY}(learning only){RESET}" if learn_only else ""
                print(f"  {AMBER}[{idx}]{RESET}  {WHITE}{label}{RESET}{tag}")
                print(f"         {GREY}= {getter()}{RESET}")
            print()

            choice = _ask("Edit option number, or Enter to finish", default="").strip()
            if not choice:
                break
            try:
                idx = int(choice) - 1
                if not (0 <= idx < len(options)):
                    raise ValueError
            except ValueError:
                _warn(f"Unrecognised input '{choice}'. Pick 1-{len(options)} or Enter.")
                continue

            key, _, getter, editor, learn_only = options[idx]
            if learn_only and learning_only:
                _warn(
                    f"'{key}' only takes effect in iterative learning mode — "
                    f"this run is '{eval_mode}'. It will be recorded but ignored."
                )
            new_value = editor()
            if new_value is None:
                continue  # user kept the current value
            if new_value != getter():
                setattr(run_config, key, new_value)
                overrides[key] = new_value
                _ok(f"{key} = {new_value}")
            else:
                _info(f"{key} unchanged.")

        if overrides:
            _ok(f"Advanced overrides recorded: {len(overrides)}")
        else:
            _info("No advanced overrides — defaults kept.")
        return overrides

    # ------------------------------------------------------------------
    # Small editors for the advanced menu
    # ------------------------------------------------------------------

    @staticmethod
    def _edit_text(prompt: str, current: str):
        """Free-text editor; empty input keeps the current value (None)."""
        raw = _ask(f"{prompt} (Enter to keep '{current}')", default="").strip()
        return raw or None

    def _edit_selection_strategy(self, current: str):
        print(f"\n  {WHITE}Selection strategies:{RESET}")
        for idx, s in enumerate(self._SELECTION_STRATEGIES, start=1):
            tag = f"  {LOCKED}" if s == current else ""
            print(f"  {AMBER}[{idx}]{RESET}  {WHITE}{s}{RESET}{tag}")
        while True:
            raw = _ask(
                f"Select 1-{len(self._SELECTION_STRATEGIES)} or Enter to keep '{current}'",
                default="",
            ).strip().lower()
            if not raw:
                return None
            if raw in self._SELECTION_STRATEGIES:
                return raw
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(self._SELECTION_STRATEGIES):
                    return self._SELECTION_STRATEGIES[idx]
            except ValueError:
                pass
            _warn(f"Invalid choice '{raw}'.")

    @staticmethod
    def _edit_float(prompt: str, current: float, lo: float, hi: float):
        while True:
            raw = _ask(f"{prompt} [{current}]", default=str(current)).strip()
            try:
                val = float(raw)
                if not (lo <= val <= hi):
                    _warn(f"Must be between {lo} and {hi}.")
                    continue
                return val
            except ValueError:
                _warn(f"Invalid number '{raw}'.")

    @staticmethod
    def _edit_int(prompt: str, current: int, lo: int = 1):
        while True:
            raw = _ask(f"{prompt} [{current}]", default=str(current)).strip()
            try:
                val = int(raw)
                if val < lo:
                    _warn(f"Must be at least {lo}.")
                    continue
                return val
            except ValueError:
                _warn(f"Invalid number '{raw}'.")


    # ------------------------------------------------------------------
    # Queue summary
    # ------------------------------------------------------------------

    def _print_queue_summary(self) -> None:
        """Print a summary table of all queued runs."""
        print()
        frame_top(f"EVALUATION QUEUE · {len(self._queue)} RUN(S)")
        for spec in self._queue:
            addrs = spec.config.discovery_addresses
            port_str = ", ".join(f"{a.ip}:{a.port_min}-{a.port_max}" for a in addrs)
            print(f"\n    {AMBER}RUN #{spec.run_id}{RESET}")
            kv("model", spec.config.smolagent_model_id)
            kv("mode", spec.eval_mode)
            kv("tasks", str(spec.csv_runs_limit))
            kv("ports", port_str)
            kv("workspace", spec.config.workspace_dir)
            kv("mcps", str(len(spec.mcp_list)))
            if spec.overrides:
                over_str = ", ".join(f"{k}={v}" for k, v in spec.overrides.items())
                kv("overrides", over_str)
        print()
        frame_bottom()
        if len(self._queue) > 1:
            _info("Execution: sequential — runs launch one at a time in queue order.")
        print()

    # ------------------------------------------------------------------
    # Save & update run notes
    # ------------------------------------------------------------------

    def _save_start_notes(self, spec: EvalRunSpec) -> None:
        """Write initial run metadata to run_notes/evaluations/.

        Best-effort: if note creation fails (read-only fs, permissions, disk
        full, …) we warn and clear ``spec.notes_path`` so subsequent
        ``_update_notes`` calls are skipped — execution should not abort just
        because we cannot save run notes.
        """
        eval_dir = Path("run_notes") / "evaluations"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = spec.config.smolagent_model_id
        if isinstance(model_id, list):
            model_id = model_id[0] if model_id else ""
        model_tag = model_id.replace("/", "_")
        filename = f"{ts}_run{spec.run_id}_{model_tag}_{spec.eval_mode}.json"
        notes_path = eval_dir / filename

        notes = {
            "started_at": datetime.now().isoformat(),
            "run_id": spec.run_id,
            "smolagent_model_id": spec.config.smolagent_model_id,
            "workflow_llm_model": spec.config.workflow_llm_model,
            "judge_model": spec.config.judge_model,
            "eval_mode": spec.eval_mode,
            "csv_runs_limit": spec.csv_runs_limit,
            "discovery_addresses": [
                {"ip": a.ip, "port_min": a.port_min, "port_max": a.port_max}
                for a in spec.config.discovery_addresses
            ],
            "workspace_dir": spec.config.workspace_dir,
            "detected_mcps": spec.mcp_list,
            "dataset": "datasets/ScienceAgentBench.csv",
            "advanced_overrides": spec.overrides,
            "evolution_config": {
                "orchestrator_choose_model": spec.config.orchestrator_choose_model,
                "literrature_grounding": spec.config.literrature_grounding,
                "selection_strategy": spec.config.selection_strategy,
                "novelty_weight": spec.config.novelty_weight,
                "max_learning_evolve_iterations": spec.config.max_learning_evolve_iterations,
                "learned_score_threshold": spec.config.learned_score_threshold,
                "crossover_rate": spec.config.crossover_rate,
                "population_size": spec.config.population_size,
            },
            "status": "running",
            "queue_size": len(self._queue),
        }

        try:
            eval_dir.mkdir(parents=True, exist_ok=True)
            with open(notes_path, "w", encoding="utf-8") as fh:
                json.dump(notes, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            _warn(
                f"Could not write notes for Run #{spec.run_id} "
                f"({notes_path}): {exc}. Continuing without note tracking."
            )
            spec.notes_path = None
            return

        spec.notes_path = notes_path
        _ok(f"Run #{spec.run_id} notes → {spec.notes_path}")

    @staticmethod
    def _update_notes(notes_path: Path | None, updates: dict) -> None:
        """Merge *updates* into the on-disk run notes JSON."""
        if not notes_path or not notes_path.exists():
            return
        try:
            with open(notes_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data.update(updates)
            with open(notes_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
                fh.write("\n")
        except Exception:
            pass  # best-effort

    # ------------------------------------------------------------------
    # Sequential launch
    # ------------------------------------------------------------------

    async def _launch_queue(self) -> None:
        """
        Execute all queued runs sequentially, in queue order.

        Each run finishes before the next one starts.  A failing run is
        recorded (status + run notes) and does not stop the rest of the
        queue.
        """
        total = len(self._queue)
        section(f"IGNITION · {total} EVALUATION(S)")
        print()

        for position, spec in enumerate(self._queue, start=1):
            _info(f"Run #{spec.run_id} ({position}/{total}) launching…")
            try:
                await self._run_single_eval(spec)
            except Exception as exc:
                _err(f"Run #{spec.run_id} failed: {exc}")
                if spec.status == "pending":  # failed before the run recorded anything
                    spec.status = "error"
                    self._update_notes(spec.notes_path, {
                        "status": "error",
                        "error": str(exc),
                        "finished_at": datetime.now().isoformat(),
                    })
                continue
            if spec.status == "completed":
                _ok(f"Run #{spec.run_id} completed.")
            else:
                _warn(f"Run #{spec.run_id} finished with status: {spec.status}")

        print()
        _ok(f"All {total} evaluation(s) finished.")
        self._print_final_queue_report()

    async def _run_single_eval(self, spec: EvalRunSpec) -> None:
        """Execute a single evaluation run from its spec."""
        from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode

        run_config = spec.config

        # Isolate mutable directories per run so runs cannot contaminate each other
        run_config.workflow_dir = f"sources/workflows/run_{spec.run_id}"
        run_config.memory_dir = f"sources/memory/run_{spec.run_id}"
        run_config.runner_temp_dir = f"./tmp/run_{spec.run_id}"
        run_config.runs_capsule_dir = f"runs_capsule/run_{spec.run_id}"

        run_config.create_paths()
        try:
            run_config.validate_paths()
        except AssertionError as exc:
            _err(f"Run #{spec.run_id} config validation failed: {exc}")
            spec.status = "config_error"
            self._update_notes(spec.notes_path, {
                "status": "config_error", "error": str(exc)
            })
            return

        single_agent = spec.eval_mode == "single_agent"
        learning = spec.eval_mode == "iterative"

        max_concurrent = getattr(run_config, "max_concurrent_eval_tasks", 4)
        task_start_delay = getattr(run_config, "task_start_delay", 30.0)

        evaluator = CsvEvaluationMode(
            run_config,
            csv_runs_limit=spec.csv_runs_limit,
            max_concurrent_tasks=max_concurrent,
            task_start_delay=task_start_delay,
            run_notes_dir=Path("run_notes") / f"run_{spec.run_id}",
        )
        # Attach the run notes path so csv_mode can write final results there
        evaluator._evaluation_cli_notes_path = spec.notes_path

        _info(f"Run #{spec.run_id} starting "
              f"(model={run_config.smolagent_model_id}, mode={spec.eval_mode}, "
              f"tasks={spec.csv_runs_limit})")

        try:
            await evaluator.start_evaluation(
                dataset_type="science_agent_bench",
                dataset_path="datasets/ScienceAgentBench.csv",
                learning=learning,
                single_agent_mode=single_agent,
                concurrent=max_concurrent > 1,
                start_row=spec.start_row,
                restore_cache=spec.restore_cache,
            )
            # A run aborted by a persistent network outage is NOT a completed
            # benchmark — most rows may never have been evaluated.
            spec.status = (
                "network_failure"
                if getattr(evaluator, "_network_aborted", False)
                else "completed"
            )
            self._update_notes(spec.notes_path, {
                "status": spec.status,
                "finished_at": datetime.now().isoformat(),
            })
        except KeyboardInterrupt:
            spec.status = "interrupted"
            self._update_notes(spec.notes_path, {
                "status": "interrupted",
                "finished_at": datetime.now().isoformat(),
            })
            raise
        except Exception as exc:
            spec.status = "error"
            self._update_notes(spec.notes_path, {
                "status": "error",
                "error": str(exc),
                "finished_at": datetime.now().isoformat(),
            })
            raise

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    def _print_final_queue_report(self) -> None:
        """Print a summary of all queue execution results."""
        print()
        frame_top("QUEUE EXECUTION REPORT")
        print()

        completed = sum(1 for s in self._queue if s.status == "completed")
        errors = sum(1 for s in self._queue if s.status == "error")
        other = len(self._queue) - completed - errors

        for spec in self._queue:
            line = (f"{WHITE}Run #{spec.run_id}{RESET}  "
                    f"{GREY}{spec.config.smolagent_model_id}  "
                    f"({spec.eval_mode}, {spec.csv_runs_limit} tasks){RESET} "
                    f"{EMBER}·{RESET} {spec.status}")
            if spec.status == "completed":
                _ok(line)
            elif spec.status == "error":
                _err(line)
            else:
                _warn(line)
            if spec.notes_path:
                print(f"          {GREY}Notes: {spec.notes_path}{RESET}")

        print()
        kv("completed", str(completed), accent=True)
        kv("errors", str(errors), accent=errors > 0)
        kv("other", str(other))
        print()
        frame_bottom()
        print()
