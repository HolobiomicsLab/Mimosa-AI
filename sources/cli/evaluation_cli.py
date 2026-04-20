"""
Interactive evaluation CLI for Mimosa-AI.

Guides the user through evaluation setup: model selection (smolagent only),
port range, workspace folder, evaluation mode, then launches CsvEvaluationMode
on the ScienceAgentBench dataset.  Supports queuing multiple evaluation runs
with different configurations, validates non-overlapping port ranges and unique
workspaces, and executes them with adaptive parallelism (starts with 2
concurrent runs, doubles when RAM allows).

Saves run metadata (including detected MCPs) to ``run_notes/evaluations/``
at start and appends final results at the end.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from config import Config, AddressMCP
from sources.core.tools_manager import ToolManager
from sources.cli.onboard_cli import (
    _MODEL_PRESETS,
    _print_step,
    _ok,
    _warn,
    _err,
    _info,
    _ask,
    _ask_yn,
    _wrap,
    CYAN,
    GREEN,
    BOLD,
    DIM,
    RESET,
)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_EVAL_BANNER = f"""
{CYAN}{BOLD}
  ███╗   ███╗██╗███╗   ███╗ ██████╗ ███████╗ █████╗
  ████╗ ████║██║████╗ ████║██╔═══██╗██╔════╝██╔══██╗
  ██╔████╔██║██║██╔████╔██║██║   ██║███████╗███████║
  ██║╚██╔╝██║██║██║╚██╔╝██║██║   ██║╚════██║██╔══██║
  ██║ ╚═╝ ██║██║██║ ╚═╝ ██║╚██████╔╝███████║██║  ██║
  ╚═╝     ╚═╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
{RESET}
{DIM}  Evaluation CLI  ·  ScienceAgentBench{RESET}
"""

YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"

TOTAL_STEPS = 6  # Updated: Config → Model → Connectivity → Mode → Tasks → Queue/Launch
_CONFIG_DEFAULT_PATH = "config_default.json"

# Adaptive parallelism constants
_INITIAL_CONCURRENCY = 2
_RAM_SAFETY_FACTOR = 2  # ram_used_last_batch * 2 < available_ram


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
    # Populated after execution
    status: str = "pending"
    peak_ram_mb: float = 0.0


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

        # Build queue: loop of configure-run → validate → "add another?"
        while True:
            run_spec = await self._configure_single_run()
            self._queue.append(run_spec)
            _ok(f"Run #{run_spec.run_id} added to queue.")

            # Validate entire queue so far
            ok = self._validate_queue()
            if not ok:
                _err("Queue validation failed. Removing last run – please reconfigure.")
                self._queue.pop()
                continue

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

        # Launch with adaptive parallelism
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
    # Configure a single run (Steps 2–5 for each queued run)
    # ------------------------------------------------------------------

    async def _configure_single_run(self) -> EvalRunSpec:
        """Walk the user through configuring one evaluation run."""
        run_id = self._next_run_id
        self._next_run_id += 1

        print()
        print(f"  {MAGENTA}{BOLD}{'═' * 54}{RESET}")
        print(f"  {MAGENTA}{BOLD}  CONFIGURING RUN #{run_id}{RESET}")
        print(f"  {MAGENTA}{BOLD}{'═' * 54}{RESET}")
        print()

        # Deep-copy the base config so each run is independent
        run_config = copy.deepcopy(self.config)

        # Step 2 – Agent model
        _print_step(2, TOTAL_STEPS, f"Agent Model Selection (Run #{run_id})")
        self._choose_agent_model(run_config)

        # Step 3 – Port range & workspace
        _print_step(3, TOTAL_STEPS, f"Toolomics / Workspace (Run #{run_id})")
        mcp_list = await self._setup_connectivity(run_config)

        # Step 4 – Evaluation mode
        _print_step(4, TOTAL_STEPS, f"Evaluation Mode (Run #{run_id})")
        eval_mode = self._choose_eval_mode()

        # Step 5 – Number of tasks (csv_runs_limit)
        _print_step(5, TOTAL_STEPS, f"Task Limit (Run #{run_id})")
        csv_runs_limit = self._ask_csv_runs_limit()

        return EvalRunSpec(
            run_id=run_id,
            config=run_config,
            eval_mode=eval_mode,
            csv_runs_limit=csv_runs_limit,
            mcp_list=mcp_list,
        )

    # ------------------------------------------------------------------
    # Step 2 – Agent model (smolagent_model_id only)
    # ------------------------------------------------------------------

    def _choose_agent_model(self, run_config: Config) -> None:
        _info("workflow_llm_model is fixed to anthropic/claude-opus-4-5 for evaluations.")
        run_config.workflow_llm_model = "anthropic/claude-opus-4-5"

        available: list[tuple[str, str]] = [
            (label, model_id)
            for env_key, label, model_id in _MODEL_PRESETS
            if os.getenv(env_key)
        ]

        print(_wrap(
            "Choose the LLM for agent execution (SmolAgents). "
            "This is the only model you can change for evaluations.",
            width=70, indent=2,
        ))

        suggested = run_config.smolagent_model_id or (
            available[0][1] if available else ""
        )

        if run_config.smolagent_model_id:
            _info(f"Current value: {run_config.smolagent_model_id}")

        if available:
            print(f"\n  {BOLD}Available presets:{RESET}")
            for idx, (label, model_id) in enumerate(available, start=1):
                is_default = model_id == suggested
                tag = f"{GREEN}← default{RESET}" if is_default else ""
                num_color = GREEN if is_default else CYAN
                print(f"  {num_color}[{idx}]{RESET}  {label}  {tag}")
                print(f"         {DIM}{model_id}{RESET}")
            print(f"  {CYAN}[c]{RESET}  Enter a custom model ID")
        else:
            _warn("No matching API key found – enter a model ID manually.")

        choice = _ask(
            "Select number, 'c' for custom, or Enter to keep current",
            default="",
        )

        if not choice and suggested:
            model = suggested
        elif choice.lower() == "c" or not available:
            custom = _ask("Enter model ID (e.g. anthropic/claude-sonnet-4-5)")
            model = custom.strip() if custom.strip() else suggested
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    model = available[idx][1]
                else:
                    _warn(f"Invalid selection '{choice}'. Using default.")
                    model = suggested
            except ValueError:
                _warn(f"Unrecognised input '{choice}'. Using default.")
                model = suggested

        run_config.smolagent_model_id = model
        _ok(f"Agent model: {model}")

    # ------------------------------------------------------------------
    # Step 3 – Toolomics connectivity & workspace
    # ------------------------------------------------------------------

    async def _setup_connectivity(self, run_config: Config) -> list[str]:
        """Configure port range, discover MCPs, set workspace. Returns MCP list."""
        # Port range
        current = run_config.discovery_addresses
        _info(
            f"Current discovery addresses: "
            f"{', '.join(f'{a.ip}:{a.port_min}-{a.port_max}' for a in current)}"
        )
        change = _ask_yn("Change port range?", default=False)
        if change:
            ip = _ask("IP address", default="0.0.0.0")
            port_min = _ask("Port min", default="5000")
            port_max = _ask("Port max", default="5200")
            try:
                run_config.discovery_addresses = [
                    AddressMCP(ip=ip, port_min=int(port_min), port_max=int(port_max))
                ]
                _ok(f"Discovery: {ip}:{port_min}-{port_max}")
            except Exception as exc:
                _warn(f"Invalid address ({exc}). Keeping previous value.")

        # Discover MCPs
        tool_manager = ToolManager(config=run_config)
        mcp_list: list[str] = []
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
        else:
            _err("No MCP servers found. Evaluation may fail at runtime.")

        # Workspace — loop until we have a valid directory
        while True:
            ws = run_config.workspace_dir
            if os.path.isdir(ws):
                _ok(f"Workspace directory found: {ws}")
                break

            _err(f"Workspace directory not found: {ws}")
            print(_wrap(
                "This path must point to the Toolomics workspace folder — the shared "
                "directory where Mimosa reads and writes task artifacts. "
                "Please enter the correct absolute path.",
                width=70, indent=2,
            ))
            new_ws = _ask("Workspace directory path")
            if not new_ws:
                _warn("No path provided. Workspace must be set for evaluation to work.")
                continue
            new_ws = os.path.expanduser(new_ws.strip())
            if os.path.isdir(new_ws):
                run_config.workspace_dir = new_ws
                _ok(f"Workspace set to: {new_ws}")
                break
            _err(f"Directory does not exist: {new_ws}. Please try again.")

        return mcp_list

    # ------------------------------------------------------------------
    # Step 4 – Evaluation mode
    # ------------------------------------------------------------------

    def _choose_eval_mode(self) -> str:
        print(_wrap(
            "Choose how Mimosa should run each benchmark task:",
            width=70, indent=2,
        ))
        print()
        print(f"  {CYAN}[1]{RESET}  Single-agent       – one agent per task (baseline comparison)")
        print(f"  {CYAN}[2]{RESET}  One-shot            – multi-agent workflow, no learning")
        print(f"  {CYAN}[3]{RESET}  Iterative learning  – multi-agent with evolution loop")
        print()

        choice = _ask("Select mode", default="2")
        if choice == "1":
            _ok("Mode: Single-agent")
            return "single_agent"
        elif choice == "3":
            _ok("Mode: Iterative learning")
            return "iterative"
        else:
            _ok("Mode: One-shot (no learning)")
            return "one_shot"

    # ------------------------------------------------------------------
    # Step 5 – csv_runs_limit
    # ------------------------------------------------------------------

    def _ask_csv_runs_limit(self) -> int:
        """Ask how many tasks to evaluate (csv_runs_limit)."""
        print(_wrap(
            "How many benchmark tasks should this run evaluate? "
            "This is equivalent to --csv_runs_limit. "
            "Enter a number (default: 200 = all tasks).",
            width=70, indent=2,
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

    # ------------------------------------------------------------------
    # Queue validation
    # ------------------------------------------------------------------

    def _validate_queue(self) -> bool:
        """
        Validate the full queue:
        - No overlapping port ranges between any two runs
        - All workspace paths are unique (resolved)

        Returns True if valid, False otherwise.
        """
        ok = True

        # Collect port ranges and workspaces
        port_ranges: list[tuple[int, set[int]]] = []  # (run_id, set_of_ports)
        workspaces: list[tuple[int, str]] = []  # (run_id, resolved_path)

        for spec in self._queue:
            # Build set of all ports for this run
            ports: set[int] = set()
            for addr in spec.config.discovery_addresses:
                ports.update(range(addr.port_min, addr.port_max + 1))
            port_ranges.append((spec.run_id, ports))

            # Resolved workspace path
            ws = os.path.realpath(spec.config.workspace_dir)
            workspaces.append((spec.run_id, ws))

        # Check port overlaps (pairwise)
        for i in range(len(port_ranges)):
            for j in range(i + 1, len(port_ranges)):
                rid_a, ports_a = port_ranges[i]
                rid_b, ports_b = port_ranges[j]
                overlap = ports_a & ports_b
                if overlap:
                    sample = sorted(overlap)[:5]
                    _err(
                        f"Port conflict between Run #{rid_a} and Run #{rid_b}: "
                        f"{len(overlap)} overlapping port(s) (e.g. {sample})"
                    )
                    ok = False

        # Check workspace uniqueness
        for i in range(len(workspaces)):
            for j in range(i + 1, len(workspaces)):
                rid_a, ws_a = workspaces[i]
                rid_b, ws_b = workspaces[j]
                if ws_a == ws_b:
                    _err(
                        f"Workspace conflict: Run #{rid_a} and Run #{rid_b} "
                        f"share the same workspace: {ws_a}"
                    )
                    ok = False

        if ok and len(self._queue) > 1:
            _ok(f"Queue validated: {len(self._queue)} runs, no port/workspace conflicts.")

        return ok

    # ------------------------------------------------------------------
    # Queue summary
    # ------------------------------------------------------------------

    def _print_queue_summary(self) -> None:
        """Print a summary table of all queued runs."""
        print()
        print(f"  {BOLD}{'═' * 70}{RESET}")
        print(f"  {BOLD}EVALUATION QUEUE — {len(self._queue)} run(s){RESET}")
        print(f"  {'═' * 70}")
        for spec in self._queue:
            addrs = spec.config.discovery_addresses
            port_str = ", ".join(f"{a.ip}:{a.port_min}-{a.port_max}" for a in addrs)
            print(f"  {CYAN}Run #{spec.run_id}{RESET}")
            print(f"    Model:      {spec.config.smolagent_model_id}")
            print(f"    Mode:       {spec.eval_mode}")
            print(f"    Tasks:      {spec.csv_runs_limit}")
            print(f"    Ports:      {port_str}")
            print(f"    Workspace:  {spec.config.workspace_dir}")
            print(f"    MCPs:       {len(spec.mcp_list)}")
            print(f"  {'─' * 70}")
        if len(self._queue) > 1:
            print(f"  {BOLD}Execution: adaptive parallelism (starting with "
                  f"{min(_INITIAL_CONCURRENCY, len(self._queue))} concurrent){RESET}")
        print()

    # ------------------------------------------------------------------
    # Save & update run notes
    # ------------------------------------------------------------------

    def _save_start_notes(self, spec: EvalRunSpec) -> None:
        """Write initial run metadata to run_notes/evaluations/."""
        eval_dir = Path("run_notes") / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = spec.config.smolagent_model_id.replace("/", "_")
        filename = f"{ts}_run{spec.run_id}_{model_tag}_{spec.eval_mode}.json"
        spec.notes_path = eval_dir / filename

        notes = {
            "started_at": datetime.now().isoformat(),
            "run_id": spec.run_id,
            "smolagent_model_id": spec.config.smolagent_model_id,
            "workflow_llm_model": spec.config.workflow_llm_model,
            "eval_mode": spec.eval_mode,
            "csv_runs_limit": spec.csv_runs_limit,
            "discovery_addresses": [
                {"ip": a.ip, "port_min": a.port_min, "port_max": a.port_max}
                for a in spec.config.discovery_addresses
            ],
            "workspace_dir": spec.config.workspace_dir,
            "detected_mcps": spec.mcp_list,
            "dataset": "datasets/ScienceAgentBench.csv",
            "status": "running",
            "queue_size": len(self._queue),
        }

        with open(spec.notes_path, "w", encoding="utf-8") as fh:
            json.dump(notes, fh, indent=2)
            fh.write("\n")

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
    # Adaptive parallel launch
    # ------------------------------------------------------------------

    async def _launch_queue(self) -> None:
        """
        Execute all queued runs with adaptive parallelism.

        Strategy:
        - Start with min(INITIAL_CONCURRENCY, queue_length) concurrent runs
        - After each batch completes, measure peak RAM usage
        - Double concurrency for the next batch if:
            peak_ram_last_batch * 2 < available_ram
        - Otherwise keep the same concurrency level
        """
        try:
            import psutil
        except ImportError:
            _warn("psutil not installed — running all evaluations sequentially.")
            _info("Install psutil for adaptive parallel execution: pip install psutil")
            for spec in self._queue:
                await self._run_single_eval(spec)
            return

        remaining = list(self._queue)
        concurrency = min(_INITIAL_CONCURRENCY, len(remaining))
        batch_num = 0

        print()
        print(f"  {MAGENTA}{BOLD}{'═' * 60}{RESET}")
        print(f"  {MAGENTA}{BOLD}  LAUNCHING {len(remaining)} EVALUATION(S){RESET}")
        print(f"  {MAGENTA}{BOLD}  Initial concurrency: {concurrency}{RESET}")
        print(f"  {MAGENTA}{BOLD}{'═' * 60}{RESET}")
        print()

        while remaining:
            batch_num += 1
            batch = remaining[:concurrency]
            remaining = remaining[concurrency:]

            _info(f"Batch #{batch_num}: launching {len(batch)} run(s) "
                  f"(concurrency={concurrency}, {len(remaining)} remaining)")

            # Snapshot RAM before batch
            mem_before = psutil.virtual_memory()
            ram_available_before = mem_before.available / (1024 * 1024)  # MB

            # Launch batch concurrently
            tasks = [
                asyncio.create_task(
                    self._run_single_eval(spec),
                    name=f"eval_run_{spec.run_id}",
                )
                for spec in batch
            ]

            # Wait for all tasks in this batch, capturing exceptions
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log results
            peak_ram_batch = 0.0
            for spec, result in zip(batch, results):
                if isinstance(result, Exception):
                    _err(f"Run #{spec.run_id} failed: {result}")
                    spec.status = "error"
                    self._update_notes(spec.notes_path, {
                        "status": "error",
                        "error": str(result),
                        "finished_at": datetime.now().isoformat(),
                    })
                else:
                    _ok(f"Run #{spec.run_id} completed.")
                peak_ram_batch = max(peak_ram_batch, spec.peak_ram_mb)

            # Adaptive scaling: decide concurrency for next batch
            if remaining:
                mem_after = psutil.virtual_memory()
                ram_available_now = mem_after.available / (1024 * 1024)  # MB
                ram_used_by_batch = max(0, ram_available_before - ram_available_now)

                # Use the larger of measured usage or peak_ram from specs
                ram_estimate = max(ram_used_by_batch, peak_ram_batch)

                _info(f"Batch #{batch_num} RAM estimate: {ram_estimate:.0f} MB "
                      f"(available: {ram_available_now:.0f} MB)")

                if ram_estimate > 0 and (ram_estimate * _RAM_SAFETY_FACTOR) < ram_available_now:
                    new_concurrency = concurrency * 2
                    _ok(f"RAM headroom OK — scaling concurrency: {concurrency} → {new_concurrency}")
                    concurrency = new_concurrency
                else:
                    _info(f"RAM headroom insufficient — keeping concurrency at {concurrency}")

                # Never exceed remaining count
                concurrency = min(concurrency, len(remaining))

        print()
        _ok(f"All {len(self._queue)} evaluation(s) finished.")
        self._print_final_queue_report()

    async def _run_single_eval(self, spec: EvalRunSpec) -> None:
        """Execute a single evaluation run from its spec."""
        from sources.evaluation.csv_mode import CsvEvaluationMode

        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            process = None

        run_config = spec.config

        # Isolate mutable directories per run to prevent race conditions
        run_config.workflow_dir = f"sources/workflows/run_{spec.run_id}"
        run_config.memory_dir = f"sources/memory/run_{spec.run_id}"
        run_config.runner_temp_dir = f"./tmp/run_{spec.run_id}"

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
            )
            spec.status = "completed"
            self._update_notes(spec.notes_path, {
                "status": "completed",
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
        finally:
            # Record peak RAM for adaptive scaling
            if process is not None:
                try:
                    mem_info = process.memory_info()
                    spec.peak_ram_mb = mem_info.rss / (1024 * 1024)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    def _print_final_queue_report(self) -> None:
        """Print a summary of all queue execution results."""
        print()
        print(f"  {BOLD}{'═' * 60}{RESET}")
        print(f"  {BOLD}QUEUE EXECUTION REPORT{RESET}")
        print(f"  {'═' * 60}")

        completed = sum(1 for s in self._queue if s.status == "completed")
        errors = sum(1 for s in self._queue if s.status == "error")
        other = len(self._queue) - completed - errors

        for spec in self._queue:
            if spec.status == "completed":
                icon = f"{GREEN}✓{RESET}"
            elif spec.status == "error":
                icon = f"{RED}✗{RESET}"
            else:
                icon = f"{YELLOW}?{RESET}"

            print(f"  {icon}  Run #{spec.run_id}  "
                  f"{spec.config.smolagent_model_id}  "
                  f"({spec.eval_mode}, {spec.csv_runs_limit} tasks)  "
                  f"→ {spec.status}")
            if spec.notes_path:
                print(f"      {DIM}Notes: {spec.notes_path}{RESET}")

        print(f"  {'─' * 60}")
        print(f"  Completed: {completed}  |  Errors: {errors}  |  Other: {other}")
        print(f"  {'═' * 60}")
        print()
