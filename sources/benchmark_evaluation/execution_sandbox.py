"""
Execution Sandbox - Safe execution utilities for evaluating generated code.

Provides isolated execution environment with automatic dependency management.
"""

from __future__ import annotations

import ast
import atexit
import logging
import os
import re
import subprocess
import shutil
import tempfile
import threading
from pathlib import Path


logger = logging.getLogger(__name__)

SANDBOX_PYTHON_VERSION = "3.12"


class EvalInfraError(RuntimeError):
    """Raised when the eval harness/environment fails — not the agent's code.

    Signals the task should be EXCLUDED from VER/SR metrics rather than counted
    as a failure. Examples: sandbox/venv build failure, missing gold_results,
    or a figure-judged task with no OPENAI_API_KEY / AZURE_OPENAI_KEY set.
    """


class ExecutionSandbox:
    """
    Execution sandbox for safely running generated code with dependency management.

    Follows ScienceAgentBench approach:
    - Uses virtual environment to avoid package conflicts
    - Initializes with basic packages: numpy, pandas, matplotlib, pytorch, tensorflow, rdkit, tf-keras
    - Uses pipreqs to analyze generated code for dependencies
    - Uses pip-tools for dependency resolution and installation
    """

    # Basic packages to install in every environment (for ScienceAgentBench)
    BASIC_PACKAGES = [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",  # sklearn
        "torch",  # pytorch
        "tensorflow",
        "rdkit",  # rdkit
        "pipreqs",  # for dependency analysis
        "pip-tools",  # for dependency resolution
        "openai"
    ]

    # Process-wide base venv, created once and reused across tasks.
    _shared_venv_path: Path | None = None
    _shared_base_dir: str | None = None
    _shared_venv_lock = threading.Lock()

    def __init__(
        self,
        capsule_path: Path,
        cpu_only: bool = True,
        base_packages: list[str] | None = None,
    ):
        """
        Initialize execution sandbox and set up its Python environment.

        The base venv is created ONCE per process and reused across tasks (see
        _create_or_reuse_base_venv); only the per-task working directories are
        rebuilt each time. This avoids re-installing torch/tensorflow per task.

        Args:
            capsule_path: Path to capsule directory containing generated code
            cpu_only: If True (default), force CPU execution by hiding any GPU
                from the spawned scripts. Sidesteps CUDA/XLA plumbing issues
                (e.g. missing libdevice) that would otherwise fail VER on
                machines with a partial CUDA install.
            base_packages: Packages to ensure in the shared venv. Defaults to
                BASIC_PACKAGES; pass a lighter set for fast, targeted runs.
        """
        self.capsule_path = Path(capsule_path)
        self.cpu_only = cpu_only
        self.base_packages = list(base_packages) if base_packages is not None else list(self.BASIC_PACKAGES)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Per-instance temp dir for execution/eval working copies (cleaned per task).
        self._temp_dir_context = tempfile.TemporaryDirectory(prefix="mimosa_sandbox_")
        self.temp_dir = Path(self._temp_dir_context.name)

        # Reuse the process-wide base venv instead of rebuilding it per task.
        self.venv_path = self._create_or_reuse_base_venv()
        self.python_exe = self.venv_path / "bin" / "python"
        self.pip_exe = self.venv_path / "bin" / "pip"

        # Ensure base + capsule dependencies are present in the shared venv.
        self._setup_environment()

    def _resolve_sandbox_python(self) -> str:
        """Resolve a Python SANDBOX_PYTHON_VERSION interpreter for the venv.

        Resolution order: the ``$MIMOSA_SANDBOX_PYTHON`` override, then
        ``python<version>`` on PATH. The interpreter's ``--version`` is verified
        so we never silently build the venv with the wrong Python (which is how
        the eval previously drifted onto the harness interpreter).
        """
        candidates = []
        override = os.environ.get("MIMOSA_SANDBOX_PYTHON")
        if override:
            candidates.append(override)
        which = shutil.which(f"python{SANDBOX_PYTHON_VERSION}")
        if which:
            candidates.append(which)

        for cand in candidates:
            try:
                out = subprocess.run(
                    [cand, "--version"], capture_output=True, text=True, timeout=30
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if (out.stdout or out.stderr).strip().startswith(
                f"Python {SANDBOX_PYTHON_VERSION}."
            ):
                return cand

        raise RuntimeError(
            f"Eval sandbox requires Python {SANDBOX_PYTHON_VERSION}; none found "
            f"(looked at $MIMOSA_SANDBOX_PYTHON and python{SANDBOX_PYTHON_VERSION} on "
            f"PATH). Install it (apt install python{SANDBOX_PYTHON_VERSION} "
            f"python{SANDBOX_PYTHON_VERSION}-venv) or set $MIMOSA_SANDBOX_PYTHON."
        )

    def _create_or_reuse_base_venv(self) -> Path:
        """Create the process-wide base venv once, then reuse it across tasks.

        Building a fresh venv and re-installing heavy packages (torch,
        tensorflow) for every task dominated eval time. The venv is created
        once, cached on the class, and reused; per-task working directories stay
        isolated. Creation is lock-guarded. Concurrent per-task installs into the
        shared venv rely on the eval loop being effectively serialized by its
        blocking subprocess calls.
        """
        cls = type(self)
        with cls._shared_venv_lock:
            existing = cls._shared_venv_path
            if existing is not None and (existing / "bin" / "python").exists():
                self.logger.info(f"[SANDBOX] Reusing shared base venv at {existing}")
                return existing

            base_dir = tempfile.mkdtemp(prefix="mimosa_base_venv_")
            venv_path = Path(base_dir) / "venv"
            py = self._resolve_sandbox_python()
            self.logger.info(
                f"[SANDBOX] Creating shared Python {SANDBOX_PYTHON_VERSION} venv at {venv_path} via {py}"
            )
            result = subprocess.run(
                [py, "-m", "venv", str(venv_path)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                shutil.rmtree(base_dir, ignore_errors=True)
                raise RuntimeError(
                    f"venv creation failed via {py}: {result.stderr[:500]} "
                    f"(missing module? apt install python{SANDBOX_PYTHON_VERSION}-venv)"
                )

            python_exe = venv_path / "bin" / "python"
            if not python_exe.exists():
                shutil.rmtree(base_dir, ignore_errors=True)
                raise RuntimeError(
                    f"Virtual environment created but Python executable not found at {python_exe}."
                )

            # Fail loudly on a version mismatch instead of evaluating on the wrong interpreter.
            ver = subprocess.run(
                [str(python_exe), "--version"], capture_output=True, text=True, timeout=10
            )
            got = (ver.stdout or ver.stderr).strip()
            if not got.startswith(f"Python {SANDBOX_PYTHON_VERSION}."):
                shutil.rmtree(base_dir, ignore_errors=True)
                raise RuntimeError(
                    f"Eval venv is {got}, expected Python {SANDBOX_PYTHON_VERSION}.x"
                )

            cls._shared_venv_path = venv_path
            cls._shared_base_dir = base_dir
            atexit.register(cls.cleanup_shared_venv)
            self.logger.info(f"[SANDBOX] Shared base venv ready at {venv_path} ({got})")
            return venv_path

    @classmethod
    def cleanup_shared_venv(cls) -> None:
        """Remove the process-wide base venv (also registered with atexit)."""
        with cls._shared_venv_lock:
            base_dir = cls._shared_base_dir
            if base_dir and Path(base_dir).exists():
                shutil.rmtree(base_dir, ignore_errors=True)
            cls._shared_venv_path = None
            cls._shared_base_dir = None

    def _setup_environment(self) -> None:
        """Ensure base packages and capsule dependencies exist in the shared venv."""
        try:
            # pip skips already-satisfied packages, so this is cheap after the first task.
            self.logger.info("[SANDBOX] Ensuring base packages...")
            self._install_packages(self.base_packages)

            # Analyze capsule code and install additional dependencies
            self._install_capsule_dependencies()

        except Exception as e:
            self.logger.error(f"[SANDBOX] Failed to setup environment: {e}")
            raise

    def _install_packages(self, packages: list[str]) -> None:
        """Install packages in the virtual environment."""
        if not packages:
            return

        cmd = [str(self.pip_exe), "install", "--quiet"] + packages

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                self.logger.warning(f"[SANDBOX] Package installation warnings: {result.stderr[:500]}")
            else:
                self.logger.info(f"[SANDBOX] Installed packages: {', '.join(packages)}")

        except subprocess.TimeoutExpired:
            self.logger.error("[SANDBOX] Package installation timed out")
            raise
        except Exception as e:
            self.logger.error(f"[SANDBOX] Package installation failed: {e}")
            raise

    def _install_capsule_dependencies(self) -> None:
        """Analyze capsule code with pipreqs and install dependencies using pip-tools."""
        self.logger.info(f"[SANDBOX] Analyzing capsule {self.capsule_path.name}...")

        # First, check if there's a requirements.txt in the capsule directory
        requirements_txt = self.capsule_path / "requirements.txt"
        if requirements_txt.exists():
            self.logger.info("[SANDBOX] Found requirements.txt in capsule, installing dependencies...")
            try:
                cmd_install = [str(self.pip_exe), "install", "-r", str(requirements_txt)]
                result = subprocess.run(
                    cmd_install,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    self.logger.info("[SANDBOX] Dependencies from requirements.txt installed successfully")
                    return
                else:
                    self.logger.warning(f"[SANDBOX] Failed to install from requirements.txt: {result.stderr[:500]}")
                    # Continue to try pipreqs as fallback
            except subprocess.TimeoutExpired:
                self.logger.error("[SANDBOX] requirements.txt installation timed out")
            except Exception as e:
                self.logger.error(f"[SANDBOX] requirements.txt installation failed: {e}")
                # Continue to try pipreqs as fallback

        # Find Python files in capsule
        python_files = list(self.capsule_path.glob("*.py"))
        if not python_files:
            self.logger.info("[SANDBOX] No Python files found in capsule")
            return
        try:
            self.logger.info("[SANDBOX] Analyzing code dependencies with pipreqs...")
            temp_path = self.temp_dir / "deps_analysis"
            temp_path.mkdir(exist_ok=True)
            # Copy capsule files to temp directory for analysis
            for file_path in python_files:
                shutil.copy2(file_path, temp_path / file_path.name)
            # Run pipreqs (installed as console script in venv)
            pipreqs_exe = self.venv_path / "bin" / "pipreqs"

            # Check if pipreqs is available (may not be if basic package installation failed)
            if not pipreqs_exe.exists():
                self.logger.warning("[SANDBOX] pipreqs not found in venv, skipping dependency analysis")
                return

            cmd_pipreqs = [
                str(pipreqs_exe),
                "--savepath", str(temp_path / "requirements.in"),
                "--mode", "no-pin",
                str(temp_path)
            ]

            result = subprocess.run(
                cmd_pipreqs,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                self.logger.warning(f"[SANDBOX] pipreqs failed: {result.stderr[:512]}")
                return

            requirements_in = temp_path / "requirements.in"
            if not requirements_in.exists():
                self.logger.info("[SANDBOX] No additional dependencies found")
                return
            # Use pip-tools to compile requirements
            self.logger.info("[SANDBOX] Compiling requirements with pip-tools...")
            requirements_txt = temp_path / "requirements.txt"
            cmd_compile = [
                str(self.python_exe), "-m", "piptools", "compile",
                "--output-file", str(requirements_txt),
                str(requirements_in)
            ]
            result = subprocess.run(
                cmd_compile,
                capture_output=True,
                text=True,
                timeout=180
            )

            if result.returncode != 0:
                self.logger.warning(f"[SANDBOX] pip-tools compile failed: {result.stderr[:512]}")
                # Fall back to direct installation from .in file
                requirements_txt = requirements_in

            # Install the compiled requirements
            self.logger.info("[SANDBOX] Installing additional dependencies...")
            cmd_install = [str(self.pip_exe), "install", "-r", str(requirements_txt)]

            result = subprocess.run(
                cmd_install,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                self.logger.info("[SANDBOX] Additional dependencies installed successfully")
            else:
                self.logger.warning(f"[SANDBOX] Dependency installation warnings: {result.stderr[:500]}")

        except Exception as e:
            self.logger.error(f"[SANDBOX] Dependency analysis/installation failed: {e}")
            # Continue execution even if dependency installation fails

    def _subprocess_env(self) -> dict:
        """Build the env dict for spawned scripts, applying cpu_only if set."""
        env = os.environ.copy()
        if self.cpu_only:
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["TF_CPP_MIN_LOG_LEVEL"] = env.get("TF_CPP_MIN_LOG_LEVEL", "2")
        return env

    def run_generated_code(
        self,
        script_path: Path = None,
        script_name: str = None,
        expected_output: str = "",
        timeout: int = 3600
    ) -> tuple[bool, str]:
        """
        Run the generated code in the capsule to produce output.

        Args:
            eval_script_path: Path to evaluation script (used for smart file selection)
            expected_output: Expected output filename to check for
            timeout: Execution timeout in seconds

        Returns:
            (success: bool, message: str)
        """
        try:
            self.logger.info("[SANDBOX] Running generated code for VER evaluation")

            # Find Python file in capsule
            py_files = list(self.capsule_path.glob("*.py"))
            if not py_files:
                return False, "No Python file found in capsule"
            if script_path is not None and script_path not in py_files:
                self.logger.warning(f"[SANDBOX] Specified script {script_path.name} not found in capsule, using smart file selection")
            # Smart file selection based on eval_script_path
            generated_script = self._select_best_matching_file(py_files, script_name)
            self.logger.info(f"[SANDBOX] Selected generated script: {generated_script.name}")
            # Use the sandbox's persistent temp directory for code execution
            temp_path = self.temp_dir / "execution"
            # Clean up previous execution if any
            if temp_path.exists():
                shutil.rmtree(temp_path)
            temp_path.mkdir(exist_ok=True)

            self._copy_capsule_contents_to_temp(temp_path)

            # SAB output convention is pred_results/; provide it so a program
            # that writes there without mkdir doesn't fail VER on the last line.
            (temp_path / "pred_results").mkdir(exist_ok=True)

            # Run the generated script
            cmd = [str(self.python_exe), generated_script.name]

            result = subprocess.run(
                cmd,
                cwd=str(temp_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._subprocess_env()
            )

            if result.returncode != 0:
                error_msg = f"Generated code failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr[:100000]}"
                self.logger.error(f"[SANDBOX] {error_msg}")
                return False, error_msg

            # Check if expected output was created
            if expected_output:
                # Handle case where expected_output already contains 'pred_results/' prefix
                expected_output_clean = expected_output
                if expected_output.startswith("pred_results/"):
                    expected_output_clean = expected_output[len("pred_results/"):]
                elif expected_output.startswith("pred_results\\"):
                    expected_output_clean = expected_output[len("pred_results\\"):]
                expected_path = temp_path / "pred_results" / expected_output_clean
                if not expected_path.exists():
                    return False, f"Expected output file not created: {expected_output}"

            # Copy results back to capsule if pred_results was created
            pred_results_src = temp_path / "pred_results"
            if pred_results_src.exists():
                pred_results_dst = self.capsule_path / "pred_results"
                if pred_results_dst.exists():
                    shutil.rmtree(pred_results_dst)
                shutil.copytree(pred_results_src, pred_results_dst)
                self.logger.info("[SANDBOX] Copied pred_results back to capsule")

            output = result.stdout.strip()
            self.logger.info("[SANDBOX] Generated code executed successfully")
            return True, f"Code executed. Output: {output[:100000]}"

        except subprocess.TimeoutExpired:
            self.logger.error(f"[SANDBOX] Generated code timeout after {timeout}s")
            return False, f"Code execution timeout after {timeout} seconds"
        except Exception as e:
            self.logger.error(f"[SANDBOX] Generated code error: {str(e)}")
            return False, f"Code execution error: {str(e)}"

    def select_generated_script(self, script_name: str = None) -> Path | None:
        """
        Return the capsule Python file that best matches script_name.

        Single source of truth for "which file is the generated program", so
        VER execution and CBS scoring judge the same file instead of drifting
        onto an arbitrary glob order.

        Args:
            script_name: Reference name to match against (e.g. gold program name)

        Returns:
            Best-matching Python file, or None if the capsule has no .py files
        """
        py_files = list(self.capsule_path.glob("*.py"))
        if not py_files:
            return None
        return self._select_best_matching_file(py_files, script_name)

    def _select_best_matching_file(self, py_files: list[Path], script_name: str = None) -> Path:
        """
        Select the best matching Python file from candidates based on eval script name.

        Uses simple string similarity without heavy models.
        """
        if not script_name or not py_files:
            return py_files[0] if py_files else None
        best_match = None
        best_score = 0

        for py_file in py_files:
            file_name = py_file.name.lower()
            score = self._calculate_similarity_score(script_name.lower(), file_name)
            if score > best_score:
                best_score = score
                best_match = py_file
        if best_score < 0.2:
            return py_files[0]
        return best_match

    def _calculate_similarity_score(self, str1: str, str2: str) -> float:
        """
        Calculate simple string similarity score (0.0 to 1.0).
        Uses substring matching and length ratio.
        """
        if not str1 or not str2:
            return 0.0

        # Exact match gets perfect score
        if str1 == str2:
            return 1.0

        # Substring matching
        shorter = min(str1, str2, key=len)
        longer = max(str1, str2, key=len)

        # Check if shorter is contained in longer
        if shorter in longer:
            return len(shorter) / len(longer)

        # Find longest common substring
        lcs_length = self._longest_common_substring_length(str1, str2)
        if lcs_length > 0:
            # Weight by the proportion of the shorter string that matches
            return lcs_length / len(shorter)

        # No match found
        return 0.0

    def _longest_common_substring_length(self, str1: str, str2: str) -> int:
        """
        Find length of longest common substring using dynamic programming.
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_length = max(max_length, dp[i][j])

        return max_length

    def _copy_capsule_contents_to_temp(self, temp_path: Path) -> None:
        """
        Copy all contents from capsule directory to temp directory.
        Args:
            temp_path: Destination temporary directory path
        """
        copied_files = []
        copied_dirs = []
        errors = []
        total_bytes = 0

        try:
            self.logger.info("=" * 60)
            self.logger.info("[SANDBOX] Starting capsule content copy operation")
            self.logger.info(f"[SANDBOX] Source: {self.capsule_path}")
            self.logger.info(f"[SANDBOX] Destination: {temp_path}")
            if not self.capsule_path.exists():
                self.logger.warning(f"[SANDBOX] Capsule path does not exist: {self.capsule_path}")
                return

            temp_path.mkdir(parents=True, exist_ok=True)

            items = list(self.capsule_path.iterdir())
            for idx, item in enumerate(items, 1):
                dest = temp_path / item.name
                try:
                    if item.is_file():
                        size = item.stat().st_size
                        shutil.copy2(item, dest)
                        copied_files.append((item.name, size))
                        total_bytes += size
                    elif item.is_dir():
                        dir_items = sum(1 for _ in item.rglob('*') if _.is_file())
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                        copied_dirs.append((item.name, dir_items))
                except (shutil.Error, OSError, PermissionError) as e:
                    errors.append((item.name, str(e)))
                    continue
            if copied_files:
                for name, size in copied_files:
                    self.logger.info(f"  {name:<30} {size:>12,} bytes")
            if copied_dirs:
                for name, count in copied_dirs:
                    self.logger.info(f"{name}/ ({count} nested files)")
            if errors:
                self.logger.warning(f"[SANDBOX] Failed items ({len(errors)}):")
                for name, err in errors:
                    self.logger.warning(f"  ⚠️  {name}: {err}")
            self.logger.info(f"[SANDBOX] Total Size: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.2f} MB)")
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.error(f"[SANDBOX] Fatal error during copy: {str(e)}", exc_info=True)
            raise

    def run_eval_script(
        self,
        eval_script_path: Path,
        visual_judge_path: Path = None,
        timeout: int = 600
    ) -> tuple[bool, str]:
        """
        Run a ScienceAgentBench evaluation script.

        The eval script expects:
        - pred_results/ directory in current working directory
        - benchmark/eval_programs/gold_results/ directory for reference data

        Args:
            eval_script_path: Path to evaluation script
            timeout: Execution timeout in seconds

        Returns:
            (success: bool, message: str)

        Raises:
            EvalInfraError: if gold_results is missing, or the task is figure-
                judged but no OPENAI_API_KEY / AZURE_OPENAI_KEY is set. These are
                harness/setup problems, so the task is excluded from metrics
                rather than scored as an SR failure.
        """
        try:
            self.logger.info(f"[SANDBOX] Running eval script: {eval_script_path.name}")

            # Detect figure-judged tasks by an ACTUAL import of the judge, not a
            # loose substring (a comment mentioning it must not force exclusion).
            needs_judge = self._eval_needs_judge(
                eval_script_path.read_text(encoding="utf-8", errors="ignore")
            )
            if needs_judge:
                if visual_judge_path is None or not Path(visual_judge_path).exists():
                    raise EvalInfraError(
                        f"Figure-judged task '{eval_script_path.name}' needs "
                        f"gpt4_visual_judge.py — not available; excluding from metrics"
                    )
                if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")):
                    raise EvalInfraError(
                        f"Figure-judged task '{eval_script_path.name}' needs OPENAI_API_KEY "
                        f"or AZURE_OPENAI_KEY; none set — excluding from metrics"
                    )

            # Use the sandbox's persistent temp directory for eval script execution
            temp_path = self.temp_dir / "eval"
            # Clean up previous eval if any
            if temp_path.exists():
                shutil.rmtree(temp_path)
            temp_path.mkdir(exist_ok=True)

            self._copy_capsule_contents_to_temp(temp_path)

            benchmark_dir = temp_path / "benchmark" / "eval_programs"
            benchmark_dir.mkdir(parents=True, exist_ok=True)

            gold_results_src = eval_script_path.parent / "gold_results"
            if gold_results_src.exists():
                gold_results_dst = benchmark_dir / "gold_results"
                shutil.copytree(gold_results_src, gold_results_dst)
                self.logger.info("[SANDBOX] Copied gold_results for evaluation")
            else:
                raise EvalInfraError(
                    f"gold_results not found next to {eval_script_path.name} "
                    f"({gold_results_src}) — benchmark eval data incomplete"
                )

            shutil.copy2(eval_script_path, temp_path / eval_script_path.name)
            if visual_judge_path:
                shutil.copy2(visual_judge_path, temp_path / visual_judge_path.name)

            # Use virtual environment's Python executable
            cmd = [str(self.python_exe), eval_script_path.name]

            result = subprocess.run(
                cmd,
                cwd=str(temp_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._subprocess_env()
            )

            if result.returncode != 0:
                error_msg = f"Eval script failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr[:100000]}"
                self.logger.error(f"[SANDBOX] {error_msg}")
                return False, error_msg
            output = result.stdout.strip()
            self.logger.debug(f"[SANDBOX] Eval output: {output}")

            return self._parse_eval_output(output)

        except EvalInfraError:
            raise  # infra problem — let the caller exclude the task
        except subprocess.TimeoutExpired:
            self.logger.error(f"[SANDBOX] Eval script timeout after {timeout}s")
            return False, f"Evaluation timeout after {timeout} seconds"
        except Exception as e:
            self.logger.error(f"[SANDBOX] Eval script error: {str(e)}")
            return False, f"Evaluation error: {str(e)}"

    def cleanup(self) -> None:
        """
        Clean up the sandbox temporary directory.

        This should be called when the sandbox is no longer needed to free up disk space.
        """
        if hasattr(self, '_temp_dir_context'):
            self.logger.info(f"[SANDBOX] Cleaning up sandbox temp directory: {self.temp_dir}")
            self._temp_dir_context.cleanup()
            self.logger.info("[SANDBOX] Sandbox temp directory cleaned up successfully")

    def __del__(self):
        """Destructor to ensure cleanup is performed."""
        try:
            self.cleanup()
        except Exception:
            pass

    def _parse_eval_output(self, output: str) -> tuple[bool, str]:
        """
        Parse a ScienceAgentBench eval script's stdout into (success, message).

        Every SAB eval script ends with ``print(eval())`` where ``eval`` returns
        an ``(int_status, message)`` tuple, so the RESULT is the last printable
        line. Scientific libraries (tensorflow, sklearn, rdkit) commonly print
        banners first, so we scan from the last non-empty line for the first
        line that parses as that tuple.

        Unparseable output is treated as failure, never success. The previous
        heuristic returned success whenever the raw text contained "1", "True"
        or "success" — a stray line number or float silently turned a failing
        task into a pass.

        Args:
            output: Captured stdout from the eval script

        Returns:
            (success, message) — success is False if no result tuple is found
        """
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        for line in reversed(lines):
            parsed = self._parse_result_tuple(line)
            if parsed is not None:
                return parsed
        return False, f"No (status, message) result tuple in eval output: {output[:1000]}"

    @staticmethod
    def _eval_needs_judge(eval_text: str) -> bool:
        """True if the eval script actually imports the GPT figure judge.

        Matches a real import line, not any mention — so a comment referencing
        the judge never forces a task to be excluded for lack of an API key.
        """
        return bool(re.search(
            r"^\s*(?:from\s+gpt4_visual_judge\s+import|import\s+gpt4_visual_judge)",
            eval_text, re.MULTILINE,
        ))

    @staticmethod
    def _parse_result_tuple(line: str) -> tuple[bool, str] | None:
        """Return (success, message) if line is a SAB result tuple, else None."""
        try:
            parsed = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return None
        if not (isinstance(parsed, tuple) and len(parsed) >= 2):
            return None
        try:
            success = bool(int(parsed[0]))
        except (ValueError, TypeError):
            success = bool(parsed[0])
        return success, str(parsed[1])


if __name__ == "__main__":
    # Lightweight smoke check: exercise output parsing without building a venv.
    logging.basicConfig(level=logging.INFO)
    _sb = object.__new__(ExecutionSandbox)  # bypass heavy env setup
    assert _sb._parse_eval_output("(1, 'ok')") == (True, "ok")
    assert _sb._parse_eval_output("(0, 'nope')")[0] is False
    # Library banners before the result tuple must not fool the parser
    _noisy = "tensorflow: using CPU\n(1, \"{'data_correctness': True}\")"
    assert _sb._parse_eval_output(_noisy)[0] is True
    # A stray '1' with no result tuple is a failure, not a spurious success
    assert _sb._parse_eval_output("Traceback: error on line 12")[0] is False
    assert _sb._parse_eval_output("")[0] is False
    print("execution_sandbox smoke check passed")
