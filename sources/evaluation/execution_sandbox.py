"""
Execution Sandbox - Safe execution utilities for evaluating generated code.

Provides isolated execution environment with automatic dependency management.
"""

import logging
import os
import subprocess
import shutil
import sys
import tempfile
import venv
from pathlib import Path
from typing import Tuple


logger = logging.getLogger(__name__)


class ExecutionSandbox:
    """
    Execution sandbox for safely running generated code with dependency management.

    Follows ScienceAgentBench approach:
    - Uses virtual environment to avget_eval_script_pathoid package conflicts
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

    def __init__(self, capsule_path: Path):
        """
        Initialize execution sandbox and set up virtual environment with dependencies.

        Args:
            capsule_path: Path to capsule directory containing generated code
        """
        self.capsule_path = Path(capsule_path)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create a single temporary directory for this sandbox instance
        # This will be used for venv, dependency analysis, and code execution
        self._temp_dir_context = tempfile.TemporaryDirectory(prefix="mimosa_sandbox_")
        self.temp_dir = Path(self._temp_dir_context.name)

        # Create virtual environment inside the temp directory
        self.venv_path = self._create_virtual_environment()
        self.python_exe = self.venv_path / "bin" / "python"
        self.pip_exe = self.venv_path / "bin" / "pip"

        # Install basic packages and analyze/setup dependencies
        self._setup_environment()

    def _create_virtual_environment(self) -> Path:
        """Create a virtual environment for isolated execution."""
        venv_path = self.temp_dir / "venv"

        self.logger.info(f"[SANDBOX] Creating virtual environment at {venv_path}")

        try:
            venv.create(venv_path, with_pip=True)
            self.logger.info("[SANDBOX] Virtual environment created successfully")
            return venv_path
        except Exception as e:
            self.logger.error(f"[SANDBOX] Failed to create virtual environment: {e}")
            raise

    def _setup_environment(self) -> None:
        """Set up the virtual environment with basic packages and capsule dependencies."""
        try:
            # Install basic packages
            self.logger.info("[SANDBOX] Installing basic packages...")
            self._install_packages(self.BASIC_PACKAGES)

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
                timeout=300
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
                    timeout=300
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
                timeout=60
            )

            if result.returncode != 0:
                self.logger.warning(f"[SANDBOX] pipreqs failed: {result.stderr[:200]}")
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
                self.logger.warning(f"[SANDBOX] pip-tools compile failed: {result.stderr[:200]}")
                # Fall back to direct installation from .in file
                requirements_txt = requirements_in

            # Install the compiled requirements
            self.logger.info("[SANDBOX] Installing additional dependencies...")
            cmd_install = [str(self.pip_exe), "install", "-r", str(requirements_txt)]

            result = subprocess.run(
                cmd_install,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                self.logger.info("[SANDBOX] Additional dependencies installed successfully")
            else:
                self.logger.warning(f"[SANDBOX] Dependency installation warnings: {result.stderr[:500]}")

        except Exception as e:
            self.logger.error(f"[SANDBOX] Dependency analysis/installation failed: {e}")
            # Continue execution even if dependency installation fails

    def run_generated_code(
        self,
        script_path: Path = None,
        script_name: str = None,
        expected_output: str = "",
        timeout: int = 1000
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
            if script_path not in py_files:
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

            # Run the generated script
            cmd = [str(self.python_exe), generated_script.name]

            result = subprocess.run(
                cmd,
                cwd=str(temp_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy()
            )

            if result.returncode != 0:
                error_msg = f"Generated code failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr[:2048]}"
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
            return True, f"Code executed successfully. Output: {output[:200]}"

        except subprocess.TimeoutExpired:
            self.logger.error(f"[SANDBOX] Generated code timeout after {timeout}s")
            return False, f"Code execution timeout after {timeout} seconds"
        except Exception as e:
            self.logger.error(f"[SANDBOX] Generated code error: {str(e)}")
            return False, f"Code execution error: {str(e)}"

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
                    self.logger.info(f"  📄 {name:<30} {size:>12,} bytes")
            if copied_dirs:
                for name, count in copied_dirs:
                    self.logger.info(f"  📁 {name}/ ({count} nested files)")
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
        timeout: int = 60
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
        """
        try:
            self.logger.info(f"[SANDBOX] Running eval script: {eval_script_path.name}")

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
                self.logger.warning("[SANDBOX] Could not find gold results.")
                return False, "Failed to find gold results folder."

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
                env=os.environ.copy()
            )

            if result.returncode != 0:
                error_msg = f"Eval script failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr[:4096]}"
                self.logger.error(f"[SANDBOX] {error_msg}")
                return False, error_msg
            output = result.stdout.strip()
            self.logger.debug(f"[SANDBOX] Eval output: {output}")

            return self._parse_eval_output(output)

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
        self.cleanup()

    def _parse_eval_output(self, output: str) -> tuple[bool, str]:
        """Parse evaluation script output."""
        try:
            import ast
            parsed = ast.literal_eval(output)
            if isinstance(parsed, tuple) and len(parsed) >= 2:
                success = bool(parsed[0])
                message = str(parsed[1])
                return success, message
            else:
                if output.startswith("(1,") or output.startswith("(True,"):
                    return True, output
                else:
                    return False, output
        except (ValueError, SyntaxError):
            if "1" in output or "True" in output or "success" in output.lower():
                return True, output
            else:
                return False, output
