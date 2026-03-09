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

    # Basic packages to install in every environment (matching ScienceAgentBench)
    BASIC_PACKAGES = [
        "numpy",
        "pandas",
        "matplotlib",
        "torch",  # pytorch
        "tensorflow",
        "rdkit",  # rdkit
        "pipreqs",  # for dependency analysis
        "pip-tools"  # for dependency resolution
    ]

    def __init__(self, capsule_path: Path):
        """
        Initialize execution sandbox and set up virtual environment with dependencies.

        Args:
            capsule_path: Path to capsule directory containing generated code
        """
        self.capsule_path = Path(capsule_path)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create virtual environment
        self.venv_path = self._create_virtual_environment()
        self.python_exe = self.venv_path / "bin" / "python"
        self.pip_exe = self.venv_path / "bin" / "pip"

        # Install basic packages and analyze/setup dependencies
        self._setup_environment()

    def _create_virtual_environment(self) -> Path:
        """Create a virtual environment for isolated execution."""
        temp_dir = Path(tempfile.mkdtemp(prefix="mimosa_sandbox_"))
        venv_path = temp_dir / "venv"

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
        # Find Python files in capsule
        python_files = list(self.capsule_path.glob("*.py"))
        if not python_files:
            self.logger.info("[SANDBOX] No Python files found in capsule")
            return

        try:
            # Use pipreqs to analyze dependencies
            self.logger.info("[SANDBOX] Analyzing code dependencies with pipreqs...")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy capsule files to temp directory for analysis
                for file_path in python_files:
                    shutil.copy2(file_path, temp_path / file_path.name)

                # Run pipreqs
                cmd_pipreqs = [
                    str(self.python_exe), "-m", "pipreqs.pipreqs",
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
                    str(self.python_exe), "-m", "piptools.compile",
                    "--output-file", str(requirements_txt),
                    str(requirements_in)
                ]

                result = subprocess.run(
                    cmd_compile,
                    capture_output=True,
                    text=True,
                    timeout=60
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
        eval_script_path: Path = None,
        expected_output: str = "",
        timeout: int = 300
    ) -> tuple[bool, str]:
        """
        Run the generated code in the capsule to produce output.

        Args:
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

            generated_script = eval_script_path
            self.logger.info(f"[SANDBOX] Found generated script: {generated_script.name}")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
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
                    expected_path = temp_path / "pred_results" / expected_output
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

    def _copy_capsule_contents_to_temp(self, temp_path: Path) -> None:
        """
        Copy all contents from capsule directory to temp directory.
        Args:
            temp_path: Destination temporary directory path
        """
        try:
            self.logger.info("[SANDBOX] Copying all capsule contents to temp directory")

            if not self.capsule_path.exists():
                self.logger.warning(f"[SANDBOX] Capsule path does not exist: {self.capsule_path}")
                return

            for item in self.capsule_path.iterdir():
                dest = temp_path / item.name

                if item.is_file():
                    shutil.copy2(item, dest)
                    self.logger.debug(f"[SANDBOX] Copied file: {item.name}")
                elif item.is_dir():
                    shutil.copytree(item, dest)
                    self.logger.debug(f"[SANDBOX] Copied directory: {item.name}/")

            self.logger.info("[SANDBOX] Successfully copied all capsule contents")

        except Exception as e:
            self.logger.error(f"[SANDBOX] Error copying capsule contents: {str(e)}")
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

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
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
