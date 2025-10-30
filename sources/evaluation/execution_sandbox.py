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
from pathlib import Path
from typing import Tuple


logger = logging.getLogger(__name__)


class ExecutionSandbox:
    """
    Execution sandbox for safely running generated code with dependency management.
    """
    
    def __init__(self, capsule_path: Path):
        """
        Initialize execution sandbox and install dependencies.
        
        Args:
            capsule_path: Path to capsule directory containing generated code
        """
        self.capsule_path = Path(capsule_path)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self._install_dependencies()
    
    def _install_dependencies(self) -> None:
        """
        Install dependencies from requirements.txt if present in capsule.
        """
        requirements_file = self.capsule_path / "requirements.txt"
        
        if not requirements_file.exists():
            self.logger.info(f"[SANDBOX] No requirements.txt found in {self.capsule_path.name}")
            return
        try:
            self.logger.info(f"[SANDBOX] Installing dependencies from {requirements_file.name}")
            cmd_install = [
                sys.executable, "-m", "pip", "install",
                "-r", str(requirements_file),
                "--quiet"
            ]
            
            result = subprocess.run(
                cmd_install,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                self.logger.info("[SANDBOX] Dependencies installed successfully")
            else:
                self.logger.warning(f"[SANDBOX] Dependency installation warnings: {result.stderr[:200]}")
        except Exception as e:
            raise e
    
    def execute_safely(
        self,
        script_path: Path,
        expected_output: str = "",
        timeout: int = 600
    ) -> tuple[bool, str, bool]:
        """
        Execute a Python script safely with timeout and error handling.
        Args:
            script_path: Path to Python script to execute
            expected_output: Expected output file path (relative to capsule)
            timeout: Execution timeout in seconds
        Returns:
            (success: bool, message: str, output_created: bool)
        """
        try:
            self.logger.info(f"[SANDBOX] Executing script: {script_path.name}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                self._copy_capsule_contents_to_temp(temp_path)

                python_exe = sys.executable
                cmd = [python_exe, temp_path / script_path.name]
            
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=os.environ.copy()
                )
                if result.returncode != 0:
                    return self._handle_execution_failure(
                        result,
                        expected_output
                    )
                return self._handle_execution_success(
                    expected_output
                )
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"[SANDBOX] Execution timeout after {timeout}s")
            return False, f"Timeout after {timeout} seconds", False
            
        except Exception as e:
            self.logger.error(f"[SANDBOX] Execution error: {str(e)}")
            return False, f"Execution error: {str(e)}", False
    
    def _handle_execution_failure(
        self,
        result: subprocess.CompletedProcess,
    ) -> tuple[bool, str, bool]:
        """Handle script execution failure with intelligent warning detection."""
        
        self.logger.error(f"[SANDBOX] Script execution failed with return code {result.returncode}")
        stderr_file = self.capsule_path / "execution_error.log"
        stdout_file = self.capsule_path / "execution_output.log"
        
        with open(stderr_file, 'w', encoding='utf-8') as f:
            f.write(result.stderr if result.stderr else "(no stderr)")
        with open(stdout_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout if result.stdout else "(no stdout)")
        error_msg = f"Script exited with code {result.returncode}"
        if result.stderr:
            stderr_lines = result.stderr.strip().split('\n')
            relevant_lines = stderr_lines[-16:]
            error_msg += "\n\nLast stderr lines:\n" + "\n".join(relevant_lines)
        error_msg += f"\n\nFull logs saved to: {stderr_file.name} and {stdout_file.name}"
        return False, error_msg, False
    
    def _handle_execution_success(
        self,
        expected_output: str
    ) -> tuple[bool, str, bool]:
        """Handle successful script execution."""
        output_created = False
        if expected_output:
            output_path = self.capsule_path / expected_output
            output_created = output_path.exists()
            if not output_created:
                self.logger.warning(f"[SANDBOX] Expected output not found: {expected_output}")
        self.logger.info("[SANDBOX] Execution successful")
        return True, "Execution completed successfully", output_created
    
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
                
                python_exe = sys.executable
                cmd = [python_exe, eval_script_path.name]
                
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
                        error_msg += f": {result.stderr[:1024]}"
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
