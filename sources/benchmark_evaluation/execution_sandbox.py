"""
Execution Sandbox - Safe execution utilities for evaluating generated code.

Provides isolated execution environment with automatic dependency management.
"""

from __future__ import annotations

import ast
import atexit
import hashlib
import logging
import os
import re
import signal
import subprocess
import shutil
import tempfile
import threading
from pathlib import Path


logger = logging.getLogger(__name__)

# ScienceAgentBench pins its eval environment to Python 3.10 (config_conda_env.py).
# Old rdkit/deepchem-era wheels do not exist for 3.12, so we match 3.10 to reproduce.
SANDBOX_PYTHON_VERSION = "3.10"

# Version caps mirroring the authors' pinned eval environment. Applied as a pip
# constraints file to the base AND every per-task install, so pipreqs-discovered
# deps (e.g. deepchem) cannot pull an incompatible numpy/rdkit.
PINNED_CONSTRAINTS = [
    "numpy<2.0",
    "scipy<1.14.0",
    "pandas<=1.5.3",
    "matplotlib<3.8.0",
    "torch<=2.3.0",
    "tensorflow<=2.17.0",
    "tf_keras<=2.17.0",
    "rdkit<=2023.09.5",
    "pymatgen<=2024.5.1",
    "oggm<=1.6.1",
    # Beyond the authors' list: transitive deps they leave floating, pinned here
    # because their drift breaks the gold programs at runtime.
    # tensorflow<=2.17 holds ml_dtypes<0.5 while newer jax needs >=0.5
    # (AttributeError: module 'ml_dtypes' has no attribute 'float8_e3m4' at any
    # tensorflow import). pip can't catch it — specs are satisfied either way.
    "jax<=0.4.34",
    "jaxlib<=0.4.34",
    "ml_dtypes<0.5.0",
    # phonopy v3/v4 removed the legacy set_* API (set_force_constants,
    # set_band_structure) the gold programs call; 2.29.x is the benchmark era.
    "phonopy<2.30",
    # chemprop v2 renamed the chemprop_train/chemprop_predict CLIs gold shells
    # out to (a no-op on py3.10 — v2 requires >=3.11 — but pins the era in case
    # the sandbox interpreter moves).
    "chemprop<2.0",
    # opencv 4.11+ requires numpy>=2, conflicting with the numpy<2.0 pin;
    # 4.10.0.84 is the last release built against numpy 1.x (cp37-abi3 wheels
    # cover the py3.10 sandbox on macOS arm64 and Linux).
    "opencv-python<=4.10.0.84",
    "opencv-python-headless<=4.10.0.84",
]

# pipreqs import name -> PyPI package name (authors' handcrafted remaps).
IMPORT_NAME_REMAP = {
    "scvi": "scvi-tools",
    "skimage": "scikit-image",
    "iris": "scitools-iris",
}

# Imports pipreqs may report that must not be installed.
DROP_PACKAGES = {"benchmark"}

# Extra runtime deps some libraries need but pipreqs misses (keyed lowercase).
EXTRA_DEPS = {
    "biopsykit": ["mne"],
    "oggm": ["salem", "tables", "geopandas"],
    "scanpy": ["scikit-misc", "leidenalg"],
}

# Libraries needing bespoke install commands (keyed lowercase); failures are
# treated as infra errors (task excluded), never as generated-code failures.
SPECIAL_CASE_INSTALLS = {
    "deepchem": [["dgl", "-f", "https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html"]],
    "deeppurpose": [["git+https://github.com/bp-kelley/descriptastorus"]],
    "qsprpred": [["papyrus-scaffold-visualizer", "kaleido"]],
}


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
    - Uses a virtual environment to avoid package conflicts
    - Installs the authors' pinned base stack (numpy, scipy, pandas, matplotlib,
      scikit-learn, torch, tensorflow, tf_keras, rdkit) at their exact versions
    - Uses pipreqs + the authors' handcrafted rules to add per-program deps,
      capped by a shared constraints file
    """

    # Authors' pinned base stack (config_conda_env.py); versions matter for repro.
    BASIC_PACKAGES = [
        "numpy<2.0",
        "scipy<1.14.0",
        "pandas<=1.5.3",
        "matplotlib<3.8.0",
        "scikit-learn",
        "torch<=2.3.0",
        "tensorflow<=2.17.0",
        "tf_keras<=2.17.0",
        "rdkit<=2023.09.5",
        "openai==1.54.4",
        # openai 1.54.4 passes the `proxies` kwarg to httpx.Client, which
        # httpx>=0.28 removed — leaving the default (transitive) httpx breaks
        # the GPT-4 visual judge at `OpenAI()` construction. Cap httpx below
        # 0.28 to keep the authors' exact openai pin while restoring the judge.
        "httpx<0.28",
        # pipreqs can NEVER provision cv2: it queries PyPI for a package named
        # "cv2", finds none, and silently drops the import (observed 2026-08-05:
        # BBBC002 task excluded — "No module named 'cv2'"). IMPORT_NAME_REMAP
        # cannot fix this (the remap only sees pipreqs' output, and cv2 never
        # reaches it). Install it in the base venv instead; headless because
        # the sandbox has no display and benchmark code never calls imshow.
        # The <=4.10.0.84 pin mirrors PINNED_CONSTRAINTS (numpy<2.0 compat).
        "opencv-python-headless<=4.10.0.84",
        "pipreqs",
        "pip-tools",
    ]

    # Process-wide base venvs, created once per environment fingerprint and
    # reused across tasks. Keyed by a hash of (base_packages, constraints) so
    # queued runs with differing environments never share (and drift) a venv.
    # Entry: {"venv_path": Path, "base_dir": str, "constraints_file": str}
    _shared_venvs: dict[str, dict] = {}
    _shared_venv_lock = threading.Lock()
    # Serializes pip installs into shared venvs — after the C1 fix the eval
    # loop runs sandbox setup from multiple threads (asyncio.to_thread), so
    # event-loop serialization no longer protects per-task installs.
    _shared_install_lock = threading.Lock()

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
        self._constraints_file: str | None = None  # set by _create_or_reuse_base_venv

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

    def _venv_fingerprint(self) -> str:
        """Hash of (base_packages, constraints) — the shared-venv cache key."""
        h = hashlib.sha256()
        h.update("\n".join(self.base_packages).encode())
        h.update(b"\n--constraints--\n")
        h.update("\n".join(PINNED_CONSTRAINTS).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _run_process(
        cmd: list[str],
        timeout: int,
        cwd: str | None = None,
        env: dict | None = None,
    ) -> subprocess.CompletedProcess:
        """``subprocess.run(capture_output=True, text=True)`` that kills the
        whole process group on timeout.

        Generated scripts may spawn workers (multiprocessing, DataLoader, TF);
        ``subprocess.run(timeout=)`` kills only the direct child, orphaning
        grandchildren that keep burning CPU/RAM (and lose their temp dir when
        the sandbox cleans up). ``start_new_session`` puts the child in its own
        process group so a timeout can SIGKILL the entire group before reaping.
        Raises ``subprocess.TimeoutExpired`` like ``subprocess.run`` does.
        """
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            stdout, stderr = proc.communicate()  # reap
            raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    def _create_or_reuse_base_venv(self) -> Path:
        """Create the process-wide base venv once per env fingerprint, then reuse it.

        Building a fresh venv and re-installing heavy packages (torch,
        tensorflow) for every task dominated eval time. The venv is created
        once per (base_packages, constraints) fingerprint, cached on the
        class, and reused; per-task working directories stay isolated.
        Creation is lock-guarded; per-task installs into the shared venv are
        serialized by _shared_install_lock (see _setup_environment).
        """
        cls = type(self)
        fingerprint = self._venv_fingerprint()
        with cls._shared_venv_lock:
            existing = cls._shared_venvs.get(fingerprint)
            if existing is not None and (existing["venv_path"] / "bin" / "python").exists():
                self.logger.info(f"[SANDBOX] Reusing shared base venv at {existing['venv_path']}")
                self._constraints_file = existing["constraints_file"]
                return existing["venv_path"]

            base_dir = tempfile.mkdtemp(prefix="mimosa_base_venv_")
            venv_path = Path(base_dir) / "venv"
            py = self._resolve_sandbox_python()
            self.logger.info(
                f"[SANDBOX] Creating shared Python {SANDBOX_PYTHON_VERSION} venv at {venv_path} via {py} "
                f"(fingerprint {fingerprint})"
            )
            result = self._run_process(
                [py, "-m", "venv", str(venv_path)],
                timeout=600,
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

            constraints = Path(base_dir) / "constraints.txt"
            constraints.write_text("\n".join(PINNED_CONSTRAINTS) + "\n")

            cls._shared_venvs[fingerprint] = {
                "venv_path": venv_path,
                "base_dir": base_dir,
                "constraints_file": str(constraints),
            }
            self._constraints_file = str(constraints)
            atexit.register(cls.cleanup_shared_venv)
            self.logger.info(f"[SANDBOX] Shared base venv ready at {venv_path} ({got})")
            return venv_path

    @classmethod
    def cleanup_shared_venv(cls) -> None:
        """Remove all process-wide base venvs (also registered with atexit)."""
        with cls._shared_venv_lock:
            for entry in cls._shared_venvs.values():
                base_dir = entry["base_dir"]
                if base_dir and Path(base_dir).exists():
                    shutil.rmtree(base_dir, ignore_errors=True)
            cls._shared_venvs.clear()

    # Small tools that must be installed before per-task dep discovery can run.
    _DISCOVERY_TOOLS = ("pipreqs", "pip-tools")

    def _setup_environment(self) -> None:
        """Ensure base packages and capsule dependencies exist in the shared venv.

        Install order: the pipreqs/pip-tools discovery tools first (needed to
        analyze the capsule), then the rest of the base stack and the per-task
        deps in ONE pip transaction — pip co-resolves the full graph instead of
        drifting across separate installs (matching the authors' merged
        ``pip install -r`` in config_conda_env.py).
        """
        try:
            # Serialize installs into the shared venv across worker threads.
            with type(self)._shared_install_lock:
                tools = [p for p in self.base_packages if p in self._DISCOVERY_TOOLS]
                if tools:
                    self._install_packages(tools)

                per_task, present = self._discover_capsule_dependencies()

                rest = [p for p in self.base_packages if p not in self._DISCOVERY_TOOLS]
                # Drop per-task entries already covered (pinned) by the base set —
                # pip errors on a requirement named twice in one command.
                rest_names = {self._req_name(p) for p in rest}
                merged = rest + [p for p in per_task if self._req_name(p) not in rest_names]
                # pip skips already-satisfied packages, so this is cheap after the first task.
                self.logger.info("[SANDBOX] Ensuring base + per-program packages...")
                self._install_packages(merged)
                self._run_special_case_installs(present)

        except Exception as e:
            self.logger.error(f"[SANDBOX] Failed to setup environment: {e}")
            raise

    @staticmethod
    def _req_name(req: str) -> str:
        """Canonical package name of a requirement specifier (for de-dup)."""
        return re.split(r"[=<>!~\[; ]", req.strip(), 1)[0].strip().lower().replace("_", "-")

    def _discover_capsule_dependencies(self) -> tuple[list[str], list[str]]:
        """Return (per-program packages, present names) via pipreqs + SAB rules.

        A capsule-provided requirements.txt wins: it is installed as-is (still
        constraint-capped) and discovery stops there.
        """
        # A capsule-provided requirements.txt wins (still constraint-capped).
        capsule_reqs = self.capsule_path / "requirements.txt"
        if capsule_reqs.exists():
            self._pip_install_requirements(capsule_reqs)
            return [], []

        if not list(self.capsule_path.glob("*.py")):
            self.logger.info("[SANDBOX] No Python files in capsule")
            return [], []

        requirements_in = self._run_pipreqs()
        if requirements_in is None:
            return [], []
        packages, present = self._apply_dependency_rules(requirements_in.read_text())
        if packages:
            self.logger.info(f"[SANDBOX] Per-program deps: {', '.join(packages)}")
        return packages, present

    def _install_packages(self, packages: list[str]) -> None:
        """Install packages in the venv, capped by the shared constraints file.

        A non-zero pip return code raises EvalInfraError: a broken environment
        is not an agent failure, so the task is excluded from metrics.
        """
        if not packages:
            return

        cmd = [str(self.pip_exe), "install", "--quiet"]
        if self._constraints_file:
            cmd += ["-c", self._constraints_file]
        cmd += packages

        try:
            result = self._run_process(cmd, timeout=900)
            if result.returncode != 0:
                raise EvalInfraError(
                    f"Required package install failed (exit {result.returncode}) "
                    f"for {packages}: {result.stderr[:500]}"
                )
            self.logger.info(f"[SANDBOX] Installed: {', '.join(packages)}")
        except subprocess.TimeoutExpired:
            self.logger.error("[SANDBOX] Package installation timed out")
            raise
        except EvalInfraError:
            raise
        except Exception as e:
            self.logger.error(f"[SANDBOX] Package installation failed: {e}")
            raise

    def _pip_install_requirements(self, req_file: Path) -> None:
        """pip install -r req_file, capped by the shared constraints file."""
        cmd = [str(self.pip_exe), "install", "-r", str(req_file)]
        if self._constraints_file:
            cmd += ["-c", self._constraints_file]
        result = self._run_process(cmd, timeout=900)
        if result.returncode == 0:
            self.logger.info(f"[SANDBOX] Installed deps from {req_file.name}")
        else:
            raise EvalInfraError(
                f"Required dep install from {req_file.name} failed "
                f"(exit {result.returncode}): {result.stderr[:500]}"
            )

    def _run_pipreqs(self) -> Path | None:
        """Run pipreqs over the capsule's Python files; return requirements.in or None."""
        pipreqs_exe = self.venv_path / "bin" / "pipreqs"
        if not pipreqs_exe.exists():
            self.logger.warning("[SANDBOX] pipreqs missing, skipping dependency analysis")
            return None
        temp_path = self.temp_dir / "deps_analysis"
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)
        for py in self.capsule_path.glob("*.py"):
            shutil.copy2(py, temp_path / py.name)
        req_in = temp_path / "requirements.in"
        cmd = [str(pipreqs_exe), "--savepath", str(req_in), "--mode", "no-pin", str(temp_path)]
        result = self._run_process(cmd, timeout=600)
        if result.returncode != 0 or not req_in.exists():
            self.logger.warning(f"[SANDBOX] pipreqs found no deps: {result.stderr[:300]}")
            return None
        return req_in

    @staticmethod
    def _apply_dependency_rules(req_text: str) -> tuple[list[str], list[str]]:
        """Apply SAB rules (remap/drop/extras); return (install_list, present_names)."""
        packages, present = [], []
        dropped = {d.lower() for d in DROP_PACKAGES}
        for raw in req_text.splitlines():
            name = re.split(r"[=<>!~\[; ]", raw.strip(), 1)[0].strip()
            if not name or name.startswith("#") or name.lower() in dropped:
                continue
            present.append(name)
            packages.append(IMPORT_NAME_REMAP.get(name.lower(), name))
            packages.extend(EXTRA_DEPS.get(name.lower(), []))
        seen, out = set(), []
        for pkg in packages:  # de-dup, preserve order
            if pkg not in seen:
                seen.add(pkg)
                out.append(pkg)
        return out, present

    def _run_special_case_installs(self, present: list[str]) -> None:
        """Run bespoke installs for libs pipreqs can't fully provision.

        A failure raises EvalInfraError: a dependency that cannot be provisioned
        is an infra problem (task excluded), not a generated-code failure.
        """
        for name in present:
            for extra in SPECIAL_CASE_INSTALLS.get(name.lower(), []):
                self.logger.info(f"[SANDBOX] Special-case install for {name}: {' '.join(extra)}")
                self._install_packages(extra)

    def _subprocess_env(self) -> dict:
        """Build the env dict for spawned scripts, applying cpu_only if set."""
        env = os.environ.copy()
        # Put the venv's bin/ first on PATH so console scripts installed into the
        # sandbox (e.g. chemprop_train) resolve when generated code shells out to
        # them by name.
        env["PATH"] = str(self.venv_path / "bin") + os.pathsep + env.get("PATH", "")
        if self.cpu_only:
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["TF_CPP_MIN_LOG_LEVEL"] = env.get("TF_CPP_MIN_LOG_LEVEL", "2")
        return env

    def run_generated_code(
        self,
        script_path: Path = None,
        script_name: str = None,
        expected_output: str | list[str] = "",
        timeout: int = 3600
    ) -> tuple[bool, str]:
        """
        Run the generated code in the capsule to produce output.

        Args:
            eval_script_path: Path to evaluation script (used for smart file selection)
            expected_output: Expected output filename(s) to check for — a single
                name or the full list of files the task's checker consumes
            timeout: Execution timeout in seconds

        Returns:
            (success: bool, message: str)

        Raises:
            EvalInfraError: if the program dies on a missing/broken import —
                that is a provisioning gap (task excluded), not invalid code.
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

            result = self._run_process(
                cmd,
                cwd=str(temp_path),
                timeout=timeout,
                env=self._subprocess_env()
            )

            if result.returncode != 0:
                # A missing/broken import means provisioning failed, not that the
                # code is invalid — exclude the task instead of failing VER.
                import_error = re.search(
                    r"(?:ModuleNotFoundError|ImportError):[^\n]*", result.stderr or ""
                )
                if import_error:
                    raise EvalInfraError(
                        f"Generated code import failed — provisioning gap: "
                        f"{import_error.group(0)}"
                    )
                error_msg = f"Generated code failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr[:100000]}"
                self.logger.error(f"[SANDBOX] {error_msg}")
                return False, error_msg

            # Check that every expected output was created
            expected_outputs = (
                [expected_output] if isinstance(expected_output, str) else list(expected_output)
            )
            missing = []
            for name in expected_outputs:
                if not name:
                    continue
                # Handle case where the name already contains 'pred_results/' prefix
                clean = name
                if clean.startswith("pred_results/"):
                    clean = clean[len("pred_results/"):]
                elif clean.startswith("pred_results\\"):
                    clean = clean[len("pred_results\\"):]
                if not (temp_path / "pred_results" / clean).exists():
                    missing.append(name)
            if missing:
                return False, f"Expected output file(s) not created: {', '.join(missing)}"

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

        except EvalInfraError:
            raise  # infra problem — let the caller exclude the task
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

            result = self._run_process(
                cmd,
                cwd=str(temp_path),
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
    # dependency-rule mapping (remap, drop, extra deps)
    _pkgs, _present = _sb._apply_dependency_rules("scvi\nskimage\nbenchmark\nbiopsykit\ndeepchem\n")
    assert "scvi-tools" in _pkgs and "scikit-image" in _pkgs
    assert "benchmark" not in _pkgs and "benchmark" not in _present
    assert "mne" in _pkgs and "deepchem" in _present  # biopsykit adds mne
    print("execution_sandbox smoke check passed")
