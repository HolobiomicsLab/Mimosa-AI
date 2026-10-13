"""Fresh re-execution runner for the policy_bt verifier (plan §1 P1, §4 3b).

The workflow runner executes the Mimosa LangGraph workflow and produces no
VER facts; this module is the verifier's OWN re-execution channel. For one
generation it:

1. hashes a guard manifest of the ORIGINAL workspace (never mutated),
2. copies the workspace into a verifier-owned task sandbox
   (``<task_sandbox>/runs/<uuid>/``),
3. provisions a LOCAL ``./venv`` at the task sandbox root (NOT the offline
   parent-folder shared-venv convention), installing the workspace's
   ``requirements.txt`` — or an AST import scan of the main script when
   absent — under the era pins imported from
   ``sources.benchmark_evaluation/execution_sandbox.PINNED_CONSTRAINTS``,
4. runs the main script (largest non-test/non-helper ``.py``) with a
   wall-clock timeout and whole-process-group kill (macOS orphan lesson),
5. records the execution ledger: exit code, wall time, timeout flag, error
   signatures, log tail and the mtime-based ``produced_this_run`` artifact
   set (the stale-artifact flag),
6. asserts the original workspace manifest is unchanged before returning.

The venv is reused across generations of the same task when
``venv_reuse`` is set (pip skips satisfied requirements, so incremental
installs are cheap).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from sources.benchmark_evaluation.execution_sandbox import (
        DROP_PACKAGES,
        EXTRA_DEPS,
        IMPORT_NAME_REMAP,
        PINNED_CONSTRAINTS,
    )

    from .logging_utils import StageLogger
    from .recon import is_infra_name
except ImportError:  # direct script execution (uv run reexec.py <folder>)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from logging_utils import StageLogger
    from recon import is_infra_name

    from sources.benchmark_evaluation.execution_sandbox import (
        DROP_PACKAGES,
        EXTRA_DEPS,
        IMPORT_NAME_REMAP,
        PINNED_CONSTRAINTS,
    )

ERROR_PAT = re.compile(
    r"(Traceback \(most recent call last\)|^\w*(Error|Exception)\b|CUDA out of memory"
    r"|Killed|Segmentation fault|AssertionError|ModuleNotFoundError)",
    re.M,
)

#: Dirs excluded from the source-guard manifest and from fresh-artifact
#: scans (contents are never evidence; presence is enough) — keeps the
#: guard fast on 100MB+ model dirs, and keeps the local venv out.
MANIFEST_SKIP_DIRS = {
    "model_checkpoints",
    "best_model_dir",
    "graphconv_model_dir",
    "model_dir",
    "models",
    "model_checkpoints_smoke",
    "__pycache__",
    "venv",
    ".venv",
}
HASH_HEAD = 64 * 1024  # head-hash guard: first 64 KiB + size per file
FRESH_SHA_CAP = 8 << 20  # full-hash only files <= 8MB
LOG_TAIL_LINES = 40
#: Helper/test script prefixes never selected as the main script.
NON_MAIN_PREFIXES = ("test_", "verify_", "probe_", "conftest")

#: Import names whose PyPI project differs and that pipreqs-era remaps
# miss (P1.3: sklearn/PIL/cv2/yaml/Bio imports used to install nothing).
COMMON_IMPORT_REMAP = {
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "yaml": "pyyaml",
    "Bio": "biopython",
}

#: A contract ``python <file>.py [args]`` invocation inside run.sh/README.
#: Lines are tokenized (shlex + operator splitting) and parsed by
#: :func:`_python_command` (N5: the old regex passed ``>``, ``out``,
#: ``&&`` ... as argv to the script and never matched ``python -u x.py``
#: / ``python -m pkg``).
_PY_TOKEN = re.compile(r"^python\d*(?:\.\d+)?$")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_]\w*=")
_REDIRECT_HEAD = re.compile(r"^(?:\d*>>|\d*>|<>|<<|&>>?|>|<)")
_SHELL_CHAIN_OPS = frozenset({"&&", "||", ";", "|"})
_OP_SPLIT = re.compile(r"(&&|\|\||;|\||>>|<<|<>|&>>|&>|\d+>>|\d*>|>|<)")
_INTERP_FLAGS_SKIP = frozenset({"-c", "--command"})
_MODULE_FLAGS = frozenset({"-m", "--module"})
_PIP_LINE = re.compile(r"\bpip3?\s+install\b")

_STDLIB = set(getattr(sys, "stdlib_module_names", ()))


def json_dump(obj: Any) -> str:
    """Compact-but-readable JSON dump used for all package artifacts."""
    return json.dumps(obj, indent=1, sort_keys=True, default=str)


def sha256_file(p: Path, head_only: bool = False) -> str:
    """SHA-256 of a file (head+size only when *head_only*)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if head_only:
            h.update(f.read(HASH_HEAD))
            h.update(str(p.stat().st_size).encode())
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def frozen_manifest(d: Path) -> dict[str, list]:
    """relpath -> [size, head_sha256] guard manifest (heavy dirs skipped)."""
    man: dict[str, list] = {}
    if not d.exists():
        return man
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(d).as_posix()
        if any(part in MANIFEST_SKIP_DIRS for part in Path(rel).parts):
            continue
        try:
            man[rel] = [p.stat().st_size, sha256_file(p, head_only=True)]
        except OSError:
            continue
    return man


def select_main_script(root: Path) -> Path | None:
    """Largest non-test/non-helper top-level ``.py`` in *root* (or None).

    FALLBACK only (P1.1): the deliverable contract is ``run.sh`` / the
    README — see :func:`contract_commands`.
    """
    candidates = [
        p
        for p in root.glob("*.py")
        if not p.name.startswith(NON_MAIN_PREFIXES) and not p.name.startswith(".")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def clone_tree(src: Path, dst: Path) -> None:
    """Copy *src* to *dst* (CoW clonefile on macOS, plain copy elsewhere).

    N11: the destination is removed with ``ignore_errors`` (a partial
    ``cp -c`` copy must not poison the next attempt with
    ``FileExistsError``) and the fallback copytree preserves symlinks
    without following dangling ones.
    """
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    if sys.platform == "darwin":
        r = subprocess.run(
            ["cp", "-c", "-R", str(src), str(dst)],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return
        # clonefile unsupported mid-tree -> plain copy
    shutil.copytree(src, dst, symlinks=True, ignore_dangling_symlinks=True)


def _tokenize_line(ln: str) -> list[str]:
    """Shell-ish token stream of one script line (N5).

    shlex handles quoting; embedded shell operators (``&&``, ``;``,
    redirects) are then split out as standalone tokens so chain groups
    and argv cuts see them (``b.py;`` and ``a.py>out`` glue otherwise).
    """
    try:
        toks = shlex.split(ln)
    except ValueError:
        return []  # unbalanced quotes: not a parseable contract line
    out: list[str] = []
    for t in toks:
        for part in _OP_SPLIT.split(t):
            if part:
                out.append(part)
    return out


def _chain_groups(tokens: list[str]) -> list[list[str]]:
    """Split a token stream at shell chain operators (N5)."""
    groups: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SHELL_CHAIN_OPS:
            groups.append([])
        else:
            groups[-1].append(tok)
    return [g for g in groups if g]


def _python_command(tokens: list[str]) -> tuple[str, list[str]] | None:
    """One shell command group -> ``(script_slot, args)`` when it is a
    python invocation, else ``None`` (N5).

    Accepts leading env assignments and ``uv run`` wrappers, interpreter
    flags before the target (``python -u x.py``), and ``-m <module>``
    invocations; argv collection CUTS at shell redirections so
    ``python a.py > out && python b.py`` runs exactly ``a.py`` then
    ``b.py`` — never ``a.py '>' 'out'``. The script slot may hold
    interpreter flags plus the target (``"-u x.py"``, ``"-m pkg"``); the
    runner splits it on whitespace before the script path.
    """
    i = 0
    while i < len(tokens) and _ENV_ASSIGN.match(tokens[i]):
        i += 1
    if tokens[i : i + 2] == ["uv", "run"]:
        i += 2
    if i >= len(tokens) or not _PY_TOKEN.match(tokens[i]):
        return None
    i += 1

    def _args_from(j: int) -> list[str]:
        out: list[str] = []
        for tok in tokens[j:]:
            if tok == "#" or _REDIRECT_HEAD.match(tok):
                break  # redirection targets / comments are not argv
            out.append(tok)
        return out

    if i < len(tokens) and tokens[i] in _MODULE_FLAGS:
        if i + 1 >= len(tokens):
            return None
        return f"-m {tokens[i + 1]}", _args_from(i + 2)
    pre: list[str] = []
    while i < len(tokens) and tokens[i].startswith("-") and len(tokens[i]) > 1:
        if tokens[i] in _INTERP_FLAGS_SKIP:
            return None  # inline code: no file contract to honour
        pre.append(tokens[i])
        i += 1
    if i >= len(tokens):
        return None
    target = tokens[i].strip("\"'")
    if not target or not target.endswith(".py"):
        return None
    script = " ".join([*pre, target]) if pre else target
    return script, _args_from(i + 1)


def contract_commands(root: Path) -> tuple[str, list[tuple[str, list[str]]]]:
    """Python invocations demanded by ``run.sh`` (then ``README*``).

    The deliverable contract, not the largest ``.py`` (P1.1: the HP run's
    gen 3 re-executed a STALE leftover script because ``run.sh`` named a
    different, smaller one). Each line is shlex-tokenized and split at
    ``&&``/``||``/``;``/``|`` chains; redirections cut the argv;
    ``pip install`` lines are skipped — installs happen up front under
    the era pins (P1.2).

    Args:
        root: The sandbox copy whose contract files to parse.

    Returns:
        ``(source, commands)`` where *source* is ``"run.sh"``, ``"readme"``
        or ``""`` and *commands* is ``[(script_relpath, args), ...]`` in
        file order (empty when no contract invocation was found).
    """
    for source, pattern in (("run.sh", "run.sh"), ("readme", "README*")):
        for sh in sorted(root.glob(pattern)):
            if not sh.is_file():
                continue
            try:
                txt = sh.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            cmds: list[tuple[str, list[str]]] = []
            for ln in txt.splitlines():
                if _PIP_LINE.search(ln):
                    continue  # installs are done under the era pins
                for group in _chain_groups(_tokenize_line(ln)):
                    cmd = _python_command(group)
                    if cmd is not None:
                        cmds.append(cmd)
            if cmds:
                return source, cmds
    return "", []


def scan_imports(script: Path) -> list[str]:
    """Pip package names for the top-level imports of *script*.

    stdlib and LOCAL modules AND local package dirs are skipped; import
    names are remapped through the common table (P1.3: PIL→pillow,
    sklearn→scikit-learn, cv2→opencv-python-headless, yaml→pyyaml,
    Bio→biopython) then the benchmark era table, expanded with their
    ``EXTRA_DEPS`` runtime requirements. Sorted, deduplicated.
    """
    try:
        tree = ast.parse(script.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    local = {p.stem for p in script.parent.glob("*.py")}
    local |= {
        d.name
        for d in script.parent.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    }
    pkgs: set[str] = set()
    for name in sorted(roots):
        if name in _STDLIB or name in local or name.lower() in DROP_PACKAGES:
            continue
        pkg = COMMON_IMPORT_REMAP.get(name) or IMPORT_NAME_REMAP.get(name, name)
        pkgs.add(pkg)
        pkgs.update(EXTRA_DEPS.get(pkg.lower(), ()))
    return sorted(pkgs)


def scan_imports_all(root: Path) -> list[str]:
    """Union import scan over EVERY top-level ``.py`` in *root* (P1.3)."""
    pkgs: set[str] = set()
    for p in sorted(root.glob("*.py")):
        if p.name.startswith(NON_MAIN_PREFIXES):
            continue
        pkgs.update(scan_imports(p))
    return sorted(pkgs)


def _is_pip_requirements(path: Path) -> bool:
    """True if *path* looks like a pip requirements file (not prose).

    Agents following a goal that says "save a requirement.txt" sometimes
    write a documentation/prose file instead of pip package specs (HP
    folding 2026-10-10: the file described the algorithm in 80+ char
    paragraphs). ``pip install -r`` then fails on the prose and the
    re-execution is marked as a pip failure even though the script needs
    no packages. This check detects the mismatch and skips the install.

    Heuristics (any one rejects):
    - Average non-comment line length > 60 chars (prose)
    - A section header line (``====``, ``----`` under a text line)
    - Fewer than 50% of non-comment lines match a pip-spec pattern
    """
    try:
        text = path.read_text(errors="replace")[:4000]
    except OSError:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    content = [ln for ln in lines if not ln.startswith("#")]
    if not content:
        return True  # comment-only file is a valid (empty) pip requirements
    # Section headers: a line of only =, -, or ~ characters
    if any(re.fullmatch(r"[=\-~]{3,}", ln) for ln in content):
        return False
    # Long prose lines
    avg_len = sum(len(ln) for ln in content) / len(content)
    if avg_len > 60:
        return False
    # Pip spec pattern: package, package>=ver, package==ver, -r, --flag, etc.
    spec_re = re.compile(
        r"^(?:-[rce]\s+\S+|--\w[\w-]*|"
        r"[A-Za-z0-9][A-Za-z0-9._\[\]-]*\s*(?:[<>=!~]=?\s*[\w.,]+)?)\s*$"
    )
    matching = sum(1 for ln in content if spec_re.match(ln))
    return matching >= len(content) * 0.5


def _find_requirements_file(copy_dir: Path) -> Path | None:
    """The workspace's pip requirements file (P1.2), content-validated.

    ``requirements*.txt`` (sorted) first, then ``requirement.txt`` — the
    HP goal's singular spelling. Only files that pass _is_pip_requirements
    (not prose documentation) count.
    """
    cands = sorted(copy_dir.glob("requirements*.txt"))
    singular = copy_dir / "requirement.txt"
    if singular.exists():
        cands.append(singular)
    for c in cands:
        try:
            if c.stat().st_size > 0 and _is_pip_requirements(c):
                return c
        except OSError:
            continue
    return None


class ReexecRunner:
    """Re-execute generations of ONE task inside a verifier-owned sandbox.

    Attributes:
        task_sandbox: Verifier-owned root for this task (``runs/`` +
            the local ``venv/`` + ``constraints.txt`` live here).
        runs_dir: Per-generation sandbox copies (``runs/<uuid>/``).
        venv_path: The LOCAL ``./venv`` at the task sandbox root.
        timeout_s: Wall-clock cap for the main-script run.
        venv_reuse: Keep the venv across generations (incremental pip).
    """

    def __init__(
        self,
        task_sandbox: Path,
        timeout_s: int = 900,
        venv_reuse: bool = True,
        install_timeout_s: int = 900,
        python_exe: str | None = None,
        log: StageLogger | None = None,
    ) -> None:
        """Prepare (but not create) the task sandbox layout.

        Args:
            task_sandbox: Root directory owned by the verifier for this
                task; created on first use.
            timeout_s: Wall-clock cap per main-script execution.
            venv_reuse: Reuse one local venv across the task's
                generations (era pins + incremental installs).
            install_timeout_s: Wall-clock cap per pip install.
            python_exe: Interpreter used to build the venv (defaults to
                ``$MIMOSA_POLICY_BT_PYTHON`` then ``sys.executable``).
            log: Optional stage logger for loud failure lines.
        """
        self.task_sandbox = Path(task_sandbox)
        self.runs_dir = self.task_sandbox / "runs"
        self.venv_path = self.task_sandbox / "venv"
        self.timeout_s = max(5, int(timeout_s))
        self.venv_reuse = bool(venv_reuse)
        self.install_timeout_s = max(60, int(install_timeout_s))
        self.python_exe = (
            python_exe or os.environ.get("MIMOSA_POLICY_BT_PYTHON") or sys.executable
        )
        self.log = log
        self._constraints_file = self.task_sandbox / "constraints.txt"
        self._last_pip_error: str | None = None
        self._last_install_source: str = ""
        self._py_version: str | None = None  # cached interpreter version

    def _python_version(self) -> str:
        """Version string of the venv's interpreter (cached; P5.7)."""
        if self._py_version is None:
            try:
                r = subprocess.run(
                    [self.python_exe, "-c", "import sys; print(sys.version_info[:3])"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self._py_version = (r.stdout or "").strip() or sys.version
            except (OSError, subprocess.SubprocessError):
                self._py_version = sys.version
        return self._py_version

    def venv_fingerprint(self) -> str:
        """sha256(PINNED_CONSTRAINTS + python version) — venv identity.

        P5.7: a venv is only reusable while the era pins and the
        interpreter version it was built for are CURRENT; otherwise it
        is rebuilt (a stale venv silently installs against outdated
        pins).
        """
        h = hashlib.sha256()
        h.update("\n".join(PINNED_CONSTRAINTS).encode("utf-8"))
        h.update(self._python_version().encode("utf-8"))
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_copy_path(self, uuid: str) -> Path:
        """Sandbox copy location for *uuid* (the digest builder reads it)."""
        return self.runs_dir / uuid

    def run(self, workspace: Path, uuid: str) -> dict[str, Any]:
        """Re-execute one generation's workspace; return the ledger.

        The ORIGINAL *workspace* is never mutated: a size+head-hash
        manifest is asserted identical before/after, and execution happens
        inside the sandbox copy. The executed program comes from the
        deliverable contract (``run.sh`` / README python lines) with the
        largest-``.py`` heuristic as FALLBACK only (P1.1).
        """
        workspace = Path(workspace)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        pre_man = frozen_manifest(workspace)
        ledger: dict[str, Any] = {"uuid": uuid, "source_manifest_intact": True}

        if not workspace.exists() or not workspace.is_dir():
            ledger["launch_error"] = "missing_workspace_dir"
            return ledger
        if not any(workspace.iterdir()):
            ledger["launch_error"] = "empty_workspace"
            return ledger

        dst = self.run_copy_path(uuid)
        try:
            clone_tree(workspace, dst)
        except OSError as e:  # P2.7: copy failure is a launch error, not a crash
            ledger["launch_error"] = f"clone_tree: {type(e).__name__}: {e}"
            return ledger
        ledger["n_files_copied"] = sum(1 for p in dst.rglob("*") if p.is_file())

        source, commands = contract_commands(dst)
        if commands:
            ledger["main_script_source"] = source
        else:
            main = select_main_script(dst)
            if main is None:
                ledger["launch_error"] = "no_main_script"
                ledger["source_manifest_intact"] = pre_man == frozen_manifest(workspace)
                return ledger
            ledger["main_script_source"] = "fallback"
            commands = [(main.name, [])]
        ledger["main_script"] = commands[0][0]
        ledger["commands"] = [{"script": s, "args": a} for s, a in commands]

        try:
            venv_created = self._ensure_venv()
            pkgs = self._install_requirements(dst)
        except Exception as e:  # noqa: BLE001 — ledger must survive infra errors
            ledger["launch_error"] = f"venv_provisioning: {type(e).__name__}: {e}"
            ledger["source_manifest_intact"] = pre_man == frozen_manifest(workspace)
            return ledger
        ledger["venv_created"] = venv_created
        ledger["packages_installed"] = pkgs
        ledger["install_source"] = self._last_install_source

        run_sh = dst / "run.sh"
        if run_sh.exists() and run_sh.stat().st_size > 0:
            # Execute run.sh as a SHELL SCRIPT — preserves pipes (| tee),
            # redirects (>, >>), and command chains (&&, ;) that the
            # python-argv extraction drops. Files written through shell
            # operators are correctly marked fresh by _fresh_artifacts.
            ledger["main_script_source"] = "shell_run_sh"
            out = self._execute_shell_script(dst, run_sh)
        else:
            out = self._execute_scripts(dst, commands)
        ledger.update(out)

        fresh = self._fresh_artifacts(dst, out.get("t0_wall", time.time()))
        ledger["fresh_artifacts"] = fresh
        ledger["produced_pred_results"] = any(
            k.startswith("pred_results/") for k in fresh
        )
        ledger["source_manifest_intact"] = pre_man == frozen_manifest(workspace)
        ledger.pop("t0_wall", None)
        return ledger

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_venv(self) -> bool:
        """Create the LOCAL venv when missing (or when reuse is off).

        P5.7: the pins file is REWRITTEN every run and the venv is
        fingerprinted by ``sha(PINNED_CONSTRAINTS + python version)`` —
        a venv built for different pins or a different interpreter (or
        a pre-fingerprint legacy venv) is rebuilt, never trusted.

        Returns True when this call created it. Raises on failure.
        """
        python_bin = self.venv_path / "bin" / "python"
        fp = self.venv_fingerprint()
        fp_file = self.venv_path / ".mimosa_venv_fp"
        self.task_sandbox.mkdir(parents=True, exist_ok=True)
        self._constraints_file.write_text(
            "\n".join(PINNED_CONSTRAINTS) + "\n", encoding="utf-8"
        )
        if python_bin.exists() and self.venv_reuse:
            try:
                current = fp_file.read_text(encoding="utf-8").strip()
            except OSError:
                current = ""
            if current == fp:
                return False
        if self.venv_path.exists():
            shutil.rmtree(self.venv_path)
        r = subprocess.run(
            [self.python_exe, "-m", "venv", str(self.venv_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode != 0 or not python_bin.exists():
            raise RuntimeError(f"venv creation failed: {(r.stderr or r.stdout)[-500:]}")
        fp_file.write_text(fp + "\n", encoding="utf-8")
        return True

    def _install_requirements(self, copy_dir: Path) -> int:
        """Install workspace requirements (or the import scan) under pins.

        Any requirements file name is honoured (P1.2: ``requirements*.txt``
        and the goal's ``requirement.txt`` alike), installed with
        ``pip install -r <file> -c constraints.txt``; on failure the
        import scan over every top-level ``.py`` is the fallback.

        Returns the number of requirement specifiers handed to pip. pip
        skips already-satisfied specs, so reused venvs install increment
        by increment. Failures are recorded (``pip_error``) but not
        raised: the run still executes and its crash signature lands in
        the ledger where the policy can see it.
        """
        self._last_pip_error = None
        pip = self.venv_path / "bin" / "pip"
        req_file = _find_requirements_file(copy_dir)
        if req_file is not None:
            specs = [
                ln.strip()
                for ln in req_file.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            self._last_install_source = req_file.name
            if self._pip_install(pip, ["-r", str(req_file)]):
                return len(specs)
            # pip failed on the requirements file -> import-scan fallback
            self._last_install_source = "import_scan(fallback)"
        else:
            self._last_install_source = "import_scan"
        specs = scan_imports_all(copy_dir)
        if not specs:
            return 0
        self._pip_install(pip, specs)
        return len(specs)

    def _pip_install(self, pip: Path, spec_args: list[str]) -> bool:
        """One pinned pip install; True on success (failure recorded)."""
        cmd = [
            str(pip),
            "install",
            "--quiet",
            "-c",
            str(self._constraints_file),
            *spec_args,
        ]
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.install_timeout_s,
            )
            if r.returncode != 0:
                self._last_pip_error = (r.stderr or r.stdout)[-800:]
                if self.log is not None:
                    self.log.warn(
                        f"pip install failed ({self._last_install_source}, "
                        f"{len(spec_args)} specs): {self._last_pip_error[-200:]}"
                    )
                return False
        except subprocess.TimeoutExpired:
            self._last_pip_error = f"pip timeout after {self.install_timeout_s}s"
            if self.log is not None:
                self.log.warn(f"pip install timed out ({self._last_install_source})")
            return False
        return True

    @property
    def pip_error(self) -> str | None:
        """Last pip failure recorded by ``_install_requirements``."""
        return self._last_pip_error

    def _execute_shell_script(self, copy_dir: Path, script: Path) -> dict[str, Any]:
        """Execute a shell script (run.sh) with the venv environment.

        Unlike :meth:`_execute_scripts` (which extracts python argv and
        drops shell operators), this runs ``bash <script>`` directly so
        pipes (``| tee``), redirects (``>``, ``>>``), and multi-command
        chains (``&&``, ``;``) execute exactly as the workflow intended.
        Files written through shell operators are then correctly marked
        fresh by ``_fresh_artifacts``.

        Args:
            copy_dir: The sandbox copy to execute in (cwd).
            script: Path to the shell script (typically ``run.sh``).

        Returns:
            Ledger fields: exit_code, wall_time_s, timed_out,
            error_signature, log_tail, output_log_sha256, pip_error.
        """
        t0 = time.monotonic()
        t0_wall = time.time()
        timed_out = False
        venv_bin = str(self.venv_path / "bin")
        env = {
            "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "VIRTUAL_ENV": str(self.venv_path),
            "PYTHONNOUSERSITE": "1",
            "HOME": os.environ.get("HOME", str(copy_dir)),
        }
        proc = subprocess.Popen(
            ["bash", str(script.name)],
            cwd=str(copy_dir),
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        out = ""
        exit_code: int | None = None
        try:
            out, _ = proc.communicate(timeout=self.timeout_s)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            partial, _ = proc.communicate()
            out = partial or out
            if self.log is not None:
                self.log.warn(
                    f"shell re-execution timed out after {self.timeout_s}s "
                    f"— killed process group ({script.name})"
                )
        wall = round(time.monotonic() - t0, 1)
        out_log = copy_dir / "output.log"
        with contextlib.suppress(OSError):
            out_log.write_text(out or "", encoding="utf-8")
        lines = (out or "").splitlines()
        err_sig = sorted({m.group(0)[:120] for m in ERROR_PAT.finditer(out or "")})[:8]
        return {
            "exit_code": exit_code,
            "wall_time_s": wall,
            "timed_out": timed_out,
            "error_signature": err_sig,
            "log_tail": "\n".join(lines[-LOG_TAIL_LINES:]),
            "output_log_sha256": (sha256_file(out_log) if out_log.exists() else None),
            "pip_error": self.pip_error,
            "t0_wall": t0_wall,
        }

    def _execute_scripts(
        self, copy_dir: Path, commands: list[tuple[str, list[str]]]
    ) -> dict[str, Any]:
        """Run the contract commands in order, process-group kill on timeout.

        ``set -e`` semantics: commands stop at the first failure (the
        ledger records that exit code). The venv's ``bin`` leads ``PATH``
        with ``VIRTUAL_ENV`` set and ``PYTHONNOUSERSITE=1`` so the
        deliverable resolves the sandbox interpreter, not the operator's
        (P2.6). On timeout the whole process group is SIGKILLed and the
        PARTIAL stdout/stderr is still collected into ``log_tail``
        (P2.6: a timed-out run's partial evidence must not be discarded).
        """
        t0 = time.monotonic()
        t0_wall = time.time()
        timed_out = False
        killed_pgid: int | None = None
        exit_code: int | None = None
        parts: list[str] = []
        python_bin = str(self.venv_path / "bin" / "python")
        venv_bin = str(self.venv_path / "bin")
        env = {
            "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "VIRTUAL_ENV": str(self.venv_path),
            "PYTHONNOUSERSITE": "1",
            "HOME": os.environ.get("HOME", str(copy_dir)),
        }
        for script, args in commands:
            # N5: the script slot may carry interpreter flags
            # ("-u x.py", "-m pkg") before the target; split on
            # whitespace into argv (targets/args are shell-clean tokens).
            proc = subprocess.Popen(
                [python_bin, *script.split(), *args],
                cwd=str(copy_dir),
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            out = ""
            try:
                out, _ = proc.communicate(timeout=self.timeout_s)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                killed_pgid = proc.pid
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                partial, _ = proc.communicate()  # keep partial evidence
                out = partial or out
                if self.log is not None:
                    self.log.warn(
                        f"re-execution timed out after {self.timeout_s}s — "
                        f"killed process group {killed_pgid} ({script})"
                    )
            parts.append(f"$ python {script} {' '.join(args)}".rstrip())
            parts.append(out or "")
            if timed_out or (exit_code not in (None, 0)):
                break  # set -e: stop at first failure/timeout
        combined = "\n".join(parts)
        wall = round(time.monotonic() - t0, 1)
        out_log = copy_dir / "output.log"
        with contextlib.suppress(OSError):
            out_log.write_text(combined or "", encoding="utf-8")
        lines = (combined or "").splitlines()
        err_sig = sorted(
            {m.group(0)[:120] for m in ERROR_PAT.finditer(combined or "")}
        )[:8]
        return {
            "exit_code": exit_code,
            "wall_time_s": wall,
            "timed_out": timed_out,
            "killed_pgid": killed_pgid,
            "error_signature": err_sig,
            "log_tail": "\n".join(lines[-LOG_TAIL_LINES:]),
            "output_log_sha256": (sha256_file(out_log) if out_log.exists() else None),
            "pip_error": self.pip_error,
            "t0_wall": t0_wall,
        }

    def _fresh_artifacts(
        self, copy_dir: Path, t0_wall: float
    ) -> dict[str, dict[str, Any]]:
        """Files written/rewritten by THIS run (mtime >= start, wall clock)."""
        fresh: dict[str, dict[str, Any]] = {}
        for p in sorted(copy_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(copy_dir).as_posix()
            if is_infra_name(rel):  # P1.5: one infra rule everywhere
                continue
            if any(part in MANIFEST_SKIP_DIRS for part in Path(rel).parts):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime >= t0_wall - 1:
                fresh[rel] = {
                    "bytes": st.st_size,
                    "sha256": (sha256_file(p) if st.st_size <= FRESH_SHA_CAP else None),
                }
        return fresh


def write_ledger_file(ledger: dict[str, Any], out_path: Path) -> None:
    """Persist one generation's ledger as JSON (plan §6b artifact)."""
    clean = dict(ledger)
    clean.pop("t0_wall", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_dump(clean), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: re-execute one workspace folder in a sandbox.

    The input folder is never mutated (manifest guard): execution runs
    in a sandbox copy under ``--sandbox`` (default
    ``<tmp>/policy_bt_reexec/<folder name>``), whose venv is provisioned
    under the era pins — first run may take minutes on pip installs.
    The ledger JSON goes to stdout (and ``--out``, default
    ``<sandbox>/ledger_<uuid>.json``). Run from this package directory::

        uv run reexec.py <workspace> [--uuid FAKE] [--sandbox DIR]
            [--timeout S] [--no-venv-reuse] [--out ledger.json]

    or as a module from the repository root::

        uv run python -m sources.evaluators.policy_bt_verifier.reexec ...

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 normally — a crashing workspace is evidence, not a CLI
        failure; 1 only when the ledger records a ``launch_error`` (the
        verifier itself could not run).
    """
    ap = argparse.ArgumentParser(
        description=("Re-execute one workspace folder in a verifier-owned sandbox."),
    )
    ap.add_argument(
        "folder",
        type=Path,
        help="generation workspace to re-execute (never mutated)",
    )
    ap.add_argument(
        "--uuid",
        default="cli-reexec-00000000",
        help="fake generation uuid for the sandbox copy (default: %(default)s)",
    )
    ap.add_argument(
        "--sandbox",
        type=Path,
        default=None,
        help="task sandbox root (default: <tmp>/policy_bt_reexec/<folder name>)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="wall-clock cap per script execution, seconds (default: %(default)s)",
    )
    ap.add_argument(
        "--no-venv-reuse",
        action="store_true",
        help="rebuild the sandbox venv instead of reusing it",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="ledger JSON path (default: <sandbox>/ledger_<uuid>.json)",
    )
    args = ap.parse_args(argv)

    ws = args.folder.resolve()
    if not ws.is_dir():
        ap.error(f"not a directory: {ws}")
    sandbox = (
        args.sandbox or Path(tempfile.gettempdir()) / "policy_bt_reexec" / ws.name
    ).resolve()
    runner = ReexecRunner(
        sandbox,
        timeout_s=args.timeout,
        venv_reuse=not args.no_venv_reuse,
        log=StageLogger(args.uuid),
    )
    ledger = runner.run(ws, args.uuid)
    out = args.out or sandbox / f"ledger_{args.uuid}.json"
    write_ledger_file(ledger, out)
    print(json_dump(ledger))
    print(f"LEDGER {out}", file=sys.stderr)
    print(f"SANDBOX {sandbox}", file=sys.stderr)
    return 1 if ledger.get("launch_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
