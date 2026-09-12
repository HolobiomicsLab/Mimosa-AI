"""Validate and package the opt-in native CodeAgent harness transport."""

import hashlib
import math
from pathlib import Path
import shutil
import subprocess


def validate_native_config(config):
    """Validate explicit native settings without dispatching or loading a bridge."""
    from .harness_model import validate_native_harness_settings
    settings = getattr(config, "native_harness_config", None)
    validate_native_harness_settings(settings)
    models = config.smolagent_model_id
    models = models if isinstance(models, list) else [models]
    if not models or any(not isinstance(m, str) or not m.startswith("codex-cli/") for m in models):
        raise ValueError("Native harness mode requires explicit codex-cli agent models")
    if getattr(config, "harness_auth_mode", "subscription") != "subscription":
        raise ValueError("Native Codex agents require subscription authentication")
    if getattr(config, "orchestrator_choose_model", False):
        raise ValueError("Native harness mode requires a fixed configured agent model")
    timeout = getattr(config, "agent_execution_timeout", None)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Native agent execution timeout must be finite and positive")


def package_native_harness(config):
    """Embed canonical runtime modules into a standalone generated workflow."""
    if getattr(config, "native_harness_config", None) is None:
        return ""
    validate_native_config(config)
    source_dir = Path(__file__).parent
    names = ("completion_backends", "harness_budget", "harness_model", "process_lifecycle", "native_agent")
    sources = {name: (source_dir / (name + ".py")).read_text() for name in names}
    # A content-named package keeps simultaneous generated scripts independent.
    digest = hashlib.sha256(repr(sources).encode()).hexdigest()[:16]
    package = "_mimosa_native_" + digest
    return f'''import sys as _native_sys
import types as _native_types
_native_package = _native_types.ModuleType({package!r})
_native_package.__path__ = []
_native_sys.modules[{package!r}] = _native_package
for _native_name, _native_source in {sources!r}.items():
    _native_fullname = {package!r} + '.' + _native_name
    _native_module = _native_types.ModuleType(_native_fullname)
    _native_module.__package__ = {package!r}
    _native_module.__file__ = '<embedded:' + _native_fullname + '>'
    _native_sys.modules[_native_fullname] = _native_module
    exec(compile(_native_source, _native_module.__file__, 'exec'), _native_module.__dict__)
from {package}.harness_model import HarnessCompletionModel
from {package}.native_agent import run_native_agent
NATIVE_HARNESS_CONFIG = {config.native_harness_config!r}
'''


def preflight_native_harness(config):
    """Check bridge bytes, executable and ChatGPT login without a model request."""
    validate_native_config(config)
    settings = config.native_harness_config
    bridge = Path(settings["bridge_path"])
    if hashlib.sha256(bridge.read_bytes()).hexdigest() != settings["bridge_sha256"]:
        raise ValueError("Native completion bridge content changed")
    executable = shutil.which("codex")
    if not executable:
        raise ValueError("Codex executable is unavailable in this runtime")
    status = subprocess.run([executable, "login", "status"], capture_output=True, timeout=10)
    if status.returncode or b"ChatGPT" not in status.stdout + status.stderr:
        raise ValueError("Codex ChatGPT subscription login is unavailable")
    version = subprocess.run([executable, "--version"], capture_output=True, timeout=10)
    if version.returncode or not version.stdout.startswith(b"codex-cli "):
        raise ValueError("Codex version check failed")
    return {"status": "passed", "method": "local_runtime_no_model_call",
            "executable": executable, "cli_version": version.stdout.decode().strip(),
            "bridge_sha256": settings["bridge_sha256"], "auth_mode": "subscription"}
