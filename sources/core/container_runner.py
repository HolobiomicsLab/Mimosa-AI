"""Opt-in containment for complete workflows and credential-free completions.

The evaluator stages workflow.py and its public assets. The default host runner
is unchanged. This runner never executes generated code or solver tool requests
on the host; only bounded text messages enter the pinned completion adapter.
"""

import asyncio
import json
import math
import os
import re
import stat
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from .container_completion import FRAME_PREFIX, MAX_FRAME_BYTES, parse_request
from .harness_budget import validate_native_harness_settings
from .process_lifecycle import OwnedProcessTree

MAX_OUTPUT_BYTES = 4 * MAX_FRAME_BYTES
CLEANUP_TIMEOUT = 10
COMPLETION_SETTLEMENT_SECONDS = 5
CLEANUP_DIAGNOSTIC_LIMIT = 4096


class ProcessCleanupError(RuntimeError):
    """An owned host process could not be verified stopped."""

    def __init__(self, message, *, cancelled=False):
        super().__init__(message)
        self.cancelled = cancelled


@dataclass
class ContainerRuntime:
    """Evaluator-owned mounts, pinned image and private native completion policy."""

    image: str
    public_dir: Path
    work_dir: Path
    model_id: str
    settings: dict
    timeout_seconds: float = 30
    memory_mb: int = 512
    cpus: float = 1
    pids_limit: int = 64
    docker_executable: str = "docker"
    private_paths: tuple[Path, ...] = ()


@dataclass
class ContainerResult:
    """Execution outcome; cleanup proof is distinct from successful computation."""

    status: str
    return_code: int | None
    stdout: str
    stderr: str
    execution_time: float
    cleanup_verified: bool
    container_name: str
    completion_count: int
    cleanup_diagnostics: list[dict] = field(default_factory=list)


def _validated(config):
    if not isinstance(config.image, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", config.image
    ):
        raise ValueError("container image must be a pinned local image ID")
    for name in ("timeout_seconds", "cpus", "memory_mb", "pids_limit"):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if type(config.memory_mb) is not int or type(config.pids_limit) is not int:
        raise ValueError("memory_mb and pids_limit must be integers")
    public, work = Path(config.public_dir).resolve(), Path(config.work_dir).resolve()
    if any("," in str(path) or "\x00" in str(path) for path in (public, work)):
        raise ValueError("mount paths cannot contain Docker mount separators")
    if not public.is_dir() or not work.is_dir() or any(work.iterdir()):
        raise ValueError("public directory and fresh empty work directory are required")
    if public.is_relative_to(work) or work.is_relative_to(public):
        raise ValueError("public and work directories must not overlap")
    private_paths = [config.settings[name] for name in ("bridge_path", "ledger_path")]
    for path in [*private_paths, *config.private_paths]:
        private = Path(path).resolve()
        if any(
            private.is_relative_to(mount) or mount.is_relative_to(private)
            for mount in (public, work)
        ):
            raise ValueError("private completion paths cannot be mounted")
    workflow = public / "workflow.py"
    if workflow.is_symlink() or not workflow.is_file():
        raise ValueError("public workflow.py must be a regular file")
    for path in public.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("public staging requires unaliased regular files")
    settings = validate_native_harness_settings(config.settings)
    return replace(config, public_dir=public, work_dir=work, settings=dict(settings))


async def _watch(tree):
    while True:
        tree.capture()
        await asyncio.sleep(0.02)


async def _finish_process(process, tree, watcher):
    try:
        tree.kill(include_root=process.returncode is None)
        await asyncio.wait_for(process.wait(), CLEANUP_TIMEOUT)
        await asyncio.to_thread(tree.wait_children)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


async def _settle_task(task):
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled


async def _finish_despite_cancellation(task):
    result, cancelled = await _settle_task(task)
    if cancelled:
        raise asyncio.CancelledError
    return result


@asynccontextmanager
async def _owned_process(argv, **kwargs):
    launch = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=MAX_FRAME_BYTES + len(FRAME_PREFIX) + 1,
            **kwargs,
        )
    )
    process, cancelled = await _settle_task(launch)
    tree = OwnedProcessTree(process.pid)
    watcher = asyncio.create_task(_watch(tree))
    try:
        if cancelled:
            raise asyncio.CancelledError
        yield process
    except asyncio.CancelledError:
        cancelled = True
        raise
    except ProcessCleanupError as error:
        cancelled = cancelled or error.cancelled
        raise
    finally:
        try:
            await _finish_despite_cancellation(
                asyncio.create_task(_finish_process(process, tree, watcher))
            )
        except Exception as error:
            raise ProcessCleanupError(
                "owned host process cleanup is unverified",
                cancelled=cancelled or bool(asyncio.current_task().cancelling()),
            ) from error


async def _read_limited(stream, maximum):
    raw = bytearray()
    while chunk := await stream.read(65536):
        raw.extend(chunk)
        if len(raw) > maximum:
            raise ValueError("process output exceeds byte limit")
    return bytes(raw)


async def _exchange(process, payload, maximum):
    async def write():
        process.stdin.write(payload)
        await process.stdin.drain()
        process.stdin.close()

    tasks = [
        asyncio.create_task(write()),
        asyncio.create_task(_read_limited(process.stdout, maximum)),
        asyncio.create_task(_read_limited(process.stderr, maximum)),
    ]
    try:
        _, stdout, stderr = await asyncio.gather(*tasks)
        await process.wait()
        return stdout, stderr
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class ContainerWorkflowRunner:
    """Execute one staged workflow with cancellable private host completions."""

    def __init__(self, config):
        self.config = _validated(config)
        self._used = False
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._completion_count = 0
        self._return_code = None
        self.container_name = None
        self._cleanup_diagnostics = []

    def container_argv(self, name):
        """Build the fixed containment policy; no solver fields reach Docker."""
        config = self.config
        return [
            config.docker_executable,
            "run",
            "--name",
            name,
            "--rm",
            "--pull=never",
            "-i",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--memory={config.memory_mb}m",
            f"--cpus={config.cpus}",
            f"--pids-limit={config.pids_limit}",
            "--user=1000:1000",
            "--entrypoint=python3",
            "--tmpfs=/tmp:size=64m,mode=1777",
            "--mount",
            f"type=bind,src={config.public_dir},dst=/data,readonly",
            "--mount",
            f"type=bind,src={config.work_dir},dst=/work",
            "--workdir=/work",
            "--env=HOME=/tmp",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            config.image,
            "-u",
            "/data/workflow.py",
        ]

    async def _completion(self, raw):
        request = parse_request(raw)
        payload = json.dumps(
            {
                "model_id": self.config.model_id,
                "settings": self.config.settings,
                "request": request,
            },
            allow_nan=False,
        ).encode()
        argv = [sys.executable, "-P", "-m", "sources.core.container_completion_worker"]
        source_root = str(Path(__file__).resolve().parents[2])
        keys = (
            "HOME",
            "PATH",
            "CODEX_HOME",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        )
        env = {key: os.environ[key] for key in keys if key in os.environ}
        env.update(PYTHONPATH=source_root, PYTHONDONTWRITEBYTECODE="1")
        # Leave the bridge time to persist its terminal receipt.
        worker_timeout = (
            self.config.settings["call_timeout_seconds"] + COMPLETION_SETTLEMENT_SECONDS
        )
        async with asyncio.timeout(worker_timeout):
            async with _owned_process(argv, env=env, cwd=source_root) as process:
                response, _ = await _exchange(process, payload, MAX_FRAME_BYTES)
                if process.returncode:
                    raise RuntimeError(
                        "private completion worker failed; inspect its private budget"
                    )
        result = json.loads(response)
        if set(result) != {"id", "result"} or result["id"] != request["id"]:
            raise ValueError("invalid host completion response")
        self._completion_count += 1
        return response

    def _retain(self, target, raw):
        if len(self._stdout) + len(self._stderr) + len(raw) > MAX_OUTPUT_BYTES:
            raise ValueError("workflow output exceeds byte limit")
        target.extend(raw)

    async def _stdout_stream(self, process):
        while line := await process.stdout.readline():
            if line.startswith(FRAME_PREFIX):
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete completion frame")
                response = await self._completion(line[len(FRAME_PREFIX) :])
                process.stdin.write(response)
                await process.stdin.drain()
            else:
                self._retain(self._stdout, line)

    async def _stderr_stream(self, process):
        while chunk := await process.stderr.read(65536):
            self._retain(self._stderr, chunk)

    async def _workflow(self, name):
        async with _owned_process(self.container_argv(name)) as process:
            tasks = [
                asyncio.create_task(self._stdout_stream(process)),
                asyncio.create_task(self._stderr_stream(process)),
            ]
            try:
                await asyncio.gather(*tasks)
                await process.wait()
                self._return_code = process.returncode
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    def _record_cleanup_error(self, phase, error):
        """Retain the failed verification phase and bounded exception evidence."""
        record = {
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error)[:CLEANUP_DIAGNOSTIC_LIMIT],
        }
        if error.__cause__ is not None:
            record.update(
                cause_type=type(error.__cause__).__name__,
                cause=str(error.__cause__)[:CLEANUP_DIAGNOSTIC_LIMIT],
            )
        self._cleanup_diagnostics.append(record)

    async def _remove_container(self, name):
        async def control(args, phase):
            async with _owned_process(
                [self.config.docker_executable, *args]
            ) as process:
                stdout, stderr = await _exchange(process, b"", MAX_FRAME_BYTES)
                self._cleanup_diagnostics.append({
                    "phase": phase,
                    "return_code": process.returncode,
                    "stdout": stdout[:CLEANUP_DIAGNOSTIC_LIMIT].decode(errors="replace"),
                    "stderr": stderr[:CLEANUP_DIAGNOSTIC_LIMIT].decode(errors="replace"),
                    "stdout_truncated": len(stdout) > CLEANUP_DIAGNOSTIC_LIMIT,
                    "stderr_truncated": len(stderr) > CLEANUP_DIAGNOSTIC_LIMIT,
                })
                return process.returncode, stdout

        phase = "container_remove"
        try:
            await asyncio.wait_for(control(["rm", "--force", name], phase), CLEANUP_TIMEOUT)
            phase = "container_inspect"
            code, remaining = await asyncio.wait_for(
                control(["ps", "-aq", "--filter", f"name=^/{name}$"], phase), CLEANUP_TIMEOUT
            )
            return code == 0 and not remaining.strip()
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            self._record_cleanup_error(phase, error)
            return False

    async def execute(self):
        """Run once; cancellation propagates only after owned cleanup finishes."""
        if self._used:
            raise RuntimeError("container runner has already been used")
        self._used = True
        self.config = _validated(self.config)
        name = "mimosa-oracle-" + uuid.uuid4().hex
        self.container_name = name
        started, status = time.monotonic(), "completed"
        cancelled, processes_clean = False, True
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                await self._workflow(name)
                if self._return_code:
                    status = "failed"
        except TimeoutError:
            status = "timeout"
        except asyncio.CancelledError:
            cancelled = True
        except ProcessCleanupError as error:
            status, processes_clean = "cleanup_failed", False
            cancelled = error.cancelled
            self._record_cleanup_error("host_processes", error)
        except (OSError, ValueError, RuntimeError) as error:
            status = "failed"
            remaining = MAX_OUTPUT_BYTES - len(self._stdout) - len(self._stderr)
            self._stderr.extend(
                (type(error).__name__ + ": " + str(error)).encode()[
                    : min(4096, remaining)
                ]
            )
        finally:
            container_clean, cleanup_cancelled = await _settle_task(
                asyncio.create_task(self._remove_container(name))
            )
            cancelled = cancelled or cleanup_cancelled
            cleanup = container_clean and processes_clean
        if not cleanup:
            status = "cleanup_failed"
            if cancelled:
                raise RuntimeError(f"cancelled workflow cleanup is unverified: {name}")
        if cancelled:
            raise asyncio.CancelledError
        return ContainerResult(
            status,
            self._return_code,
            self._stdout.decode(errors="replace"),
            self._stderr.decode(errors="replace"),
            time.monotonic() - started,
            cleanup,
            name,
            self._completion_count,
            list(self._cleanup_diagnostics),
        )
