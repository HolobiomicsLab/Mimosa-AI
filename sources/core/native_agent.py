"""Run the native agent once with a fatal deadline and durable cost receipts."""

import json
import os
from pathlib import Path
import threading

from .process_lifecycle import OwnedProcessTree


def _expire_agent(done, timeout):
    """End the generated workflow and its descendants when its agent overruns."""
    if done.wait(timeout):
        return
    try:
        os.write(2, b"Native Mimosa agent deadline exceeded; ending workflow.\n")
        tree = OwnedProcessTree(os.getpid())
        tree.kill(include_root=False)
        tree.wait_children()
    finally:
        # A timed-out tool thread must never survive into a retry.
        os._exit(124)


def _save_receipts(model, memory_path):
    """Persist sanitized model receipts separately from smolagents raw memory."""
    path = Path(memory_path)
    path.mkdir(parents=True, exist_ok=True)
    for receipt in model.receipts:
        filename = path / ("native_completion_" + receipt["reservation_id"] + ".json")
        if filename.exists():
            if json.loads(filename.read_text()) != {"completion_metadata": receipt}:
                raise ValueError("Native completion receipt changed")
            continue
        with filename.open("x") as handle:
            json.dump({"completion_metadata": receipt}, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())


def run_native_agent(agent, instructions, options):
    """Run once; options supplies finite timeout_seconds and memory_path."""
    done = threading.Event()
    watchdog = threading.Thread(target=_expire_agent, args=(done, options["timeout_seconds"]), daemon=True)
    watchdog.start()
    try:
        result = agent.run(instructions)
        agent.model.assert_healthy()
        return result
    finally:
        done.set()
        watchdog.join()
        _save_receipts(agent.model, options["memory_path"])
