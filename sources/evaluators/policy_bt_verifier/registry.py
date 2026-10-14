"""Task-scoped tournament persistence with serialized atomic commits."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

REGISTRY_VERSION = 1


def task_key(goal: str, session: str) -> str:
    """Name a registry without mixing independent search sessions.

    Args:
        goal: Exact user goal from workflow state.
        session: Evolution session identifier supplied by the engine.

    Returns:
        SHA-256 key for the goal and session.
    """
    return hashlib.sha256(json.dumps([goal, session]).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """Commit registry or state data without exposing partial JSON.

    Args:
        path: Destination owned by the verifier or workflow engine.
        value: JSON-compatible state to persist.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class TaskRegistry:
    """Keep snapshots and one committed tournament for a task session.

    Created by the evaluator per goal and session, at
    ``<workflow_dir>/.policy_bt/<sha256(goal, session)>``. All reads and the
    single writer go through ``transaction()``, which holds an exclusive file
    lock and commits atomically, so a crashed evaluation leaves the previous
    tournament intact.
    """

    def __init__(self, root: Path, identity: tuple[str, str]):
        """Locate the registry for evaluator-supplied task identity.

        Args:
            root: Verifier-owned registry directory.
            identity: Exact goal and session identifier.

        Returns:
            None.
        """
        self.goal, self.session = identity
        self.root = root / task_key(*identity)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "tournament.json"

    @contextmanager
    def transaction(self):
        """Serialize evaluator updates and commit only successful transactions.

        Args:
            None.

        Returns:
            Context manager yielding mutable tournament state.
        """
        with (self.root / "registry.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = self.load()
                yield state
                atomic_json(self.path, state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def load(self) -> dict:
        """Read a committed tournament or initialize its empty state.

        Args:
            None.

        Returns:
            Validated task state; corrupt or incompatible state raises.
        """
        if not self.path.exists():
            return {"schema_version": REGISTRY_VERSION, "goal": self.goal,
                    "session": self.session, "version": 0, "generations": [],
                    "pairs": [], "bundle": None, "ratings": {}}
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if (state["schema_version"] != REGISTRY_VERSION
                or state["goal"] != self.goal or state["session"] != self.session):
            raise ValueError("Incompatible Policy-BT registry")
        return state


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        registry = TaskRegistry(Path(directory), ("smoke goal", "smoke session"))
        with registry.transaction() as data:
            data["version"] = 1
        assert registry.load()["version"] == 1
        print("Atomic registry commit passed")
