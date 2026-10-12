"""Per-task persistence for the policy_bt verifier (plan §2, §5b).

One JSON file per task under the verifier scratch root, keyed by
``sha256(goal + registry_version)`` — the same keying scheme as the hybrid
verifier's ``TaskRegistry``. The registry is the PRIMARY artifact store
(plan §5b): each generation's digest is computed ONCE at its evaluation and
read from here for every later comparison; predecessor comparisons never
touch ``/tmp`` snapshots or the live workspace again.

Stored state:

- ``generations``: EVOLUTION/arrival-ordered records — uuid, digest,
  re-execution ledger, arrival-moment reward and pair stats (never
  retroactively rewritten; E19c online convention);
- ``verdicts``: code-policy pair cache keyed ``<policy_sha>|<a>|<b>``;
- ``visual``: kimi pair-call cache keyed
  ``<prompt_version>|<a>|<b>|<order>|<image hashes>`` (self-heal scan);
- ``recon``: the per-task reconnaissance TASK CONFIG (goal-hash keyed);
- ``policy``: the accepted policy script source + sha + build metadata.

Everything is label-free by construction (digests, verdicts, rewards are
measured verifier outputs only — gold firewall).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

REGISTRY_VERSION = "policy_bt-v1-e48-20261005"


def task_key(goal: str) -> str:
    """Stable 16-hex-char key from goal text + registry version."""
    digest = hashlib.sha256((goal or "").encode("utf-8"))
    digest.update(REGISTRY_VERSION.encode("utf-8"))
    return digest.hexdigest()[:16]


def code_verdict_key(policy_sha: str, uuid_a: str, uuid_b: str) -> str:
    """Cache key for one code-policy pair verdict."""
    return f"{policy_sha}|{uuid_a}|{uuid_b}"


def visual_key(
    prompt_version: str, uuid_a: str, uuid_b: str, order: str, img_hash: str
) -> str:
    """Cache key for one kimi pair call (per plan §4b)."""
    return f"{prompt_version}|{uuid_a}|{uuid_b}|{order}|{img_hash}"


class TaskRegistry:
    """Load/persist/extend the per-task policy_bt state.

    Attributes:
        path: JSON file backing this registry.
        goal: Task goal text (frozen at creation).
        generations: Arrival-ordered per-generation records.
        verdicts: Code-policy pair verdict cache.
        visual: Kimi pair-call cache (append-only, self-heal scan).
        recon: The task's reconnaissance config (or None).
        policy: The accepted policy record (or None).
    """

    def __init__(self, path: Path, goal: str) -> None:
        self.path = path
        self.goal = goal
        self.generations: list[dict[str, Any]] = []
        self.verdicts: dict[str, dict[str, Any]] = {}
        self.visual: list[dict[str, Any]] = []
        self.recon: dict[str, Any] | None = None
        self.policy: dict[str, Any] | None = None
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls, root: Path, goal: str, logger: logging.Logger | None = None
    ) -> TaskRegistry:
        """Load (or start) the registry for one task goal under *root*."""
        reg = cls(Path(root) / f"policy_bt_registry_{task_key(goal)}.json", goal)
        if logger is not None:
            reg.logger = logger
        if not reg.path.exists():
            return reg
        try:
            data = json.loads(reg.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            reg.logger.warning(f"Could not read policy_bt registry {reg.path}: {e}")
            return reg
        if not isinstance(data, dict):
            return reg
        reg.generations = data.get("generations") or []
        reg.verdicts = data.get("verdicts") or {}
        reg.visual = data.get("visual") or []
        reg.recon = data.get("recon")
        reg.policy = data.get("policy")
        if data.get("version") != REGISTRY_VERSION:
            reg.logger.warning(
                f"policy_bt registry {reg.path.name} version mismatch "
                f"({data.get('version')!r}); continuing with loaded state."
            )
        return reg

    def save(self) -> None:
        """Persist the registry atomically (P2.9: tmp file + os.replace).

        A crash mid-write must never truncate the task's only artifact
        store — the previous file stays intact until the replacement
        lands on disk.
        """
        payload = {
            "version": REGISTRY_VERSION,
            "task_key": task_key(self.goal),
            "goal": self.goal,
            "generations": self.generations,
            "verdicts": self.verdicts,
            "visual": self.visual,
            "recon": self.recon,
            "policy": self.policy,
        }
        tmp = self.path.with_suffix(".json.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8"
            )
            os.replace(tmp, self.path)
        except OSError as e:
            self.logger.warning(f"Could not write policy_bt registry {self.path}: {e}")

    def record_generation(
        self,
        uuid: str,
        digest: dict[str, Any],
        ledger: dict[str, Any] | None = None,
        reward: float | None = None,
        stats: dict[str, Any] | None = None,
        session: str | None = None,
    ) -> None:
        """Upsert one generation's record (idempotent per uuid).

        Arrival order is preserved for new uuids; re-evaluating an
        existing uuid updates its record in place (its recorded reward
        may change, but history of OTHER generations is never rewritten).
        *session* tags the campaign session that wrote the record (N9:
        cross-session predecessors stay comparable but are WARNED about
        by the evaluator).
        """
        record = {
            "uuid": uuid,
            "arrival": len(self.generations),
            "digest": digest,
            "ledger": ledger or {},
            "reward": reward,
            "stats": stats or {},
        }
        if session:
            record["session"] = session
        for i, g in enumerate(self.generations):
            if g.get("uuid") == uuid:
                record["arrival"] = g.get("arrival", i)
                if not session and g.get("session"):
                    record["session"] = g["session"]  # keep original tag
                self.generations[i] = record
                return
        self.generations.append(record)

    def generation(self, uuid: str) -> dict[str, Any] | None:
        """One generation record by uuid (or None)."""
        return next((g for g in self.generations if g.get("uuid") == uuid), None)

    def previous_generations(self, uuid: str) -> list[dict[str, Any]]:
        """Every recorded generation that ARRIVED before *uuid*."""
        own = self.generation(uuid)
        cut = own["arrival"] if own else len(self.generations)
        return [g for g in self.generations if g.get("arrival", 0) < cut]

    def predecessors_with_digest(self, uuid: str) -> list[dict[str, Any]]:
        """Arrival-ordered predecessors that carry a stored digest."""
        return [
            g for g in self.previous_generations(uuid) if g.get("digest")
        ]

    def rewards(self) -> dict[str, float | None]:
        """Recorded arrival-moment reward per uuid."""
        return {g["uuid"]: g.get("reward") for g in self.generations}

    # ------------------------------------------------------------------
    # Verdict caches
    # ------------------------------------------------------------------

    def get_verdict(self, key: str) -> dict[str, Any] | None:
        """Cached code-policy verdict for *key* (or None)."""
        v = self.verdicts.get(key)
        return dict(v) if v else None

    def set_verdict(self, key: str, verdict: dict[str, Any]) -> None:
        """Cache one code-policy verdict."""
        self.verdicts[key] = verdict

    def visual_entries(self, key: str) -> list[dict[str, Any]]:
        """ALL cached entries for one visual key (self-heal scan input)."""
        return [e for e in self.visual if e.get("k") == key]

    def add_visual_entry(self, key: str, entry: dict[str, Any]) -> None:
        """Append one visual cache entry (parse-fails stay for audit)."""
        rec = {"k": key, **entry}
        self.visual.append(rec)

    # ------------------------------------------------------------------
    # Recon + policy
    # ------------------------------------------------------------------

    def set_recon(
        self, recon: dict[str, Any], prompt_version: str,
        recon_version: str = "",
    ) -> None:
        """Store the task's recon config (prompt version + content hash).

        ``recon_version`` is a hash of the config content; when it changes
        (incremental refresh), the policy is rewritten with the same ladder
        but updated schema/metric knowledge.
        """
        self.recon = {
            "prompt_version": prompt_version,
            "config": recon,
            "recon_version": recon_version,
        }

    def get_recon(self, prompt_version: str) -> dict[str, Any] | None:
        """Recon config when cached for *prompt_version* (else None)."""
        r = self.recon
        if not r or r.get("prompt_version") != prompt_version:
            return None
        return r.get("config")

    def get_recon_version(self) -> str:
        """Content hash of the current recon config ("" if none)."""
        return (self.recon or {}).get("recon_version", "")

    def set_policy(self, record: dict[str, Any]) -> None:
        """Store the accepted policy record (source/sha/metadata)."""
        self.policy = record

    def get_policy(self, policy_sha: str) -> dict[str, Any] | None:
        """Cached policy record when the sha matches (else None)."""
        p = self.policy
        if p and p.get("sha") == policy_sha:
            return p
        return None
