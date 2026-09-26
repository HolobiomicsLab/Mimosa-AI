"""Per-task hybrid-verifier registry (rubric-cache-style persistence).

One JSON file per task under the verifier scratch root, keyed by
``sha256(goal + registry_version)`` — the same keying scheme as the legacy
rubric cache, so tasks are shared across generations of the same goal and
distinct between goals. The registry stores:

- the current claim set (statement + scoring rule + cached scorer script),
- every generation's per-claim score vector seen so far,
- the union workspace inventory with presence counts,
- per-claim discriminativeness stats (variance filter input),
- how many replacement claims were requested per generation.

Everything in the registry is label-free by construction (only measured
scorer outputs), mirroring the E19b extract -> dry-run -> refine loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

REGISTRY_VERSION = "hybrid-v1-e19b-20260924"

_DEAD_STATES = ("zero_variance", "all_fail")


def task_key(goal: str) -> str:
    """Stable 16-hex-char key from goal text + registry version."""
    digest = hashlib.sha256((goal or "").encode("utf-8"))
    digest.update(REGISTRY_VERSION.encode("utf-8"))
    return digest.hexdigest()[:16]


class TaskRegistry:
    """Load/persist/extend the per-task hybrid-verifier state.

    Attributes:
        path: JSON file backing this registry.
        goal: Task goal text (frozen at creation).
        claims: Current claim list; each claim dict carries the E19b fields
            (``id``/``category``/``statement``/``target``/``scoring_rule``)
            plus ``scorer`` (cached script source or ``None``), ``state``
            (``alive``/``dead``), ``drop_reason`` and ``added_round``.
        generations: Ordered per-generation records
            (``uuid``, ``scores``: claim_id -> float|None, ``evidence``:
            claim_id -> str, ``reward``).
        inventory: Union inventory across generations
            (``{rel_path: {"count": int, "signal": str}}``).
        refinements: Replacements requested per generation uuid.
    """

    def __init__(self, path: Path, goal: str) -> None:
        self.path = path
        self.goal = goal
        self.claims: list[dict[str, Any]] = []
        self.generations: list[dict[str, Any]] = []
        self.inventory: dict[str, dict[str, Any]] = {}
        self.refinements: dict[str, int] = {}
        self.digests: dict[str, str] = {}
        self.n_inventories = 0
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls, root: Path, goal: str, logger: logging.Logger | None = None
    ) -> TaskRegistry:
        """Load (or start) the registry for one task goal under *root*."""
        reg = cls(root / f"hybrid_registry_{task_key(goal)}.json", goal)
        if logger is not None:
            reg.logger = logger
        if not reg.path.exists():
            return reg
        try:
            data = json.loads(reg.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            reg.logger.warning(f"Could not read hybrid registry {reg.path}: {e}")
            return reg
        if not isinstance(data, dict):
            return reg
        reg.claims = data.get("claims") or []
        reg.generations = data.get("generations") or []
        reg.inventory = data.get("inventory") or {}
        reg.refinements = data.get("refinements") or {}
        reg.digests = data.get("digests") or {}
        reg.n_inventories = int(data.get("n_inventories") or len(reg.generations))
        reg._migrate_claim_stages()
        # Guard against a version drift wiping measurements: the version is
        # part of the file key, so a mismatch here means a renamed file.
        if data.get("version") != REGISTRY_VERSION:
            reg.logger.warning(
                f"Hybrid registry {reg.path.name} version mismatch "
                f"({data.get('version')!r}); continuing with loaded state."
            )
        return reg

    def _migrate_claim_stages(self) -> None:
        """Back-fill ladder metadata on legacy (untagged) registry claims.

        Untagged claims default to stage ``result`` with
        ``temporal_index`` = original claim order, so pre-ladder
        registries keep scoring under the temporal modes without
        invalidation. The E26 ``visual`` stage (figure tasks) is
        canonical since E35.
        """
        for i, c in enumerate(self.claims):
            if "stage" not in c:
                c["stage"] = "result"
            if c.get("stage") not in ("visual", "script", "log", "result"):
                c["stage"] = "result"
            if "temporal_index" not in c:
                c["temporal_index"] = i


    def save(self) -> None:
        """Persist the registry atomically-enough for single-writer use."""
        payload = {
            "version": REGISTRY_VERSION,
            "task_key": task_key(self.goal),
            "goal": self.goal,
            "claims": self.claims,
            "generations": self.generations,
            "inventory": self.inventory,
            "refinements": self.refinements,
            "digests": self.digests,
            "n_inventories": self.n_inventories,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as e:
            self.logger.warning(f"Could not write hybrid registry {self.path}: {e}")

    # ------------------------------------------------------------------
    # Claim set management
    # ------------------------------------------------------------------

    def claim(self, claim_id: str) -> dict[str, Any] | None:
        """Claim dict by id, or None."""
        return next((c for c in self.claims if c.get("id") == claim_id), None)

    def seed_claims(self, claims: list[dict[str, Any]]) -> None:
        """Set the claim set on first extraction (no-op when seeded)."""
        if self.claims:
            return
        self.claims = [
            {
                **c,
                "scorer": None,
                "state": "alive",
                "drop_reason": None,
                "added_round": 0,
            }
            for c in claims
        ]

    def replace_claims(
        self,
        dead_ids: list[str],
        replacements: list[dict[str, Any]],
        generation: int,
    ) -> list[str]:
        """Drop dead claims, append replacements; returns replacement ids."""
        dead = set(dead_ids)
        self.claims = [c for c in self.claims if c.get("id") not in dead]
        added: list[str] = []
        for c in replacements:
            self.claims.append(
                {
                    **c,
                    "scorer": None,
                    "state": "alive",
                    "drop_reason": None,
                    "added_round": generation,
                }
            )
            added.append(c["id"])
        return added

    def mark_dead(self, claim_id: str, reason: str) -> None:
        """Flag a claim non-discriminative (kept in the file for audit)."""
        c = self.claim(claim_id)
        if c is not None:
            c["state"] = "dead"
            c["drop_reason"] = reason

    def mark_alive(self, claim_id: str) -> None:
        """Clear the dead flag (a claim revived by later variance)."""
        c = self.claim(claim_id)
        if c is not None:
            c["state"] = "alive"
            c["drop_reason"] = None

    def append_claims(
        self, claims: list[dict[str, Any]], generation: int = 0
    ) -> list[str]:
        """Append layer-owned claims (e.g. the E26 visual rung) to the set.

        Unlike :meth:`seed_claims` this works on an already-seeded task:
        the visual rung is added on first sight of a figure task even when
        text claims exist. Ids already present are skipped.
        """
        taken = {c.get("id") for c in self.claims}
        added: list[str] = []
        for c in claims:
            cid = c.get("id")
            if cid in taken:
                continue
            self.claims.append(
                {
                    **c,
                    "scorer": None,
                    "state": "alive",
                    "drop_reason": None,
                    "added_round": generation,
                }
            )
            taken.add(cid)
            added.append(cid)
        return added

    def alive_claims(self) -> list[dict[str, Any]]:
        """Claims not currently marked dead."""
        return [c for c in self.claims if c.get("state", "alive") != "dead"]

    def dead_claims(self) -> list[dict[str, Any]]:
        """Claims currently marked dead (with their reasons)."""
        return [c for c in self.claims if c.get("state") == "dead"]

    def set_scorer(self, claim_id: str, script: str | None) -> None:
        """Cache the scorer script for one claim (None = failed build)."""
        c = self.claim(claim_id)
        if c is not None:
            c["scorer"] = script

    # ------------------------------------------------------------------
    # Generations
    # ------------------------------------------------------------------

    def record_generation(
        self,
        uuid: str,
        scores: dict[str, float | None],
        evidence: dict[str, str],
        reward: float | None = None,
    ) -> None:
        """Upsert one generation's score vector (idempotent per uuid)."""
        self.generations = [g for g in self.generations if g.get("uuid") != uuid]
        self.generations.append(
            {
                "uuid": uuid,
                "scores": scores,
                "evidence": evidence,
                "reward": reward,
            }
        )

    def previous_generations(self, uuid: str) -> list[dict[str, Any]]:
        """Every recorded generation except *uuid* (the comparison set)."""
        return [g for g in self.generations if g.get("uuid") != uuid]

    def observed_scores(
        self, claim_id: str, exclude_uuid: str | None = None
    ) -> list[float]:
        """All non-None scores observed for one claim across generations."""
        out = []
        for g in self.generations:
            if exclude_uuid is not None and g.get("uuid") == exclude_uuid:
                continue
            s = (g.get("scores") or {}).get(claim_id)
            if s is not None:
                out.append(float(s))
        return out

    # ------------------------------------------------------------------
    # Union inventory
    # ------------------------------------------------------------------

    def merge_inventory(self, current: dict[str, dict[str, Any]]) -> None:
        """Fold one workspace's inventory into the per-task union.

        Presence counts follow E19: each generation contributes +1 for
        every path it contains; the first non-empty content signal wins.
        """
        self.n_inventories += 1
        for rel, info in current.items():
            entry = self.inventory.setdefault(
                rel, {"count": 0, "signal": "", "kind": info.get("kind", "other")}
            )
            entry["count"] += 1
            if not entry.get("signal") and info.get("signal"):
                entry["signal"] = info["signal"]

    @property
    def n_workspaces(self) -> int:
        """Number of workspaces whose inventory was merged."""
        return self.n_inventories
