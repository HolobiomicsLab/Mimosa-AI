"""Champion-plateau stop control for the evolution engine (2026-10-08 plan).

Why this module exists: the legacy ``learned_score_threshold`` early stop
assumes ``overall_score`` is an ABSOLUTE quality measure ("0.9 = good
enough"). The two pairwise verifiers produce RELATIVE rewards instead:

- ``hybrid`` / ``policy_bt`` win-rate mode: any generation that edges
  every predecessor on any rung scores 1.0, so a 0.9 threshold means
  "stop on the first clean sweep" — both HP-folding runs stopped at
  generation 4 of 16 on scientifically meaningless edges.
- ``policy_bt`` Bradley-Terry mode: a generation beating every
  predecessor reaches only ~0.58-0.86 within 16 generations, so the
  threshold silently becomes "never stop".

A relative signal can only say the search stopped improving. The
replacement rule is therefore a CHAMPION PLATEAU: a new generation
becomes the champion only by beating the current champion head-to-head;
when no new champion arrives for ``patience`` consecutive generations
(and at least ``min_iter`` generations have completed), the run stops
and ships the final champion as the capsule.

The controller is deliberately pure (no I/O, no engine imports): the
engine creates it once per session, feeds it one ``observe()`` per
completed generation, and asks ``should_stop()`` after each iteration.
"""

from __future__ import annotations

#: Verifier kinds whose reward is pairwise (relative), not absolute.
RELATIVE_VERIFIER_KINDS = frozenset({"hybrid", "policy_bt"})

#: Accepted ``early_stop_mode`` values (see ``Config.early_stop_mode``).
VALID_STOP_MODES = ("auto", "threshold", "plateau", "none")


class StopController:
    """Session-scoped stop decision for one evolution run.

    The controller tracks a single champion uuid plus the number of
    generations since the champion last changed, and converts that state
    into a stop/no-stop decision. Mode semantics:

    - ``"auto"`` — resolved once at construction from the verifier kind:
      ``plateau`` for relative verifiers (:data:`RELATIVE_VERIFIER_KINDS`),
      ``threshold`` for absolute ones (``legacy``, ``gold``, generic /
      scenario evaluators).
    - ``"threshold"`` — today's behaviour (``reward >= threshold``),
      additionally gated by ``min_iter``.
    - ``"plateau"`` — champion-plateau rule; the threshold is ignored.
    - ``"none"`` — no early stop; only the ``max_iter`` hard cap applies.

    Args:
        mode: Requested stop mode (``auto`` is resolved, never stored).
        min_iter: No early stop (any mode) before this many completed
            generations. Clamped into ``[1, max_iter]``; ``1`` restores
            the ungated legacy behaviour.
        patience: Champion-plateau patience — stop once this many
            consecutive generations failed to produce a new champion.
        threshold: Reward threshold used by ``threshold`` mode only.
        max_iter: The engine's hard generation cap; ``should_stop``
            reports a ``max`` stop at the last iteration.
        verifier_kind: Verifier kind used to resolve ``auto`` and to
            decide whether capsule selection ships the champion.
    """

    def __init__(
        self,
        mode: str = "auto",
        min_iter: int = 6,
        patience: int = 3,
        threshold: float = 0.9,
        max_iter: int = 16,
        verifier_kind: str | None = None,
    ) -> None:
        kind = str(verifier_kind or "").strip().lower()
        self._relative = kind in RELATIVE_VERIFIER_KINDS
        resolved = str(mode or "auto").strip().lower()
        if resolved not in VALID_STOP_MODES:
            raise ValueError(
                f"early_stop_mode must be one of {VALID_STOP_MODES}, "
                f"got {mode!r}"
            )
        if resolved == "auto":
            resolved = "plateau" if self._relative else "threshold"
        cap = max(1, int(max_iter))
        self.mode = resolved
        self.min_iter = max(1, min(int(min_iter), cap))
        self.patience = max(1, int(patience))
        self.threshold = float(threshold)
        self.max_iter = cap

        self._champion: str | None = None
        self._since_change = 0
        self._last_reward: float | None = None

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def observe(
        self,
        uuid: str,
        iteration: int,
        reward: float | None,
        pair_outcomes: dict[str, str] | None,
        failed: bool,
    ) -> None:
        """Record one completed generation.

        Args:
            uuid: The generation's workflow uuid (may be empty when
                orchestration produced none).
            iteration: 0-based iteration index of the generation.
            reward: The generation's arrival-moment reward (any scale;
                only ``threshold`` mode reads it).
            pair_outcomes: ``{prev_uuid: "won" | "lost" | "tie"}`` — the
                normalised head-to-head outcomes of THIS generation
                against its predecessors, as persisted by the verifier.
            failed: True for errored / crashed / short-circuited
                generations: they can never become champion and count as
                a generation without improvement.
        """
        self._last_reward = None if reward is None else float(reward)
        if failed or not uuid:
            # Budget spent, no improvement: advance the plateau counter
            # once a champion exists (before that there is nothing to
            # compare against).
            if self._champion is not None:
                self._since_change += 1
            return
        if self._champion is None:
            # First generation with a real (non-error) verdict.
            self._champion = uuid
            self._since_change = 0
            return
        # Only the head-to-head outcome against the CURRENT champion
        # matters; ties keep the incumbent (a tie-tolerant rule would
        # let the champion drift through equal generations and never
        # plateau), and a missing outcome (pair excluded / errored /
        # champion not compared) counts as no change.
        outcome = (pair_outcomes or {}).get(self._champion)
        if outcome == "won":
            self._champion = uuid
            self._since_change = 0
        else:
            self._since_change += 1

    # ------------------------------------------------------------------ #
    # Decision
    # ------------------------------------------------------------------ #
    def should_stop(self, iteration: int) -> tuple[bool, str]:
        """Decide whether the run stops after generation ``iteration``.

        Args:
            iteration: 0-based iteration index that just completed.

        Returns:
            ``(stop, reason)`` — ``reason`` is a human-readable string
            when ``stop`` is True (``"max: ..."``, ``"threshold: ..."``
            or ``"plateau: ..."``) and empty otherwise.
        """
        done = iteration + 1  # completed generations
        if done >= self.max_iter:
            return True, f"max: {done}/{self.max_iter} generations run"
        if done < self.min_iter:
            return False, ""
        if self.mode == "threshold":
            if (
                self._last_reward is not None
                and self._last_reward >= self.threshold
            ):
                return True, (
                    f"threshold: reward {self._last_reward:.3f} >= "
                    f"{self.threshold:.2f} after {done} generations"
                )
            return False, ""
        if self.mode == "plateau":
            if self._champion is not None and self._since_change >= self.patience:
                return True, (
                    f"plateau: champion {self._tag(self._champion)} "
                    f"unchanged for {self._since_change} generations"
                )
            return False, ""
        return False, ""  # mode "none": only the max cap stops the run

    # ------------------------------------------------------------------ #
    # State exposure
    # ------------------------------------------------------------------ #
    @property
    def champion(self) -> str | None:
        """Current champion uuid, or None while no real verdict arrived."""
        return self._champion

    @property
    def since_change(self) -> int:
        """Generations since the champion last changed (failed incl.)."""
        return self._since_change

    @property
    def last_reward(self) -> float | None:
        """Most recently observed reward (any scale)."""
        return self._last_reward

    @property
    def relative(self) -> bool:
        """Whether the verifier reward is pairwise (relative)."""
        return self._relative

    @property
    def selects_champion(self) -> bool:
        """Whether capsule selection should ship the final champion.

        True for the plateau/none modes with a relative verifier — the
        argmax over arrival-moment rewards is meaningless across
        generations there (a 1.0 vs 1 predecessor ranks equal to a 1.0
        vs 10, and earlier generations can never be re-ranked).
        """
        return self.mode in ("plateau", "none") and self._relative

    def resolution_note(self) -> str:
        """One-line explanation of the resolved mode (for the [STOP] log)."""
        if self.mode == "plateau":
            return (
                "relative reward: threshold disabled; champion-plateau "
                f"patience={self.patience}, min_iterations={self.min_iter}, "
                f"max_iterations={self.max_iter}"
            )
        if self.mode == "threshold":
            return (
                f"absolute reward: threshold={self.threshold:.2f}, "
                f"min_iterations={self.min_iter}, "
                f"max_iterations={self.max_iter}"
            )
        return f"no early stop; run to max_iterations={self.max_iter}"

    @staticmethod
    def _tag(uuid: str) -> str:
        """Short log tag for a uuid (full Mimosa uuids are long)."""
        return uuid[-8:] if len(uuid) > 8 else uuid
