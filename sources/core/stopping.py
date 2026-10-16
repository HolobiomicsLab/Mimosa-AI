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

In Mimosa-AI's layered architecture this is the Layer 2 evolution engine's
stop policy: ``sources/core/evolution_engine.py`` builds the controller
from the run configuration's early-stop knobs at session start, and the
stop-controller tests (``tests/evolution_engine_stop_test.py``,
``tests/test_policy_bt_selection.py``) drive it directly.
"""

from __future__ import annotations

#: Verifier kinds whose reward is pairwise (relative), not absolute.
RELATIVE_VERIFIER_KINDS = frozenset({"hybrid", "policy_bt"})

#: Accepted ``early_stop_mode`` values (see ``Config.early_stop_mode``).
VALID_STOP_MODES = ("auto", "threshold", "plateau", "none")


class StopController:
    """Session-scoped stop decision for one evolution run.

    The evolution engine builds exactly one controller per session from the
    run configuration's early-stop knobs, and the stop-controller tests
    drive it directly. The controller tracks a single champion uuid plus the number
    of generations since the champion last changed, and converts that state
    into a stop/no-stop decision.

    - ``"auto"`` — resolved once at construction from the verifier kind:
      ``plateau`` for relative verifiers (:data:`RELATIVE_VERIFIER_KINDS`),
      ``threshold`` for absolute ones (``legacy``, ``gold``, generic /
      scenario evaluators).
    - ``"threshold"`` — today's behaviour (``reward >= threshold``),
      additionally gated by ``min_iter``.
    - ``"plateau"`` — champion-plateau rule; the threshold is ignored.
    - ``"none"`` — no early stop; only the ``max_iter`` hard cap applies.

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
        """Validate the mode, resolve ``auto``, and clamp the knobs.

        The evolution engine calls it once per session; all inputs come from
        the run configuration's early-stop knobs and the active verifier
        kind. An unknown mode raises; ``auto`` resolves to
        ``plateau`` for relative verifiers and ``threshold`` otherwise.

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

        Returns:
            None.

        Raises:
            ValueError: If ``mode`` is not one of :data:`VALID_STOP_MODES`.
        """
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

        The evolution engine calls it once per completed generation, passing
        the arrival reward and the persisted head-to-head pair outcomes from
        the verifier. It updates the champion and the plateau counter; the
        next ``should_stop`` call reads that state.

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

        The evolution engine calls it after each iteration to gate the next
        one; the input is the completed iteration index. The hard cap wins
        first, then ``min_iter`` gates every early stop, then the mode rule
        applies (threshold reads the last reward; plateau reads champion
        stagnation; "none" never stops early).

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
        """Current champion uuid, or None while no real verdict arrived.

        The engine logs it between generations. Returns ``str | None``: the
        reigning champion's uuid, None before the first non-error verdict.
        """
        return self._champion

    @property
    def since_change(self) -> int:
        """Generations since the champion last changed (failed incl.).

        The engine reads it for plateau logging. Returns ``int``: the count
        of generations without a champion change, failed ones included.
        """
        return self._since_change

    @property
    def last_reward(self) -> float | None:
        """Most recently observed reward (any scale).

        The engine logs it between generations. Returns ``float | None``:
        the reward of the last ``observe`` call, None before the first one.
        """
        return self._last_reward

    @property
    def relative(self) -> bool:
        """Whether the verifier reward is pairwise (relative).

        The engine reads it when picking capsule-selection semantics.
        Returns ``bool``: True for the relative verifier kinds.
        """
        return self._relative

    @property
    def selects_champion(self) -> bool:
        """Whether capsule selection should ship the final champion.

        The engine reads it when exporting the capsule after the run. True
        for the plateau/none modes with a relative verifier — the argmax
        over arrival-moment rewards is meaningless across generations there
        (a 1.0 vs 1 predecessor ranks equal to a 1.0 vs 10, and earlier
        generations can never be re-ranked). Returns ``bool``.
        """
        return self.mode in ("plateau", "none") and self._relative

    def resolution_note(self) -> str:
        """One-line explanation of the resolved mode (for the [STOP] log).

        The engine logs it once at session start. Returns ``str``: the
        threshold, patience, and iteration bounds of the resolved mode.
        """
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
        """Short log tag for a uuid (full Mimosa uuids are long).

        Internal helper; the controller's own messages call it. Returns
        ``str``: the last 8 characters, or the input unchanged when shorter.
        """
        return uuid[-8:] if len(uuid) > 8 else uuid
