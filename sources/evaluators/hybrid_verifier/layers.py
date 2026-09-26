"""EvidenceLayer extension interface + the verifier-v3 evidence layers.

The aggregator, registry, reward and gradient code are layer-agnostic:
they consume ``(claim_id, score)`` pairs keyed only by id. An evidence
layer produces those pairs for one workspace; a layer may also cap the
final reward through the optional ``gate`` hook. Three layers implement
the E35 PRIME design:

- ``ClaimsEvidenceLayer`` — the measured E19/E19b/E30 pipeline: key-claim
  extraction (content lever + T1 firewall), deterministic per-claim
  scorers, variance-filtered refinement.
- ``VisualEvidenceLayer`` — the E26 visual rung for figure tasks
  (detection >=50% image deliverables; goal-anchored visual claims frozen
  before scoring; per-generation vision scoring, same rubric both sides),
  ordered EARLY on the ladder.
- ``ExecutionGateLayer`` — the E24 execution gate (crash/no-entry/timeout
  re-execution caps the reward at 0.0, divergent at 0.5).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sources.evaluators.base import extract_json_payload

from . import claims as claims_mod
from . import digest as digest_mod
from . import inventory as inv_mod
from . import scorers as scorers_mod
from .registry import TaskRegistry
from .scorers import ExecOutcome

# Hard ceiling on live claims per task incl. replacements (E19b MAX_CLAIMS_TOTAL).
MAX_CLAIMS_TOTAL = 14


@dataclass
class LayerScore:
    """One measured evidence item for one workspace.

    Attributes:
        claim_id: Stable id; the aggregator's join key across generations.
        statement: Human-readable claim text (rendered into gradients).
        score: Measured 0..1 score; ``None`` = not measured (scorer
            failure) — excluded from the discriminating set, recorded
            with a reason in ``evidence``.
        evidence: Short measured-numbers evidence string or failure
            reason.
        category: E19b claim category (diagnostic).
        meta: Layer-specific telemetry (scorer attempts, status, ...).
    """

    claim_id: str
    statement: str
    score: float | None
    evidence: str = ""
    category: str = "other"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerContext:
    """Everything a layer needs to score one workspace.

    Attributes:
        uuid: Workflow identifier of the generation being evaluated.
        goal: Task goal text (defines the per-task registry key).
        workspace: Absolute workspace root being scored.
        registry: The task's registry (claims, score history, inventory).
        generation_index: 0-based index of this generation for the task.
    """

    uuid: str
    goal: str
    workspace: Path
    registry: TaskRegistry
    generation_index: int


@runtime_checkable
class EvidenceLayer(Protocol):
    """One source of ``(claim_id, score)`` evidence for a workspace.

    ``collect`` runs once per evaluated generation and returns the
    layer's scores; ``revise`` (optional) runs after the aggregator's
    variance filter has marked non-discriminative claims dead — a layer
    may replace them and return the replacement scores; ``gate`` is an
    optional post-collection hook that may cap the final reward (return
    ``None`` for no cap). Layers may hold per-task state through
    ``context.registry``.
    """

    name: str

    def collect(self, context: LayerContext) -> list[LayerScore]:
        """Measure one workspace; never raises (degrade per-item)."""
        ...

    def revise(
        self, context: LayerContext, dead_claim_ids: list[str]
    ) -> list[LayerScore]:
        """Optional: replace dead claims and score the replacements."""
        ...

    def gate(self, context: LayerContext, collected: list[LayerScore]) -> float | None:
        """Optional cap on the final reward (None = no cap)."""
        ...


class ClaimsEvidenceLayer:
    """Default layer: E19/E19b key claims + deterministic scorers.

    Owns the per-task claim lifecycle: extraction on first sight of a
    task, variance-driven replacement of dead claims (budgeted per
    generation), scorer-script generation with a bounded repair loop, and
    scorer execution against the workspace being scored. Scorers are
    cached in the registry and reused verbatim across generations — the
    same script grades every workspace of the task.
    """

    name = "claims"

    def __init__(
        self,
        llm_text: Callable[[str, str, str], str],
        run_script: Callable[[Path, str, str], ExecOutcome],
        logger: Any = None,
        num_claims: int = 10,
        refinement_rounds: int = 2,
        scorer_timeout_s: int = 60,
        gen_parallelism: int = 8,
        digest_max_files: int = 8,
    ) -> None:
        """Wire the layer to its host evaluator's LLM + runner helpers.

        Args:
            llm_text: ``(uuid, agent_name, prompt) -> raw text`` — the
                host evaluator's judge call (cached, retried).
            run_script: ``(workspace, code, execution_id) -> ExecOutcome``
                — pinned-subprocess execution (WorkflowRunner).
            logger: Logger shared with the host evaluator.
            num_claims: Target claim count; extraction accepts
                ``num_claims-2 .. num_claims+2`` (E19b: 8-12 for 10).
            refinement_rounds: Max replacement claims per generation.
            gen_parallelism: Max concurrent LLM/execution workers.
            digest_max_files: Cap on files sampled by the format digest.
        """
        self._llm_text = llm_text
        self._run_script = run_script
        self._log = logger
        self.num_claims = max(3, int(num_claims))
        self.refinement_rounds = max(0, int(refinement_rounds))
        self.scorer_timeout_s = int(scorer_timeout_s)
        self.gen_parallelism = max(1, int(gen_parallelism))
        self.digest_max_files = max(1, int(digest_max_files))

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _llm_json(
        self, context: LayerContext, agent: str, prompt: str
    ) -> tuple[Any, str | None]:
        """One judge call parsed as JSON; one repair round on bad JSON."""
        last_err: str | None = None
        cur = prompt
        for attempt in (1, 2):
            name = agent if attempt == 1 else f"{agent}_retry"
            try:
                raw = self._llm_text(context.uuid, name, cur)
            except Exception as e:  # noqa: BLE001 — degrade per-claim
                return None, f"judge call failed: {type(e).__name__}: {e}"
            payload = extract_json_payload(raw or "")
            if payload:
                try:
                    return json.loads(payload), None
                except json.JSONDecodeError as e:
                    last_err = f"invalid JSON: {e}"
            else:
                last_err = "no JSON object found in response"
            if attempt == 1 and self._log is not None:
                self._log.warning(f"[{name}] JSON parse failed ({last_err}); retrying")
            cur = (
                f"{prompt}\n\nPREVIOUS ATTEMPT FAILED: {last_err}\n"
                "Reply with ONLY the strict JSON object requested, no prose, "
                "no markdown fences, no commentary."
            )
        return None, last_err

    # ------------------------------------------------------------------
    # Claim lifecycle
    # ------------------------------------------------------------------

    def _extract(self, context: LayerContext) -> str | None:
        """First-sight claim extraction; returns None on total failure.

        E35 semantics: the T1 firewall (E29) screens every extracted
        claim for smuggled numeric answer keys — ONE batched LLM call
        per attempt over the whole claim set — violating claims are
        rejected and the extraction re-issued with feedback naming them
        (max 3 attempts; survivors of the last attempt are kept). The
        E30 composition mandate (>=2 core_computation, >=2
        method_identity) is enforced by ``validate_claims`` with the same
        retry budget, then the model's best effort is accepted and the
        shortfall disclosed (E35: 18/19 groups met it on prompt power
        alone).
        """
        reg = context.registry
        lo = max(3, self.num_claims - 2)
        hi = self.num_claims + 2
        current = inv_mod.scan_workspace(context.workspace)
        union = inv_mod.render_union_inventory(
            reg.inventory,
            reg.n_workspaces,
            current=current.keys(),
        )
        previews = inv_mod.deliverable_previews(current, context.workspace)
        prompt = claims_mod.extract_claims_prompt(context.goal, union, previews, lo, hi)
        max_attempts = 3
        claims: list[dict[str, Any]] = []
        err: str | None = None
        t1_rejects: dict[str, list[str]] = {}
        for attempt in range(1, max_attempts + 1):
            cur = prompt
            if attempt > 1:
                cur = (
                    f"{prompt}\n\nPREVIOUS ATTEMPT REJECTED (T1 FIREWALL / "
                    "COMPOSITION): claims "
                    f"{json.dumps(t1_rejects)} hard-code specific numeric "
                    "constants the goal does not state — answer-smuggling, "
                    "unfalsifiable — or the composition mandate went unmet "
                    "(at least 2 core_computation AND at least 2 "
                    "method_identity). Re-issue the FULL claim set with those "
                    "claims re-phrased as workspace-verifiable properties (no "
                    "numeric answer constants) and every mandate satisfied. "
                    "Reply with ONLY the strict JSON object requested."
                )
            parsed, jerr = self._llm_json(context, "hybrid_extract_claims", cur)
            if jerr is not None:
                err = jerr
                continue
            out, verr = claims_mod.validate_claims(
                parsed,
                lo,
                hi,
                require_composition=(attempt < max_attempts),
                llm_text=self._llm_text,
                uuid=context.uuid,
            )
            if verr is None:
                t1_rejects = {
                    k: v
                    for k, v in claims_mod.t1_violations(
                        out, context.goal, self._llm_text, context.uuid
                    ).items()
                    if v
                }
                claims = [c for c in out if c["id"] not in t1_rejects]
                if not t1_rejects or attempt == max_attempts:
                    break
                continue  # rejected claims -> re-extract with feedback
            err = verr
            if out and verr.startswith("composition"):
                claims = out  # valid rubric, mandate unmet: keep best effort
            continue
        if not claims:
            if self._log is not None and err is not None:
                self._log.warning(
                    f"[{context.uuid}] hybrid claim extraction failed: {err}"
                )
            return err or "extraction produced no usable claims"
        if t1_rejects and self._log is not None:
            self._log.warning(
                f"[{context.uuid}] T1 firewall dropped claims after "
                f"{max_attempts} attempts: {sorted(t1_rejects)}"
            )
        if (cerr := claims_mod.composition_error(claims)) and self._log is not None:
            self._log.warning(
                f"[{context.uuid}] E30 composition mandate unmet, "
                f"disclosed: {cerr}"
            )
        reg.seed_claims(claims)
        return None

    def revise(
        self, context: LayerContext, dead_claim_ids: list[str]
    ) -> list[LayerScore]:
        """Replace dead claims (budgeted) and score the replacements.

        Called by the evaluator right after the variance filter marked
        claims dead; up to ``refinement_rounds`` replacement claims per
        generation are requested in ONE LLM call (E19b refinement), then
        each replacement gets a scorer built and executed against the
        current workspace like any other claim.
        """
        reg = context.registry
        # The visual rung (E26) owns its claims; never replace them here.
        dead = [
            c
            for c in reg.dead_claims()
            if c["id"] in set(dead_claim_ids) and c.get("stage") != "visual"
        ]
        alive = [c for c in reg.alive_claims() if c.get("stage") != "visual"]
        budget = self.refinement_rounds - int(reg.refinements.get(context.uuid, 0))
        if not dead or budget <= 0:
            return []
        n_new = min(len(dead), budget, MAX_CLAIMS_TOTAL - len(alive))
        if n_new <= 0:
            return []
        to_replace = dead[:n_new]
        union = self._union_for_prompt(context)
        prompt = claims_mod.refinement_prompt(
            goal=context.goal,
            union_inventory=union,
            outcome_report=claims_mod.outcome_report(reg.claims, reg.generations),
            dead_block="\n".join(
                f"- [{c['id']}] ({c.get('category', 'other')}) {c['statement']}\n"
                f"  target: {c['target']}\n  happened: " + self._dead_reason(reg, c)
                for c in to_replace
            ),
            alive_block="\n".join(
                f"- [{c['id']}] {c['statement'][:140]}" for c in alive
            ),
            n_new=n_new,
            next_id=claims_mod.next_claim_id(reg.claims),
        )
        parsed, err = self._llm_json(context, "hybrid_refine_claims", prompt)
        new_claims: list[dict[str, Any]] = []
        if err is None:
            new_claims, err = claims_mod.validate_claims(
                parsed, 1, n_new, llm_text=self._llm_text, uuid=context.uuid
            )
        if err is not None or not new_claims:
            if self._log is not None:
                self._log.warning(f"[{context.uuid}] hybrid refinement failed: {err}")
            return []
        new_claims = claims_mod.dedupe_claim_ids(new_claims, reg.claims)[:n_new]
        t1 = {
            k: v
            for k, v in claims_mod.t1_violations(
                new_claims, context.goal, self._llm_text, context.uuid
            ).items()
            if v
        }
        if t1:
            if self._log is not None:
                self._log.warning(
                    f"[{context.uuid}] T1 firewall rejected replacement "
                    f"claims: {sorted(t1)}"
                )
            new_claims = [c for c in new_claims if c["id"] not in t1]
        if not new_claims:
            return []
        reg.replace_claims(
            [c["id"] for c in to_replace[: len(new_claims)]],
            new_claims,
            context.generation_index,
        )
        reg.refinements[context.uuid] = int(reg.refinements.get(context.uuid, 0)) + len(
            new_claims
        )
        results = [self._score_claim_safe(context, c) for c in new_claims]
        return sorted(results, key=lambda s: s.claim_id)

    # ------------------------------------------------------------------
    # Scorers
    # ------------------------------------------------------------------

    @staticmethod
    def _dead_reason(reg: TaskRegistry, claim: dict[str, Any]) -> str:
        """Why one claim died (rendered into the refinement prompt)."""
        if claim.get("drop_reason") == "all_fail":
            return "scorer failed on all workspaces"
        n = len(reg.observed_scores(claim["id"]))
        return f"all {n} workspaces scored equally"

    def _union_for_prompt(self, context: LayerContext) -> str:
        reg = context.registry
        return inv_mod.render_union_inventory(
            reg.inventory,
            reg.n_workspaces,
            current=inv_mod.scan_workspace(context.workspace).keys(),
        )

    def _ensure_digests(self, context: LayerContext) -> str:
        """One-shot format digests for the task (cached in the registry).

        Returns the rendered digests block for policy prompts; empty when
        the task has no digestible deliverables or the call soft-failed.
        """
        reg = context.registry
        if reg.digests:
            return digest_mod.render_digests(reg.digests)
        current = inv_mod.scan_workspace(context.workspace)
        digests = digest_mod.ensure_digests(
            workspace=context.workspace,
            inventory=current,
            cached=reg.digests,
            llm_text=lambda agent, prompt: self._llm_text(context.uuid, agent, prompt),
            uuid=context.uuid,
            max_files=self.digest_max_files,
            logger=self._log,
        )
        if digests:
            reg.digests = digests
        return digest_mod.render_digests(digests)

    def _build_scorer(
        self, context: LayerContext, claim: dict[str, Any]
    ) -> tuple[str | None, dict[str, Any], dict[str, Any] | None]:
        """Generate + repair one policy script (ladder rung).

        Returns ``(script|None, telemetry, parsed|None)`` — the parse of
        the first successful execution on this workspace, mirroring E19's
        build loop (no re-execution after a working attempt).
        """
        union = self._union_for_prompt(context)
        digests_block = digest_mod.render_digests(context.registry.digests)
        feedback: str | None = None
        code: str | None = None
        attempts: list[dict[str, Any]] = []
        best_ok: str | None = None
        best_parsed: dict[str, Any] | None = None
        for attempt in range(1, scorers_mod.MAX_CODE_ATTEMPTS + 1):
            if attempt == 1:
                prompt = scorers_mod.scorer_prompt(
                    context.goal, claim, union, digests_block=digests_block
                )
                agent = f"hybrid_scorer_{claim['id']}"
            else:
                prompt = scorers_mod.repair_prompt(feedback or "", code or "")
                agent = f"hybrid_scorer_{claim['id']}_repair{attempt}"
            rec: dict[str, Any] = {"attempt": attempt}
            try:
                raw = self._llm_text(context.uuid, agent, prompt)
            except Exception as e:  # noqa: BLE001 — degrade per-claim
                rec["parse"] = f"call_failed: {type(e).__name__}: {e}"
                attempts.append(rec)
                break
            code = scorers_mod.parse_code(raw)
            if code is None:
                rec["parse"] = "no_code_block"
                attempts.append(rec)
                feedback = "model returned no python code block"
                continue
            violations = scorers_mod.static_violations(code)
            if violations:
                rec["parse"] = "static_violation"
                rec["static_violations"] = violations[:5]
                attempts.append(rec)
                feedback = "static policy violations: " + "; ".join(violations[:5])
                continue
            outcome = self._run_script(
                context.workspace,
                scorers_mod.build_launcher(context.workspace, claim, code),
                f"hybrid_score_{claim['id']}",
            )
            parsed = (
                scorers_mod.parse_scorer_output(outcome.stdout)
                if outcome.rc == 0
                else None
            )
            rec["executed"] = True
            rec["ok"] = parsed is not None
            attempts.append(rec)
            if parsed is not None:
                best_ok = code
                best_parsed = parsed
                break
            feedback = scorers_mod.feedback_lines(outcome, self.scorer_timeout_s)
        return best_ok, {"attempts": attempts}, best_parsed

    def _score_claim(self, context: LayerContext, claim: dict[str, Any]) -> LayerScore:
        """Ensure a scorer exists, run it, return the LayerScore."""
        reg = context.registry
        cid = claim["id"]
        script = claim.get("scorer")
        parsed: dict[str, Any] | None = None
        if not script:
            script, telemetry, parsed = self._build_scorer(context, claim)
            reg.set_scorer(cid, script)
            claim["scorer"] = script
        else:
            telemetry = {"attempts": []}
        if not script:
            return LayerScore(
                claim_id=cid,
                statement=claim["statement"],
                score=None,
                evidence="scorer failed: could not produce a working script",
                category=claim.get("category", "other"),
                meta={"scorer_status": "failed", **telemetry},
            )
        if parsed is None:
            # Cached scorer (or a build that never parsed): execute now.
            outcome = self._run_script(
                context.workspace,
                scorers_mod.build_launcher(context.workspace, claim, script),
                f"hybrid_score_{cid}",
            )
            parsed = (
                scorers_mod.parse_scorer_output(outcome.stdout)
                if outcome.rc == 0
                else None
            )
            if parsed is None:
                return LayerScore(
                    claim_id=cid,
                    statement=claim["statement"],
                    score=None,
                    evidence="scorer failed: "
                    + scorers_mod.feedback_lines(outcome, self.scorer_timeout_s)[-200:],
                    category=claim.get("category", "other"),
                    meta={"scorer_status": "exec_failed", **telemetry},
                )
        return LayerScore(
            claim_id=cid,
            statement=claim["statement"],
            score=float(parsed["score"]),
            evidence=str(parsed["evidence"]),
            category=claim.get("category", "other"),
            meta={"scorer_status": "ok", **telemetry},
        )

    def _score_claim_safe(
        self, context: LayerContext, claim: dict[str, Any]
    ) -> LayerScore:
        """Soft-fail wrapper: one bad claim never crashes the layer."""
        try:
            return self._score_claim(context, claim)
        except Exception as e:  # noqa: BLE001
            if self._log is not None:
                self._log.warning(
                    f"[{context.uuid}] scoring claim {claim.get('id')} raised: "
                    f"{type(e).__name__}: {e}"
                )
            return LayerScore(
                claim_id=str(claim.get("id", "?")),
                statement=str(claim.get("statement", "")),
                score=None,
                evidence=f"scorer raised: {type(e).__name__}: {e}",
                category=claim.get("category", "other"),
                meta={"scorer_status": "raised"},
            )

    # ------------------------------------------------------------------
    # EvidenceLayer interface
    # ------------------------------------------------------------------

    def collect(self, context: LayerContext) -> list[LayerScore]:
        """Extract claims on first sight, then score the workspace on each."""
        reg = context.registry
        if not reg.claims:
            err = self._extract(context)
            if err is not None or not reg.claims:
                return []
        # Pre-policy digest stage: ONE LLM call per task, cached in the
        # registry, injected into every policy-writer prompt.
        self._ensure_digests(context)
        # The visual rung (E26) scores its own claims via the vision
        # model — never build Python scorers for stage "visual".
        alive = [c for c in reg.alive_claims() if c.get("stage") != "visual"]
        if not alive:
            return []
        results: list[LayerScore] = []
        lock = threading.Lock()
        workers = max(1, min(len(alive), self.gen_parallelism))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(self._score_claim_safe, context, c): c["id"] for c in alive
            }
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    score = fut.result()
                except Exception as e:  # noqa: BLE001 — belt and braces
                    score = LayerScore(
                        claim_id=cid,
                        statement="",
                        score=None,
                        evidence=f"scorer task failed: {e}",
                    )
                with lock:
                    results.append(score)
        results.sort(key=lambda s: s.claim_id)
        return results

    def gate(self, context: LayerContext, collected: list[LayerScore]) -> float | None:
        """No cap by default — the claims layer is the primary signal."""
        return None

# ---------------------------------------------------------------------
# E26 visual rung (E35: ordered EARLY — visual -> script -> log -> result)
# ---------------------------------------------------------------------

MAX_IMAGES_SCORE = 3  # <=3 largest deliverable images scored per workspace
MAX_IMAGES_EXTRACT = 2  # <=2 sample figures ground the claims extraction
MIN_VISUAL_CLAIMS = 3
MAX_VISUAL_CLAIMS = 6

# E26 prompt conventions, verbatim structure (experiments E26/E35).
VISUAL_CLAIMS_PROMPT = """You are designing VISUAL acceptance checks for a scientific figure deliverable, to be scored later against many independently generated workspaces of the SAME task.

TASK GOAL (verbatim):
<<<
{goal}
>>>

WORKSPACE FILE LISTING (union over all evolved workspaces of this task; the figure deliverable(s) are tagged [FIGURE]):
{listing}

Attached below: up to {n_img} sample figure(s) from the most complete workspace of this task. They are provided ONLY to ground what the deliverable looks like; do NOT tune the claims to idiosyncrasies of these particular samples.

Produce between {mn} and {mx} VISUAL claims that a grader can check by LOOKING AT THE FIGURE(S) of any workspace for this task. Requirements:
- Each claim MUST be anchored in the TASK GOAL: the goal-specified plot type and its required axes/elements/annotations, the relationship or content the goal asks to visualize, or a legibility/rendering requirement (axis labels present and readable, no clipped or overlapping elements).
- FORBIDDEN: anything about code, scripts, logs, stdout, CSV/text tables, or file existence (other verification stages cover those); anything ANY figure for this task would trivially satisfy (e.g. "a figure is produced", "the image is a plot"); anything requiring the sample figures' specific numbers or layout.
- Each claim gets a 0-1 scoring rubric with explicit partial credit (what earns 0.0, partial credit conditions, what earns 1.0).

Output STRICT JSON exactly:
{{"claims": [{{"id": "V1", "statement": "<one checkable visual property>", "rubric": "1.0 if ...; 0.5 if ...; 0.0 if ..."}}, ...]}}
with consecutive ids V1..Vk."""

VISUAL_SCORE_PROMPT = """You are scoring ONE workspace's figure deliverable against FIXED visual acceptance criteria. The same criteria and rubric are used for every workspace of this task; your job is only to judge the attached figure(s) honestly against them.

TASK GOAL (verbatim):
<<<
{goal}
>>>

CRITERIA (fixed; score ONLY what you can see in the attached figure(s)):
{criteria}

Attached below: this workspace's figure deliverable(s) (pred_results images, largest first).

For EACH criterion, output:
- "score": 0.0-1.0 strictly per the criterion's rubric (partial credit allowed);
- "evidence": <=40 words describing what you actually saw (colors/labels/elements), or what is missing.

Output STRICT JSON exactly:
{{"scores": {{"V1": {{"score": 0.0, "evidence": "..."}}, "V2": {{"...": "..."}}}}}}"""


class VisualEvidenceLayer:
    """E26 visual rung: goal-anchored visual claims scored by a vision LLM.

    Detection (E26, production adaptation): the task is a figure task
    when >=50% of the union inventory's deliverable files are images
    (``inventory.is_figure_task``). On first sight of a figure task the
    layer extracts 3-6 goal-anchored visual claims in ONE vision call
    grounded with <=2 sample figures, freezes them into the task registry
    BEFORE any scoring, then scores every generation's <=3 largest
    deliverable figures 0-1 per claim in one call per generation — same
    rubric text both sides, temperature 0, no pairwise calls (E26 gates:
    coverage 50/50, determinism 3/3). The claims join the ladder at stage
    ``visual``, ordered BEFORE ``script`` (E26: visual-early 0.833 vs
    0.646 as 4th rung).
    """

    name = "visual"

    def __init__(
        self,
        vision_call: Callable[[str, str, str, list[Path]], str],
        logger: Any = None,
        min_claims: int = MIN_VISUAL_CLAIMS,
        max_claims: int = MAX_VISUAL_CLAIMS,
        max_images_score: int = MAX_IMAGES_SCORE,
        max_images_extract: int = MAX_IMAGES_EXTRACT,
    ) -> None:
        """Wire the layer to the host evaluator's vision-model call.

        Args:
            vision_call: ``(uuid, agent_name, prompt, images) -> raw text``
                — one vision-LLM round-trip with the image files attached
                (kimi-k3 transport, temperature 0).
            logger: Logger shared with the host evaluator.
            min_claims / max_claims: Visual-claim count window (E26: 3-6).
            max_images_score: Figures scored per workspace (E26: <=3).
            max_images_extract: Sample figures grounding extraction (<=2).
        """
        self._vision_call = vision_call
        self._log = logger
        self.min_claims = max(1, int(min_claims))
        self.max_claims = max(self.min_claims, int(max_claims))
        self.max_images_score = max(1, int(max_images_score))
        self.max_images_extract = max(1, int(max_images_extract))

    # ------------------------------------------------------------------
    # Vision plumbing
    # ------------------------------------------------------------------

    def _vision_json(
        self, context: LayerContext, agent: str, prompt: str, images: list[Path]
    ) -> tuple[Any, str | None]:
        """One vision call parsed as JSON; one repair round on bad JSON."""
        last_err: str | None = None
        cur = prompt
        for attempt in (1, 2):
            name = agent if attempt == 1 else f"{agent}_retry"
            try:
                raw = self._vision_call(context.uuid, name, cur, images)
            except Exception as e:  # noqa: BLE001 — degrade per-call
                return None, f"vision call failed: {type(e).__name__}: {e}"
            payload = extract_json_payload(raw or "")
            if payload:
                try:
                    return json.loads(payload), None
                except json.JSONDecodeError as e:
                    last_err = f"invalid JSON: {e}"
            else:
                last_err = "no JSON object found in response"
            if attempt == 1 and self._log is not None:
                self._log.warning(f"[{name}] JSON parse failed ({last_err}); retrying")
            cur = (
                f"{prompt}\n\nPREVIOUS ATTEMPT FAILED: {last_err}\n"
                "Reply with ONLY the strict JSON object requested, no prose, "
                "no markdown fences, no commentary."
            )
        return None, last_err

    # ------------------------------------------------------------------
    # Visual claims lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _deliverable_images(
        inventory: dict[str, dict[str, Any]], workspace: Path, cap: int
    ) -> list[Path]:
        """Largest deliverable images first; results-dir figures preferred."""
        rels = [p for p, i in inventory.items() if i.get("kind") == "png"]
        rels.sort(
            key=lambda p: (
                not ("pred_results" in p or "results" in Path(p).parent.as_posix().lower()),
                -int(inventory[p].get("bytes", 0)),
                p,
            )
        )
        return [workspace / p for p in rels[:cap] if (workspace / p).is_file()]

    def _figure_listing(self, reg: TaskRegistry) -> str:
        """Union-inventory listing with [FIGURE] tags for the extractor."""
        n = max(1, reg.n_workspaces)
        lines = []
        for rel in sorted(reg.inventory):
            info = reg.inventory[rel]
            kind = info.get("kind", "other")
            tag = " [FIGURE]" if kind == "png" else ""
            sig = f" ({info['signal']})" if kind == "png" and info.get("signal") else ""
            lines.append(
                f"  {rel}{tag}{sig}  [in {info.get('count', 0)}/{n} generations]"
            )
        return "\n".join(lines[:300]) or "(empty inventory)"

    def _extract_claims(self, context: LayerContext) -> list[dict[str, Any]]:
        """One vision call: 3-6 goal-anchored visual claims (frozen after)."""
        current = inv_mod.scan_workspace(context.workspace)
        samples = self._deliverable_images(
            current, context.workspace, self.max_images_extract
        )
        prompt = VISUAL_CLAIMS_PROMPT.format(
            goal=context.goal[:4000],
            listing=self._figure_listing(context.registry),
            n_img=len(samples),
            mn=self.min_claims,
            mx=self.max_claims,
        )
        parsed, err = self._vision_json(
            context, "hybrid_visual_extract", prompt, samples
        )
        if err is not None:
            if self._log is not None:
                self._log.warning(
                    f"[{context.uuid}] visual claims extraction failed: {err}"
                )
            return []
        raw = parsed.get("claims") if isinstance(parsed, dict) else None
        out: list[dict[str, Any]] = []
        for i, c in enumerate(raw or []):
            if not isinstance(c, dict):
                continue
            stmt = str(c.get("statement", "")).strip()
            rubric = str(c.get("rubric", "")).strip()
            if stmt and rubric:
                out.append(
                    {
                        "id": f"V{i + 1}",
                        "stage": "visual",
                        "temporal_index": i + 1,
                        "category": "visual",
                        "statement": stmt,
                        "target": "figure deliverables (results-dir images)",
                        "scoring_rule": rubric,
                    }
                )
        if not (self.min_claims <= len(out) <= self.max_claims):
            if self._log is not None:
                self._log.warning(
                    f"[{context.uuid}] visual extraction yielded "
                    f"{len(out)} claims (need {self.min_claims}-"
                    f"{self.max_claims}); visual rung disabled for now"
                )
            return []
        return out

    def _score_figures(
        self,
        context: LayerContext,
        claims: list[dict[str, Any]],
        images: list[Path],
    ) -> list[LayerScore]:
        """One vision call: every frozen claim scored 0-1 on the figures."""
        criteria = "\n".join(
            f"- {c['id']}: {c['statement']}\n  RUBRIC: {c['scoring_rule']}"
            for c in claims
        )
        prompt = VISUAL_SCORE_PROMPT.format(goal=context.goal[:4000], criteria=criteria)
        parsed, err = self._vision_json(
            context, "hybrid_visual_score", prompt, images
        )
        if err is not None:
            return [
                LayerScore(
                    claim_id=c["id"],
                    statement=c["statement"],
                    score=None,
                    evidence=f"visual scorer failed: {err}",
                    category=c.get("category", "visual"),
                    meta={"scorer_status": "vision_failed"},
                )
                for c in claims
            ]
        raw = parsed.get("scores") if isinstance(parsed, dict) else {}
        out: list[LayerScore] = []
        for c in claims:
            item = raw.get(c["id"]) if isinstance(raw, dict) else None
            score: float | None = None
            if isinstance(item, dict) and item.get("score") is not None:
                try:
                    score = max(0.0, min(1.0, float(item["score"])))
                except (TypeError, ValueError):
                    score = None
            ev = str((item or {}).get("evidence", ""))[:240] if item else ""
            if score is None:
                ev = ev or f"visual scorer returned no score for {c['id']}"
            out.append(
                LayerScore(
                    claim_id=c["id"],
                    statement=c["statement"],
                    score=score,
                    evidence=ev,
                    category=c.get("category", "visual"),
                    meta={"scorer_status": "ok" if score is not None else "no_score"},
                )
            )
        return out

    # ------------------------------------------------------------------
    # EvidenceLayer interface
    # ------------------------------------------------------------------

    def collect(self, context: LayerContext) -> list[LayerScore]:
        """Detect figure tasks; freeze visual claims once; score this one."""
        reg = context.registry
        if not inv_mod.is_figure_task(reg.inventory):
            return []
        claims = [
            c
            for c in reg.alive_claims()
            if c.get("stage") == "visual"
        ]
        if not claims:
            claims = self._extract_claims(context)
            if not claims:
                return []
            reg.append_claims(claims, context.generation_index)
            if self._log is not None:
                self._log.info(
                    f"[{context.uuid}] visual rung armed: {len(claims)} "
                    f"goal-anchored visual claims frozen before scoring"
                )
        current = inv_mod.scan_workspace(context.workspace)
        images = self._deliverable_images(
            current, context.workspace, self.max_images_score
        )
        if not images:
            # E26 semantics: a figure task workspace without figures
            # scores 0 on every visual claim, not "unmeasured".
            return [
                LayerScore(
                    claim_id=c["id"],
                    statement=c["statement"],
                    score=0.0,
                    evidence="no figure produced",
                    category=c.get("category", "visual"),
                    meta={"scorer_status": "no_figures"},
                )
                for c in claims
            ]
        return self._score_figures(context, claims, images)

    def revise(
        self, context: LayerContext, dead_claim_ids: list[str]
    ) -> list[LayerScore]:
        """Visual claims are frozen per task (E26); no replacements."""
        return []

    def gate(self, context: LayerContext, collected: list[LayerScore]) -> float | None:
        """No cap — the visual rung contributes evidence, not gates."""
        return None


# ---------------------------------------------------------------------
# E24 execution gate (E35 stage 5)
# ---------------------------------------------------------------------

# E24 re-execution status -> reward cap. Timeout counts with crash (E24's
# strict F4 convention; disclosed E35 mapping). ``None`` = no cap.
GATE_CAPS: dict[str, float | None] = {
    "crash": 0.0,
    "no_entry_script": 0.0,
    "timeout": 0.0,
    "divergent": 0.5,
    "clean_recover": None,
}

# Frozen E24 re-execution statuses (benchmark reproduction path). Live
# re-execution is a future enhancement through this same layer interface.
FROZEN_REEXEC_LEDGER = (
    Path(__file__).resolve().parents[3]
    / "experiments_verifiers"
    / "results"
    / "E24_reexec_holdout_2026-09-24.json"
)


class ExecutionGateLayer:
    """E24 execution gate: re-execution status caps the final reward.

    E24 measured that 29% of workspaces fail clean re-execution — a
    generation whose delivered entry script crashes (or has no entry, or
    times out) under clean-room re-execution cannot outrank rivals on
    artifact claims alone, so its reward is capped at 0.0; a divergent
    re-execution (regenerated outputs differ from the frozen ones) caps
    it at 0.5; a clean recovery leaves the reward uncapped. Status data
    comes from a lookup (frozen E24 ledger by default; a live
    re-execution check plugs in here) — an unknown generation has no
    re-execution evidence and is never capped.
    """

    name = "execution_gate"

    def __init__(
        self,
        status_lookup: Callable[[str], tuple[str | None, float | None]] | None = None,
        logger: Any = None,
    ) -> None:
        """Wire the gate to its status source.

        Args:
            status_lookup: ``uuid -> (status, runtime_s)`` where status is
                one of the E24 classes (``crash`` / ``no_entry_script`` /
                ``timeout`` / ``divergent`` / ``clean_recover``) or None
                when no re-execution evidence exists. Defaults to the
                frozen E24 ledger keyed by uuid prefix.
            logger: Logger shared with the host evaluator.
        """
        self._lookup = status_lookup or self._frozen_lookup
        self._log = logger
        self._frozen: dict[str, tuple[str | None, float | None]] | None = None
        self.last_facts: dict[str, Any] | None = None

    def _load_frozen(self) -> dict[str, tuple[str | None, float | None]]:
        """uuid8 -> (reexec_status, runtime_s) from the frozen E24 ledger."""
        if self._frozen is not None:
            return self._frozen
        table: dict[str, tuple[str | None, float | None]] = {}
        try:
            data = json.loads(FROZEN_REEXEC_LEDGER.read_text(encoding="utf-8"))
            for row in data.get("gen_rows", []):
                key = str(row.get("uuid8") or "")
                if key:
                    table[key] = (
                        row.get("reexec_status"),
                        row.get("runtime_s"),
                    )
        except (OSError, json.JSONDecodeError) as e:
            if self._log is not None:
                self._log.warning(f"could not read frozen re-exec ledger: {e}")
        self._frozen = table
        return table

    def _frozen_lookup(self, uuid: str) -> tuple[str | None, float | None]:
        """Frozen-ledger lookup by uuid8 prefix (miss -> no evidence)."""
        return self._load_frozen().get(uuid[:8], (None, None))

    # ------------------------------------------------------------------
    # EvidenceLayer interface
    # ------------------------------------------------------------------

    def collect(self, context: LayerContext) -> list[LayerScore]:
        """The gate produces no claim scores — only the reward cap."""
        return []

    def revise(
        self, context: LayerContext, dead_claim_ids: list[str]
    ) -> list[LayerScore]:
        return []

    def gate(self, context: LayerContext, collected: list[LayerScore]) -> float | None:
        """Cap from the generation's re-execution status (None = no cap)."""
        status: str | None = None
        runtime: float | None = None
        try:
            status, runtime = self._lookup(context.uuid)
        except Exception as e:  # noqa: BLE001 — no evidence -> no cap
            if self._log is not None:
                self._log.warning(
                    f"[{context.uuid}] execution-gate lookup failed: {e}"
                )
        cap = GATE_CAPS.get(str(status or "clean_recover"))
        self.last_facts = {
            "status": status,
            "runtime_s": runtime,
            "cap": cap,
        }
        if cap is not None and self._log is not None:
            self._log.info(
                f"[{context.uuid}] execution gate: {status} re-execution "
                f"caps the reward at {cap}"
            )
        return cap
