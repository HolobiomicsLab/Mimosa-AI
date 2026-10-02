"""
Claim extraction (six independent sources) and importance rating.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


# ----- Importance rating fan-out ---------------------------------------------
# Phase B (rating) is split into batches of this size and run in a thread pool.
# Single-shot rating on 30+ claims with rationales is the slowest verifier step
# because it generates one big structured JSON output; fanning it out collapses
# wall-clock without changing the rubric.
_IMPORTANCE_BATCH_SIZE = 10
_IMPORTANCE_PARALLELISM = 4

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sources.cli.pretty_print import (
    print_info,
    print_ok,
    print_warn,
)
from sources.evaluators.base import LLMEvaluationError
from sources.evaluators.verifier_claim_sources import ClaimContext, SOURCES


# Importance anchors shown to the rater LLM so it doesn't collapse to
# the middle of the scale. Kept short on purpose — long anchors waste
# tokens and tend to confuse small judges.
_IMPORTANCE_ANCHOR_BLOCK = """
Anchored scale (for all other claims):
- 10  The deliverable named in the goal does not exist or is wrong without this.
     e.g. "the conformation achieves energy -7 or lower" (headline metric);
            "the lower-bound proof establishes -11 via a valid counting argument"
-  9  Not the named deliverable, but the result is invalid without it — a core
       methodology step the goal's correctness depends on.
       e.g. "every step uses the exact 20-mer 'HPHPPHHPHPPHPHHPPHPH'"
-  8  A required methodology step that invalidates or fakes the result if missing AND claims about requirements.txt present and pinned.
       e.g.  "the search does genuine algorithmic exploration, not a hardcoded coordinate list";
-  7  A literature-required step that materially changes the result if skipped.
         e.g. "the energy minimisation must converge to a stationary point";
-  6  A required-by-convention property whose absence weakens but does not invalidate the result.
       e.g. "the conformation is non-degenerate (not a straight line or hairpin)";
-  5  A non-negotiable sanity property — cheap to check, embarrassing if wrong.
       e.g. "the conformation has exactly 20 coordinates, matching sequence length"
-  4  A literature-recommended best practice that improves trust, not validity.
        eg. "The used algorithm is Monte Carlo search"
-  3  Advisory / hygiene. Affects maintainability, not the result.
-  2  Nice-to-have, not expected by the goal.
       e.g. "the workspace is free of pathological clutter / junk-file dumps"
-  1  Tangential.

FIXED-FLOOR CLAIMS
- Reproducibility artifacts (requirements.txt present, dependencies pinned) = 8.
  A run without these FAILS, so they are never advisory. Score them 8 regardless of what the goal is about.

Use the FULL scale; Goal-alignment dominates."""


class _VerifierClaimExtractionMixin:
    """Claim-extraction and importance-rating methods.

    Sentinel claim emitted by ``_extract_claims`` when the workflow produced
    nothing actionable; rated importance 10 because failure to execute is
    failure of the literal deliverable.
    """

    # ------------------------------------------------------------------
    # Stage 1 — claim extraction
    # ------------------------------------------------------------------

    def _extract_claims(
        self,
        uuid: str,
        goal: str,
        execution_text: str,
        workspace_listing: str,
        is_truly_empty: bool,
        grounding: str = "",
    ) -> list[dict[str, Any]]:
        """Extract atomic claims by polling six independent source prompts.

        Args:
            uuid: Workflow identifier (used for logging and judge calls).
            goal: Original workflow goal text.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.
            is_truly_empty: When True, returns a single sentinel "execution succeeded" claim and skips extraction.
            grounding: Optional peer-reviewed literature grounding block.

        Returns:
            Filtered claim list with stable ids, source labels, importance
            (1-10) and a one-sentence importance rationale on every entry.
        """
        if is_truly_empty:
            return [{
                "id": "c0_execution_succeeded",
                "description": "The workflow executed to completion and produced a non-empty answer.",
                "importance": 10,
                "importance_rationale": "literal deliverable; absent here",
                "likely_relevant_files": [],
            }]

        task_key = self._task_cache_key(goal)
        cached = self._load_cached_rubric(task_key)
        if cached is not None:
            adapted = self._adapt_rubric_to_workspace(cached)
            print_ok(
                f"Rubric cache HIT for {uuid}: reusing {len(adapted)} claims "
                f"(skipped extraction, dedup, importance)"
            )
            return adapted

        per_source_min, per_source_max = self._per_source_targets(n_sources=len(SOURCES))
        base_ctx = ClaimContext(
            goal=goal,
            workspace_listing=workspace_listing,
            target_min=per_source_min,
            target_max=per_source_max,
            grounding=grounding,
            execution_text=execution_text,
        )
        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        per_source_elapsed: list[tuple[str, float, int]] = []
        t_sources = time.time()
        for source in SOURCES:
            label = source.label
            prompt = source.build(base_ctx)
            t_src = time.time()
            data, err = self._call_judge_for_json(
                uuid, f"verifier_extract_claims_{label}", prompt
            )
            if err is not None:
                print_warn(f"Claim extraction source {label} failed for {uuid}: {err}")
                self.logger.warning(
                    f"Claim extraction source {label} failed for {uuid}: {err}"
                )
                per_source_elapsed.append((label, time.time() - t_src, 0))
                continue
            new_claims = self._parse_and_filter_claims(uuid, data)
            for claim in new_claims:
                print_info(f"Extracted claim {claim['id']} from source {label} for {uuid}")
                claim_id = claim["id"]
                if claim_id in seen_ids:
                    claim_id = f"{claim_id}_{label}"
                claim["id"] = claim_id
                claim["source"] = f"source_{label}"
                seen_ids.add(claim_id)
                merged.append(claim)
            per_source_elapsed.append((label, time.time() - t_src, len(new_claims)))

        sources_dt = time.time() - t_sources
        breakdown = "\n".join(
            f"  source {lbl}: {dt:>6.1f}s  ({n} claims)"
            for lbl, dt, n in per_source_elapsed
        )
        print_ok(
            f"Extracted claims for workflow {uuid} from {len(SOURCES)} sources "
            f"in {sources_dt:.1f}s ({len(merged)} merged):\n{breakdown}"
        )
        if len(merged) < self.min_claims:
            print_warn("Claim extraction yielded fewer than the minimum required claims ")
            self.logger.warning(
                f"Claim extraction yielded only {len(merged)} claims "
                f"(min_claims={self.min_claims}); proceeding with what we got"
            )
        t_rate = time.time()
        ranked = self._declare_claim_importance(uuid, goal, merged, grounding)
        print_ok(
            f"Importance rating for {uuid}: {len(ranked)} claims kept "
            f"in {time.time() - t_rate:.1f}s"
        )
        self._persist_rubric(task_key, ranked)
        return ranked

    def _per_source_targets(self, n_sources: int = 3) -> tuple[int, int]:
        """Per-source min/max claim targets derived from the global bounds.

        Scaled by the number of extraction sources to ensure the overall target is met.

        Args:
            n_sources: Number of extraction sources to share the global budget across.

        Returns:
            Tuple ``(per_source_min, per_source_max)`` of target claim counts.
        """
        n_sources = max(1, n_sources)
        per_min = max(2, self.min_claims // n_sources)
        per_max = max(per_min, max(2, self.max_claims // n_sources))
        return per_min, per_max

    # ------------------------------------------------------------------
    # Per-source parsing + path validation
    # ------------------------------------------------------------------

    def _parse_and_filter_claims(
        self,
        uuid: str,
        data: Any,
    ) -> list[dict[str, Any]]:
        """Validate the LLM JSON, normalise each claim, drop confabulated paths.

        Args:
            uuid: Workflow identifier (used in error context).
            data: Parsed JSON payload from the judge; expected to contain a
                ``claims`` list or to be a list itself.

        Returns:
            List of normalised claim dicts with stable ids and workspace-validated
            relevant file lists.

        Raises:
            LLMEvaluationError: When ``data`` does not carry a usable claims list.
        """
        claims = data.get("claims", []) if isinstance(data, dict) else data
        if not isinstance(claims, list):
            raise LLMEvaluationError(f"Claim extractor JSON has no 'claims' list for {uuid}")

        cleaned: list[dict[str, Any]] = []
        for idx, c in enumerate(claims):
            if not isinstance(c, dict) or "description" not in c:
                continue
            raw_files = c.get("likely_relevant_files") or []
            relevant = self._validate_workspace_paths(
                raw_files,
                allowed=self._workspace_files or None,
                label=str(c.get("id") or f"c{idx}"),
            )
            cleaned.append({
                "id": str(c.get("id") or f"c{idx}"),
                "description": str(c["description"]).strip(),
                "likely_relevant_files": relevant,
            })
        return cleaned

    # ------------------------------------------------------------------
    # Stage 1b — rate claim importance (1-10) against the goal, dedupe
    # ------------------------------------------------------------------

    def _declare_claim_importance(
        self,
        uuid: str,
        goal: str,
        claims: list[dict[str, Any]],
        grounding: str = "",
    ) -> list[dict[str, Any]]:
        """Drop near-duplicates and rate every surviving claim 1-10 vs the goal.

        Args:
            uuid: Workflow identifier (used for the judge call).
            goal: Workflow goal text — primary anchor for importance.
            claims: Merged, source-tagged claims from the per-source extractors.
            grounding: Optional peer-reviewed literature grounding block.

        Returns:
            Filtered claim list with ``importance`` (int 1-10) and
            ``importance_rationale`` (short phrase) populated on every entry.
        """
        if not claims:
            return claims

        t_dedup = time.time()
        drop_ids = self._run_dedup_pass(uuid, goal, claims, grounding)
        surviving = [c for c in claims if c.get("id") not in drop_ids]
        print_ok(
            f"Importance dedup for {uuid}: dropped {len(drop_ids)} of "
            f"{len(claims)} in {time.time() - t_dedup:.1f}s"
        )

        if not surviving:
            return surviving

        t_rate = time.time()
        importance_by_id = self._rate_importance_parallel(
            uuid, goal, surviving, grounding
        )
        print_ok(
            f"Importance rating for {uuid}: rated {len(importance_by_id)} of "
            f"{len(surviving)} in {time.time() - t_rate:.1f}s "
            f"(batches of {_IMPORTANCE_BATCH_SIZE}, parallelism={_IMPORTANCE_PARALLELISM})"
        )

        kept: list[dict[str, Any]] = []
        for c in surviving:
            imp, rationale = importance_by_id.get(
                c.get("id"), (self._DEFAULT_CLAIM_IMPORTANCE, "")
            )
            kept.append({**c, "importance": imp, "importance_rationale": rationale})
        return kept

    def _run_dedup_pass(
        self,
        uuid: str,
        goal: str,
        claims: list[dict[str, Any]],
        grounding: str,
    ) -> set[str]:
        """Single cheap LLM call returning only ids to drop as near-duplicates."""
        prompt = self._build_dedup_prompt(goal, claims, grounding)
        data, err = self._call_judge_for_json(
            uuid, "verifier_dedup_claims", prompt
        )
        if err is not None or not isinstance(data, dict):
            self.logger.warning(
                f"dedup pass failed for {uuid} ({err or 'non-dict JSON'}); "
                f"keeping all {len(claims)} claims"
            )
            return set()
        return self._extract_drop_ids(data)

    def _rate_importance_parallel(
        self,
        uuid: str,
        goal: str,
        claims: list[dict[str, Any]],
        grounding: str,
    ) -> dict[str, tuple[int, str]]:
        """Split rating into batches and fan out across threads."""
        batches = [
            claims[i : i + _IMPORTANCE_BATCH_SIZE]
            for i in range(0, len(claims), _IMPORTANCE_BATCH_SIZE)
        ]
        if not batches:
            return {}
        workers = max(1, min(len(batches), _IMPORTANCE_PARALLELISM))
        merged: dict[str, tuple[int, str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    self._rate_one_importance_batch,
                    uuid, goal, batch, grounding, idx,
                ): idx
                for idx, batch in enumerate(batches)
            }
            for f in as_completed(futures):
                merged.update(f.result())
        return merged

    def _rate_one_importance_batch(
        self,
        uuid: str,
        goal: str,
        batch: list[dict[str, Any]],
        grounding: str,
        batch_idx: int,
    ) -> dict[str, tuple[int, str]]:
        """Rate one batch; default-importance fallback on parse/call error."""
        prompt = self._build_importance_batch_prompt(goal, batch, grounding)
        agent_name = f"verifier_rate_importance_b{batch_idx}"
        data, err = self._call_judge_for_json(uuid, agent_name, prompt)
        if err is not None or not isinstance(data, dict):
            self.logger.warning(
                f"importance batch {batch_idx} failed for {uuid} "
                f"({err or 'non-dict JSON'}); defaulting {len(batch)} claim(s)"
            )
            return {
                str(c.get("id")): (self._DEFAULT_CLAIM_IMPORTANCE, "")
                for c in batch
                if c.get("id")
            }
        return self._extract_importance_map(data)

    def _build_dedup_prompt(
        self,
        goal: str,
        claims: list[dict[str, Any]],
        grounding: str,
    ) -> str:
        """Render the dedup-only prompt. Output is just a list of ids."""
        grounding_block = (
            grounding.strip() if grounding else "(no literature grounding available)"
        )
        claim_lines = "\n".join(
            f"- id={c.get('id')!r}  source={c.get('source', 'unknown')}  "
            f"desc={str(c.get('description', '')).strip()[:200]}"
            for c in claims
        )
        return f"""You are pruning near-duplicate verification claims.
CLAIMS:
{claim_lines}

TASK:
Identify near-duplicate claims (same checked property, different wording or
source). Return the redundant ids to drop. Keep the clearest version of each
cluster. Do NOT drop claims that check different facets — only true duplicates.
If nothing is duplicated, return an empty list.
Also remove any claim that's only about the sources dataset files. (eg: the file data/train.csv has duplicate rows)
Do not remove claims about the output artefacts, even if they mention input files, as long as they check a property of the output (eg: the output predictions.csv has duplicate rows)
Also drop any claim that in it's claims assume a "_prob"  at the end of a prediction data column.

Return STRICT JSON only, in this exact shape:
{{"drop_ids": ["<id>", ...]}}
"""

    def _build_importance_batch_prompt(
        self,
        goal: str,
        batch: list[dict[str, Any]],
        grounding: str,
    ) -> str:
        """Render the per-batch rating prompt. Rationale is a short phrase."""
        grounding_block = (
            grounding.strip() if grounding else "(no literature grounding available)"
        )
        claim_lines = "\n".join(
            f"- id={c.get('id')!r}  source={c.get('source', 'unknown')}  "
            f"desc={str(c.get('description', '')).strip()[:300]}"
            for c in batch
        )
        return f"""You are rating verification claims by how much they matter for the task success.
You assign an IMPORTANCE weight (1–10) to each claim. Importance answers ONE
question: if this claim turns out false or missing, how much does result break?

WORKFLOW GOAL:
{goal}

LITERATURE GROUNDING:
{grounding_block}

{_IMPORTANCE_ANCHOR_BLOCK}

CLAIMS TO RATE:
{claim_lines}

TASK:
For every claim listed above, return an integer importance 1–10 anchored on
the scale (goal-alignment dominates). For each claim also give a TERSE
rationale: a short phrase, MAX 8 words / 60 characters, no full sentence,
no punctuation at the end. Examples of acceptable rationales:
  "literal deliverable; named in goal"
  "sanity property; prevents silent corruption"
  "style hygiene; advisory only"

Return STRICT JSON only, in this exact shape:
{{
  "importance": [
    {{"id": "<id>", "importance": <int 1-10>, "rationale": "<≤8 words>"}},
    ...
  ]
}}
"""

    @staticmethod
    def _extract_drop_ids(data: dict[str, Any]) -> set[str]:
        """Best-effort parse of the ``drop_ids`` list from the rater JSON."""
        raw = data.get("drop_ids") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return set()
        return {str(x) for x in raw if isinstance(x, str) and x}

    def _extract_importance_map(
        self,
        data: dict[str, Any],
    ) -> dict[str, tuple[int, str]]:
        """Project the ``importance`` array into ``{id: (importance, rationale)}``.

        Args:
            data: Parsed rater JSON.

        Returns:
            Map from claim id to ``(importance, rationale)``.
        """
        out: dict[str, tuple[int, str]] = {}
        raw = data.get("importance") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return out
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("id")
            if not isinstance(cid, str) or not cid:
                continue
            try:
                imp = int(entry.get("importance", self._DEFAULT_CLAIM_IMPORTANCE))
            except (TypeError, ValueError):
                imp = self._DEFAULT_CLAIM_IMPORTANCE
            imp = max(1, min(10, imp))
            rationale = str(entry.get("rationale") or "").strip()
            out[cid] = (imp, rationale)
        return out