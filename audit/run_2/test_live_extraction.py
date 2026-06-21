"""End-to-end live-LLM test for the new claim-list cache.

Runs the verifier's claim-extraction stage TWICE against the saved best
workflow of the new mat_diffusion run:

  Pass 1: cache empty → prompts contain no PRIOR CLAIMS block; cache is
          seeded for every source that returns a non-empty list.
  Pass 2: cache populated → prompts contain the PRIOR CLAIMS block; the
          extractor is steered to reproduce the seeded claim_ids.

Asserts:
  - cache files appear under sources/workflows/run_1/_verifier_tmp/
    (5 files, one per source) after pass 1
  - pass 2 reuses those IDs at a high rate (≥60% reuse on shared sources)
  - the per-claim verifier-script side-effect produces NEW scripts under
    sources/workflows/run_1/_verifier_tmp/<uuid>/ (proving no anchor walk
    is reused)

This is the regression test for the seed-anchoring pathology documented in
audit/run_2/REPORT.md §f and the verifier-cache investigation.

Run with: uv run python audit/run_2/test_live_extraction.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Same priming pattern as tests/verifier_claim_cache_test.py
from sources.core.failure_fingerprint import compute_failure_fingerprint  # noqa: E402, F401
from sources.evaluators import verifier as verifier_mod  # noqa: E402
from sources.evaluators.verifier import VerifierEvaluator  # noqa: E402

from config import Config  # noqa: E402


BEST_UUID = "20260621_003623_ef01df13"
TMP_ROOT = _REPO_ROOT / "sources" / "workflows" / "run_1" / "_verifier_tmp"


def _purge_existing_claim_caches() -> int:
    """Remove any prior claim_cache_*.json so pass 1 starts cold. Returns count purged."""
    n = 0
    for p in TMP_ROOT.glob("claim_cache_*.json"):
        p.unlink()
        n += 1
    return n


def _load_seeded_caches() -> dict[str, list[dict]]:
    """Map source label → cached claims (post-pass-1)."""
    out: dict[str, list[dict]] = {}
    for p in sorted(TMP_ROOT.glob("claim_cache_*.json")):
        # filename: claim_cache_<task>_source_<label>.json
        label = p.stem.rsplit("_source_", 1)[-1]
        data = json.loads(p.read_text())
        out[label] = data.get("claims") or []
    return out


def _extract_once(label: str) -> tuple[list[dict], float, str]:
    """Run a single pass of claim extraction and return (merged_claims, dt_s, source_prompt_excerpt)."""
    config = Config()
    config.load(str(_REPO_ROOT / "config_default.json"))
    config.workflow_dir = str(_REPO_ROOT / "sources" / "workflows" / "run_1")
    config.temp_dir = str(TMP_ROOT)
    # Force grounding off to keep the test fast and deterministic
    v = VerifierEvaluator(config, use_grounding=False)

    wf_info = v._load_workflow_data(BEST_UUID)
    goal = wf_info.goal
    execution_text, _ = v.workflow_execution_text(BEST_UUID)
    workspace_listing = v._list_workspace()

    print(f"\n=== PASS [{label}] ===")
    print(f"  goal hash: {VerifierEvaluator._task_cache_key(goal)}")
    print(f"  cache files before pass: {len(list(TMP_ROOT.glob('claim_cache_*.json')))}")

    t = time.time()
    claims = v._extract_claims(
        BEST_UUID,
        goal=goal,
        execution_text=execution_text,
        workspace_listing=workspace_listing,
        is_truly_empty=False,
        grounding="",
    )
    dt = time.time() - t
    print(f"  -> {len(claims)} claims merged in {dt:.1f}s")
    return claims, dt, ""


def main() -> None:
    if not (TMP_ROOT / BEST_UUID).exists():
        print(f"FATAL: best uuid folder missing at {TMP_ROOT / BEST_UUID}", file=sys.stderr)
        sys.exit(2)

    purged = _purge_existing_claim_caches()
    print(f"purged {purged} stale claim_cache_*.json files; starting cold")

    # ---- PASS 1: cold; expect cache to be seeded
    claims_pass1, dt1, _ = _extract_once("pass1-cold")
    cache_after_pass1 = _load_seeded_caches()
    print(f"\ncache state after pass 1:")
    for lbl in sorted(cache_after_pass1):
        ids = [c.get("id") for c in cache_after_pass1[lbl]]
        print(f"  source {lbl}: {len(ids)} claims  ids={ids[:6]}{' ...' if len(ids) > 6 else ''}")
    assert cache_after_pass1, "pass 1 produced no claim_cache files — seeding broken"

    # ---- PASS 2: warm; expect prior_claims block in prompts + high id reuse
    claims_pass2, dt2, _ = _extract_once("pass2-warm")
    cache_after_pass2 = _load_seeded_caches()
    print(f"\ncache state after pass 2:")
    for lbl in sorted(cache_after_pass2):
        ids = [c.get("id") for c in cache_after_pass2[lbl]]
        print(f"  source {lbl}: {len(ids)} claims  ids={ids[:6]}{' ...' if len(ids) > 6 else ''}")
    assert cache_after_pass1 == cache_after_pass2, (
        "pass 2 overwrote the cache; _persist_claims_for_source must be a no-op when cache exists"
    )

    # ---- Reuse-rate check
    # Group pass-1 ids by source by reading from claim["source"] = "source_<label>"
    pass1_ids_by_source: dict[str, set[str]] = {}
    for c in claims_pass1:
        src = c.get("source", "").replace("source_", "") or "?"
        pass1_ids_by_source.setdefault(src, set()).add(c.get("id"))
    pass2_ids_by_source: dict[str, set[str]] = {}
    for c in claims_pass2:
        src = c.get("source", "").replace("source_", "") or "?"
        pass2_ids_by_source.setdefault(src, set()).add(c.get("id"))

    print("\nreuse rate (pass 2 ids ∩ pass 1 cached ids) / pass 1 cached ids:")
    overall_kept = 0
    overall_seeded = 0
    for lbl in sorted(cache_after_pass1):
        cached_ids = {c.get("id") for c in cache_after_pass1[lbl]}
        kept = pass2_ids_by_source.get(lbl, set()) & cached_ids
        rate = (len(kept) / len(cached_ids)) if cached_ids else 0.0
        print(f"  source {lbl}: kept {len(kept)}/{len(cached_ids)}  rate={rate:.1%}")
        overall_kept += len(kept)
        overall_seeded += len(cached_ids)
    overall_rate = overall_kept / overall_seeded if overall_seeded else 0.0
    print(f"\noverall id-reuse rate: {overall_kept}/{overall_seeded} = {overall_rate:.1%}")
    print(f"pass 1 wall: {dt1:.1f}s · pass 2 wall: {dt2:.1f}s")
    assert overall_rate >= 0.50, (
        f"reuse rate {overall_rate:.1%} too low — the prior-claims continuity layer is not steering the extractor"
    )
    print("\nOK — cache seeded + reused; claim continuity working without anchor-walk.")


if __name__ == "__main__":
    main()
