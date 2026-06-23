"""Cross-cutting: did the new claim_cache fix the noise-drift pathology?

Part 6's baseline finding on mat_diffusion (prior run):
  - iter 0 ∩ iter 1   = 1 claim_id  (only `workspace_not_cluttered`)
  - iter 0 ∩ iter 10  = 14 claim_ids
  - iter 0 ∩ iter 1 ∩ iter 10 = 1
  - i.e. the rubric *churned* across the lineage — claim sets weren't comparable.

After the refactor, the per-(task, source) claim-text cache should seed
claim continuity. This script computes shared-id rates across iterations
for the 3 tasks of the new run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path("/home/martin/Projects/CNRS/Mimosa-AI")
TMP = ROOT / "sources" / "workflows" / "run_1" / "_verifier_tmp"


TASKS = {
    "clintox":      ("edc4def39a21daa0", ["20260623_092931_faff3fe1"]),
    "mat_diffusion": ("bcc62483f627f138", [
        "20260623_100554_7bbee0c9", "20260623_103432_a487b01e",
        "20260623_111821_0d0dc035", "20260623_114955_d8befb05",
        "20260623_122620_f78ee91c", "20260623_130347_ef57ce97",
        "20260623_132955_81b35f04", "20260623_135928_5d5866c8",
        "20260623_143921_625d03ad", "20260623_152043_bb682237",
    ]),
    "bulk_modulus": ("cbf1704cc7aae656", [
        "20260623_155721_103086cc", "20260623_155908_276b7b36",
        "20260623_164547_0e3d612c", "20260623_172504_e901c27f",
        "20260623_175250_84c7a767", "20260623_181511_40cd1628",
        "20260623_185311_2c3aad28", "20260623_192329_a55eb592",
        "20260623_192501_970cdd18", "20260623_195824_e9aedc0b",
    ]),
}


CLAIM_HEAD_RE = re.compile(r"^\[([^\]]+)\]\s*\(")


def claim_ids_in_eval(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(errors="replace").splitlines():
        m = CLAIM_HEAD_RE.match(line)
        if m:
            cid = m.group(1)
            # strip any "_a/_b/_c/_d/_e" source suffix added by _extract_claims
            # on collision so we compare canonical ids
            for suf in ("_a", "_b", "_c", "_d", "_e"):
                if cid.endswith(suf):
                    cid = cid[: -len(suf)]
                    break
            out.add(cid)
    return out


def cached_ids(task_key: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for label in ("a", "b", "c", "d", "e"):
        p = TMP / f"claim_cache_{task_key}_source_{label}.json"
        if not p.exists():
            out[label] = set()
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            out[label] = set()
            continue
        out[label] = {c.get("id") for c in (data.get("claims") or []) if c.get("id")}
    return out


def main() -> None:
    for task, (task_key, uuids) in TASKS.items():
        print("=" * 76)
        print(f"TASK: {task}  (task_key={task_key}, n_iters={len(uuids)})")
        print("=" * 76)

        cached = cached_ids(task_key)
        cached_total = set().union(*cached.values())
        print(f"\n[cache] seeded claims per source:")
        for label in "abcde":
            print(f"  source {label}: {len(cached[label])}  ids={sorted(cached[label])[:6]}{'...' if len(cached[label]) > 6 else ''}")
        print(f"  TOTAL cached ids: {len(cached_total)}")

        per_iter_ids: dict[str, set[str]] = {}
        for uid in uuids:
            eval_path = ROOT / "sources" / "workflows" / "run_1" / uid / "evaluation.txt"
            per_iter_ids[uid] = claim_ids_in_eval(eval_path)

        # exclude iterations with no evaluation (e.g. crashed)
        with_eval = {uid: ids for uid, ids in per_iter_ids.items() if ids}
        if not with_eval:
            print("  (no evaluation.txt files found — task may have crashed before scoring)")
            continue

        print(f"\n[per-iter] claim-id counts (only iters with evaluation.txt):")
        for uid, ids in with_eval.items():
            print(f"  {uid}: {len(ids)} claims")

        # pairwise overlap across iterations (which is the part-6 metric)
        if len(with_eval) >= 2:
            uids = list(with_eval.keys())
            print(f"\n[pairwise] |iter_i ∩ iter_j| / |iter_i|  (rubric stability):")
            print(f"  {'pair':<70}  shared  rate")
            sums = []
            for i, ui in enumerate(uids):
                for j, uj in enumerate(uids):
                    if j <= i:
                        continue
                    ids_i = with_eval[ui]
                    ids_j = with_eval[uj]
                    shared = ids_i & ids_j
                    rate = len(shared) / len(ids_i) if ids_i else 0
                    sums.append(rate)
                    if j == i + 1 or (i == 0 and j == len(uids) - 1):
                        # log only adjacent + first-last for brevity
                        print(f"  {ui[:18]}..{uj[:18]}  {len(shared):>3}/{len(ids_i):<3}  {rate:.1%}")
            print(f"\n  mean pairwise share = {sum(sums)/len(sums):.1%}  (Part-6 baseline on prior mat_diff: ~7%)")

            # all-iter intersection (Part 6 baseline = 1)
            all_shared = set.intersection(*with_eval.values())
            print(f"\n[all-iter intersection] |∩ all iters| = {len(all_shared)}  ids={sorted(all_shared)[:10]}{'...' if len(all_shared) > 10 else ''}")
            print(f"  (Part-6 baseline on prior mat_diff: 1 — only `workspace_not_cluttered`)")

        # how much do iter claim_ids overlap with the cache
        if cached_total:
            print(f"\n[cache reuse] |iter_ids ∩ cached_ids| / |cached_ids|:")
            for uid, ids in with_eval.items():
                hit = ids & cached_total
                rate = len(hit) / len(cached_total)
                print(f"  {uid}: {len(hit):>3}/{len(cached_total):<3}  {rate:.1%}")


if __name__ == "__main__":
    main()
