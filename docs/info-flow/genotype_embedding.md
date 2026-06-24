# Genotype-embedding descriptor — info flow

> **Reader's note.** This page is an info-flow audit, not a user-facing
> tutorial. It traces every variable that feeds the QD behaviour
> descriptor from its source — the workflow's generated code — to the
> point where novelty distance is computed. If you want to *understand*
> the descriptor, start with
> [`concepts/evolution-engine.md`](../concepts/evolution-engine.md) and
> [`concepts/evaluation-pipeline.md`](../concepts/evaluation-pipeline.md).

## Why this descriptor

QD selection compares candidates with two scalars:

- **quality** — `reward_uncapped`, i.e. how well the workflow scored.
- **novelty** — k-NN distance in *behaviour-descriptor space* to the
  rest of the archive.

The behaviour descriptor is the **genotype embedding**: a dense vector
embedding of the workflow's generated source code (its *genotype*),
L2-normalised. Two workflows whose code is semantically similar collapse
to nearly the same point, so keeping both is redundant; two that explored
different approaches land far apart and each earns a seat in the archive.
The descriptor therefore reads "how different is the generated approach"
directly, instead of inferring it from a proxy.

> **History.** Earlier versions derived this descriptor from a *failure
> fingerprint* (a 6-D centered vector of per-source verifier pass rates)
> and, before that, from a structural descriptor `[n_agents, n_edges,
> n_branches, prompt_chars]`. The failure fingerprint is still computed
> and persisted by the verifier as a diagnostic (see
> [`concepts/evaluation-pipeline.md`](../concepts/evaluation-pipeline.md#failure-fingerprint-diagnostic)),
> but it no longer feeds novelty. The structural descriptor is gone.

## The signal: an embedding of the workflow code

The embedder takes the workflow source string and returns a unit-norm
vector:

```
vec = backend.encode(code)          # raw embedding
vec = vec / ||vec||                 # L2-normalised
descriptor = list(vec)              # plain list[float], len = backend dim
```

The backend is pluggable
([`genotype_embedding.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/genotype_embedding.py)):

| Backend | When | Dim | Notes |
| --- | --- | --- | --- |
| `all-MiniLM-L6-v2` (sentence-transformers) | **default** | 384 | Local, offline, deterministic, free. |
| OpenAI `text-embedding-3-small` | `MIMOSA_GENOTYPE_EMBEDDING_BACKEND=openai` **and** `OPENAI_API_KEY` set | 1536 | API call per uncached genotype; falls back to MiniLM if the client can't be built. |

Identical code text is embedded once per process — `GenotypeEmbedder`
caches by the SHA-1 of the source — so the QD inner loop stays cheap even
when archive refresh walks dozens of members.

## Novelty: mean cosine distance

Novelty is the mean **cosine distance** from the candidate descriptor to
its comparison set:

```
cosine_distance(a, b) = 1 − cosine_similarity(a, b)      ∈ [0, 2]
```

Two comparison modes
([`SelectionPressure._compute_novelty`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/selection.py)):

- **`archive_knn`** (default) — mean distance to the `k =
  novelty_k_neighbours = 15` nearest archive members.
- **`previous_n`** — mean distance to the `previous_n` most recently
  produced genotypes (ignores eviction).

`novelty` then enters the QD score additively:
`qd_score = (1 − w)·quality_norm + w·novelty_norm`, `w = novelty_weight =
0.25`. Quality and novelty are never multiplied, so high novelty cannot
drag a broken run above its peers.

## Sources of variables

```
                              ┌────────────────────────────────────┐
   workflow run               │ orchestrator → workflow_genotype   │
   ──────────────►            │   code = generated Python source   │
   uuid, code,                │   (the genotype)                   │
   execution_text             └─────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────────────┐
                              │ SelectionPressure                  │
                              │   ._extract_behaviour_descriptor   │
                              │     → genotype_embedding_descriptor│
                              │        (run.code)                  │
                              │     → None when code is degenerate │
                              │        (treated as neutral novelty)│
                              └─────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────────────┐
                              │ code_features                      │
                              │   .genotype_embedding_descriptor   │
                              │     → embed_genotype(code)         │
                              └─────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────────────┐
                              │ genotype_embedding                 │
                              │   GenotypeEmbedder.embed           │
                              │     SHA-1 cache → backend.encode   │
                              │     → L2-normalised np.ndarray     │
                              │     → None on degenerate / error   │
                              └─────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────────────┐
                              │ PopulationMember.behaviour_        │
                              │ descriptor  (list[float], unit-norm)│
                              │   snapshot at admission time       │
                              └─────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────────────┐
                              │ k-NN cosine novelty                │
                              │   _compute_novelty /               │
                              │   _knn_novelty_against_peers       │
                              └────────────────────────────────────┘
```

## Variable inventory (audit table)

| Symbol | Type | Set by | Read by | Notes |
| --- | --- | --- | --- | --- |
| `run.code` | str | orchestrator / workflow factory | `_extract_behaviour_descriptor` | The workflow genotype (generated Python source). |
| `genotype_embedding_descriptor(code)` | `list[float]` \| `None` | `code_features` | `_extract_behaviour_descriptor` | L2-normalised embedding; `None` for degenerate input. |
| `embed_genotype(code)` | `np.ndarray` \| `None` | `genotype_embedding.GenotypeEmbedder` | `genotype_embedding_descriptor` | Unit-norm vector; SHA-1 cached per process. |
| `PopulationMember.behaviour_descriptor` | `list[float]` | `SelectionPressure._validate_open_ended` | `_compute_novelty`, `_knn_novelty_against_peers` | Snapshot of the descriptor at admission; `[]` when no usable genotype. |
| `novelty_score` | float | `_compute_novelty` / `_refresh_member_metrics` | `qd_score` | Mean cosine distance ∈ `[0, 2]` to the comparison set. |
| `qd_score` | float | `_validate_open_ended`, `_refresh_member_metrics` | parent draw, archive eviction | `(1 − w)·quality_norm + w·novelty_norm`; quality and novelty are *additive*, never multiplied. |

## Failure modes the audit checks

- **Degenerate genotype** (missing / empty / non-string source, or a
  backend exception): `embed_genotype` returns `None`,
  `_extract_behaviour_descriptor` yields `None`, the member stores `[]`,
  and `_compute_novelty` returns `0.0`. A missing embedding contributes
  **nothing** to QD score — it is never rewarded as max-novel.
- **Empty comparison set** (cold start, archive still empty):
  `_compute_novelty` returns `0.0`, so the first candidate is neither
  novel nor stale.
- **Non-finite / empty embedding** from a backend: dropped to `None` by
  `GenotypeEmbedder.embed` before it can poison a descriptor.
- **Mismatched descriptor length across archive members** (only happens
  if the embedding backend is switched mid-run): `_cosine_distance`
  returns the neutral sentinel `1.0` for that pair. Drain the archive
  when you change backends — the descriptor dimension is backend-defined,
  not fixed.

## Invariants

- Descriptors are **L2-normalised**, so `_cosine_distance` lands in
  `[0, 2]` and equals `1 − dot(a, b)` for two unit vectors.
- Identical source text → identical descriptor (SHA-1 cache), so two
  byte-identical workflows have distance `0` and the second is treated as
  redundant.
- `_cosine_distance` returns the neutral `1.0` for empty, zero-norm, or
  shape-mismatched vectors — no pair ever raises or returns NaN.
- A `None` descriptor never maps to maximum novelty; it maps to `0.0`
  (neutral), so broken offspring cannot win a slot on novelty alone.
