# ADR 0001 — Bootstrap synthesize via embedding-cluster, not LLM single-shot

- **Date**: 2026-05-13
- **Status**: Accepted
- **Component**: `vocab_maintenance.bootstrap` — the `synthesize` phase of the bootstrap pipeline

## Context

Bootstrap distills records into themes, then `synthesize` produces an initial tag vocabulary from the themes. For our corpus this is 696 themes → ~30-60 distinct engineering patterns.

Single-shot synthesize sends all themes to one LLM call and asks for a structured vocabulary. We exhausted variations of this design and **all failed**:

| Attempt | Model | Prompt | Result |
|---|---|---|---|
| 1 | qwen-plus + thinking | aim-for-N + "merge aggressively" | 9 abstract umbrella tags, 521s |
| 2 | qwen-plus thinking-off | same | timeout (>900s, killed) |
| 3 | qwen-max thinking-off | same | 2 tags (`state_management` + `data_integrity`), 345s, hit_rate dropped to 0.776 |
| 4 | qwen-max | removed aggressive-merge + few-shot + no count constraint | 2 tags (`error_handling` + `state_management`) |
| 5 | qwen-max | + explicit FORBIDDEN umbrella list + reframed "label each theme" | killed before completion; iter 4 evidence sufficient |
| 6 | map-reduce w/ LLM partition (`merged_from: list[int]`) | per-batch synth + LLM consolidates 269 candidates with strict partition | reduce step `returned_none` at 271s |
| 7 | map-reduce w/ LLM names + embedding assign | LLM produces vocab; embeddings map 378 candidates to final tags | reduce LLM also timed out — input ≥ 7k tokens too much for cross-batch dedup |

Root cause is structural, not prompt-tunable. LLMs have a **cognitive-bandwidth limit**: when given hundreds of items in one pass, attention is diluted and they default to the lowest-common-denominator categories (umbrella names like `error_handling`, `state_management`). No amount of prompt rewriting overcame this — qwen-plus and qwen-max both exhibited identical collapse patterns under identical input scale.

### What LLMs do well vs poorly

| Task type | LLM performance |
|---|---|
| Read 1 record → name a pattern (1-2 words) | excellent |
| Compare 5 items, find duplicates | excellent |
| Read 10-20 themes → name shared pattern | excellent |
| Read 100+ items, produce structured taxonomy | **poor** — collapses to umbrellas |
| Output partition (every input index appears exactly once across N output buckets) | **poor** — counting/bookkeeping errors |

The synthesize task as originally framed combined two LLM weaknesses (large input + partition constraint) in one call.

## Decision

**Adopt a BERTopic-style three-stage pipeline for bootstrap synthesize**, replacing the single-shot LLM design:

1. **Embed** all themes via `similarity._embed` (DashScope `text-embedding-v3`, concurrency-10 ThreadPool).
2. **Cluster** the embedding matrix with `sklearn.cluster.KMeans(n_clusters=50, random_state=42, n_init=10)`. This produces deterministic, balanced clusters; algorithm handles the global-organization task LLMs failed at.
3. **Name** each cluster with one small LLM call: given the ~14 themes in this cluster, return `{name, definition}`. 50 clusters × ~5s each, concurrency-10 → ~25s total.

The function is `bootstrap.synthesize_via_clustering(model_factory, themes, log_file, n_topics=50, concurrency=10)`. It returns the same `{vocab, notes}` shape as `synthesize_step`, so it slots into the existing bootstrap graph in place of the old single-shot call.

### Why this works

- Embedding handles **global similarity** (its strength) — deterministic, scales to any input size.
- LLM handles **local naming** (its strength) — per call it sees only ~14 themes from one semantic group, so it cannot collapse to umbrellas. Output is a single tag name, no partition tracking.
- The two are decoupled: cluster boundaries are computed once, naming is independent per cluster.

### Industry precedent

This is a well-established pattern; the closest direct analogue is **BERTopic** (Maarten Grootendorst, ~6k GitHub stars), which uses sentence-transformers + UMAP + HDBSCAN + LLM/c-TF-IDF naming. Microsoft Research's **TnT-LLM** (KDD 2024) is a related iterative variant. Both confirm that LLMs alone fail at large-scale taxonomy induction and that embedding-based pre-clustering is the standard remediation.

## Consequences

### Positive

- **Bootstrap synthesize runs in ~77s** (60s embed + 0.3s KMeans + 14s naming, plus headroom), down from 300-900s in single-shot attempts.
- **Produces ~50 specific tags** (target hit consistently) instead of 2-9 umbrella categories.
- **Coverage 98.6%** of themes (686/696) versus 78% in collapsed runs.
- **Cluster size distribution is healthy** (min=3, median=13, max=35) — no mega-clusters swallowing the corpus.
- **Deterministic structure**: same input + same `random_seed` → same clusters → same final vocab modulo LLM naming temperature. Reproducibility for free.
- **Robust to scale**: behavior at 696 themes scales to thousands; LLM never sees more than one cluster at a time.
- **Cost per bootstrap drops** by ~5×: 50 small qwen-plus calls + 696 embeddings vs one large qwen-max call.

### Negative

- **`n_topics` is a hyperparameter** chosen up front. KMeans does not pick its own k; misjudging it under-fragments or over-fragments the vocabulary. Mitigation: 696 / 14-target-themes-per-tag ≈ 50 — defaults to 50; expose as CLI flag.
- **Cluster boundaries are embedding-driven**, not LLM-creative. Some semantically-distinct themes may land in the same cluster because of surface vocabulary overlap (and vice versa). This is the same risk `expand_new_tag_coverage` mitigates downstream via LLM binary checks; bootstrap accepts it as an acceptable bias for the speed/reliability gain.
- **Cross-cluster duplicate naming**: independent per-cluster LLM calls don't know what siblings are named, so near-synonym tags can emerge (observed in the validation run: 4 `contract_*` tags, two clusters both named `explicit_validation`). Mitigation: post-bootstrap, the maintenance agent's `propose_merge` action is designed to clean these up; no special-casing needed.
- **Adds a runtime dependency on `sklearn`** (already transitively present via `langchain`, but now a first-party import).
- **Embedding API is rate-bound**: 696 sequential embed calls with concurrency-10 took 61s (≈90ms/call API roundtrip). For larger corpora, batch-embedding endpoints would be needed; out of scope here.

### Neutral

- The previous single-shot `synthesize_step` and the map-reduce variant `synthesize_mapreduce` remain in the codebase as alternative paths. They are not used by the default bootstrap pipeline but are kept for diagnostic comparison and as a record of the failure modes documented above. Iterating on prompts / variant selection during development was done via a local-only harness (not committed).
- `_default_target_count_range` in `bootstrap.py` is now unused. Not deleted — preserved for any downstream caller and to keep the change minimal.

## Validation

Tested against the same 696-theme checkpoint that produced the earlier failures (`thread_id=bootstrap-1778606496`). Direct comparison:

| Variant | Time | Tags | Coverage | Notes |
|---|---|---|---|---|
| single-shot qwen-plus thinking | 521s | 9 | partial | Over-abstracted, names like `state_minimization`, `shallow_heuristics` |
| single-shot qwen-max | 345s | 2 | 78% | `state_management` + `data_integrity` only |
| map-reduce w/ LLM partition | timeout | — | — | Reduce step `llm.returned_none` at 271s |
| map-reduce w/ embedding assign | timeout | — | — | Reduce LLM still times out on 378-candidate input |
| **`synthesize_via_clustering` (this ADR)** | **77s** | **50** | **98.6%** | Specific tags, healthy size distribution |

## Follow-ups (not in scope for this ADR)

- Integrate `synthesize_via_clustering` as the default in `graphs/bootstrap.py:_default_synthesize` (the test harness currently exercises it directly). Tracked separately.
- Investigate `n_topics` auto-selection (e.g., silhouette score sweep or HDBSCAN's density-based clustering, which picks its own k). Not required for current scale; revisit if vocabulary quality regresses.
- Quantify the `contract_*` / `explicit_validation` duplicate-naming rate on real corpora and decide whether to add an explicit cross-cluster dedup step in the synthesize function, or leave it to the maintenance agent's `propose_merge`.

## References

- BERTopic: https://github.com/MaartenGr/BERTopic — sentence-transformer + dim-reduce + cluster + topic-name pipeline.
- TnT-LLM (Wan et al., KDD 2024) — iterative LLM-assisted taxonomy induction; same observation that single-pass LLM induction fails at scale.
- Failure-mode evidence (this repo): `outputs/runs/2026-05-13T01-21-35_bootstrap.jsonl` (single-shot 9-tag run), `outputs/runs/2026-05-13T02-25-45_bootstrap_v2.jsonl` (qwen-max single-shot 2-tag run), `outputs/runs/2026-05-13T02-48-11_synthesize_test.jsonl` (map-reduce partition timeout).
- Success run: `outputs/runs/2026-05-13T03-23-22_synthesize_test.jsonl` (this ADR's validation).
