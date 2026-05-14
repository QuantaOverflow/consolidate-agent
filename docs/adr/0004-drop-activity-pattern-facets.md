# ADR-0004: Drop Activity and Pattern facets from Stage 1 vocab

**Status**: Accepted
**Date**: 2026-05-13
**Supersedes**: parts of `docs/plans/2026-05-tag-network-evolution-roadmap.md`, `docs/plans/2026-05-stage1-acceptance.md`

## Context

Stage 1 of the tag network evolution roadmap proposed evolving the flat 49-tag vocab into a 3-facet faceted classification: **Matter** (technical subject), **Activity** (engineering work type), **Pattern** (abstract lesson), plus a **lesson_type** enum (anti_pattern/discovery/best_practice).

The hypothesis: each facet would be **internally orthogonal**, and after isolating tags by facet, within-facet silhouette would turn positive (records would land in a clear nearest tag rather than competing across abstraction levels).

We tested this hypothesis with two iterations:

- **v2** (handcrafted): I wrote 16 Matter + 9 Activity + 7 Pattern tags + 3 lesson_type values based on bottom-up regex frequency analysis + engineering intuition + light reference to SE literature (SWEBOK, faceted classification).
- **v2.1** (data-driven): regenerated Activity (8) and Pattern (6) facets via "extract aspect phrase per axis → embed → KMeans → LLM-name" pipeline (mirroring the v1 bootstrap BERTopic approach). Matter and lesson_type retained from v2.

## Decision

**Drop Activity and Pattern facets from Stage 1 scope.** Vocab v3 will contain only:
- **Matter facet** (16 tags) — retained from v2.1
- **lesson_type enum** (3 values) — retained from v2.1

Activity and Pattern dimensions are deferred to Stage 2 (knowledge graph), where they will be expressed as **typed edges and pattern nodes**, not as flat tag classifications.

## Rationale: what the data showed

### Diagnostic summary (per-facet silhouette + healthy ratio after each iteration)

| Iteration | Matter healthy | Activity healthy | Pattern healthy | LLM top-1 (Activity / Pattern) |
|---|---|---|---|---|
| **v1** (flat 49 tags) | 8.2% global (no facets) | n/a | n/a | n/a |
| **v2** (handcrafted facets) | **50%** | 33% | 14% | 25% / 24% |
| **v2.1** (data-driven facets) | **56%** | 25% (↓) | 17% | 24% / 15% |

Source: `outputs/network_v2_diagnostics.txt`, `outputs/network_v2_1_diagnostics.txt`.

### What was proven

1. **Matter facet works** — faceted classification is correctly applied to the dimension of technical objects. Healthy ratio improved 7× from v1 (8%) to v2.1 (56%). This validates the *faceted classification* hypothesis for that one dimension.

2. **lesson_type enum works** — 100% coverage in both v2 and v2.1, 0 hallucinations. The 3-value polarity classification is well-defined and applied consistently.

3. **The faceted-partition mechanism itself is sound** — separating facets lifted Matter from 8% to 50%+ silhouette health. This is the right architectural approach for the dimension it fits.

### What was disproven

1. **Activity and Pattern do not admit clean faceted classification on this corpus.**
   - v2.1 (data-driven, ideally suited to the method) produced *worse* Activity health than v2 (25% vs 33%).
   - In both versions, one or two generic tags absorbed 25-35% of records (v2: `designing` 34%, `explicit_contract` 26%; v2.1: `debugging` 28%, `explicit_contract` 35%). These generic tags carry uniformly negative silhouette.
   - Switching methodology (handcrafted ↔ clustering) did not move the failure pattern. **The problem is the domain, not the method.**

2. **KMeans clustering does not avoid generic-tag failure mathematically.**
   - n_clusters=8 over 696 records *must* produce 1-2 mega-clusters (100+ records).
   - A 100+ record cluster's embedding centroid sits at a generic-verb position in vector space (e.g., "debugging" or "designing").
   - LLM forced to name such a cluster outputs that generic verb.
   - LLM then re-applies that generic tag as the safe default in every ambiguous case during tagging.
   - This loop is invariant to method choice — increasing K to 15-20 only shifts the threshold; the largest cluster is still a generic-verb cluster, just smaller.

3. **SE engineering lessons are inherently multi-aspect on the Activity and Pattern axes.**
   - Example record: "LLM node JSON parsing failure causes fallback to template text"
     - Activity: debugging? designing? integrating? — all defensible, none canonical.
     - Pattern: silent_failure? fallback_strategy? explicit_contract? early_validation? — all apply simultaneously.
   - Forcing single-tag (or 0-2) selection on these axes systematically misclassifies the majority of records, regardless of vocab quality.

### What this means architecturally

The roadmap implicitly assumed all three facets had similar structure (orthogonal discrete categories). In fact:

| Dimension | Structure | Suited to |
|---|---|---|
| **Matter** (what technical object) | Discrete, orthogonal: a record is *about* git or about a database, rarely both at equal weight | ✅ Faceted classification |
| **lesson_type** (polarity) | 3 mutually-exclusive values, easy boundary | ✅ Enum field |
| **Activity** (what work is happening) | Continuous flow: debugging-while-refactoring, designing-during-integration | ❌ Not discrete categories; better expressed as record metadata or KG context |
| **Pattern** (abstract lesson) | Multi-layer composition: every record exemplifies 2-4 patterns at different abstraction levels | ❌ Not flat classifiable; better expressed as KG nodes referenced by many records |

Faceted classification only fits the first two. The other two need different machinery.

## Consequences

### Immediate (Stage 1 scope change)

- `docs/plans/vocab_v2.1.json` is the artifact for v2.1; superseded but retained for diagnostic comparison.
- New `docs/plans/vocab_v3.json` contains only `facets.matter` (16 tags) + `lesson_type` (3 values). No Activity or Pattern facet.
- New `scripts/run_minimal_tagging.py` (or extension of existing) produces per-record output: `{matter_tags: [...], lesson_type: ...}`.
- Stage 1 acceptance plan dimensions ② (coverage) and ③ (silhouette quality) re-scoped to evaluate only Matter + lesson_type.
- The `record_aspects.jsonl` data (LLM-extracted activity_phrase + pattern_phrase per record) is **retained as freeform metadata** — useful for full-text search and as raw material for Stage 2's KG construction, even if not consumed by the tagging pipeline.

### Stage 2 redesign (planned, not yet executed)

The original roadmap framed Stage 2 as "sparse typed record-record edges (refines/contradicts/requires)". With Activity/Pattern dropped, Stage 2 expands to also include:

- **Pattern nodes** (separate from record nodes): write ~15-20 GoF-style pattern entries with name/context/problem/solution. Records reference patterns via typed `instance_of` edges. A single record can `instance_of` multiple patterns naturally — no forced single-tag.
- **Activity inferred at retrieval time**: rather than tagging activity per record, the agent retrieval pipeline can infer "this query is about debugging-class records" via record content + lesson_type filters.

### What we are NOT doing

- ❌ Not retrying Activity/Pattern facet with different K, different cluster method, or different LLM prompts. Two attempts (one ideal-method) is sufficient evidence; further tuning is sunk-cost.
- ❌ Not introducing a Pydantic `Activity` enum with finer subcategories. The underlying issue is dimensional structure, not value count.
- ❌ Not abandoning lesson_type (clean enum, 100% coverage).
- ❌ Not abandoning Matter facet (50%+ healthy and improving).

## Related artifacts

- `outputs/network_v2.json` + `outputs/network_v2_diagnostics.txt` — v2 (handcrafted) full data
- `outputs/network_v2_1.json` + `outputs/network_v2_1_diagnostics.txt` — v2.1 (data-driven) full data
- `outputs/record_aspects.jsonl` — 696 records × 2 axis phrases (preserved for Stage 2 use)
- `outputs/activity_clusters.json` / `outputs/pattern_clusters.json` — failed clustering artifacts (preserved for analysis)
- `~/.claude/projects/.../memory/project_tag_taxonomy_lesson.md` — updated with this ADR's conclusion

## References

- Ranganathan (1933) — Colon Classification; faceted classification works when facets are orthogonal *and* internally discrete. The latter condition is now empirically known to fail for SE lessons on the Activity and Pattern axes.
- Gene Ontology Consortium — biology's faceted classification succeeds because its 3 facets (biological_process, molecular_function, cellular_component) all have discrete underlying ontology. Software engineering lessons do not have that property uniformly.
- Christopher Alexander (1977) — *A Pattern Language* explicitly treats patterns as a *network* with `uses` and `is_used_by` relations, not as a partition. This is what Stage 2 will adopt.
