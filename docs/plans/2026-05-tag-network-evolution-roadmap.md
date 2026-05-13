# Tag Network Evolution Roadmap

**Date**: 2026-05-13
**Status**: Decision recorded; Stage 1 not yet started
**Author**: shiwj (planner) — based on diagnostic findings in this session

## TL;DR

The vocab is a **controlled vocabulary for an Experience Repository** (业界 30 年成熟领域, e.g. NASA LLIS, Basili's Experience Factory). The current 49-tag flat vocab is broken because it **混合了三种本质不同的 facet**（Matter / Activity / Pattern）into one table, forcing them to compete in the same retrieval space.

**Decision**: Evolve in 3 stages. **Do NOT start with knowledge graph** — classification is the backbone, KG edges are a layer on top.

```
Stage 1 (now): Refactor flat vocab → 3-facet faceted classification
Stage 2 (next): Add sparse typed edges between records (refines / contradicts / requires)
Stage 3 (later): HippoRAG2-style hybrid retrieval (dense + sparse KG + facet filter)
```

Each stage produces standalone value. Skip-ahead optional only if Stage 1 outcomes shift the picture.

## Why this order (the architectural call)

**Classification answers**: "**WHAT** is each record about" — assigns properties to records
**KG answers**: "**HOW** are records related to each other" — types edges between records

These are orthogonal capabilities. They are NOT mutually exclusive, but classification is **prerequisite to KG**:

- A KG node still needs a "what is this about" label — that's classification.
- KG edges are valuable only when there's real relationship signal. On 696 records, the density of true `refines`/`contradicts`/`requires` relations is empirically very low (estimated < 100 pairs).
- Going KG-first means paying for 696×696/2 = ~240K LLM relation-extraction queries for ~0.04% useful edges — bad ROI before classification is correct.

The project's existing HippoRAG2 exploration (per `~/.claude/projects/.../memory/project_state.md`) is consistent with this layering — HippoRAG2 is **hybrid**, combining KG and dense retrieval **on top of** structured anchors. We are currently only at the "Document + record→tag edges + dense embedding" layer with **broken tag layer**. Stage 1 fixes that layer; Stage 2+3 build on it.

## Diagnostic Evidence (motivating this roadmap)

Source: `outputs/network_diagnostics.txt` (regenerated 2026-05-13 after fixing v3/v4 embedding inconsistency)

| Metric | Value | Interpretation |
|---|---|---|
| Total edges analyzed | 765 (full coverage) | Diagnostic now trustworthy |
| Mean cosine(rec, assigned) | 0.401 | Mediocre semantic match |
| Silhouette < 0 | **80.8%** | LLM's tag pick is rarely the cosine-nearest |
| LLM picked top-1 cosine match | 19.2% | LLM optimizes for different objective than cosine |
| Healthy tags (sil > 0) | **4 / 49** (date_semantics, llm_output_contract, symbolic_link_semantics, staged_diff_fidelity) — **all in Matter facet** |
| Worst hubs (sil ≤ -0.30) | abstraction_leak, failure_observability, single_source_of_truth, consistency_contract — **all in Pattern facet** |
| 17 orphan records | All real concept gaps clustering into 4-5 missing tag families | Vocab undersized too |

**Root cause**: Mixed-facet flat vocabulary. The 4 healthy tags are concrete technical concepts (Matter facet); the worst tags are abstract themes (Pattern facet). They compete in the same vector space, and the abstract Pattern tags lose every time on cosine basis.

**Critical**: This is **not** fixable by rewriting abstract definitions to add concrete words. That treats the symptom. The structural fix is separating facets.

## Stage 1 — Faceted Classification Refactor

**Goal**: From `flat vocab (49 tags)` → `3-facet vocab (3 separate controlled lists)`

### Facet Definitions

| Facet | Semantics | Tag style | Per-record count |
|---|---|---|---|
| **Matter** | 技术主语 / 对象 (what the record technically touches) | Noun, concrete: `state_schema`, `cli`, `llm_output`, `date`, `directory`, `frontend`, `api`, `git_diff` | 1-2 |
| **Activity** | 工程活动 (what kind of work was happening) | Gerund/verb: `debugging`, `refactoring`, `testing`, `migrating`, `version_management`, `merge` | 1 |
| **Pattern** | 失败/解决模式 (the recurring concept the lesson represents) | Concept: `silent_failure`, `single_source_of_truth`, `defensive_integration`, `abstraction_leak`, `early_validation` | 0-2 |

Mean tags per record: from current **1.13** → target **3-4**.

### Reference Frameworks

- **Ranganathan PMEST** (1933) — original faceted classification theory
- **W3C SKOS** (2009) — standard for representing concept schemes with `prefLabel`, `definition`, `broader`, `narrower`, `related`, `inScheme`
- **Gene Ontology** (industrial 25-year example) — 3 namespace (= facet) + within-facet hierarchy, directly analogous to our scale

### Concrete Deliverables

1. **Data model change**: extend `KnowledgeRecord` `selected_tags` schema to include `facet` field per tag, or use 3 separate fields (`matter_tags`, `activity_tags`, `pattern_tags`).
2. **Re-bootstrap vocab per facet** (not from scratch — partition existing 49 into facets, fill gaps):
   - Matter: ~20-25 tags, ~70% from existing concrete ones
   - Activity: ~8-12 tags, mostly new
   - Pattern: ~12-15 tags, from existing abstract ones (cleaner now since they don't compete with Matter)
3. **Re-run tagging pipeline** (reverse_check) to assign all 696 records to the 3-facet scheme.
4. **Re-run diagnostic** (silhouette per facet) — expect within-facet silhouette > 0 for healthy facets; cross-facet comparison no longer applies.

### Success Criteria

- ≥ 70% of records have a tag in all 3 facets
- Within Matter facet: mean silhouette ≥ 0
- Within Pattern facet: mean silhouette ≥ 0
- Orphan rate < 1% (down from 2.4%)
- Mean tags/record ≥ 3.0 (up from 1.13)

### Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Facet boundary disputes (a tag fits in 2 facets) | Treat as primary-facet decision with cross-references; SKOS `related` solves it |
| Pattern facet still drifts abstract | Stage 1 itself doesn't solve abstraction; relies on facet separation removing cross-facet competition. If still bad: add concrete anchors in defs (per CO Stage 1.5) |
| Re-bootstrap loses good existing assignments | Run side-by-side, compare silhouette; keep best |

## Stage 2 — Sparse Typed Record-Record Edges

**Goal**: Add KG layer with high-precision, low-density edges between records.

**Prerequisite**: Stage 1 complete (classification reliable).

### Edge Types (in priority order)

| Edge type | Definition | Expected count / 696 records |
|---|---|---|
| `refines` | A record updates / supersedes / strengthens B's conclusion | ~30 pairs |
| `contradicts` | A's conclusion conflicts with B's | ~10 pairs |
| `requires` | A's lesson is meaningful only if B's holds | ~10 pairs |
| `instance_of` | record → Pattern facet tag (collapses to Stage 1 edges) | already covered |

### Acquisition Strategy

**NOT** all-pairs LLM evaluation (240K pairs × 1 LLM call = prohibitive). Instead:

1. **Candidate generation**: for each record, take top-K (K=10) most similar by cosine — only ~7K candidate pairs
2. **LLM filtering**: per candidate pair, 1 LLM call asking "is there a refines/contradicts/requires relationship? otherwise return none"
3. **High precision threshold**: only commit edges the LLM rates `confidence: high`

Estimated cost: ~7K LLM calls one-time. Acceptable.

### Success Criteria

- Edge precision (manual sample of 30) ≥ 80%
- Edge density: 30-100 edges total (sparse, not noise)
- 0 contradictions go uncaught between obviously-contradictory record pairs (regression test)

## Stage 3 — HippoRAG2-Style Hybrid Retrieval

**Goal**: Production-grade query pipeline combining all layers.

**Prerequisite**: Stages 1 + 2 complete.

### Pipeline

```
user query
   │
   ├──► dense top-K (cosine over record embeddings)
   │
   ├──► facet filter (parse query for Matter/Activity/Pattern hints, narrow set)
   │
   ▼
   union → PPR over sparse KG (expand via refines/related edges)
   ▼
   re-rank by composite score (cosine + PPR weight + facet match)
   ▼
   top-N records → answer composition
```

### Open Questions (deferred to Stage 3 design)

- How to surface facet hints from natural-language queries (LLM parse vs. heuristic vs. structured input)
- PPR damping factor + edge weights
- Whether to expose facet-filter UI or fully embed in retrieval

## Anti-Patterns to Avoid

These were considered and rejected:

1. **❌ Start with knowledge graph** — KG nodes still need classification; without it, KG is "things connected to other things" with no semantic anchors. Sparse-edge KG without classification is a worse version of dense vector retrieval.
2. **❌ Rewrite abstract tag definitions to add concrete words** — Treats symptom. Even with technical anchors, an abstract Pattern tag will still lose to a Matter tag in the same flat vocab.
3. **❌ Pure pattern catalog (GoF-style)** — Records are *instances* not *patterns*. 696 records ≫ ~20 patterns. Doesn't scale.
4. **❌ Pure folksonomy (free tags)** — Already worse than current 49-tag controlled vocab. Loss of vocabulary discipline.
5. **❌ Big-bang facet design** — Don't try to finalize all 3 facet vocabs before any tagging. Use Stage 1 success criteria as gates; iterate.

## Related Memory & Docs

- `~/.claude/projects/.../memory/project_tag_taxonomy_lesson.md` — the lesson behind this roadmap
- `outputs/network_diagnostics.txt` — current quality baseline (regenerate after each stage)
- `scripts/diagnose_assignment_quality.py` — the metric tool

## Industry References (for future research)

- W3C SKOS Primer (2009)
- "Faceted Classification of Information" — Denton, 2003
- Basili et al. "The Experience Factory" 1994
- NASA Lessons Learned Information System (LLIS), 1996
- Gene Ontology Consortium — *operational* faceted classification at scale
- IBM Orthogonal Defect Classification (ODC) — multi-facet for software defects
- HippoRAG / HippoRAG2 papers — hybrid retrieval architecture
