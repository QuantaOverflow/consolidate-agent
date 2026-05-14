# ADR-0005: Deprecate silhouette as quality metric in multi-tag setting

**Status**: Accepted
**Date**: 2026-05-13
**Supersedes**: silhouette PASS/FAIL gates in `docs/plans/2026-05-stage1-acceptance.md` dimension ③

## Context

Silhouette score (Rousseeuw 1986) was adopted as the primary quality metric
for record-tag assignments in Stage 1. It assumed each record-tag edge
should have the assigned tag as the **single best fit** among the vocab.

After moving from v3 (mean 1.10 tags/record) → v3_p1 (mean 1.63 tags/record),
silhouette and ground-truth metrics diverged:

| Metric | v3 baseline | v3_p1 (multi-tag) | Direction |
|---|---|---|---|
| Matter exact vs golden | 46.2% | **51.2%** | ↑ improved |
| Matter partial vs golden | 88.8% | **91.2%** | ↑ improved |
| Mean matter/record | 1.10 | 1.63 | (multi-tag) |
| **Silhouette mean** | +0.027 | **-0.040** | ↓ "worsened" |
| **Silhouette healthy ratio** | 56% | **31%** | ↓ "worsened" |
| LLM top-1 (cosine) hit | 58% | 44% | ↓ "worsened" |

Ground-truth (golden) improved; silhouette regressed. They cannot both be right.

## Root cause

Silhouette is defined per-edge: `(a - b) / max(a, b)` where `a` = cosine
to assigned tag's def, `b` = max cosine to any OTHER tag's def. It assumes
the assigned tag should be the unique top match.

In multi-tag assignment, a record legitimately has 2 (or more) tags.
For its **secondary** tag:
- `a` = cosine to secondary tag's def (legitimate match, ~0.4-0.5)
- `b` = cosine to primary tag's def (also assigned, even higher)
- silhouette becomes negative even though the secondary tag is correct

Example: record "HTTP client environment variables can break local service calls"
- Tagged `[http_api, config_env]` (both correct per golden truth)
- http_api edge silhouette: -0.3 (config_env is even higher) — **looks bad**
- config_env edge silhouette: +0.2 (http_api is also high but slightly lower) — **looks ok**
- Aggregate: this record has 1 "negative" edge in our healthy-ratio computation
- But the assignment is exactly right

Silhouette mechanically penalizes the secondary tag in every multi-tag
record, regardless of correctness. This is a property of the metric, not
a quality problem in the data.

## Decision

**Silhouette is deprecated as a PASS/FAIL gate in multi-tag scenarios.**
It is retained as a **diagnostic** signal (worst-N edges to investigate),
but no longer drives:
- Acceptance plan dimension ③ PASS/WARN/FAIL judgments
- Agent maintenance propose_refine targeting decisions
- "Tag healthy" verdicts

## Replacement: 3-tier metric stack for multi-tag setting

### Primary (drives ship/no-ship)

1. **Golden set match** — human ground truth on 80 labeled records
   - Matter exact match (LLM tag set == golden tag set)
   - Matter partial overlap (≥1 tag in common)
   - lesson_type exact match

2. **Intra-cluster coherence** — mean pairwise cosine among records sharing a tag
   - High = tag's members semantically tight (concept-unified)
   - Multi-tag friendly: doesn't penalize secondary tags
   - Already implemented in `scripts/intra_cluster_coherence.py`

### Secondary (diagnostic)

3. **Tag plausibility per record** — fraction of assigned tags appearing in top-K
   cosine neighbors of the record (K=5 default)
   - Each record's full tag set evaluated together
   - 1.0 = all assigned tags are top-K plausible matches
   - <1.0 = at least one assigned tag is far from the record by cosine

4. **Hallucination rate** — % of LLM-emitted tag names not in vocab

5. **Coverage rate** — % records with ≥1 matter_tag

### Sanity (hygiene)

6. **Mean tags/record** in target range (1.3-2.0 for current scope)

7. **Tag size distribution** — no tag dominating (>30%) or starving (<3 records)

8. **Silhouette** — retained as **diagnostic only** (worst-N edges to inspect),
   marked clearly in reports as "not a PASS/FAIL gate"

## Consequences

### Immediate

- `docs/plans/2026-05-stage1-acceptance.md` dimension ③ thresholds invalid;
  superseded by this ADR
- `scripts/diagnose_faceted_quality.py` retained as diagnostic but its
  PASS/FAIL verdicts removed or marked deprecated
- New `scripts/quality_scorecard.py` combines all primary + sanity metrics
  into a single dashboard with verdicts based on the 3-tier stack

### For Stage 1 ship decision

Stage 1 v3 + v3_p1 PASS criteria (replacing dimension ③):
- Golden Matter exact match: ≥ 50% overall (was ≥65% easy / ≥45% medium / ≥25% hard — keep difficulty breakdown)
- Golden Matter partial overlap: ≥ 85% overall
- Golden lesson_type exact: ≥ 80% (acknowledge currently at 70%, P0 prompt fix deferred)
- Intra-cluster coherence: ≥ 50% tags at "OK" (≥0.45) or better
- Hallucination rate: ≤ 1%
- Coverage: ≥ 98%
- Mean tags/record: 1.3-2.0

### For agent maintenance (P2)

`propose_refine` currently targets tags with low mean_fit (forced_fit_candidates
in `compute_fit_signals`). After this ADR:
- Maintenance still useful for **definition sharpening** (LLM-driven refactor of
  tag definitions)
- Maintenance prune operations now risk removing legitimate secondary tags
- Decision: keep agent loop unchanged for now; **validate maintenance output
  against golden set** rather than silhouette improvement; if golden regresses,
  reconsider agent's internal targeting

## What this does NOT mean

- Silhouette is not "wrong" — it's correctly measuring what it measures.
  It's just measuring the wrong thing for multi-tag classification quality.
- v1 → v2 → v2.1 → v3 silhouette comparisons in earlier ADRs/diagnostics
  remain valid for single-tag-dominant comparisons (all those iterations
  had mean tags/record ≈ 1.1, where silhouette was reasonable).
- Single-tag-dominant settings (mean ~1.0-1.1) could still use silhouette
  legitimately; only multi-tag (mean ≥ 1.3) deprecates it.

## References

- Rousseeuw 1986 — original silhouette formulation, defined for single-cluster
  membership (k-means style). Multi-label classification literature
  (Tsoumakas 2007, Read 2009) uses different metrics: Hamming loss, F1 per
  label, subset accuracy — none translate directly to "is this assignment
  the optimal one" the way silhouette tried to.
- The bias surfaced because we expanded mean tags/record from 1.10 → 1.63,
  pushing the corpus into a regime where single-membership metrics break.
