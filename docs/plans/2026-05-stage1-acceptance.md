# Stage 1 Acceptance Plan — Flat Vocab → 3 Facets + lesson_type Enum

**Date**: 2026-05-13
**Status**: Acceptance criteria finalized; Gate 1 (vocab draft) not yet started
**Parent**: `2026-05-tag-network-evolution-roadmap.md`
**Author**: shiwj (planner)

## Scope

Move from current state (1 flat vocab, 49 tags, no facet) to:
- **3 tag-based facets**: Matter / Activity / Pattern
- **1 enum field**: lesson_type ∈ {anti_pattern, discovery, best_practice}
- ~30 total tags + 3-value enum
- Mean tags/record from 1.13 → target 3-5

Does **NOT** include: re-extracting records from raw jsonl, knowledge-graph edges, hierarchical sub-tags within facets, retrieval pipeline changes. Those are deferred to Stage 2/3.

## Key Metric Definitions

These two metrics drive most of the acceptance gates. Both grounded in record/tag embeddings (text-embedding-v4).

### Silhouette score (per record-tag edge)

For each `(record, assigned_tag)` edge, compute:

```
a = cosine(record_emb, assigned_tag_def_emb)            ← similarity to assigned tag
b = max cosine(record_emb, other_tag_def_emb)           ← similarity to best alternative
silhouette = (a - b) / max(a, b)
```

**Range** `[-1, +1]`:

| Value | Meaning |
|---|---|
| `+1` | assigned tag is the obvious winner; very different from alternatives |
| `+0.3` | clear winner with close neighbors |
| `0` | tie with best alternative; coin flip |
| `-0.3` | better alternative exists; assignment is suboptimal |
| `-1` | assigned tag is nearly unrelated to record; obvious mis-assignment |

**Worked example from current vocab** (`outputs/network_diagnostics.txt`):

```
record   : "LLM node JSON parsing failure causes fallback to template text"
assigned : failure_observability
  cosine(record, failure_observability_def)  = 0.194  ← a

best_alt : llm_output_contract
  cosine(record, llm_output_contract_def)    = 0.600  ← b

silhouette = (0.194 - 0.600) / max(0.194, 0.600) = -0.677
```

Interpretation: `llm_output_contract` is objectively a much better fit; the assignment is wrong.

**Why silhouette > raw cosine**: pure cosine can't distinguish "everything is low-similarity" from "this is the worst choice among low-similarity options". Silhouette is **relative**, so it pinpoints mis-assignments even when absolute scores are uniform.

**Origin**: Rousseeuw 1986, the classic clustering quality metric. Independent of any human-defined threshold.

### Healthy tag ratio (per facet)

For each facet:

```
For each tag T in facet F:
    edges_T          = all (record, T) edges
    mean_sil(T)      = mean(silhouette of edges in edges_T)
    is_healthy(T)    = (mean_sil(T) > 0)

healthy_ratio(F) = count(healthy tags in F) / count(all tags in F)
```

**Worked example on current flat vocab** (baseline):
- 49 tags total
- Tags with mean silhouette > 0: `date_semantics`, `llm_output_contract`, `symbolic_link_semantics`, `staged_diff_fidelity` (4 tags)
- `healthy_ratio = 4/49 = 8.2%`

Stage 1 success means **Matter facet healthy_ratio ≥ 70%**, **Activity ≥ 70%**, **Pattern ≥ 50%**. This is an 8× improvement on the Matter facet — but achievable because cross-facet competition is eliminated once tags are partitioned.

## Acceptance Dimensions

5 dimensions, each PASS / WARN / FAIL. All must PASS for Stage 1 ship; any FAIL triggers rollback.

### ① Vocab Structure Compliance

| Check | PASS | WARN | FAIL |
|---|---|---|---|
| Matter facet size | 12-18 | 8-11 or 19-22 | <8 or >22 |
| Activity facet size | 6-10 | 4-5 or 11-12 | <4 or >12 |
| Pattern facet size | 5-9 | 3-4 or 10-12 | <3 or >12 |
| lesson_type enum values | exactly 3 (anti_pattern / discovery / best_practice) | — | ≠ 3 |
| Tag name uniqueness across facets | 100% unique | — | any duplicate |
| Definition contains tag's name keywords | ≥ 90% | 70-89% | <70% |
| Definition word count (mean) | 15-40 | — | <10 or >60 |

**Validation**:
```bash
python scripts/validate_vocab_structure.py docs/plans/vocab_v2.json
```
(new script; outputs PASS/WARN/FAIL table + non-compliant tag list)

### ② Tagging Coverage

| Check | PASS | WARN | FAIL |
|---|---|---|---|
| Records with ≥1 matter_tag | ≥ 98% | 90-97% | <90% |
| Records with activity_tag | ≥ 95% | 85-94% | <85% |
| Records with ≥1 pattern_tag (optional facet) | 60-85% | 40-59% or >90% | <40% |
| Records with lesson_type assigned | 100% | — | <100% |
| Mean tags/record | 3.0-4.5 | 2.5-2.9 or 4.6-5.5 | <2.5 or >5.5 |
| Orphan rate (no tags in any facet) | <1% | 1-3% | >3% |
| Long-tail tags per facet (≤2 records) | ≤ 2 | 3-4 | >4 |

**Validation**:
```bash
python scripts/validate_tagging_coverage.py outputs/network_v2.json
```

### ③ Tagging Quality (most critical)

Silhouette computed **within each facet** (compared to other tags **in the same facet**, not across facets).

| Check | PASS | WARN | FAIL |
|---|---|---|---|
| Matter facet mean silhouette | ≥ +0.05 | -0.05 to +0.05 | < -0.05 |
| Activity facet mean silhouette | ≥ +0.05 | -0.05 to +0.05 | < -0.05 |
| Pattern facet mean silhouette | ≥ 0 (Pattern is abstract — lower bar) | -0.10 to 0 | < -0.10 |
| Matter healthy_ratio | ≥ 70% | 50-69% | <50% |
| Activity healthy_ratio | ≥ 70% | 50-69% | <50% |
| Pattern healthy_ratio | ≥ 50% | 30-49% | <30% |
| LLM top-1 cosine hit rate (within facet) | ≥ 50% | 30-49% | <30% |
| Total edges count | 2100-3100 | 1800-2099 | <1800 |

**Validation** (existing diagnostic script extended with `--facet`):
```bash
python scripts/diagnose_assignment_quality.py --facet matter
python scripts/diagnose_assignment_quality.py --facet activity
python scripts/diagnose_assignment_quality.py --facet pattern
```

### ④ Golden Set Consistency (the human-grounded anchor)

**Why this matters**: cosine-derived metrics measure self-consistency, not correctness. Golden set is the only human ground truth in the acceptance pipeline. Without it, silhouette-PASS could still mean "vocab is internally consistent but semantically wrong".

#### Golden Set Composition (80 records)

```
80 records:
  ├ 30 easy    — clean cases, single obvious classification
  │   ├ 10 anti_pattern   (e.g., "StateGraph silently drops unknown keys")
  │   ├ 10 discovery      (e.g., "git status on untracked dir prints empty")
  │   └ 10 best_practice  (e.g., "asyncio.to_thread for blocking IO")
  ├ 30 medium  — require judgment
  │   ├ 10 cross-Matter records (state + LLM, test + DB)
  │   ├ 10 Activity ambiguous (refactor vs migration vs debugging)
  │   └ 10 Pattern boundary (between explicit_contract and silent_failure)
  └ 20 hard    — intentional edge cases
      ├  5 orphan candidates (no tag fits well in current vocab)
      ├  5 multi-Matter (refactor is both Activity and Matter object)
      ├  5 lesson_type ambiguous (anti_pattern + discovery)
      └  5 currently-orphan from v1 vocab
```

#### Labeling Protocol

- **Manually labeled by planner BEFORE seeing any v2 LLM output** (anchoring-bias prevention is non-negotiable)
- Stored in `tests/fixtures/golden_80.jsonl`, one record per line:
  ```json
  {
    "record_id": "knowledge_xxx",
    "difficulty": "easy" | "medium" | "hard",
    "expected": {
      "matter_tags": ["state_management", "llm_agent_runtime"],
      "activity_tag": "debugging",
      "pattern_tags": ["silent_failure"],
      "lesson_type": "anti_pattern"
    },
    "acceptable_matter_alternatives": ["..."],  // optional, for genuine multi-answer cases
    "notes": "matter is multi-axis here; both state + LLM count"
  }
  ```
- For genuinely multi-answer cases (5-10 records expected), `acceptable_*_alternatives` allows LLM to pick any listed option without penalty.

#### Scoring Thresholds (80-set)

Sample size 80 lets us tighten thresholds vs. a 30-set baseline (±5% CI tighter):

| Check | PASS | WARN | FAIL |
|---|---|---|---|
| Matter tag exact match | ≥ 65% easy / ≥ 45% medium / ≥ 25% hard | each -10% | each -20% |
| Matter tag partial overlap (≥1 in common) | ≥ 88% easy / ≥ 72% medium / ≥ 45% hard | each -8% | each -15% |
| Activity tag exact match | ≥ 82% easy / ≥ 62% medium | each -10% | each -20% |
| Pattern tag partial overlap | ≥ 72% easy / ≥ 52% medium | each -10% | each -20% |
| lesson_type exact match | ≥ 82% overall | 65-81% | <65% |

**How to use golden set results**:
- If easy fails: vocab definitions are unclear or wrong — fix vocab
- If medium fails but easy passes: prompt needs more disambiguation guidance — fix prompt
- If hard fails but easy + medium pass: acceptable, hard is hard by design
- If only lesson_type fails: enum prompt section is broken — isolated fix

**Validation**:
```bash
python scripts/eval_golden_set.py outputs/network_v2.json --golden tests/fixtures/golden_80.jsonl
```

### ⑤ Downstream Compatibility

| Component | PASS | FAIL |
|---|---|---|
| `scripts/diagnose_assignment_quality.py --facet <X>` | runs all 3 facets, reports per-facet | errors |
| `scripts/export_obsidian_vault.py` | vault generated with tags/ split by facet subdir | errors |
| `scripts/analyze_maintenance_health.py` | unchanged behavior (decoupled from facet) | regression |
| `vocab_maintenance` maintenance run | propose_merge / propose_deprecate work within single facet; refuse cross-facet ops | errors |
| Existing unit tests | 75/75 PASS | any regression |
| Legacy `selected_tags` field | retained read-only for ≥1 maintenance cycle as fallback | deleted before observation window |

## Decision Gates

```
┌──────────────────────────────────────────────────────────────────┐
│ Gate 1: Vocab Draft Review                                       │
│   Planner drafts ~30 tags across 3 facets + 3-value enum         │
│   + 80 golden set labels                                         │
│   → human review of vocab definitions and facet boundaries       │
│   → APPROVE → Gate 2                                             │
│                                                                  │
│ Gate 2: Schema + Pipeline Build                                  │
│   coder: add facet fields to schema, rewrite reverse_check       │
│   for 4-output structured generation, write 2 validation scripts │
│   → ① PASS → Gate 3                                              │
│                                                                  │
│ Gate 3: Full Re-Tag                                              │
│   Run new tagging pipeline on all 696 records                    │
│   → ② + ③ PASS → Gate 4                                          │
│   → ② FAIL or ③ FAIL → rollback to v1 vocab + redesign           │
│   → ③ WARN → analyze worst tags, decide iterate or ship          │
│                                                                  │
│ Gate 4: Golden Set Evaluation                                    │
│   Run eval_golden_set against the 80 manual labels               │
│   → ④ PASS → Gate 5                                              │
│   → ④ WARN → fix vocab definitions or prompt, re-run Gate 3      │
│   → ④ FAIL → rollback                                            │
│                                                                  │
│ Gate 5: Downstream Adaptation                                    │
│   Update obsidian export, diagnostic, maintenance to facet-aware │
│   → ⑤ PASS → Stage 1 SHIP                                        │
└──────────────────────────────────────────────────────────────────┘
```

## Rollback Strategy

| Trigger | Action |
|---|---|
| Gate 3 dimension ③ FAIL (per-facet silhouette negative) | Keep v2 vocab draft as artifact; revert `selected_tags` to v1 in DB; analyze whether facet boundaries are wrong |
| Gate 4 golden set FAIL | Don't touch schema; only revise vocab definitions or LLM prompt; re-run Gate 3 |
| Any stage downstream-script breakage | `git revert` all Stage 1 commits; restore network.json + vault from backup |

**Mandatory 1-cycle coexistence window**: `selected_tags` v1 field is preserved read-only for ≥ 1 week after Stage 1 ship. Only deleted after observed system-stable.

## Time Estimates

| Stage | Work | Estimate | Owner |
|---|---|---|---|
| Gate 1 | Vocab draft + 80 golden labels | **3-4 h** | Planner |
| Gate 2 | Schema migration + pipeline rewrite + 2 validation scripts | **4-6 h** | Coder |
| Gate 3 | Re-tag 696 records via LLM | **~30 min LLM cost** + **1 h review** | Coder + planner |
| Gate 4 | Golden eval + prompt iterations | **1-2 h** | Planner |
| Gate 5 | Adapt obsidian/diagnostic/maintenance to facet-aware | **2-3 h** | Coder |

**Total: 2-2.5 working days.**

## Open Risks (acknowledged before starting)

| Risk | Probability | Mitigation |
|---|---|---|
| Facet boundaries unclear in practice (a tag fits both Matter and Activity) | medium | Allow SKOS-style `related` cross-facet links; treat as primary-facet decision |
| Pattern facet still has negative silhouette even after isolation | medium | Pattern is genuinely abstract; lower threshold to 0 (vs +0.05 for Matter/Activity) accommodates this |
| Golden set 80 still too small to detect subtle issues | low | 80 gives ±5% CI; further increase only if WARN-band noise is unmanageable |
| Re-tag LLM cost spikes due to multi-output structured generation | low | Estimate before run: 696 × 1 call × $0.002 ≈ $1.4. Bounded. |
| lesson_type enum proves to be more than 3 values | medium | Allow Gate 1 to revise; not locked in vocab schema |

## Related Documents

- `docs/plans/2026-05-tag-network-evolution-roadmap.md` — parent roadmap; Stage 2/3 deferred
- `outputs/network_diagnostics.txt` — current baseline metrics for comparison
- `~/.claude/projects/.../memory/project_tag_taxonomy_lesson.md` — the lesson informing this work

## References (for facet design grounding)

- Rousseeuw, P. (1986) — original silhouette score formulation
- Ranganathan (1933) — Colon Classification, faceted classification founder
- W3C SKOS Primer (2009) — concept scheme standard
- ODC (Chillarege et al., 1992) — multi-attribute defect classification
- SWEBOK V4 (IEEE, 2024) — 18 SE Knowledge Areas; reference for Activity facet
- 2024 SLR on SE taxonomies — confirms 39% of SE taxonomies use faceted classification
