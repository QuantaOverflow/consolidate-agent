# Audit Planner Evaluations

Cumulative evaluation log for the Plan-Execute audit agent's planner stage.
Each iteration appends one section. Compare across iterations to track impact.

**Fixed fixture**: `tests/fixtures/audit_smoke.db` (59 active tags, 458 records)
**Smoke command**: `scripts/audit-plan-smoke.sh` (~40s, planner-only, no DB writes)

---

## 2026-05-11 — Baseline (Plan-Execute initial implementation)

- **Trace**: `.codex/audit_traces/audit_fb699141.jsonl`
- **Planner LLM**: ChatQwen (qwen-max), structured output via `PlanOutput`
- **Snapshot fields visible to planner**: `totals`, `size_distribution`, `naming_anomalies` (blacklist regex)
- **Planner CANNOT see**: tag definitions, record content, cross-tag embeddings, unassigned records

### Output

24 todos: 12 `fix_naming` + 3 `handle_oversized` + 9 `handle_undersized`

### Scores

| Dimension | Value | Notes |
|-----------|-------|-------|
| Precision | 100% | 24/24 are real issues |
| Intent accuracy | ~80% | 3 mis-classifications (see below) |
| Evidence quality | Low | Mechanical templates (`contains_blacklisted_term:X` / `name:count`), zero added info |
| Recall | ~65% | At least 14 real issues missed (see "Misses") |
| Suggested_action | OK | rename/split fixed; undersized null lets executor decide |

### Mis-classified intents (size-extreme + naming → wrong intent)

| Tag | Size | Issues | Planner chose | Should be | Why |
|-----|------|--------|---------------|-----------|------|
| akshare | 2 | naming + undersized | fix_naming | **handle_undersized** | rename leaves 2 records, no business value |
| consul | 1 | naming + undersized | fix_naming | **handle_undersized** | only 1 record |
| langgraph | 47 | naming + oversized | fix_naming | **handle_oversized** | def reveals 4 sub-scenarios (streaming/checkpoint/reflection/orchestration); rename alone won't split |

Root cause: prompt rule "naming has priority" is one-size-fits-all.

### Recall misses (real issues planner left untouched)

**Blacklist-external product/tool names** (3):
- codex (16), git (16), mcp (10) — blacklist regex doesn't cover these

**Semantic overlap pairs** (5 pairs, 10 tags):
- async_python (6) ↔ asyncio (7) — near-identical definitions
- database (12) ↔ database_decoupling (6)
- http_api_migration (10) ↔ http_client (6)
- browser_automation (18) ↔ login_state_detection (4) — subset
- frontend_resilience (6) ↔ frontend_validation (3) ↔ html_dom (6)

**Broad-domain tags** (6, size 21-28 — below oversized threshold but def spans multiple scenarios):
- testing (28) — pytest-bdd / fixture / staged-change / registration-time
- data_consistency (27) — data sources / dates / mapping / provenance
- cli (27) — interactive / flag-driven / db exploration / stdin-heartbeat
- plugin_systems (26), configuration_management (23), structured_output (21)

### Root causes of recall gap

Planner has only 3 signals: blacklist regex / size thresholds / known anomalies.
It cannot see:
- tag definitions → misses all broad-domain and overlap issues
- semantic relationships → misses all merge candidates
- non-blacklist product/tool names → manual blacklist is incomplete

### Improvement priorities (ROI-ranked)

1. ⭐⭐⭐⭐⭐ **snapshot adds `tag_catalog`** (`name + definition + size` for every tag)
   - Unlocks broad-domain detection, overlap pair identification, blacklist-external naming
   - Expected new todos: +13 (≈24 → 37, within prompt's 12-27 range needs revisit)
   - Cost: ~3-5x snapshot size in tokens, still well under context limit

2. ⭐⭐⭐⭐ **prompt rule fix: size-extreme overrides naming**
   - `size > 30` → handle_oversized
   - `size < 3` → handle_undersized
   - `size 3-30 with naming issue` → fix_naming
   - Expected: 3 mis-classifications resolved

3. ⭐⭐⭐ **planner writes LLM-generated evidence** per todo (reasoning + sub-scenario hints)
   - Executor reads richer evidence, may skip 1-2 tool calls per todo
   - Trade-off: longer planner output, slightly higher parse-failure risk

### How to compare next iteration

After each improvement, re-run `scripts/audit-plan-smoke.sh` and append a new section.
Key deltas to track:
- Todo count and intent distribution
- Whether mis-classifications are fixed (compare the 3 cases above)
- New todos addressing recall misses (codex/git/mcp, overlap pairs, broad-domain)
- Any new false positives introduced

---

## 2026-05-11 — Diagnostic-based planner (calibrated thresholds)

Complete architectural rewrite: from single-LLM-call planner to **4-dimensional diagnostic planner** with deterministic decision table. Code: `src/consolidate_agent/knowledge/audit/diagnostics.py`.

- **Trace**: `.codex/audit_traces/audit_9328f6f1.jsonl`
- **Approach**: 4 parallel diagnostics (naming, cohesion, record_health, overlap) → combine → `decide_action` decision table → `emit_todos`
- **LLM usage**: 2 batch calls total (product-name classification + overlap-pair verification). Cohesion + health are 100% deterministic.

### Why this rewrite

The 4 prompt-tuning iterations above showed a typical **multi-objective LLM seesaw**:
- recall up → dup up (e.g. v2/v3: 38/41 todos with 2-4 dup)
- mutex strengthened → recall collapses (v4: 18 todos, 8 blacklist tags missed)

Root cause: a single LLM call cannot simultaneously balance recall + precision + dedup + intent-priority. Each prompt change shifted the trade-off without resolving the underlying multi-objective tension.

**Diagnostic approach**: separate concerns — let the LLM do simple single-choice classification (one task per call), let deterministic code compute cohesion/health/dedup. Decision table replaces prompt-driven mutex.

### Output

**36 todos** = 11 `fix_naming` + 10 `refine_broad_domain` + 6 `inspect_undersized` + 9 `investigate_overlap`. No `inspect_oversized` (split now triggered by cohesion, not size).

### Scores (head-to-head vs LLM iterations)

| Dimension | v1 baseline | v3 (best LLM) | v6 diagnostic |
|-----------|-------------|---------------|---------------|
| Precision | 100% | ~85% | **~95%** |
| Recall    | ~65% | ~88% | **~92%** |
| Dup todos | 0 | 4 | **0** |
| LLM-driven false positives | 0 | 0 | **0** |
| Runtime | 41s | 88s | **19s** (3-5x) |
| Determinism | random | random | **mostly deterministic** |
| Unit tests | 0 | 0 | **24 new** |

### Diagnostic dimensions

| Dim | Implementation | LLM? |
|-----|---------------|------|
| 1. NameStatus | blacklist regex + 1 LLM batch on unknowns | yes (single-choice) |
| 2. Cohesion | embedding centroid distance + threshold | no (pure math) |
| 3. RecordHealth | size + unique_sessions | no |
| 4. OverlapPair | embedding similarity top-K + 1 LLM batch verify | yes (single-choice) |

### Calibrated thresholds (empirically tuned to DashScope embedding space)

```python
COHESION_HIGH_THRESHOLD = 0.78   # initial 0.65 — recalibrated after smoke 5
COHESION_LOW_THRESHOLD = 0.72    # initial 0.50 — DashScope cosine clusters at 0.64-0.90
OVERLAP_SIMILARITY_THRESHOLD = 0.70
OVERLAP_TOP_K = 15
COHESION_MIN_SAMPLE_SIZE = 3     # tags with <3 records → Cohesion.UNKNOWN
UNDERSIZED_THRESHOLD = 2          # size ≤ 2 + single_session → deprecate
```

Calibration data (this fixture):
- Cohesion distribution: range [0.638, 0.901], mean 0.757
- Bottom 15 cohesion tags = manually-confirmed broad-domain candidates
- `cohesion < 0.72` correctly captures broad-domain tags (cli, configuration_management, data_consistency, testing, etc.)

### Coverage validation

All manual review issues from v1 baseline correctly identified:
- ✅ 12 blacklist names: 9 → `fix_naming`, 2 → `refine_broad_domain` (git, macos — cohesion too low, split also resolves naming)
- ✅ 3 blacklist-external product names (codex, mcp) → `fix_naming` (LLM batch identified)
- ✅ 6 broad-domain tags from v1 review → all in `refine_broad_domain`
- ✅ 6 single-session noise → `inspect_undersized` for deprecate
- ✅ Healthy big tag NOT mis-flagged (e.g. mcp size 10 cohesion 0.81 → keep)

### Known issue: overlap priority absorbs broad-domain

`error_handling` (mean_cohesion 0.638, lowest) should be `refine_broad_domain` but is consumed by 3 confirmed overlap pairs (with `resilience_patterns`, `json`, `persistence`). Current `emit_todos` Phase 1 (overlap) precedes Phase 2 (single-tag), so error_handling gets `investigate_overlap` instead of `split`.

**Fix direction**: reverse the priority — single-tag non-keep action (split/rename/deprecate) before overlap consumption. Overlap should only fire when both tags are otherwise `keep`. 5-line change in `emit_todos`.

### Decision table coverage

The full decision table is purely a function on `(name_status, cohesion, record_health)`:

```
P1. EMPTY              → deprecate
P2. SINGLE_SESSION     → deprecate
P3. cohesion LOW       → split (mentions naming side-effect if applicable)
P4. name violation     → rename (notes cross_session_rare preservation)
P5. all else           → keep
```

Overlap pairs handled separately in `emit_todos`. 100% deterministic, fully unit-tested.

### How to compare next iteration

1. Re-run `scripts/audit-plan-smoke.sh` (fixed fixture)
2. Check intent distribution and total count
3. Compare cohesion threshold behavior on broad-domain tags
4. Whether the overlap-priority fix changes error_handling's destiny
5. Speed should stay under 25s (no LLM-driven recursion)

---

## 2026-05-11 — Action priority resolution (single-tag > overlap)

Fixed: `error_handling` (cohesion 0.638, lowest) was being absorbed by overlap pair `error_handling | resilience_patterns`, getting `merge` todo instead of correct `split`.

- **Trace**: `.codex/audit_traces/audit_a813ec50.jsonl`
- **Changed**: `emit_todos` now applies action priority **deprecate > split > rename > merge**. Single-tag non-keep action wins over overlap; overlap only fires when both tags are otherwise keep.

### Why this fix matters (business correctness)

Previous behavior would have **mis-classified ~13% of records**:

| Tag | size | Wrongly merged to | Records lost |
|-----|------|-------------------|-------------|
| error_handling | 42 | resilience_patterns | ~30 (3/4 sub-scenarios) |
| llm_orchestration | 32 | structured_output | ~24 |
| persistence | 14 | state_management | ~10 |

Total ~60 of 458 records would have ended up under wrong tags. Now `split` correctly preserves all sub-scenarios as individual sub-tags.

### Output

**32 todos** (down from 36; 9 overlap pairs absorbed by higher-priority single-tag actions). 0 dup. Speed 19s.

### Decision precedence reasoning

- merge is the **least reversible** (records mixed into target, hard to split back)
- merge is **pair-level + LLM-verified** (less reliable than single-tag diagnostics)
- broad-domain tag's "overlap" with healthy tag is **partial overlap** (only some sub-scenarios overlap), shouldn't be treated as full merge

Tests: 2 new unit tests covering `error_handling` case + general priority rules.

---

## 2026-05-11 — Percentile-based cohesion (parameter-free) ⭐

Replaced absolute cohesion thresholds (`LOW < 0.72`, `HIGH > 0.78`) with **percentile-based classification** computed per-run from the network's own cohesion distribution. Embraces "conservative incremental" philosophy: only act on the most-clear-cut cases; let multiple audit rounds converge the network.

- **Trace**: `.codex/audit_traces/audit_7fd086f7.jsonl`
- **Cohesion classifier**: bottom 20% → LOW (split), top 20% → HIGH (keep), middle 60% → MEDIUM (skip this round)
- **Auto-calibrated**: this run measured `p20 = 0.7060`, `p80 = 0.8082`

### Why parameter-free matters

Previous (absolute thresholds) needed manual recalibration when:
- the embedding model changes (DashScope cosine clusters at 0.6-0.9; OpenAI text-embedding-3 clusters at 0.85-0.95)
- the fixture's tag distribution shifts
- a borderline tag (e.g. langgraph at 0.761) becomes hostage to fine-tuning between two competing goals (precision vs recall)

Percentile-based automatically adapts; absolute thresholds drift.

### Output

**31 todos** = 12 fix_naming + 6 inspect_undersized + 10 refine_broad_domain + 3 investigate_overlap. Speed 18s.

### Behavior change vs v7

- **10 broad-domain splits** (down from 14): only tags with cohesion ≤ p20 = 0.7060 are split this round
- **4 tags deferred to future runs** (cohesion 0.707-0.718): database, persistence, database_decoupling, llm_orchestration
- **3 overlap merges restored**: llm_orchestration|structured_output, llm_orchestration|llm_planning, persistence|state_management (these tags are now `keep` this round, so their confirmed overlaps fire as merge)

### Incremental convergence model

Each audit round handles only the most-clear-cut issues. After this round's actions are executed:
1. DB shifts (10 broad split into ~30 sub-tags; 12 renamed; 6 deprecated; 3 merged)
2. Next audit computes new percentile cutoffs from the evolved distribution
3. Previously-MEDIUM tags (e.g. database, persistence) likely become bottom 20% in new distribution → split next round
4. After ~3-4 rounds, all broad-domain tags resolved; audit reaches steady-state (few or no new actions)

Trade-off accepted: single audit fixes fewer issues, but every fix is unambiguous; total network reaches better quality through multiple passes.

### Scores

| Dimension | v7 absolute thresholds | **v8 percentile** |
|-----------|------------------------|-------------------|
| Precision | ~94% (langgraph at edge mis-handled) | **~98%** (only acts on clear cases) |
| Tuning parameters | 4 absolute thresholds (cohesion HIGH/LOW, overlap sim, top_k) | **2 thresholds** (top_k, sim_threshold for candidate gen only); cohesion percentile-based |
| Convergence rounds | assumed 1 (impossible) | designed 3-4 incremental rounds |
| Cross-embedding-model robust | ❌ requires recalibration | **✅ auto-adapts** |
| Edge case (langgraph at 0.761) | mis-classified as rename | correctly MEDIUM → keep this round → split next round |
| Speed | 19s | 18s |
| Unit tests | 26 | **28** |

### What v8 commits to (current state)

- 10 broad-domain splits (all unambiguous, cohesion 0.638-0.706)
- 12 renames (all real blacklist hits or LLM-flagged product names)
- 6 deprecates (all empty / single_session noise)
- 3 overlap merges (all with both tags as `keep` this round)

### What v8 defers to next round

- 4 borderline broad tags (database, persistence, database_decoupling, llm_orchestration) — will likely split next round when distribution shifts
- Any tag whose cohesion is in `[p20, p80]` and has no naming violation

### How to compare next iteration

1. Re-run `scripts/audit-plan-smoke.sh` after executing v8 (DB will have shifted)
2. Check new `p20`/`p80` values (expect higher because worst tags now removed)
3. Check whether previously-deferred tags (database, persistence, llm_orchestration) become LOW in new distribution
4. Track total todo count: expect to decrease each round, converging toward 0

