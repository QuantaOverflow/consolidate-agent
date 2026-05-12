# Technical Debt & Known Limitations

Snapshot of known issues in the `vocab_maintenance` subsystem. Read before
running on real data.

Last updated: 2026-05-13

---

## 1. `expand_new_tag_coverage`: LLM keyword confusion (residual)

**Path**: `propose/additive.py` — when propose_new adds a new tag, this
function attaches it to already-assigned records via embedding-filter +
LLM binary check.

**Risk**: both layers can be fooled by surface keyword overlap.

**Concrete failure mode** (observed in fake-corpus iter 0, old prompt):
- Record: `"Output token cap truncates JSON mid-string"` (LLM output token limit)
- New tag: `rate_limiting` (API request throttling)
- System said "yes, also apply rate_limiting to this record" — wrong.
- Both contain `token / cap / limit` keywords, but the mechanisms are unrelated.

**Mitigation applied** (commit pending after this doc):
- ADDITIVE_SYSTEM prompt now lists common confusions (`token`, `limit`,
  `session`, `rotation`, `timeout`) with worked examples
- Prompt: "prefer false when uncertain"
- Embedding threshold 0.6 (not 0.5 — fewer surface-similar candidates)

**Residual risk**: prompt cannot enumerate every domain-specific confusion.
Expected false-positive rate ~5-15% on multi-domain real data.

**Recommended monitoring**:
1. After ingest run with `expand_coverage.done` events showing additions,
   spot-check the affected records.
2. If precision degrades, raise embedding threshold (0.6 → 0.75) or set
   `similarity_threshold` kwarg explicitly.
3. Long term: add a HITL review step before committing expand additions.

---

## 2. `propose_merge_fn` focus bug — FIXED

**Path**: `propose/merge.py` — `propose_merge_fn(vocab, assignments, focus, ...)`

**Status**: FIXED. `propose_merge_fn` reuses `_extract_focus_tags` from
`deprecate.py`. When focus names vocab tags, candidate pairs are restricted:
1 focus tag → pairs containing it; ≥2 focus tags → pairs entirely within
the focus set. Both paths bypass `min_cooccur` so agent-directed evaluation
isn't suppressed by the default cooccur threshold. Empty focus preserves
the original top-K behavior.

**Not yet validated with real LLM**: the partial-apply path in `apply_node`
has only been exercised end-to-end on deprecate proposals; merge proposals
share the same code so should work, but no real run has confirmed it.

---

## 3. HITL real-interactive path not verified end-to-end

**Path**: `graphs/bootstrap.py` — `vocab_review_node` calls `interrupt()`
when `auto_accept=False`.

**Status**: works in unit tests (MemorySaver + faked interrupt return);
not exercised with real stdin user input + real SqliteSaver.

**Risk**: edge cases (terminal encoding, EOF, malformed input, very long
vocab display) may cause runtime errors not caught by current tests.

**Mitigation**: `run_bootstrap.py` defaults to `--auto-accept`, so
unattended runs avoid the path. Interactive mode is opt-in.

---

## 4. Crash + thread_id resume not verified end-to-end

**Path**: `graphs/checkpointer.py` — `SqliteSaver` writes per-node-boundary.

**Status**: SqliteSaver verified to write rows during normal runs (commit
`3e05bd3`). The resume-after-crash path is structurally in place but has
not been tested by killing a process and reinvoking with same `thread_id`.

**Risk**: state serialization issues (e.g., dataclass `proposals` in
agent's `rec`) may not roundtrip cleanly through SqliteSaver across
process restarts.

**Fix when needed**: integration test that kills mid-run and reinvokes.

---

## 5. Phase 3 incomplete: `ingest_batch` is not a LangGraph StateGraph

**Path**: `network.py` — `TagRecordNetwork.ingest_batch`

**Status**: Original 3-phase migration plan completed phases 1 (bootstrap)
and 2 (agent). Phase 3 (ingest → graph) was deferred as low-value:
- Ingest is a short operation (~30-90s)
- Doesn't need crash resume
- Doesn't need HITL (no human-decision point)

**Trade-off**: bootstrap + agent are graphs; ingest is a Network method.
Mental model inconsistency.

**Fix when needed**: ~3-5h work. Defer until ingest gains a need for HITL
(e.g., review-before-add-new-tag flow).

---

## 6. Network backup — FIXED (single rolling `.bak`)

**Path**: `network.py` — `Network.save(path, *, keep_backup=True)`

**Status**: FIXED. Every `save()` copies the previous file content to
`{path}.bak` before writing the new content (atomic tmp + replace). Single
rolling backup — `.bak` always contains the version that existed
immediately before the most recent save. To roll back:
`cp network.json.bak network.json`.

**Limitation**: only one level deep. If two consecutive saves are both
bad, the original state is lost. For longer history, version-control
the file or copy to dated archives externally.

---

## 7. `network.themes` grows unbounded

**Path**: `network.py` — `themes: dict[str, str]` field

**Status**: each new record adds one entry to themes dict. For long-running
networks (years of ingest), 100k+ entries possible.

**Impact**:
- `network.json` file size grows linearly (each entry ~50-200 bytes)
- LRU embedding cache (`similarity._embed`) is bounded at 1024 but new
  theme texts beyond that pay re-embedding cost
- expand_new_tag_coverage iterates over all themes — O(N) per propose_new

**Fix when needed**: TTL eviction (drop themes older than X), or
"archive" mechanism that moves stale entries out of the active network.

---

## 8. No LLM cost monitoring

**Path**: `observability.py` RunLogger — records elapsed time but not
token counts.

**Status**: hard to know how much each operation costs at the LLM API
level.

**Fix**: extract usage from DashScope response metadata or LangChain
callbacks. Emit `llm.tokens` events alongside `llm.success` / `llm.error`.
~30 lines.

---

## 9. Concurrent runs — partially fixed

**Path**: any code that loads + saves the same `network.json`

**Status**: `save()` now holds a cross-process `fcntl.LOCK_EX` at
`{path}.lock`, so two simultaneous `save()` calls serialize correctly
(no torn writes, no lost atomic-replace).

**Still unsafe**: the lost-update problem across full load → mutate → save.
Process A loads state v1, process B loads v1, both mutate independently,
both save serially → second save overwrites first's mutations.

**Workaround**: keep the single-process discipline (one bootstrap/ingest
at a time), or have callers acquire `_file_lock(network.json.lock)`
externally around the full load → mutate → save cycle.

**Real fix**: migrate to SQLite (see overall storage discussion). Until
then, mutation throughput is by design single-writer.

---

## 10. `reverse_check` prompt is dataset-domain agnostic — may misfire on niche corpora

**Path**: `measure.py` — `SYSTEM_PROMPT`

**Status**: commit `d04ba36` tightened the prompt to prefer `missing=true`
over weak force-fit. Validated on engineering-knowledge fake corpus (3
iterations stable).

**Risk on real data**:
- If the user's records span domains (e.g., engineering + design + business),
  the umbrella-tag-detection heuristic may not transfer
- Negative examples in prompt are engineering-flavored

**Mitigation**: re-evaluate prompt's missing-rate after first real-data run.
If hit_rate drops too low (< 0.7), records are being marked missing too
aggressively — relax the prompt.

---

## 11. `propose_new` has no near-duplicate safety net against existing vocab

**Path**: `propose/new.py` — `propose_new_fn`

**Status**: NOT IMPLEMENTED.

`propose_new` shows the existing vocab to the LLM in its synthesize prompt
and instructs it to "fill gaps, don't recreate". This is a soft guard.
There is no deterministic check that a candidate tag isn't a near-duplicate
of an existing one before it lands in vocab.

**Why this matters**: bootstrap uses `synthesize_via_clustering` (BERTopic
style) producing 49 specific tags. `propose_new` is single-shot LLM with
the existing vocab as context. The architectures differ, so naming style
can drift across runs. Over many ingest cycles, near-duplicates accumulate.

**Compounding gap**: the maintenance agent's `diagnose` is signal-based
(probes low-usage or high-cooccur tags). A propose_new-introduced
near-duplicate that lands with moderate usage and zero co-occurrence with
its sibling will NEVER be probed and will silently rot the vocab.

**Validated this gap**: 2026-05-13 isolated ingest sim added
`rate_limiting_contract` (23 records) — manual `judge_pair` against the
closest existing tag `defensive_integration` (8 records) returned
`keep_distinct`, BUT they have 0 co-occurrence so agent's signal-based
diagnose never probed them. This time the new tag was legitimately
distinct, but the structural blind spot remains.

**Fix sketch** (~half day):

```python
# inside propose_new_fn, after synthesize:
from ..similarity import _embed, cosine
vocab_embeds = {t["name"]: _embed(t["name"] + ": " + t["definition"])
                for t in vocab}
filtered = []
for cand in synthesized:
    cv = _embed(cand.name + ": " + cand.definition)
    max_sim, closest = max(
        ((cosine(cv, v), name) for name, v in vocab_embeds.items()),
        default=(0, None),
    )
    if max_sim < 0.75:
        filtered.append(cand)
        continue
    # high similarity → LLM binary check
    if llm_judge_truly_new(cand, vocab_lookup(closest)):
        filtered.append(cand)
    else:
        logger.event("propose_new.rejected_as_duplicate",
                     candidate=cand.name, closest=closest, similarity=max_sim)
return filtered
```

**Defer trigger** (when to actually implement):

1. First time a propose_new-introduced tag is observed to cooccur > 5
   with an existing tag — that's empirical proof of redundancy entering
   via ingest.
2. Or vocab crosses 100 tags (entering risk window for accumulated drift).
3. Or audit shows propose_new tag naming visibly diverging from cluster
   bootstrap naming style.

Until one of these triggers, the soft prompt guard plus low ingest
frequency keep risk acceptable.

---

## 12. LLM-as-judge: aggregate metrics under-detect small-fraction bad merges

**Path**: `judge.py` — `llm_judge`

**Status**: known limitation, surfaced during Phase B Scenario C testing
(2026-05-13).

When a merge affects a small fraction of records (e.g. 38/696 = 5.5%), the
aggregate 5-dim health metrics barely move. Scenario C measured: coherence
delta -0.002, distinctness +0.001 — well below the judge prompt's per-action
suspicious thresholds (-0.02 to -0.05). The judge therefore commits
borderline-bad merges that should be rolled back.

**Concrete failure mode**:
- Synthetic merge `protocol_fidelity` (usage 38) → `failure_observability`
  (usage 16). Semantically distinct (protocol contracts vs observability
  gaps).
- Aggregate metrics barely moved; judge committed with high confidence
  under neutral reviewer reasoning.
- The missing signal: `failure_observability.mean_fit` would drop from
  ~0.49 to ~0.35 — a clear focal regression invisible in aggregates.

**Why this is currently acceptable**:

1. **Probability low** — `find_similar_pairs(threshold=0.80)` in
   `diagnose.py` prefilters merge candidates. Borderline-bad pairs
   (def similarity 0.65–0.79 with surface-keyword confusion) are
   long-tail.
2. **Recovery chain exists** — next iter's `forced_fit_candidates`
   surfaces the dragged-down tag; diagnose probes `inspect_outliers`;
   Phase C's `propose_refine` (planned) repairs by pruning the bad
   records and re-tuning the tag definition.
3. **Not on hot path** — maintenance is batch / manual trigger;
   downstream consumers don't see the intermediate bad state in real
   time.

**Fix sketch** (when triggered):

```python
# helper in health.py
def compute_tag_mean_fit(vocab, assignments, db_path, tag_names) -> dict[str, float]:
    """For each requested tag, mean cosine(record_emb, tag_def_emb)."""

# judge prompt extension: render per-affected-tag before/after fit
# e.g. "failure_observability mean_fit: 0.486 → 0.350 (−0.136)"
# Affected tags come from proposals (keep_tag for merge, the tag itself
# for deprecate). Runner closure captures db_path and passes fits to
# llm_judge.
```

Approximately 40–60 lines: health.py helper, judge prompt expansion,
graph node wiring, runner closure update.

**Defer trigger**:

- After ≥10 real maintenance runs, audit judge verdict accuracy. If
  false-commit rate on bad merges is > 10%, implement per-tag focal
  signals.
- Until then: Phase C `propose_refine` is the recovery path.

**Regression test**: `scripts/test_phase_b_judge_scenarios.py` (untracked)
currently fails with `verdict=commit`. Becomes a passing assertion once
focal signals land.

---

## Stability of LLM-driven steps — observed run-to-run variability

3 ingest iterations on identical fake corpus + identical seed (insofar as
LLM allows):

| Metric | iter 1 | iter 2 | iter 3 |
|--------|--------|--------|--------|
| Missing / 8 ingest | 6 | 7 | 6 |
| New tag name | rate_limiting_design | rate_limiting_architecture | rate_limiting_design |
| Records absorbed | 6 | 7 | 6 |
| Expand candidates | 3 | 3 | 3 |
| Expand yes | 0 | 0 | 0 |

**Conclusion**: behavior is consistent in structure (always propose_new
triggers, vocab grows by 1) but variable in naming (2 different new-tag
names across 3 runs). Downstream uses should not assume tag names are
deterministic across re-runs.

---

## Things NOT on this list (already addressed)

- Distill silent failure: fixed in `e375fe3` (retry + placeholder)
- TagDefinition theme_indices dropped: fixed in `e63edde`
- Decide prompt blocked-action retry loop: fixed in `f4fa519`
- propose_deprecate over-eager: fixed in `c20f24f` (focus + partial apply)
- build_diagnostics double-implementation: fixed in `f92f493`
- Broad ingest exception swallow: fixed in `0609295`
- Records fetcher full-table scan: fixed in `e28ade1`
- Measure_fn signature inconsistency: fixed in `352042a`
