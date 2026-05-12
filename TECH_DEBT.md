# Technical Debt & Known Limitations

Snapshot of known issues in the `vocab_maintenance` subsystem. Read before
running on real data.

Last updated: 2026-05-12

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

## 2. `propose_merge_fn` has same focus bug `propose_deprecate` had (now fixed)

**Path**: `propose/merge.py` — `propose_merge_fn(vocab, assignments, focus, ...)`

**Status**: NOT FIXED.

Commit `c20f24f` fixed `propose_deprecate_fn` to respect the agent's
`focus` parameter (only evaluate tags named in focus). The analogous code
path in `propose_merge_fn` was not touched — it still scans all
high-cooccurrence pairs regardless of focus.

**Risk**: if agent's diagnose returns `action_focus="target pair a + b"` and
selects propose_merge, `propose_merge_fn` may propose merges outside the
focused pair.

**Why we haven't hit it**: in fake-corpus testing, the agent's
diagnose always picked `done` or `propose_deprecate` — never propose_merge.

**Fix** (~20 lines): add `_extract_focus_tags()` parallel to deprecate's;
if focus names a tag, restrict cooccurrence-pair scan to pairs containing
that tag.

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

## 6. No backup mechanism for `network.json`

**Path**: `network.py` — `Network.save()`

**Status**: atomic write via tmp + rename, but each save overwrites the
previous. If a bad run corrupts the network state, there's no auto-rollback.

**Workaround**: caller scripts should `cp` before save.

**Fix**: `save(path, keep_backup=True)` option that moves previous to
`{path}.bak` before writing the new file.

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

## 9. Concurrent runs are unsafe

**Path**: any code that loads + saves the same `network.json`

**Status**: no file lock. Two simultaneous `bootstrap` or `ingest_batch`
calls on the same network path: the second `save()` overwrites the first's
result.

**Workaround**: ensure single-user / single-process discipline.

**Fix**: `fcntl` file lock on `network.json` before load + during save.
Or migrate to a proper DB (SQLite).

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
