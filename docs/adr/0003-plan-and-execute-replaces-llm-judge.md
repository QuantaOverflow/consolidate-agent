# ADR 0003 — Plan-and-execute supersedes LLM diagnose + LLM-as-judge

- **Date**: 2026-05-13
- **Status**: Accepted
- **Supersedes (in part)**: ADR-0002 — specifically the `diagnose` LLM plan+decide step and the LLM-as-judge gate. The 5-dim HealthMetrics infrastructure introduced in ADR-0002 Phase A is retained and reused.
- **Component**: `vocab_maintenance.{planner, agent, graphs.agent}` — the maintenance agent's loop topology, decision substrate, and rollback gate.

## Context

After landing Phase A-D (ADR-0002), four production smokes on `outputs/network.json` exposed structural issues that prompt-level tuning could not fix:

1. **Convergence threshold collided with refine signal strength.** Phase D's `is_converged(threshold=0.005)` was meant as "things stopped moving". But a single-record refine on a 696-record corpus moves aggregate coherence by ~1/300 ≈ 0.003 — *below* the noise threshold by construction. Result: Phase D smoke (max_iter=5) exited at iter 2 with status=CONVERGED after one productive refine + one blocked_empty, leaving 4 other forced_fit candidates untouched. The same blind-spot as TECH_DEBT #12 (judge can't see small-fraction *bad* merges), just mirrored on the success side (convergence can't see small-fraction *good* refines).
2. **Per-iter LLM diagnose added overhead without strategic payoff.** Across all observed runs, the diagnose LLM either (a) picked the same target repeatedly with no awareness that the previous iter just improved it, or (b) declared `done` immediately because the network was "healthy enough". The `iter_deltas` history table in `PLAN_USER` (Phase A) was implemented to enable cross-iter strategy; in practice the LLM did not use it. Each iter paid ~2-3 LLM calls (plan + decide + judge) for a decision the surface signals already implied.
3. **LLM-as-judge underdetected the failure mode it was designed to catch** (TECH_DEBT #12). Validated empirically in Phase B Scenario C: a synthetic forced merge of `protocol_fidelity` (38 records) into `failure_observability` produced an aggregate coherence delta of -0.002, well below the prompt's -0.02 suspicion threshold. With neutral reviewer reasoning, the judge committed. The judge's marginal value over "trust propose + recover via next-cycle forced_fit" was small.
4. **`blocked_empty` permanently locked the action** for the rest of the run, so even with `max_iter=10`, one `propose_refine` returning `[]` (a legitimate "no useful prune here") prevented all subsequent refines on different tags. Combined with `max_targets=1`, a single agent invocation cleaned at most one tag.

The user's framing of the redesign: *"去掉 convergence 这种硬设置的分数，用最大轮数做硬上限，将 probe 到的问题作为一个处理队列或者清单，一直处理直到触到上限"* — explicitly identifying that the LLM-driven open agent pattern was overkill for a domain where the work is enumerable.

## Decision

**Adopt a plan-and-execute architecture for the maintenance agent**, replacing the LLM-driven per-iter decision loop:

1. **`planner.py:build_work_queue`** runs once at the start of each agent invocation. Pure rule-based scan of:
   - `compute_fit_signals.forced_fit_candidates` → `propose_refine` items
   - `find_similar_pairs(threshold=0.80)` → `propose_merge` items
   - `diagnostics.unused_tags` → `propose_deprecate` items

   Each surface signal becomes one `WorkItem(action, focus, reason, priority)`. Priority bands: refine (1xxx) < merge (2xxx) < deprecate (3xxx), spaced by 1000 so per-item severity (mean_fit / similarity / 0) cannot bleed across bands. Within a band, sort ascending by severity (worst signal first).

2. **Per-iter loop** pops the next WorkItem and executes: `propose_X(focus) → apply → post_apply_measure → sanity_check → commit/rollback → record_iter → next item`.

3. **`planner.py:sanity_check_metrics`** is a pure function that replaces `llm_judge`. Rolls back only when a *monitored* dim (coherence / distinctness / granularity / multi_axis — coverage explicitly excluded because merge/deprecate/refine all legitimately drop coverage) regresses by ≥ 10%. No LLM call. The threshold is intentionally coarse — anything subtler is left to next-cycle `forced_fit` recovery via Phase C's refine action.

4. **Termination conditions**:
   - `work_queue` exhausted → `FinalStatus.COMPLETED`
   - `iter >= max_iter` → `FinalStatus.MAX_ITER_EXHAUSTED`
   - exception during propose/apply → `FinalStatus.FATAL_ERROR`

   Removed: `FinalStatus.CONVERGED`, `FinalStatus.NO_ACTIONS_REMAINING`, all convergence and "all blocked" early-stop logic.

5. **Code deletions** (per "avoid backwards-compatibility hacks"):
   - `diagnose.py` (LLM plan+decide step)
   - `judge.py` (LLM-as-judge)
   - HITL node + `interrupt()` route + `_invoke_with_batch_hitl` wrapper
   - `is_converged`, convergence rule
   - `blocked_actions` state field, `decision` / `judge_*` / `hitl_resolved` per-iter scratch
   - `route_regression`, `route_judge`, `route_after_diagnose`

   ADR-0002's Phase A artifacts that are RETAINED:
   - `health.py` 5-dim HealthMetrics — planner consumes them via `compute_fit_signals`; sanity_check consumes them directly
   - `metrics_history` + `iter_deltas` reducer state (still useful for audit / future cross-iter analysis even though planner doesn't read them)
   - Phase C's `propose_refine` action and `RefineTagProposal` invariants

### Why plan-and-execute over open agent at this scale

The maintenance task has three properties that make planner-executor optimal:
- **Issues are enumerable from signals** (forced_fit, similar_pairs, unused tags) — no need for LLM judgment to *find* problems.
- **Per-action effects are well-understood** — coverage drops on merge are normal, coherence rises on refine are expected. Per-action calibration tables (Phase B's `JUDGE_SYSTEM`) encode the same knowledge a simple sanity threshold does, just with more LLM overhead.
- **Recovery is cycle-to-cycle, not in-iter** — any LLM mistake (bad propose) surfaces as a new forced_fit in the next maintenance cycle, where `propose_refine` cleans it up. The LLM-as-judge in-iter rollback was paying for a defense that wasn't load-bearing.

This is the same insight that makes BERTopic + LLM-naming dominate single-shot LLM taxonomy induction (ADR-0001): use deterministic structure where it works, and reserve LLM for local naming.

## Consequences

### Positive

- **One invocation cleans the entire work queue.** Phase E smoke on `outputs/network.json` (max_iter=10): planner emitted 5 forced_fit items, agent processed all 5 in 5 iters, status=COMPLETED, queue_remaining=0. Two refines committed (explicit_contract, schema_coordination), three rejected by LLM as "no_change needed". Phase D's same data + same goal exited at iter 2 with 4 items unprocessed.
- **LLM call count drops ~7× per run.** Phase D max_iter=5 smoke: ~15 LLM calls (3+ per iter × 5 iters). Phase E max_iter=10 smoke: 2 LLM calls total (only propose_refine fired on the 2 tags that actually got refined; the 3 rejected ones still cost 1 LLM call each, so 5 propose calls; planner has 0 LLM calls). Net: 5 vs ~15-20, with more work done. Elapsed wall time per iter drops correspondingly.
- **Predictable scope.** At iter 0, the user knows "this run has N issues to attempt" — the audit log emits `planner.done queue_size=N items=[...]` exactly once. Phase A-D had no equivalent — work scope was emergent from LLM decisions.
- **No more convergence calibration.** The pathological collision between "single-record refine signal" and "convergence noise floor" is structurally impossible: planner produces N items, loop pops until queue empty or max_iter. No threshold to tune.
- **Phase B TECH_DEBT #12 is obsolete.** With no LLM-as-judge, the per-action sensitivity calibration problem disappears. The new sanity_check is coarse on purpose — it catches only catastrophic failures (≥10% single-dim drop), and accepts that subtle bad commits are recovered via next-cycle forced_fit.

### Negative / known limits

- **No adaptive strategy within a run.** Phase A-D's design allowed the LLM to look at iter 3's deltas and decide "let me try merge instead of refine next". Phase E processes the static queue in fixed priority order. If a refine in iter 1 changes the network in a way that makes iter 4's queued item no-longer-relevant, the queue still attempts it. Mitigation: `propose_refine_fn` already self-rejects via the `no_change` path when the tag looks healthy; this naturally handles "queue item became stale".
- **Sanity threshold is fixed at 10%.** A merge that legitimately drops distinctness 8% (large but not catastrophic) passes; a refine that accidentally drops coherence 12% fails. Both are coarse defaults. Per-action thresholds (like Phase B's calibration table) are NOT in scope — if real runs show false positives or negatives on the 10% line, revisit.
- **HITL escape valve removed.** With no LLM judgment loop, there's no `confidence=low` signal to escalate on. If a maintenance run goes wrong, the operator must rely on `outputs/network.json.bak` filesystem rollback (TECH_DEBT #6 single-level) or SqliteSaver time-travel (TECH_DEBT #14 — capability exists, no documented usage). The HITL machinery from Phase D was never observed firing in real LLM runs anyway, so this loss is theoretical.
- **iter_deltas PLAN history table is no longer rendered into any LLM prompt.** Phase A built this to give diagnose cross-iter memory. With diagnose deleted, it's audit-only data. The reducer infrastructure stays (still useful for downstream consumers / `run_agent_e2e.py` history display).
- **`max_targets=1` on `propose_refine_fn` is unchanged.** Each WorkItem triggers one tag's refine; cleaning 5 forced_fit candidates means 5 iters minimum. Theoretically `max_targets > 1` could let one iter batch-refine multiple tags, but the sanity_check would see a combined delta — harder to attribute regressions, same trade-off as before.
- **First-time-run lock-in: planner sees signals once.** If a refine in iter 1 *creates* a new forced_fit (e.g., the refined tag's neighbor now looks redundant), the new signal won't be queued in this run — it surfaces next maintenance cycle. For most cleanup work this is fine; for cascading repairs it adds latency.

### Neutral

- The 5-dim `HealthMetrics` infrastructure from Phase A is retained as-is. The planner reads `compute_fit_signals` for forced_fit, sanity_check reads the snapshot for catastrophe detection. Everything else (metrics_history reducer, observability events) keeps working.
- `propose_refine` Phase C work is retained unchanged. Invariants (orphan, prune cap, cosine threshold) still fire at apply time.
- `find_similar_pairs` was previously used inside `diagnose.py` to flag suspected merges to the reviewer LLM. It's now consumed directly by the planner with the same threshold (0.80). Behavior identical, one fewer LLM step in between.
- The runner script's `Agent` constructor signature changed (added `health_fn` + `db_path` required, removed `diagnose_fn` + `judge_fn` + `hit_rate_regression_threshold`). External callers must update.

## Validation

| Test | Result | Evidence |
|---|---|---|
| Phase E offline (22/22 unit tests) | ✓ | `scripts/test_phase_e_planner_sanity.py` (sanity edge cases + planner priority/severity + 3 graph-wiring smokes via MemorySaver) |
| Real-LLM e2e on `outputs/network.json` copy, max_iter=10 | ✓ status=COMPLETED, 5 items processed, 2 refines committed, queue_remaining=0 | `outputs/runs/2026-05-13T08-39-09_agent_e2e.jsonl` |
| Compare iter count + LLM calls vs Phase D | Phase D max_iter=5 → exited iter 2 with 4 forced_fit unprocessed, ~15 LLM calls. Phase E max_iter=10 → processed 5/5, 5 LLM calls (planner + 5 propose_refine, no diagnose, no judge) | same jsonl + Phase D's `2026-05-13T07-59-10_agent_e2e.jsonl` |

Phase A unit tests (`scripts/test_phase_a_health.py`) and Phase D unit tests (`scripts/test_phase_d_termination_and_hitl.py`) reference modules that no longer exist (`diagnose.PLAN_USER`, `is_converged`, hitl node). They are untracked and intentionally not updated — superseded by `test_phase_e_planner_sanity.py`. Phase C invariant tests (`scripts/test_phase_c_refine_invariants.py`) still pass (15/15) — refine invariants are independent of the surrounding loop.

## Follow-ups (not in scope for this ADR)

- Decide whether `max_targets` in `propose_refine_fn` should bump to 2-3 to amortize per-iter overhead. Currently 1. Re-evaluate after a few real runs show how often queue items > 5.
- Time-travel rollback walkthrough (TECH_DEBT #14) becomes more valuable now that HITL is removed — incident recovery is purely filesystem + checkpoint based.
- If sanity_check is observed to mis-fire (either false-positive rollbacks blocking legitimate consolidation, or false-negative commits letting bad changes through), consider per-action thresholds — but only after empirical signal, not preemptively.
- Long-tail paths (real-LLM `propose_merge` and `propose_deprecate` in the agent loop) remain unobserved — same as TECH_DEBT #13 noted for Phase A-D. Phase E does not change this; queue can include those items, but `outputs/network.json` has no high-similarity pairs or unused tags, so the smoke didn't exercise them.

## References

- ADR-0002 — design this supersedes in part (Phase A retained; Phase B + Phase D's convergence/HITL removed).
- Plan: `docs/plans/2026-05-maintain-agent-feedback-refactor.md` Phase E section.
- Plan-and-execute pattern: LangChain's blog post on the architecture (general AI engineering context, not a paper citation).
- TECH_DEBT.md #12 (judge sensitivity gap) — now obsolete after Phase E.
- User framing of the redesign: 2026-05-13 conversation excerpt: *"去掉 convergence 这种硬设置的分数 ... 一直处理直到触到上限"*.
