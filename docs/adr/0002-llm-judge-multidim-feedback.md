# ADR 0002 — Maintenance agent feedback loop: 5-dim metrics + LLM-as-judge + refine action

- **Date**: 2026-05-13
- **Status**: **Partially Superseded by ADR-0003 (2026-05-13)** — Phase A (5-dim HealthMetrics + iter_deltas reducer) and Phase C (`propose_refine` action + invariants) are RETAINED. Phase B (LLM-as-judge) and Phase D (convergence rule + HITL) are SUPERSEDED by the plan-and-execute redesign in ADR-0003 (`planner.build_work_queue` + `sanity_check_metrics`). See ADR-0003 for the rationale and what carried over.
- **Component**: `vocab_maintenance.{health, judge, propose.refine, graphs.agent}` — the maintenance agent's gate, diagnose memory, and repair vocabulary

## Context

The maintenance agent's job is to keep an already-bootstrapped tag vocabulary healthy over time. Pre-refactor, three structural gaps surfaced during the 2026-05-13 design session and were validated against `outputs/network.json` (49 tags, 696 records, hit_rate ~0.98):

1. **Single-metric reward hacking**. The post-apply regression gate was a `hit_rate` threshold (rollback if `hit_rate_after < hit_rate_before − 0.03`). This rewarded force-fit (LLM tagger assigning records to weak-fit tags inflates coverage) and punished legitimate cleanup (deleting false-positive attachments reduces hit_rate). Goodhart's Law: optimizing the single visible metric incentivized the wrong behavior.

2. **No record-level repair tool**. Three actions existed: `propose_new`, `propose_merge`, `propose_deprecate`. All operate at vocab granularity. When the (then-new) `compute_fit_signals` surfaced `forced_fit_candidates` — tags whose attached records sit far from the tag definition on average, indicating expand_coverage false positives — the agent had no tool to fix the specific records. The LLM literally responded "training problem, not vocab design flaw" and quit. The signal was visible; the action vocabulary was incomplete.

3. **Cross-iter amnesia**. `state["history"]` was appended each iter but never read into any prompt. The reviewer LLM re-judged identical signals each iter with no trajectory awareness. Detecting convergence was impossible; an action that just got rolled back could be picked again on the next iter (only `blocked_actions` saved us from infinite loops).

### Alternatives considered

- **ε-constraint MOO** (Pareto-style multi-objective optimization with explicit per-action target deltas): scrapped. Threshold calibration at <200 tags with no labeled data isn't feasible; thresholds become hand-tuned guesses that don't generalize.
- **Composite weighted score** (one number combining the 5 dims): scrapped for the same reason as #1 above — collapses orthogonal axes, recreates Goodhart vulnerability one level up.
- **Stop after K iters with no commits**: weaker than dim-aware convergence; would mis-fire on iters that legitimately rolled back.

## Decision

**Phase A — multi-dim health metrics + cross-iter memory** (`health.py`, state reducers, `diagnose.PLAN_USER`):

- Replace the single visible health number with five orthogonal dimensions, each ∈ [0, 1] with "higher is healthier":
  - `coverage` = 1 − missing_rate (pure aggregate; no embedding)
  - `coherence` = weighted mean over (record, selected_tag) pairs of `cosine(record_emb, tag_def_emb)`
  - `distinctness` = 1 − mean pairwise cosine of tag definitions
  - `granularity` = normalized Shannon entropy of tag usage distribution
  - `multi_axis` = fraction of non-missing records with ≥ 2 selected tags
- Embedding pass shared with `compute_fit_signals` via `probes._compute_record_tag_embeddings`; per-iter cost is one concurrent pass.
- `state.metrics_history` and `state.iter_deltas` use `Annotated[list, _append_last_n(5)]` reducers — bounded accumulators that keep the last 5 entries.
- `diagnose.PLAN_USER` renders the last 3 `iter_deltas` as a trajectory table from iter 2 onwards; the reviewer LLM now sees what was tried and how the dims moved.

**Phase B — LLM-as-judge replaces threshold gate** (`judge.py`, `graphs.agent.route_judge`):

- A dedicated LLM session runs after each apply, reading `(action, proposals_summary, reviewer_reasoning, before_metrics, after_metrics, vocab_size_delta)` and producing a structured `JudgeVerdict{verdict ∈ {commit, rollback, unsure}, primary_concern, reasoning, confidence ∈ {high, medium, low}}`.
- Conservative defaults: `unsure → commit` (don't reject changes on ambiguity); judge LLM failure → commit (don't reject changes on transient API errors). Per-action calibration tables in `JUDGE_SYSTEM` codify "what a healthy delta looks like for this action": merge expects small coverage drop + coherence rise + distinctness rise; deprecate expects bigger coverage drop; refine expects coherence rise above all else.
- `route_judge` replaces `route_regression`; `hit_rate_regression_threshold` is removed from state and `Agent.__init__`.

**Phase C — `propose_refine` action closes the see-but-can't-fix loop** (`apply.py`, `propose/refine.py`):

- `RefineTagProposal{tag, new_definition, prune_record_ids ≤ 10}` is a new proposal type. `_apply_refine` enforces structural invariants (tag exists, prune cap = 10, prune target carries the tag, no orphan after prune). `validate_refine_proposal` enforces the cosine-distance invariant (each prune target's cosine to the new definition must be < 0.7; otherwise the record still fits the refined tag and prune is invalid).
- `propose_refine_fn` reads diagnose's `focus` (which names a forced_fit candidate), consumes `inspect_outliers`, **pre-filters prune candidates to records that carry ≥ 2 tags** so the LLM cannot propose orphan-inducing prunes, then asks for a refined def + prune list.
- `diagnose.PLAN_USER` is updated to route forced_fit signals to `propose_refine`; `ALL_ACTIONS` and `_DECIDE_RULES` register the new action.

**Phase D — convergence + HITL** (`graphs.agent.is_converged`, `hitl_node`):

- `is_converged(iter_deltas, threshold=0.005)`: pure function — True iff the last 2 `iter_deltas` entries both show `max(abs(δ_dim))` < 0.005 across all 5 dims. `termination_check_node` checks this **before** `no_actions_remaining`, so a stable network exits with the informative `FinalStatus.CONVERGED` rather than the less-specific "ran out of actions" status.
- `hitl_node` calls `langgraph.types.interrupt()` whenever diagnose's `confidence == "low"`, surfacing the proposed action + uncertainty_reasons to the resumer. `Command(resume="approve")` continues; `Command(resume="reject")` (or any unrecognized value) forces `action="done"`. A `hitl_resolved` state flag prevents re-routing back into hitl.
- `Agent._invoke_with_batch_hitl` auto-approves in batch contexts (runner scripts) and emits an `agent.hitl.auto_approve` audit event so non-interactive runs don't silently swallow escalations.

Time-travel rollback (mentioned in the plan): **explicitly out of scope as code**. SqliteSaver already records per-node checkpoints; the capability exists. We deferred a walkthrough doc + helper to TECH_DEBT until empirical evidence motivates it. See "Consequences — Pending" below.

### Why LLM-as-judge over thresholds at this scale

LLM-as-judge (also called verifier-guided generation; cf. Anthropic Constitutional AI, OpenAI's o1-style reasoner-judge separation) is the right trade-off at <200 tags and zero labeled regression data. Threshold calibration assumes you have a labeled set of "good" and "bad" outcomes — we don't. The judge prompt encodes per-action heuristics ("coherence drop > 0.02 is suspicious for merge") as guard rails the LLM consults, but leaves the final synthesis to the model. At higher scale (>500 tags with thousands of labeled rollback decisions), a tuned threshold model would likely outperform; not our regime.

## Consequences

### Positive

- **The see→decide→fix loop is closed for the most common failure mode.** Before: forced_fit signals visible but no action; LLM quits. After: `forced_fit_candidates` → diagnose probes `inspect_outliers` → picks `propose_refine` with focus on that tag → LLM sharpens definition + prunes the misfit records → judge commits → next iter's `compute_fit_signals` reflects the improvement. Validated on `state_schema_design`: mean_fit rose **0.447 → 0.496** in one iter, tag dropped out of `forced_fit_candidates` afterward.
- **Reward hacking surface narrowed.** Force-fitting now visibly degrades coherence; deleting false-positive attachments visibly improves it. The judge's per-action calibration table separates "expected effects" from "regressions", so legitimate consolidation (small coverage drop) and legitimate cleanup (small multi_axis drop) are no longer punished.
- **Reviewer LLM has trajectory awareness.** The `Recent iters` table in `PLAN_USER` lets diagnose see what failed last iter and weight options accordingly. Particularly important for `blocked_actions` — the LLM now also sees *why* an action got blocked (rolled_back vs blocked_empty vs apply_error) instead of just "blocked".
- **Soft exits are explicit.** `FinalStatus.CONVERGED` is a positive signal; `no_actions_remaining` is now a fallback. Operators can distinguish "agent is happy" from "agent ran out of options".
- **HITL escape exists.** When the agent literally doesn't know (confidence=low), there's a structured pause point with the relevant context surfaced. Batch mode auto-approves with an audit trail, so noninteractive runs don't lose visibility into the escalations.
- **Independent commits per phase.** A/B were bundled (B builds on A and the shared-files surgery wasn't worth the time); C and D shipped as independent commits. Rollback granularity is at most one phase.

### Negative / known limits

- **Judge under-detects small-fraction bad merges.** Captured as TECH_DEBT #12. A merge that affects ≤ ~10% of records (e.g., 38/696 = 5.5% in the Scenario C synthetic test) moves aggregate metrics by less than the judge prompt's thresholds (coherence delta -0.002 vs threshold -0.02). With neutral reviewer reasoning, the judge committed an obviously-wrong protocol_fidelity → failure_observability merge. The recovery path is real (forced_fit surfaces post-merge → `propose_refine` cleans up next cycle), but the *latency* of recovery is one maintenance cycle, not immediate. A focal per-tag mean_fit signal in the judge prompt would tighten this — deferred per plan's "judge bias audit after ≥10 real runs" gate.
- **Long-tail paths are unit-tested, not e2e tested.** `propose_merge`, `propose_deprecate`, and judge=rollback **were never exercised in real-LLM agent runs** during the refactor because `outputs/network.json` is too healthy (hit_rate 0.98) for the diagnose LLM to pick those actions. They have offline unit coverage (Phase C invariants + the synthetic judge Scenario C) but their end-to-end behavior is unobserved on production data. If network health degrades (e.g., post-ingest with new domains), these paths will become hot — expect to revisit observability and possibly judge calibration.
- **HITL escalation has never fired in a real LLM run.** Modern instruction-tuned LLMs tend to over-confident outputs; `confidence=low` is rare. Scenario E is verified via stubbed diagnose only.
- **Convergence rule may fire too eagerly on action-blocked sequences.** Two consecutive `blocked` iters produce zero deltas and trigger convergence at iter 3. In Phase D's Scenario D smoke this was *correct* (one productive refine + one stagnant iter), but on a partially-blocked network where the agent is "stuck choosing the same bad action twice", convergence would mask the real problem. Mitigation: `blocked_actions` accumulates across iters, so after 2 blocks of the same action the agent picks something else anyway; convergence is more a "we agree" signal than a "we're stuck" one. Re-evaluate if real runs show false-positive convergence.
- **iter_deltas reducer keeps only the last 5 entries.** A long run rolls older deltas off the visible window. PLAN history shows last 3 by default; reducer keeps 5 for safety margin. If we ever want longer trajectory analysis (e.g., "this dim has been drifting down for 8 iters"), the reducer cap needs lifting and PLAN rendering needs paging.
- **Two refactors deferred:**
  - **#1 expand_coverage false positives** are no longer a standalone TECH_DEBT — they now have an automatic recovery path via `propose_refine`. Status updated, but the *upstream rate* of false positives is unchanged; refine is downstream cleanup.
  - **#11 propose_new near-duplicate guard** is unrelated to this refactor (lives in ingest, not maintenance). Not closed by Phase A/B/C/D.

### Neutral

- `route_regression` and `hit_rate_regression_threshold` are deleted, not deprecated. The `hit_rate` value itself is still computed and emitted in events (it's a useful metric, just not a routing condition).
- `propose_new` remains disabled in the maintenance agent. Vocab growth still belongs to `ingest_batch`. The maintenance agent is consolidation/repair only.
- All test files (`scripts/test_phase_*.py`) are intentionally untracked per the repo's scripts/ policy. The plan tracking section in `docs/plans/2026-05-maintain-agent-feedback-refactor.md` is the authoritative pointer to test scripts and coverage; next session should read it before assuming "no tests".

## Validation

Real-LLM end-to-end smoke runs on `outputs/network.json` (49 tags, 696 records, copied to `/tmp/network_phase_*_smoke.json` so production state is untouched):

| Phase | Acceptance scenario | Result | Evidence (jsonl) |
|---|---|---|---|
| A | `runs/*.jsonl` contains 5-dim metrics event | ✓ iter 0 `metrics.measured` with all five dims | `2026-05-13T06-33-23_agent_e2e.jsonl` |
| B Scenario A | max_iter=5 e2e clean | ✓ status COMPLETED at iter 1 (LLM picked done on healthy network) | `2026-05-13T06-47-29_agent_e2e.jsonl` |
| B Scenario C | Forced bad merge → judge says rollback | ✓ rollback with high confidence under "forced test" reviewer reasoning; ✗ commit under neutral reasoning (TECH_DEBT #12) | direct llm_judge call |
| C Scenario B | LLM picks propose_refine on forced_fit candidate, judge commits, mean_fit improves | ✓ state_schema_design refined, 1 record pruned, judge commit (high confidence), focal mean_fit 0.447 → 0.496 | `2026-05-13T07-10-41_agent_e2e.jsonl` |
| C Scenario C2 | Invariant tests | ✓ 15/15 unit tests | `scripts/test_phase_c_refine_invariants.py` |
| D Scenario D | converged before iter 8 on max_iter=5 | ✓ converged at iter 2 (iter 1 refine + iter 2 blocked_empty, both deltas < 0.005) | `2026-05-13T07-59-10_agent_e2e.jsonl` |
| D Scenario E | interrupt/approve/reject/unknown via graph.invoke + Command | ✓ 18/18 unit tests | `scripts/test_phase_d_termination_and_hitl.py` |

Phase A offline: 39/39 unit tests (5 dim functions × ≥3 cases + reducer + history rendering + PLAN template). Phase C offline: 15/15 invariant tests. Phase D offline: 18/18 (is_converged 8 cases + non-convergence + 3 HITL flows). Total: **72 offline unit tests + 4 real-LLM e2e smoke runs**, all green.

### Pending observations

- **judge=rollback in real-LLM agent loop**: never observed. Synthetic Scenario C with "forced test" reasoning is the only rollback evidence; neutral reasoning committed. Re-evaluate after ≥ 10 real maintenance cycles.
- **propose_merge / propose_deprecate in agent loop with real LLM**: never picked during the refactor's smoke runs. Their interaction with the judge in real graph context is unverified.
- **HITL real interrupt in real LLM**: same; diagnose hasn't self-assessed confidence=low in any observed run.

## Follow-ups (not in scope for this ADR)

- Quantify judge accuracy after 10+ real maintenance runs. If false-commit rate on borderline-bad merges exceeds ~10%, implement per-tag focal `mean_fit` in the judge prompt (TECH_DEBT #12 sketch).
- Time-travel walkthrough doc: a 1-page guide showing `graph.get_state_history(config)` + `graph.update_state(...)` flow with worked examples. Defer until needed; SqliteSaver checkpoint capability already exists.
- Observability for long-tail paths: when network health degrades and propose_merge/deprecate become common, add dashboards / sampling to surface judge verdict distribution.
- Reducer cap of 5 may need lifting if longer-trajectory diagnose reasoning becomes valuable; PLAN rendering would need paging.

## References

- Plan: `docs/plans/2026-05-maintain-agent-feedback-refactor.md` (master plan + per-phase tracking).
- Phase A+B commit: `5e7ab7b` `feat(metrics+judge): 5-dim HealthMetrics + LLM-as-judge gate`.
- Phase C commit: `f1cbaa8` `feat(refine): add propose_refine`.
- Phase D commit: `d7c6b5f` `feat(termination+hitl): convergence early-stop + HITL escape`.
- Anthropic Constitutional AI — verifier-guided generation pattern at the source.
- LangGraph documentation for `interrupt()`, `Command`, and SqliteSaver checkpoint semantics.
- TECH_DEBT.md #12 — judge sensitivity gap; this refactor's most concrete known limitation.
