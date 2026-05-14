# ADR-0006: Phase F — LLM as strategic reasoner with structured tool use

**Status**: Exploratory (parallel track; does not supersede Phase E yet)
**Date**: 2026-05-14
**Builds on**: ADR-0003 (Phase E plan-and-execute), ADR-0005 (metric proxy problem)

## Context

Phase E (ADR-0003) replaced the LLM-driven diagnose+judge agent (Phase A-D) with
a rule-based planner + LLM-only-for-propose architecture. This solved real
problems (LLM judge unreliability, no learning from history) but introduced a
**fundamental architectural limitation**:

> The entire execution direction depends on cosine-derived proxy metrics. The
> agent improves what metrics measure, not necessarily what users need.

Empirical evidence (this session):

1. **silhouette and ground-truth diverged** under multi-tag setting (ADR-0005):
   v3_p1 P1 prompt change improved golden match (46% → 51%) but degraded
   silhouette (56% → 31% healthy). The metrics agent uses to plan and validate
   point in opposite direction from ground truth.

2. **P2 agent maintenance** on v3_p1: applied 5 propose_refine operations
   targeting silhouette-detected forced-fit candidates. Result: Matter exact
   match dropped 51.2% → 46.2%. The agent successfully optimized its proxy
   metrics while regressing on the metric that actually matters.

3. **No grounded validation in the loop**: sanity_check uses cosine-derived
   HealthMetrics; rollback fires on 10% regression in those proxies. It cannot
   distinguish "embedding-space stable but semantically broken" from "actually
   improved."

The deeper issue is **Goodhart's Law**: when a measure becomes the optimization
target, it ceases to be a good measure. Phase E architecturally bakes this in.

## Decision

Begin **Phase F** as an exploratory parallel track. Phase E remains the
production architecture; Phase F is a sandboxed experiment in
`vocab_maintenance/agent_brain/` that does not modify Phase E behavior.

Phase F architecture:

```
┌──────────────────────────────────────────────────────────────┐
│ LLM Brain (strategic reasoner)                               │
│                                                              │
│  Per round:                                                  │
│    1. read AgentMemory (structured 4-layer state)            │
│    2. choose: call_tool OR decide                            │
│    3. tools return structured results → record_tool_call     │
│    4. when decide: output AgentDecision (action + cert +     │
│       evidence + reversibility + expected_deltas)            │
│    5. gate_decision: deterministic 7-gate filter             │
│       → AUTO / REVIEW / BLOCK                                │
│    6. AUTO: apply_proposal; REVIEW: log + skip (no HITL yet) │
│    7. archive_round to memory (procedural summary)           │
│    8. log DecisionOutcome (for post-hoc calibration)         │
└──────────────────────────────────────────────────────────────┘
```

Key design principles:

1. **Metrics are inputs to LLM reasoning, not direct decision drivers**.
   The LLM has tools to inspect specific tags, sample golden truth, and
   preview proposed changes before committing.

2. **Structured decisions over raw LLM output**. AgentDecision schema forces
   LLM to cite supporting_observations (concrete tool results), declare
   reversibility, and estimate impact. Pydantic validation rejects unstructured
   responses.

3. **Deterministic gates as safety net**. LLM's self-reported confidence is
   never trusted alone — gate logic checks evidence quantity, preview/golden
   verification, recent flip-flop history, and impact scope.

4. **Memory is engineered, not raw**. AgentMemory.render_for_llm produces
   four-section structured context (state / facts / history / current round)
   with critical-pattern detection that surfaces course-correction hints at
   the top of context.

## What Phase F adds

Built in this session (commit 408636f):

| Component | Purpose | Lines |
|---|---|---|
| `agent_brain/memory.py` | 4-layer AgentMemory + pattern detector | 175 |
| `agent_brain/decision.py` | AgentDecision schema + 7-gate logic | 90 |
| `agent_brain/outcome_log.py` | Append-only DecisionOutcome jsonl | 60 |
| `agent_brain/tools.py` | 7 LLM-callable tools + per-round dedup cache | 258 |
| `agent_brain/brain.py` | run_brain_loop with budget guards | 310 |
| `scripts/run_brain_spike.py` | Entrypoint with trace + summary output | 137 |
| 28 unit tests | Memory FIFO/render, gates, cache, patterns | 380 |

## Empirical results (2 spike runs)

### Spike v1 (no context engineering)

5 rounds, 30 LLM calls, $0.015 estimated cost.

```
R1-R5: refine filesystem_path (cert=high, preview=False) → gate=REVIEW
       (high_cert_no_preview), not applied
```

Findings:
- **Architecture mechanics work**: memory, tools, gates all execute correctly.
- **LLM reasoning quality is high**: sophisticated, cites specific record_ids,
  coherence values, golden disagreements. Quality exceeds Phase E propose
  outputs because LLM can actively inspect.
- **Behavior failure: no preview, no target switching, no certainty
  calibration**. LLM jumped straight to `refine` at `certainty=high` without
  calling `propose_refine_preview` first; gates blocked all 5 rounds; LLM
  did not update strategy in response to repeated blocks.
- **Cost is lower than estimated** (~$0.015 vs predicted $1.5/run). qwen-plus
  pricing makes deep exploration affordable.

### Spike v2 (with context engineering)

Same architecture; only changed `memory.render_for_llm` (added critical
pattern detector), tool dispatch (added per-round dedup cache), system
prompt (added worked example + certainty calibration table).

5 rounds, 27 LLM calls, **3 commits**.

```
R1: refine filesystem_path (high, no golden) → REVIEW
    (high_impact_no_golden_check + high_cert_no_preview)
R2: refine filesystem_path (high, preview+golden) → AUTO → APPLIED
R3: refine http_api (high, preview+golden) → AUTO → APPLIED  ← switched target
R4: refine http_api → AUTO → APPLIED
R5: refine filesystem_path → REVIEW (recent_flipflop, gate worked)
```

Findings:
- **Context engineering, not architecture, was the bottleneck**. Same agent
  with engineered prompts/memory rendering achieves real maintenance.
- **Pattern detector drove target switching**: R3's switch from
  filesystem_path to http_api was directly attributable to the ⚠ CRITICAL
  PATTERNS alert at top of context.
- **Worked example drove preview-before-decide**: R1-R4 all called
  `propose_refine_preview` (none did in v1).
- **Gates correctly caught remaining errors**: R5's flip-flop attempt was
  correctly blocked.
- **Confidence calibration still poor**: all 5 decisions claimed `cert=high`
  including the 2 blocked ones. LLM cannot self-correct certainty downward,
  even when context contains evidence of prior blocks. Gates remain the
  only reliable safeguard.

## What remains uncertain

1. **Calibration over longer runs**: 5-round spike doesn't reveal how the
   agent behaves over 15-30 rounds. Specifically, will pattern detector
   continue to drive useful behavior change, or will LLM habituate to the
   alerts?

2. **Whether Phase F can replace Phase E**: Phase F is more autonomous but
   more expensive (~5x LLM calls). Whether it converges to better-quality
   networks than Phase E (which we just showed regressed quality during
   P2 maintenance) needs longer experimentation across multiple maintenance
   cycles.

3. **HITL escalation UX**: Spike runs in `REVIEW = log + skip` mode.
   Production would need a UI for human to review escalated decisions,
   approve/reject/modify. Not designed yet.

4. **Outcome tracking → calibration loop**: DecisionOutcome jsonl is
   implemented but no calibration analysis pipeline exists yet. After ~50-100
   decisions accumulate across runs, build a calibration report:
   "claimed confidence X% → actual benefit rate Y%". Use this to tune gate
   thresholds.

5. **Cross-run decision journal**: Within one run, memory carries pattern
   detection. Across runs, agent has no memory of "we already split this tag
   last week and it didn't help." This causes repeated cycle risk that
   Phase E's static work_queue accidentally avoided. Need persistent
   decision history that influences planning in subsequent runs.

## Consequences

### Positive

- **Two architectures coexist**: Phase E for known-good operations
  (propose_refine via planner queue), Phase F for exploratory diagnostics.
  No regression risk to existing pipelines.

- **Scaffolding-pattern as first-class concept**: critical pattern detector
  in `memory.py` is a reusable mechanism. Future memory engineering can
  layer on detectors without changing schemas.

- **Confidence-without-calibration handled honestly**: gates encode the
  reality that LLM-reported confidence is unreliable, while still letting
  the LLM produce confidence as a signal for prioritization.

- **Spike-driven validation**: by running spikes before claiming Phase F
  "works," we documented the v1 failure (rebuts "LLM with tools is
  enough") and the v2 success (validates "context engineering is the
  fix"). This is the kind of evidence Phase A-D lacked.

### Negative / Risk

- **Phase A-D regression risk**: re-introducing LLM into the decision loop
  has historical precedent of failure (ADR-0003 documents this). Phase F
  protections (gates, structured output, pattern detection, outcome log)
  must remain rigorous. Erosion of any one of these reopens the failure
  mode.

- **Calibration assumption unproven**: we assume gates will catch
  miscalibrated LLM confidence. v2 spike validates this for 5 rounds, but
  not at scale. If LLM finds a way to satisfy gates while still being
  wrong (e.g., generating fake supporting_observations), the architecture
  fails silently.

- **Cost growth potential**: 5x LLM call increase vs Phase E. If Phase F
  becomes default, monthly LLM bill grows accordingly. Need budget caps
  as architecture matures.

- **Two-architecture maintenance burden**: keeping Phase E and Phase F
  both correct doubles the surface area. If Phase F doesn't graduate to
  production, this is wasted complexity.

## What this does NOT decide

- Whether Phase F replaces Phase E (deferred until Phase F shown to converge
  on better network quality over multiple maintenance cycles).
- HITL UI/UX (deferred to when REVIEW gates fire on real operational runs).
- Calibration tuning (deferred until ~50-100 decisions logged).
- Cross-run journal mechanism (deferred; current per-run pattern detection
  is sufficient for initial validation).

## Path forward

1. **Run extended spike** (max_rounds=15-20) on v3_p1 to observe Phase F
   behavior under richer budget. Specifically watch for: convergence to
   stop, cycle detection (re-doing previous work), calibration drift.

2. **After ~50 decisions accumulate**, run calibration analysis on
   `outputs/agent_decisions.jsonl`. Adjust gate thresholds based on data.

3. **If extended spike shows Phase F producing higher-quality networks
   than Phase E** (measured by golden match + intra-cluster + retrieval
   utility), promote Phase F to default for `vocab_maintenance`. Otherwise
   keep as exploratory and rely on Phase E for routine maintenance.

4. **When real REVIEW gates fire on operational runs**, design HITL UX
   (separate ADR).

## Related artifacts

- `outputs/spike_trace_20260514_023648.jsonl` — spike v1 (failure data)
- `outputs/spike_trace_20260514_025530.jsonl` — spike v2 (success data)
- `outputs/spike_summary_20260514_*.json` — high-level run summaries
- `src/consolidate_agent/vocab_maintenance/agent_brain/` — implementation
- ADR-0003: Phase E plan-and-execute (the architecture this builds on)
- ADR-0005: silhouette deprecation (the proxy-metric problem motivating Phase F)

## References

- Rousseeuw (1986) — silhouette score; foundational to ADR-0005's
  deprecation rationale.
- Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language
  Models" — pattern Phase F follows.
- Anthropic "Building effective agents" — workflow vs agent distinction;
  Phase F is firmly in "agent" territory while Phase E is "workflow."
- Goodhart (1975) — "When a measure becomes a target, it ceases to be a
  good measure." Diagnostic frame for why Phase E's proxy-metric
  optimization fails.
