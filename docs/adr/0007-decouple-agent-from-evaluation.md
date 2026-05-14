# ADR-0007: Decouple agent from evaluation; metrics are heuristics, not optimization targets

**Status**: Accepted
**Date**: 2026-05-14
**Builds on**: ADR-0005 (silhouette deprecation), ADR-0006 (Phase F)
**Affects**: Phase F architecture (`agent_brain/`), `quality_scorecard.py`, `tools.py`

## Context

After 4 maintenance attempts (P0 prompt fix, P1 multi-tag prompt, P2 Phase E
maintenance, Phase F spike v2), only 1 produced a net improvement on the
quality target (P1: matter exact 46% → 51%). The other 3 either regressed
or showed no measurable effect. The diagnosis pattern is consistent:

> Each agent run optimizes whatever metric we feed it. The metrics we have
> are either factual (coverage, hallucination) or proxies (golden match,
> intra-cluster coherence, plausibility). The proxies do not reliably
> predict actual network quality — they only correlate sometimes.

The deeper issue is twofold:

1. **Goodhart's Law revisited**: ADR-0005 deprecated silhouette as a gate
   in multi-tag settings. But we then proposed to use golden 80 as a
   replacement oracle, and intra-cluster coherence as a "tier 2 primary
   metric." This re-creates the same trap one level up — golden 80 is
   itself a proxy (small sample, single annotator, finite coverage).

2. **Test set contamination**: When the agent has tools to query golden
   labels mid-decision (`quick_golden_sample`), it can effectively
   optimize against the held-out evaluation set. This is the
   ML-equivalent of letting your model peek at the test set during
   training — once it happens, evaluation results are meaningless.

The user's reframing in this session crystallized the right principles:

- **Network quality is fundamentally a semantic judgment**. No quantitative
  proxy fully captures whether a tag organization "makes sense."
- **Metrics should be heuristics that orient the agent** ("this tag has
  low coherence, go look"), **not targets to optimize** ("get coherence
  above 0.45").
- **The agent and evaluator should be physically separated**. Agent
  operates on real data + heuristic signals. Evaluator (a separate LLM
  judge with access to golden) assesses outcomes post-hoc.

## Decision

### Principle 1: Strict separation of agent and evaluation data

The agent does **not** have access to held-out evaluation data (golden 80).
Specifically:

- **Removed from agent tool registry**: `quick_golden_sample`
- **Removed from `AgentDecision` schema**: `ground_truth_sampled` field
- **Removed from gate logic**: `high_impact_no_golden_check` rule
- **Replaced with**: `high_impact_no_preview` (high-impact actions require
  `propose_*_preview` call, not golden sampling)
- **Removed from system prompt**: "call quick_golden_sample to calibrate
  yourself" directives

Rationale: golden 80 was labeled by the same person who designs the agent.
If the agent reads golden during decision-making, it optimizes against its
designer's labels — this is contamination, not learning.

### Principle 2: Two classes of metrics

**Factual metrics (retained as PASS/FAIL gates)** measure structural facts
of the network, independent of semantic judgment:

- `coverage` (% records with ≥1 tag)
- `hallucination_rate` (% tags assigned that aren't in vocab)
- `mean_tags_per_record` (distribution density)
- `max_tag_share` (distribution skew)

These four cannot be "optimized" in a Goodhart sense — they are
constraints, not quality measures.

**Heuristic indicators (downgraded to informational, no PASS/FAIL)**:

- Golden match (matter exact / partial / lesson_type)
- Intra-cluster coherence
- Tag plausibility (top-K cosine)
- Silhouette (already deprecated by ADR-0005)

These remain visible in diagnostic reports for the human or judge LLM
reviewing a run. The agent is not driven by their thresholds.

### Principle 3: Outcome judged by separate LLM

After an agent run, a **judge LLM** (different role; initially Claude
playing the role manually) reviews the changes:

- Reads the agent's per-commit reports
- Uses inspection tools to verify against real data
- **Has access to golden 80 as reference** (this is its legitimate use —
  judge is the evaluator, not the agent)
- Outputs structured semantic judgment per commit + overall run verdict
- Judgment persists in `outputs/llm_judgments.jsonl`

Next agent run (when cross-run memory is implemented) sees prior
judgments as "lessons" — but never the raw golden labels.

### Principle 4: Run completion produces evaluable artifact

Agent runs output a **markdown commit report** containing:
- Each change applied with before/after state slice
- Agent's reasoning + claimed evidence
- Affected record IDs
- The 4 factual metrics (PASS/FAIL)
- The heuristic indicators (informational)

This is the artifact a judge (human or LLM) reviews. It's designed to be
human-readable.

## What this changes in code

### Removed

- `agent_brain/tools.py`: `_impl_quick_golden_sample` + `quick_golden_sample` tool spec
- `agent_brain/decision.py`: `AgentDecision.ground_truth_sampled` field
- `agent_brain/decision.py`: `high_impact_no_golden_check` gate logic
- `agent_brain/outcome_log.py`: `DecisionOutcome.metric_deltas` +
  `predicted_deltas_match` fields
- `agent_brain/brain.py`: system prompt sections referencing
  `quick_golden_sample`

### Added

- `agent_brain/decision.py`: `high_impact_no_preview` gate (replaces
  `high_impact_no_golden_check`)
- New package `vocab_maintenance/evaluator/`:
  - `judgment.py`: `Judgment` dataclass + persistence layer
  - `commit_report.py`: markdown run report generator (for judge to read)
- New script `scripts/evaluate_after_run.py`: reads agent output +
  computes heuristic indicators + presents to judge

### Modified

- `agent_brain/decision.py`: `expected_metric_deltas` →
  `expected_outcome: str` (qualitative semantic prediction)
- `quality_scorecard.py`: split into two sections (FACTUAL with
  PASS/FAIL, HEURISTIC informational); Overall PASS/FAIL based on
  FACTUAL section only

### Not changed (intentionally)

- `vocab_maintenance/graphs/agent.py` (Phase E): legacy plan-and-execute
  loop. Will be deprecated entirely when Phase F is validated; no point
  modifying it piecemeal now.
- `vocab_maintenance/measure_v3.py`: re-tagging pipeline. Uses LLM
  classification against vocab definitions (agent-side facts), no golden
  involvement. Compliant by construction.
- `intra_cluster_coherence.py` and other embedding-derived probes:
  these compute from agent-side data (record embeddings), not human
  labels. Remain valid as heuristic probes for the agent.

## What this does NOT decide

- **How judge LLM operates beyond Claude playing the role**: validating
  that a separate LLM (e.g., qwen-plus) reaches comparable judgments to
  Claude requires an empirical study after ~20-50 judgments accumulate.
  Deferred.
- **Cross-run memory injection**: agent reading prior judgments in
  future runs requires a memory channel. Deferred until single-run
  judgment cycle proves valuable.
- **HITL escalation UX**: REVIEW gate in spike still skips application.
  Production would need a human approval flow. Deferred.
- **Phase E removal**: legacy code stays until Phase F is validated.
  No removal date set.

## Consequences

### Positive

- **Eliminates the "agent optimizes against test set" failure mode**
  before it ships. Phase F infrastructure was already exposing
  `quick_golden_sample`; this ADR closes the leak before the agent runs
  at scale.
- **Restores metrics to their intended role** (heuristics that orient,
  not targets that drive). The agent reasons over data; the evaluator
  scores over outcomes.
- **Cleaner architectural boundaries**: agent and evaluator are physically
  separated (different packages, different files), enforced by code
  organization not just discipline.
- **Judge layer is cheap**: zero LLM cost during agent run; judgment is
  a separate post-hoc step. Cost roughly doubles for full validation
  cycle but only when validation is wanted.

### Negative / Risk

- **Single point of failure: the judge**. If Claude (or whichever LLM
  fills the judge role) has systematic blind spots, those propagate
  into agent's cross-run memory. Mitigation: log every judgment with
  full reasoning, audit ~20% with a different model.
- **No automated gate against semantic regression**. A run that
  silently degrades semantic quality but passes the 4 factual gates
  will commit changes. The judge catches this in post-hoc review, but
  by then changes are already applied. (Could add a "dry-run + judge
  before commit" mode in v2 if this becomes a real problem.)
- **Judge's access to golden is itself a contamination route over time**:
  if judgments influence future agent decisions via memory, and
  judgments leak golden-aligned conclusions, then agent indirectly
  learns golden bias. Less direct than peek-during-decision, but real.
  Mitigation: keep judgments at a semantic level ("split was too coarse")
  rather than label-level ("matter exact dropped on these 3 records").

## Path forward

1. **Implement code changes above** (~1-1.5 days).
2. **Re-run Phase F spike** with new tool set (no golden access).
   Verify gate `high_impact_no_preview` fires correctly in place of
   `high_impact_no_golden_check`.
3. **Run completion report generation** verified by reading sample output.
4. **Claude judges the spike output**, fills out first Judgment instance.
   Validate template captures useful signal.
5. **If judgment is useful**: build cross-run memory injection (separate
   ADR if non-trivial).
6. **After ~20 judgments**: pilot a second LLM (qwen-plus or similar)
   to judge same runs; compare verdicts; calibrate.

## Phase E note

`vocab_maintenance/graphs/agent.py` and its `sanity_check` (which uses
cosine-derived 5-dim HealthMetrics with a 10% catastrophic-regression
threshold) are not modified by this ADR. The sanity_check is technically
a proxy-metric gate, but it operates as a safety net against catastrophic
embedding-space regression rather than as an optimization target. Phase E
will be either replaced by Phase F or removed entirely as Phase F
matures. Modifying Phase E's internals now is scope creep.

## Related artifacts

- ADR-0005: silhouette deprecation (the proxy-metric problem first
  surfaced here)
- ADR-0006: Phase F architecture (this ADR amends Phase F's tool set)
- `outputs/spike_summary_20260514_025530.json`: spike v2 which
  demonstrated `quick_golden_sample` being used by the agent — the
  contamination this ADR closes

## References

- Goodhart, C. (1975) — "When a measure becomes a target, it ceases to
  be a good measure." Original framing of the metric-target problem.
- Strathern, M. (1997) — Goodhart's law reformulation; "improving by
  measuring" is the trap this ADR avoids.
- Standard ML practice on train/test separation — agent has no business
  reading evaluation data, full stop.
- Anthropic "Building effective agents" (2024) — evaluator-optimizer
  pattern with clear role separation is one of the patterns Phase F is
  now implementing.
