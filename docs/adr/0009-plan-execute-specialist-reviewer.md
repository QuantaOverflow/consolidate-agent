# ADR-0009: Plan-Execute architecture with specialist reviewer + forced commit

**Status**: Accepted
**Date**: 2026-05-14
**Builds on**: ADR-0006 (Phase F), ADR-0007 (agent/evaluator separation), ADR-0008 (LangGraph)
**Affects**: `agent_brain/graph.py`, `agent_brain/state.py`, `agent_brain/tools.py`

## Context

Spike v15-v18 (per ADR-0008) had Phase F running clean: 5 commits/run, 0
hallucinations, LLM citing preview diff facts. But review of those commits
showed only 1/3 of prunes were clearly justified — the rest were marginal.
The agent was approving the propose stage's marginal outputs because the
LLM in the single-agent loop bore too many roles:

  1. scan whole triage report (16 tags)
  2. pick target
  3. choose action type (split/refine/merge/deprecate)
  4. inspect samples
  5. generate proposal via preview
  6. read diff
  7. decide to commit

Cognitive overload → safe-default bias toward `refine`. Across v3-v18,
the agent never picked split, merge, or deprecate. Triage suggesting them
didn't move the needle.

The user's framing crystallized the issue: agent's job should be **review
a specific proposal**, not **simultaneously plan and execute**. The plan
stage (triage / hardcoded ticket queue) decides what to do; a specialist
reviewer agent evaluates each concrete (action, target) proposal in
isolation.

## Decision

Re-shape the LangGraph topology from a single ReAct-style loop into a
**plan-execute pipeline** with a specialist reviewer.

### Graph topology

```
START
  ↓
plan ────────── (seed plan_queue with tickets — hardcoded for spike;
                 triage-driven in future)
  ↓
take_next ──── (queue empty / budget) ──→ END
  ↓
auto_preview ── (graph auto-calls propose_*_preview for the ticket)
  ↓
specialist ─── (LLM reviewer with action-specific prompt + few-shot)
  ↓
  ├── verdict=approve → approve_to_decision → verify → gate → apply → ticket_archive
  ├── verdict=dig_deeper → dig_inspect (limited tools) → specialist (loop)
  │                         └── max 2 digs/ticket; same args debounced
  ├── verdict=dig_deeper after budget → forced_commit (Schema-constrained)
  │                                       ├→ approve → apply path
  │                                       └→ reject → ticket_archive
  └── verdict=reject → ticket_archive
  ↓
ticket_archive ──→ take_next
```

### Key components

#### 1. Plan stage (hardcoded for spike, triage-driven later)

`plan_node` seeds `state.plan_queue` with a list of tickets:

```python
[
  {"action": "refine",    "target": "persistence_db"},
  {"action": "split",     "target": "langgraph_state"},
  {"action": "deprecate", "target": "build_deployment"},
  {"action": "refine",    "target": "http_api"},
  {"action": "merge",     "target": "langgraph_state", "target_b": "llm_agent_runtime"},
]
```

Hardcoded covers all 4 action types so the specialist's behavior across
actions is observable. Future: triage-driven plan that emits prioritized
candidates (a separate ADR if non-trivial).

#### 2. Auto-preview

`auto_preview_node` looks at ticket's action, calls the matching
`propose_*_preview` automatically. The preview itself includes:

- LLM-generated proposal (new_definition / sub_tags / etc)
- Sandbox-applied diff: target_size_before/after, orphan_count
- `record_diffs`: up to 10 affected records with **title + insight snippet
  (500 chars)** + before/after tag sets — gives the reviewer concrete
  semantic content, not just IDs
- Action-specific samples: split has `sub_tags[].sample_records`,
  deprecate has `target_records_sample`, merge has `discard_records_sample`

The specialist never has to call propose itself. It only reviews.

#### 3. Specialist with action-specific prompt + few-shot

Four prompts in `_PROMPT_BY_ACTION`: refine / split / deprecate / merge.
Each contains:

- GOOD pattern signals (cite-able with preview data)
- BAD pattern signals
- A worked **dig walkthrough** (CoT): "Step 1 inspect_records(<suspect_ids>);
  if X → approve; if Y → reject; if unclear → Step 2 inspect_tag/compare_tags
  with DIFFERENT args; after Step 2 MUST verdict"

This addresses the v22-v23 finding where LLM knew it could dig but didn't
know what different question to ask next round.

#### 4. Dig tools with debounce + tool menu

When `verdict=dig_deeper`, the specialist names a `dig_tool` and `dig_args`.
Allowed:

- `inspect_records(record_ids)` — full insight + current tags for specific records
- `inspect_tag(tag_name, n_samples)` — see a different tag's samples
- `compare_tags(tag_a, tag_b)` — overlap between two tags

`specialist_dig_inspect_node` hashes `(tool, args)` into a per-ticket
`prior_dig_keys` set. Same key → ALREADY_INSPECTED hint, no tool call.

#### 5. Forced commit after budget

When `dig_deeper_count >= 2` and the model still wants to dig, route to
`forced_commit_node`. It uses a separate Pydantic schema:

```python
class ForcedCommitVerdict(BaseModel):
    verdict: Literal["approve", "reject"]  # no dig_deeper option
    reasoning: str
    cited_facts: list[str]
```

LangChain's `with_structured_output` enforces this enum at the API layer
(constrained decoding). The LLM cannot generate "dig_deeper" — its sampled
tokens are restricted to valid enum values. The forced-commit prompt
provides all gathered evidence (preview + every prior dig result) and
instructs the model: uncertainty in reasoning is OK, but the final verdict
must be binary.

This closes the v22-v23 failure mode where dig budget exhaustion was
indistinguishable from explicit reject — now the system gets a real
verdict, not a sysadmin-imposed timeout.

### Existing pipeline preserved

`verify_node` (fact-overrides preview_reviewed + affected_records_estimate),
`gate_node` (`decide_without_proposal`, `recent_flipflop`, etc.), and
`apply_node` (atomic mutation + cache clear) all unchanged. The specialist
emits a "fake" `AgentDecision` via `approve_to_decision_node` when verdict
is approve, so the rest of the pipeline runs as in ADR-0008.

## What this changes

| | ADR-0008 (single agent loop) | ADR-0009 (plan-execute) |
|---|---|---|
| LLM picks target | yes | no (plan does) |
| LLM picks action type | yes | no |
| LLM calls inspect/preview | yes | auto_preview does it; specialist only digs if needed |
| LLM role | planner+executor+reviewer | reviewer only |
| Hallucination ("redirect to X") | mitigated by prompt | structurally absent — `expected_outcome` field removed |
| dig loop on same args | possible | debounced |
| dig budget exhaustion | maps to silent reject | maps to forced binary verdict |
| split/merge/deprecate ever attempted | never (v3-v18) | yes (v19 split, v24 split) |

## Trade-offs

### Positive

- **Specialist actually reviews**. v24 R2 and R5 produced concrete rejects
  citing specific record IDs that the propose stage was mis-pruning. v3-v18
  approved similar marginal cases without question.
- **Split happens**. First time across all spikes (v19 R2, v24 R3).
  Bottleneck was cognitive overload of the planner-executor, not LLM
  capability.
- **Cost stable**: v24 used 9 LLM calls vs v18's 17 — fewer calls, higher
  per-call value.
- **Dig debounce + forced commit** eliminates the budget-exhaustion-as-
  silent-reject category.

### Negative / Open

- **Commit count drops** (v24: 1 commit vs v18: 5 commits). This is the
  consequence of strict review, not a regression — but it means the
  pipeline now depends critically on propose_*_fn output quality. If
  propose generates marginal proposals, specialist rejects them; nothing
  moves forward. Future work: improve propose stage quality (separate ADR).
- **Hardcoded plan_queue**. Triage already exists and can drive the plan,
  but spike kept the queue hardcoded so behavior across action types could
  be tested. Hooking triage → plan_queue is a small follow-up.
- **No judge step yet**. Specialist reject quality looks good in the trace,
  but no independent evaluator confirms commit-vs-reject decisions. Wiring
  ADR-0007 §4's judge layer is the next ADR.

## What this does NOT decide

- **Triage-driven plan**: how to convert triage's signals into a prioritized
  ticket queue. Deferred to a follow-up — current hardcoded queue is fine
  while specialist behavior is being validated.
- **Multi-agent feedback loop** (triage agent ↔ specialist agent that
  iterate plans): explicitly rejected for now. The single forward
  plan-execute is enough — reject doesn't trigger re-planning, just moves
  to the next ticket. Multi-agent loops add coordination cost without
  evidence they're needed at this scale.
- **Cross-run learning**. Episodic memory (ADR-0008's
  `excluded_attempts`) is per-run. Persistent `BaseStore` across runs is
  a separate decision.

## Related

- ADR-0006: Phase F LLM-as-strategic-reasoner — principles preserved,
  topology changed.
- ADR-0007: agent/evaluator separation — specialist *is* the evaluator
  role at proposal review time. ADR-0007 §4's post-hoc judge is still
  needed and separate.
- ADR-0008: LangGraph rewrite — this builds on the StateGraph + reducer
  foundation.

## References

- Anthropic, "Building effective agents" (2024) — workflows vs agents;
  evaluator-optimizer pattern; the position that most production agents
  succeed as workflows rather than multi-agent loops.
- LangChain Guardrails docs — validation as nodes + conditional edges;
  the same pattern this ADR uses for the gate / verify / forced_commit
  triad.
- OpenAI Structured Outputs — constrained decoding via JSON Schema enum,
  the mechanism behind `ForcedCommitVerdict` binary enforcement.
- "How to Prevent AI Agent Reasoning Loops from Wasting Tokens" (dev.to) —
  DebounceHook, terminal-state tools, hard limits; this ADR uses the
  first two ideas.
