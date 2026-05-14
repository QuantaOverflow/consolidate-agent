# ADR-0008: Re-implement Phase F brain loop on LangGraph

**Status**: Accepted
**Date**: 2026-05-14
**Builds on**: ADR-0006 (Phase F architecture), ADR-0007 (agent/evaluation separation)
**Affects**: `vocab_maintenance/agent_brain/`

## Context

Phase F (ADR-0006) was prototyped as a hand-written `while` loop in
`agent_brain/brain.py` to move fast on the LLM-as-strategic-reasoner spike.
The hand-written harness is now the bottleneck: spikes v3 through v9
iteratively exposed information-flow bugs that have nothing to do with the
LLM's reasoning quality and everything to do with the surrounding plumbing.

Inventory of plumbing problems found in spike v5-v9 (in chronological order
of when they surfaced):

1. **LLM self-reported fields trusted without verification**:
   `preview_reviewed`, `affected_records_estimate`, `certainty` come from
   the LLM's structured output. v8 R4 showed the LLM previewed http_api,
   received a proposal_id, then self-reported `preview_reviewed=False` in the
   decide schema, causing the gate to (correctly, given inputs) reject the
   action — even though a proposal was sitting in the cache.

2. **`proposal_cache` has no lifecycle**: proposals from REVIEW-gated
   rounds were never evicted. v8 R5 attempted to apply an http_api refine
   decision but `_find_proposal_for_target`'s fallback returned a stale
   persistence_db proposal from round R2, leading to an apply failure with
   the wrong proposal.

3. **`RoundSummary.compact()` flattens gate outcome into "(pending)"**: the
   LLM cannot distinguish "REVIEW-gated, do not retry" from "in progress."

4. **`_detect_critical_patterns` window/threshold too narrow**: Pattern A
   only fires on the same `(action, target)` appearing ≥2 times in the
   last 3 rounds. Spike v5/v7 had decisions roll off the window before the
   pattern could fire.

5. **Silent no-op path**: gate=AUTO with no matching proposal in cache
   produced `"gate=AUTO but no proposal in cache, skipping apply"` — looks
   like success in the summary but mutation was zero. Patched in v8 with a
   `decide_without_proposal` gate.

6. **`result_summary` displays `None` for error results**: trace shows the
   tool name but no error string when a tool returned `{"ok": False, ...}`.

7. **`applied_count` telemetry discrepancy** (per 2026-05-14 handoff):
   `brain.apply` events fire but summary's per-decision `committed` flag
   shows False.

8. **Mutations not persisted**: `_do_apply` mutates `context.vocab` and
   `context.assignments` in memory; nothing writes back to disk. Judges
   downstream can't compare before/after.

These are all state-management problems. They are exactly what LangGraph's
`StateGraph` + reducer + checkpointer model is designed to prevent:

- TypedDict state schema → all state fields explicit and typed
- Reducers → field-level update semantics (replace / append / merge)
- Checkpointer → per-step state persisted and replayable
- Conditional edges → state-driven control flow instead of hand-written
  `if/elif` branches

Phase E (`vocab_maintenance/graphs/agent.py`) already runs on LangGraph
with `MemorySaver`/`SqliteSaver` checkpointers. Phase F sitting on a
hand-written loop creates two harness implementations in the same
codebase, and the hand-written one is empirically the buggy one.

ADR-0006 did not technically reject LangGraph; it rejected Phase E's
**plan-and-execute** pattern. ReAct-style single-loop agents are a
standard LangGraph use case (e.g., `create_react_agent` in `langgraph.prebuilt`).
The LLM-as-strategic-reasoner principle from ADR-0006 is independent of
the orchestration runtime.

## Decision

Re-implement the Phase F brain loop using LangGraph `StateGraph`.

### What stays (no semantic change)

- **AgentDecision** schema (ADR-0006): `action / target / reasoning /
  certainty / supporting_observations / preview_reviewed / ...`
- **7-gate filter** (`decision.py`): unchanged. Gate is a node in the new
  graph, not a free function call site.
- **Triage** (`triage.py`): unchanged. Becomes a node that runs at round
  start and writes to `BrainState.triage_report`.
- **Tools** (`tools.py`): tool implementations unchanged. Tool dispatch
  becomes a node. The `_proposed_subject_this_round` one-proposal-per-round
  lock stays.
- **ADR-0007 agent/evaluation separation**: no golden access for agent.
- **Cost cap, max rounds, max tools per round**: configurable as graph
  inputs.

### What changes

| Before (hand-written) | After (LangGraph) |
|---|---|
| `@dataclass NetworkState/AgentMemory` | `BrainState(TypedDict)` with reducers |
| `while memory.current_round_idx < max_rounds` | `StateGraph` with conditional edges |
| `memory.archive_round()` mutates | reducer appends to `state.history` |
| `context.proposal_cache: dict` (no lifecycle) | `state.proposal_cache: dict` with cleanup reducer per round |
| `context.vocab.extend(new_vocab)` in-memory only | `state.vocab` with optional `SqliteSaver` checkpoint |
| `print` + ad-hoc `log.event()` telemetry | checkpointer event stream (replay-able) |
| LLM self-reported fields used as-is | dedicated `verify` node fact-overrides before `gate` |
| `_detect_critical_patterns` reads fuzzy strings | reads typed `RoundOutcome` enum from state |

### Graph shape

```
START
  ↓
orient ──────── (compute triage, build initial state)
  ↓
reason ─── call_tool ──→ tool ──┐
  ↑                              │
  └──────────────────────────────┘  (until LLM decides or budget hit)
  ↓ decide
verify ─────── (fact-override preview_reviewed, affected, target_exists)
  ↓
gate ────────── (7 gates → AUTO / REVIEW)
  ↓
[AUTO + has proposal]   [REVIEW or no proposal]
       ↓                        ↓
     apply                    skip
       ↓                        ↓
       └──── archive_round ────┘
              ↓
      [more rounds & not stop]?
        ↓ yes              ↓ no
       orient              END
```

### State schema (sketch — final form decided in implementation)

```python
class BrainState(TypedDict):
    # immutable inputs
    db_path: Path
    config: BrainConfig                # max_rounds, max_tools_per_round, cost_cap

    # mutable network state (single source of truth)
    vocab: list[dict]                  # reducer: replace
    assignments: list[dict]            # reducer: replace
    triage_report: TriageReport | None # reducer: replace, recomputed per round

    # round-level state
    current_round: int                 # reducer: replace
    proposal_cache: dict[str, ProposalEnvelope]  # reducer: cache_reducer (cleanup by round)
    proposed_subject_this_round: frozenset       # reducer: replace, cleared per round
    working: list[ToolCall]            # reducer: append, cleared per round

    # cross-round memory
    history: list[RoundSummary]        # reducer: append (bounded)
    facts: list[str]                   # reducer: append (bounded)
    decisions: list[DecisionRecord]    # reducer: append (audit trail)

    # control
    stop_reason: str | None            # reducer: replace
    llm_call_count: int                # reducer: increment
```

Key invariants enforced by reducers (not by ad-hoc code):
- `proposal_cache` entries carry `round_created`; cache_reducer evicts on
  round transition + on successful apply + on REVIEW for same subject.
- `RoundSummary.outcome` is a typed enum (`committed / rejected_by_gate /
  apply_failed / no_op_inspect / stop`), not a free-form string.
- `working` always cleared at round transition.

### Checkpointer

For the spike, use `MemorySaver`. For production, swap to `SqliteSaver`
(same approach as Phase E in `graphs/checkpointer.py`). Checkpointer choice
is a graph construction parameter, not an architectural decision here.

## What this does NOT decide

- **Whether Phase F replaces Phase E** (still deferred per ADR-0006).
- **Cross-run memory injection schema**: if `state.history`/`decisions`
  needs to persist across runs and feed the next run's prompt, that's
  ADR-0009 material.
- **HITL escalation UX**: REVIEW still skips apply. `interrupt()` is
  available in LangGraph but wiring it to a real UI is a separate task.

## Consequences

### Positive

- 8 plumbing bugs above become **structural impossibilities**, not patches:
  proposal_cache reducer can't leave stale entries; reducers can't write
  arbitrary fields; checkpointer can't lose round transitions.
- **One harness implementation** across Phase E and Phase F.
- **Replay-able runs**: checkpoint stream lets us re-run a failed spike
  from any node, or diff state across rounds.
- **Telemetry comes for free**: `graph.stream()` events ARE the trace.

### Negative

- **Implementation effort**: ~1–2 days, plus migrating 28 existing unit
  tests (most should port directly since core logic — gate, triage, tools —
  doesn't change).
- **LangGraph version pin**: tied to the same LangGraph API surface Phase E
  uses. Mostly fine (both use `langgraph.graph.StateGraph` + checkpointers).
- **Slightly more abstract control flow**: conditional edges + reducers are
  less "read top-to-bottom" than the current `while`. Mitigated by keeping
  node functions small and pure.

## Migration plan

1. Create `agent_brain/state.py` (BrainState TypedDict + reducers).
2. Create `agent_brain/graph.py` (StateGraph construction + node functions).
3. Keep `decision.py`, `triage.py`, `tools.py` unchanged except for
   signature: `(state) → state-update-dict` instead of
   `(context, memory) → ...`.
4. Rewrite `scripts/run_brain_spike.py` to invoke the graph.
5. Port unit tests; add tests for reducers (proposal_cache lifecycle,
   round-transition clearing).
6. Old `brain.py` + `memory.py` get archived (kept in tree but unused) for
   one cycle, then removed after the LangGraph version is validated by
   running spike v10+.

## Related

- ADR-0006: Phase F LLM strategic reasoner — this ADR replaces its
  implementation, not its principles.
- ADR-0007: agent/evaluation separation — preserved verbatim.
- ADR-0003: Phase E plan-and-execute on LangGraph — Phase F now uses
  LangGraph too, but as ReAct-style single loop, not plan-and-execute.

## References

- LangGraph state and persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph state machines in production:
  https://dev.to/jamesli/langgraph-state-machines-managing-complex-agent-task-flows-in-production-36f4
- AgentSpec runtime enforcement (related concept):
  https://arxiv.org/abs/2503.18666
