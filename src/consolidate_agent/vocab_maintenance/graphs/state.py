"""TypedDict state schemas for LangGraph StateGraphs.

TypedDict (vs Pydantic) chosen for:
  - Light merge semantics (each node returns partial dict)
  - No runtime validation overhead per state update
  - Direct JSON serialization for checkpointer
"""
from __future__ import annotations

from typing import Annotated, TypedDict


def _append_last_n(n: int):
    """LangGraph reducer: concat then keep tail. Bounded history accumulator.

    Nodes return ``{"field": [new_entry]}``; reducer merges into existing list
    and trims to the last n entries so state size stays bounded across iters.
    """
    def reducer(left, right):
        if right is None:
            return list(left or [])
        return (list(left or []) + list(right))[-n:]
    return reducer


_KEEP_N = 5  # last 5 iters' metrics + deltas — enough for PLAN history table


class BootstrapState(TypedDict, total=False):
    """State for the bootstrap pipeline.

    Stages populate fields as they run; HITL gate may loop synthesize.
    """
    # Inputs (set at graph.invoke time)
    db_path: str                # Path serialized as str for checkpointer
    batch_size: int             # distill batch size (default 30)
    concurrency: int            # ThreadPool workers for reverse_check
    auto_accept: bool           # if True, skip vocab_review interrupt
    themes_seed: dict           # record_id → theme (skips re-distill for known records)

    # Intermediate artifacts
    themes: list[dict]          # output of distill: [{record_id, title, theme}]
    vocab: list[dict]           # output of synthesize: [{name, definition}]
    synthesize_notes: str       # LLM's free-text rationale
    synthesize_attempts: int    # how many times synthesize ran (for retry tracking)

    # HITL gate
    review_decision: str        # "accept" / "regenerate" / "abort"

    # Final
    assignments: list[dict]     # output of reverse_check
    fake_tag_drops: int         # vocab-external tag refs filtered

    # Control flow
    abort_reason: str


class AgentLoopState(TypedDict, total=False):
    """State for the maintenance agent loop (Phase E — plan-and-execute).

    Topology (Phase E):
      START → initial_measure → plan_work_queue → check_continue:
        queue empty -> finalize_completed
        iter >= max -> finalize_terminate
        else -> pop_item -> propose -> apply -> post_apply_measure ->
                sanity_check -> commit/rollback -> record_iter -> check_continue
    """
    # Inputs
    initial_vocab: list
    max_iter: int
    disabled_actions: list
    run_id: str                          # = thread_id; used as audit run grouping key

    # Loop state
    iter: int
    vocab: list
    assignments: list
    diagnostics: dict
    history: list                         # IterationRecord dicts
    work_queue: list                      # Phase E: list of WorkItem dicts; popped each iter
    current_item: dict                    # Phase E: the WorkItem being processed this iter

    # Per-iter scratch
    action: str
    focus: str
    proposals: list                       # serialized proposals
    new_vocab: list
    new_assignments: list
    new_diag: dict
    new_metrics: dict                     # post-apply 5-dim metrics
    sanity_verdict: str                   # "commit" | "rollback" (pure-fn output)
    sanity_reason: str                    # 1-line explanation
    rec: dict                             # current iter record being assembled
    iter_result: str                      # "applied" / "rolled_back" / "apply_error" / "blocked_empty"

    # Termination
    final_status: str                     # FinalStatus.value
    run_summary: dict                     # populated by finalize_terminate_node

    # Bounded health-trajectory accumulators
    metrics_history: Annotated[list, _append_last_n(_KEEP_N)]   # [{iter, metrics}]
    iter_deltas: Annotated[list, _append_last_n(_KEEP_N)]        # [{iter, action, focus, result, delta, sanity}]
