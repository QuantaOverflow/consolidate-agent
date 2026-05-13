"""Maintenance agent: facade over the LangGraph StateGraph (Phase E).

Public types (AgentState, FinalStatus, IterationRecord, Agent) wrap the
plan-and-execute StateGraph in .graphs.agent. Each `agent.run()` invocation:

  1. measure (reverse_check) → initial diagnostics, assignments, metrics
  2. planner (rules-based) → ordered work queue from signal probes
  3. loop:
       pop item → propose (LLM) → apply (invariant-guarded) → re-measure
       → sanity_check (pure) → commit/rollback → record
  4. terminate when queue empty or max_iter

Stop conditions (FinalStatus):
  - completed: work queue exhausted
  - max_iter_exhausted: iter >= max_iter
  - degraded: failure_rate >= threshold (silent degradation guard)
  - fatal_error: unexpected exception during propose/apply
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class FinalStatus(str, Enum):
    COMPLETED = "completed"
    DEGRADED = "degraded"
    MAX_ITER_EXHAUSTED = "max_iter_exhausted"
    FATAL_ERROR = "fatal_error"


@dataclass
class IterationRecord:
    iter: int
    action: str
    focus: str
    work_item_reason: str = ""
    proposals: list = field(default_factory=list)
    hit_rate_before: float = 0.0
    hit_rate_after: float = 0.0
    result: str = ""            # 'applied' / 'rolled_back' / 'blocked_empty' / 'apply_error' / 'unknown_action' / 'propose_error'
    sanity_verdict: str = ""    # 'commit' / 'rollback'
    sanity_reason: str = ""
    error: str | None = None


@dataclass
class AgentState:
    vocab: list[dict]
    assignments: list[dict]
    diagnostics: dict
    iter: int = 0
    history: list[IterationRecord] = field(default_factory=list)
    work_queue_remaining: list[dict] = field(default_factory=list)
    run_summary: dict = field(default_factory=dict)


def _hit_rate(diagnostics: dict) -> float:
    n = diagnostics.get("sample_size", 0)
    if n == 0:
        return 0.0
    return diagnostics.get("total_assigned", 0) / n


class Agent:
    """Maintenance agent — plan-and-execute facade.

    measure_fn signature:
        measure_fn(
            vocab,
            *,
            action: str | None,                 # None on initial; 'propose_*' post-apply
            current_assignments: list[dict] | None,
            previous_assignments: list[dict] | None,
        ) -> tuple[diagnostics_dict, assignments_list]

    health_fn signature:
        health_fn(vocab, assignments) -> HealthMetrics

    propose_fns: dict mapping action name → fn(vocab, assignments, focus) -> list
    db_path: passed through to the planner for `compute_fit_signals`.
    """

    def __init__(
        self,
        measure_fn: Callable[..., tuple[dict, list[dict]]],
        propose_fns: dict[str, Callable[..., list]],
        health_fn: Callable,
        db_path: Path,
        max_iter: int = 5,
        disabled_actions: set[str] | frozenset[str] | None = None,
        checkpoint_db: Path | None = None,
        sanity_threshold: float = 0.10,
    ):
        self.measure_fn = measure_fn
        self.propose_fns = propose_fns
        self.health_fn = health_fn
        self.db_path = db_path
        self.max_iter = max_iter
        self.disabled_actions: set[str] = set(disabled_actions or ())
        self.checkpoint_db: Path | None = checkpoint_db
        self.sanity_threshold = sanity_threshold

    def run(
        self,
        initial_vocab: list[dict],
        *,
        thread_id: str | None = None,
    ) -> tuple[AgentState, FinalStatus]:
        from .graphs.agent import build_agent_graph

        if thread_id is None:
            thread_id = f"agent-{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "initial_vocab": list(initial_vocab),
            "max_iter": self.max_iter,
            "disabled_actions": list(self.disabled_actions),
            "run_id": thread_id,
        }
        invoke_config = {**config, "recursion_limit": max(50, 8 * self.max_iter + 20)}

        if self.checkpoint_db is None:
            from langgraph.checkpoint.memory import MemorySaver
            cp = MemorySaver()
            graph = build_agent_graph(
                cp,
                measure_fn=self.measure_fn,
                propose_fns=self.propose_fns,
                health_fn=self.health_fn,
                db_path=self.db_path,
                sanity_threshold=self.sanity_threshold,
            )
            result = graph.invoke(initial_state, invoke_config)
        else:
            from .graphs.checkpointer import sqlite_checkpointer
            with sqlite_checkpointer(self.checkpoint_db) as cp:
                graph = build_agent_graph(
                    cp,
                    measure_fn=self.measure_fn,
                    propose_fns=self.propose_fns,
                    health_fn=self.health_fn,
                    db_path=self.db_path,
                    sanity_threshold=self.sanity_threshold,
                )
                result = graph.invoke(initial_state, invoke_config)

        history = [
            IterationRecord(
                iter=rec["iter"],
                action=rec.get("action", ""),
                focus=rec.get("focus", ""),
                work_item_reason=rec.get("work_item_reason", ""),
                proposals=rec.get("proposals", []),
                hit_rate_before=rec.get("hit_rate_before", 0.0),
                hit_rate_after=rec.get("hit_rate_after", 0.0),
                result=rec.get("result", ""),
                sanity_verdict=rec.get("sanity_verdict", ""),
                sanity_reason=rec.get("sanity_reason", ""),
                error=rec.get("error"),
            )
            for rec in result.get("history", [])
        ]
        state = AgentState(
            vocab=result["vocab"],
            assignments=result["assignments"],
            diagnostics=result["diagnostics"],
            iter=result.get("iter", 0),
            history=history,
            work_queue_remaining=list(result.get("work_queue") or []),
            run_summary=result.get("run_summary", {}),
        )
        status = FinalStatus(result.get("final_status", "fatal_error"))
        return state, status
