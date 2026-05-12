"""Maintenance agent: facade over the LangGraph StateGraph.

Public types (AgentState, FinalStatus, IterationRecord, Agent) are
backward-compatible with the pre-graph implementation. Internally
Agent.run() builds and invokes a StateGraph from .graphs.agent.

Per iteration:
  1. measure (reverse_check) → diagnostics, assignments
  2. diagnose → next_action + focus
  3. if done → stop
  4. propose_X(focus) → proposals
  5. apply proposals → new vocab/assignments
  6. verify hit_rate (anti-regression) → keep or rollback
  7. record history, loop

Stop conditions (FinalStatus):
  - completed: diagnose returns done
  - max_iter_exhausted: iter >= max_iter
  - no_actions_remaining: all 3 ACTIONS in blocked_actions
  - fatal_error: unexpected exception during diagnose
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


ACTIONS = ("propose_new", "propose_merge", "propose_deprecate", "propose_refine")


class FinalStatus(str, Enum):
    COMPLETED = "completed"
    MAX_ITER_EXHAUSTED = "max_iter_exhausted"
    NO_ACTIONS_REMAINING = "no_actions_remaining"
    FATAL_ERROR = "fatal_error"


@dataclass
class IterationRecord:
    iter: int
    action: str                 # 'done' / 'propose_X' / 'blocked' / 'unknown_action' / ''
    decision: dict | None       # diagnose output
    proposals: list = field(default_factory=list)
    hit_rate_before: float = 0.0
    hit_rate_after: float = 0.0
    result: str = ""            # 'completed' / 'applied' / 'rolled_back' / 'blocked_empty' / 'apply_error' / 'fatal_error' / 'blocked' / 'unknown_action'
    blocked_actions_after: list = field(default_factory=list)
    error: str | None = None


@dataclass
class AgentState:
    vocab: list[dict]
    assignments: list[dict]
    diagnostics: dict
    iter: int = 0
    blocked_actions: set[str] = field(default_factory=set)
    history: list[IterationRecord] = field(default_factory=list)


def _hit_rate(diagnostics: dict) -> float:
    n = diagnostics.get("sample_size", 0)
    if n == 0:
        return 0.0
    return diagnostics.get("total_assigned", 0) / n


class Agent:
    """Maintenance agent. Public contract identical to pre-graph version.

    Internally compiles a LangGraph StateGraph (.graphs.agent.build_agent_graph)
    and drives it via graph.invoke(). State transitions are equivalent to the
    previous pure-Python while-loop.

    measure_fn signature contract:
        measure_fn(
            vocab,
            *,
            action: str | None,                 # None on initial call; 'propose_*' post-apply
            current_assignments: list[dict] | None,
            previous_assignments: list[dict] | None,
        ) -> tuple[diagnostics_dict, assignments_list]

    diagnose_fn signature contract:
        diagnose_fn(vocab, diagnostics, assignments, *, blocked_actions=[]) -> dict
            Returns either {"decision": {...}} or the decision dict directly.
    """

    def __init__(
        self,
        measure_fn: Callable[..., tuple[dict, list[dict]]],
        diagnose_fn: Callable[..., dict],
        propose_fns: dict[str, Callable[..., list]],
        max_iter: int = 5,
        disabled_actions: set[str] | frozenset[str] | None = None,
        checkpoint_db: Path | None = None,
        health_fn: Callable | None = None,
        judge_fn: Callable | None = None,
    ):
        self.measure_fn = measure_fn
        self.diagnose_fn = diagnose_fn
        self.propose_fns = propose_fns
        self.max_iter = max_iter
        # Permanently-blocked actions (e.g., maintenance disables propose_new
        # because the growth path lives in ingest_batch, not in the agent loop).
        self.disabled_actions: set[str] = set(disabled_actions or ())
        # Optional persistent checkpoint backend; None → MemorySaver.
        self.checkpoint_db: Path | None = checkpoint_db
        # Phase A: optional health_fn(vocab, assignments) -> HealthMetrics.
        # When provided, 5-dim metrics flow into metrics_history + iter_deltas.
        self.health_fn = health_fn
        # Phase B: optional judge_fn (LLM-as-judge). Replaces hit_rate gate.
        # Requires health_fn for the before/after metrics it consumes.
        self.judge_fn = judge_fn

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

        # Recursion limit: each iter walks 4-6 nodes; allow 6 × max_iter + buffer.
        invoke_config = {**config, "recursion_limit": max(50, 6 * self.max_iter + 20)}

        if self.checkpoint_db is None:
            from langgraph.checkpoint.memory import MemorySaver
            cp = MemorySaver()
            graph = build_agent_graph(
                cp,
                measure_fn=self.measure_fn,
                diagnose_fn=self.diagnose_fn,
                propose_fns=self.propose_fns,
                health_fn=self.health_fn,
                judge_fn=self.judge_fn,
            )
            result = graph.invoke(initial_state, invoke_config)
        else:
            from .graphs.checkpointer import sqlite_checkpointer
            with sqlite_checkpointer(self.checkpoint_db) as cp:
                graph = build_agent_graph(
                    cp,
                    measure_fn=self.measure_fn,
                    diagnose_fn=self.diagnose_fn,
                    propose_fns=self.propose_fns,
                )
                result = graph.invoke(initial_state, invoke_config)

        # Convert dict state → dataclass for backward compat.
        history = [
            IterationRecord(
                iter=rec["iter"],
                action=rec.get("action", ""),
                decision=rec.get("decision"),
                proposals=rec.get("proposals", []),
                hit_rate_before=rec.get("hit_rate_before", 0.0),
                hit_rate_after=rec.get("hit_rate_after", 0.0),
                result=rec.get("result", ""),
                blocked_actions_after=rec.get("blocked_actions_after", []),
                error=rec.get("error"),
            )
            for rec in result.get("history", [])
        ]
        state = AgentState(
            vocab=result["vocab"],
            assignments=result["assignments"],
            diagnostics=result["diagnostics"],
            iter=result.get("iter", 0),
            blocked_actions=set(result.get("blocked_actions", [])),
            history=history,
        )
        status = FinalStatus(result.get("final_status", "fatal_error"))
        return state, status
