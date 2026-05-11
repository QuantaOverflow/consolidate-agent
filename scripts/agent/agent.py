"""Agent loop: signal-driven state machine for vocab maintenance.

Per iteration:
  1. measure (reverse_check) → diagnostics, assignments
  2. diagnose → next_action + focus
  3. if done → stop
  4. propose_X(focus) → proposals
  5. apply proposals → new vocab/assignments
  6. verify hit_rate (anti-regression) → keep or rollback
  7. record history, loop

Stop conditions:
  - completed: diagnose returns done
  - max_iter_exhausted: iter >= max_iter
  - no_actions_remaining: all 3 actions blocked
  - fatal_error: unexpected exception
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from apply import (
    DeprecateProposal,
    InvalidProposal,
    MergeProposal,
    NewTagProposal,
    OrphanError,
    apply_proposal,
    check_invariants,
)


ACTIONS = ("propose_new", "propose_merge", "propose_deprecate")


class FinalStatus(str, Enum):
    COMPLETED = "completed"
    MAX_ITER_EXHAUSTED = "max_iter_exhausted"
    NO_ACTIONS_REMAINING = "no_actions_remaining"
    FATAL_ERROR = "fatal_error"


@dataclass
class IterationRecord:
    iter: int
    action: str                 # 'done' / 'propose_X' / 'blocked' / 'rolled_back'
    decision: dict | None       # diagnose output
    proposals: list = field(default_factory=list)
    hit_rate_before: float = 0.0
    hit_rate_after: float = 0.0
    result: str = ""            # 'completed' / 'applied' / 'rolled_back' / 'blocked_empty' / 'apply_error'
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
    """Signal-driven loop. Dependencies injected for testability."""

    def __init__(
        self,
        measure_fn: Callable[[list[dict]], tuple[dict, list[dict]]],
        diagnose_fn: Callable[[list[dict], dict, list[dict]], dict],
        propose_fns: dict[str, Callable[..., list]],
        max_iter: int = 5,
        hit_rate_regression_threshold: float = 0.03,
    ):
        self.measure_fn = measure_fn
        self.diagnose_fn = diagnose_fn
        self.propose_fns = propose_fns
        self.max_iter = max_iter
        self.hit_rate_regression_threshold = hit_rate_regression_threshold

    def run(self, initial_vocab: list[dict]) -> tuple[AgentState, FinalStatus]:
        # Initial measure
        diag, assignments = self.measure_fn(initial_vocab)
        state = AgentState(
            vocab=list(initial_vocab),
            assignments=list(assignments),
            diagnostics=diag,
        )
        check_invariants(state.vocab, state.assignments)

        while True:
            # Stop check: budget
            if state.iter >= self.max_iter:
                return state, FinalStatus.MAX_ITER_EXHAUSTED

            # Stop check: all actions blocked
            if state.blocked_actions.issuperset(ACTIONS):
                return state, FinalStatus.NO_ACTIONS_REMAINING

            state.iter += 1
            rec = IterationRecord(iter=state.iter, action="", decision=None,
                                  hit_rate_before=_hit_rate(state.diagnostics))

            try:
                # Diagnose
                diag_result = self.diagnose_fn(state.vocab, state.diagnostics, state.assignments)
                decision = diag_result.get("decision") if isinstance(diag_result, dict) and "decision" in diag_result else diag_result
                rec.decision = decision

                action = decision["next_action"]
                focus = decision.get("action_focus", "")

                # Done
                if action == "done":
                    rec.action = "done"
                    rec.result = "completed"
                    rec.hit_rate_after = rec.hit_rate_before
                    rec.blocked_actions_after = sorted(state.blocked_actions)
                    state.history.append(rec)
                    return state, FinalStatus.COMPLETED

                # Skip blocked
                if action in state.blocked_actions:
                    rec.action = action
                    rec.result = "blocked"
                    rec.hit_rate_after = rec.hit_rate_before
                    rec.blocked_actions_after = sorted(state.blocked_actions)
                    state.history.append(rec)
                    continue

                # Propose
                if action not in self.propose_fns:
                    rec.action = action
                    rec.result = "unknown_action"
                    rec.error = f"unknown action: {action}"
                    rec.blocked_actions_after = sorted(state.blocked_actions)
                    state.history.append(rec)
                    state.blocked_actions.add(action)
                    continue

                proposals = self.propose_fns[action](state.vocab, state.assignments, focus)
                rec.action = action
                rec.proposals = list(proposals)

                if not proposals:
                    state.blocked_actions.add(action)
                    rec.result = "blocked_empty"
                    rec.hit_rate_after = rec.hit_rate_before
                    rec.blocked_actions_after = sorted(state.blocked_actions)
                    state.history.append(rec)
                    continue

                # Apply
                new_vocab, new_assignments = state.vocab, state.assignments
                try:
                    for p in proposals:
                        new_vocab, new_assignments = apply_proposal(new_vocab, new_assignments, p)
                except (InvalidProposal, OrphanError) as e:
                    state.blocked_actions.add(action)
                    rec.result = "apply_error"
                    rec.error = f"{type(e).__name__}: {e}"
                    rec.hit_rate_after = rec.hit_rate_before
                    rec.blocked_actions_after = sorted(state.blocked_actions)
                    state.history.append(rec)
                    continue

                check_invariants(new_vocab, new_assignments)

                # Verify (re-measure)
                new_diag, new_assignments2 = self.measure_fn(new_vocab)
                # use re-measured assignments (apply's in-memory ones are pre-LLM-revalidation)
                new_hit = _hit_rate(new_diag)
                rec.hit_rate_after = new_hit

                if new_hit < rec.hit_rate_before - self.hit_rate_regression_threshold:
                    # rollback
                    state.blocked_actions.add(action)
                    rec.result = "rolled_back"
                    rec.blocked_actions_after = sorted(state.blocked_actions)
                    state.history.append(rec)
                    # state.vocab/diagnostics/assignments unchanged
                    continue

                # Commit
                state.vocab = new_vocab
                state.diagnostics = new_diag
                state.assignments = new_assignments2
                rec.result = "applied"
                rec.blocked_actions_after = sorted(state.blocked_actions)
                state.history.append(rec)

            except Exception as e:  # noqa: BLE001 — fatal path
                rec.result = "fatal_error"
                rec.error = f"{type(e).__name__}: {e}"
                rec.blocked_actions_after = sorted(state.blocked_actions)
                state.history.append(rec)
                return state, FinalStatus.FATAL_ERROR
