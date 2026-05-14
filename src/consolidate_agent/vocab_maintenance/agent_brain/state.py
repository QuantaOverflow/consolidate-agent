"""TypedDict state schema for the Phase F brain LangGraph (ADR-0008).

State is the single source of truth. Every node returns a partial dict of
field updates; reducers control how updates merge into the prior state.

LLM-self-reported fields (preview_reviewed, affected_records_estimate,
certainty) flow through a `verify` node that fact-overrides them against
this state before reaching gate logic.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict


# ── Reducers ──────────────────────────────────────────────────────────────


def _append(left, right):
    if right is None:
        return list(left or [])
    return list(left or []) + list(right)


def _append_bounded(n: int):
    def reducer(left, right):
        if right is None:
            return list(left or [])
        return (list(left or []) + list(right))[-n:]
    return reducer


def _increment(left, right):
    if right is None:
        return left or 0
    return (left or 0) + right


def _proposal_cache_reducer(left, right):
    """Merge proposal cache updates.

    `right` is one of:
      - {"add": {pid: envelope}}      add a new proposal
      - {"remove": [pid, ...]}        evict specific proposals
      - {"clear_subject": subject}    evict all proposals matching a subject
      - {"replace": dict}             wholesale replace
      - dict                          treated as additive (back-compat)
    """
    base: dict[str, Any] = dict(left or {})
    if right is None:
        return base
    if isinstance(right, dict) and "replace" in right and len(right) == 1:
        return dict(right["replace"] or {})
    if isinstance(right, dict) and ({"add", "remove", "clear_subject"} & right.keys()):
        if "add" in right:
            for pid, env in (right["add"] or {}).items():
                base[pid] = env
        if "remove" in right:
            for pid in right["remove"] or []:
                base.pop(pid, None)
        if "clear_subject" in right:
            subject = right["clear_subject"]
            if subject is not None:
                doomed = [
                    pid for pid, env in base.items()
                    if env.get("subject") == subject
                ]
                for pid in doomed:
                    base.pop(pid, None)
        return base
    if isinstance(right, dict):
        base.update(right)
        return base
    return base


# ── State ─────────────────────────────────────────────────────────────────


_HISTORY_MAX = 15
_FACTS_MAX = 30


class BrainState(TypedDict, total=False):
    # ── inputs (set at graph.invoke time, immutable through run) ──────────
    db_path: str
    max_rounds: int
    max_tools_per_round: int
    cost_cap_calls: int
    run_id: str

    # ── network SSOT (single source of truth — agent mutates via apply) ───
    vocab: list[dict]
    assignments: list[dict]

    # ── triage (recomputed at each round_start) ───────────────────────────
    triage_report: Any  # TriageReport | None; held opaque to avoid import cycle

    # ── round-level scratch (cleared on round_transition) ─────────────────
    current_round: int
    round_tool_count: int
    proposed_subject_this_round: Any  # frozenset
    tool_cache_this_round: dict
    working: list  # tool results captured this round; replace semantics, cleared by orient_node, accumulated by tool_node returning full list

    # ── proposal cache (lifecycle managed by reducer) ─────────────────────
    proposal_cache: Annotated[dict, _proposal_cache_reducer]

    # ── decision pipeline within current round ───────────────────────────
    pending_tool: Any           # (tool_name, args) | None — set by reason, consumed by tool_node
    pending_decision: Any       # AgentDecision | None — set by reason, transformed by verify, read by gate
    pending_gate: Any           # GateDecision | None — set by gate_node, read by apply/archive
    pending_apply_outcome: Any  # str | None — set by apply_node, read by archive_node

    # ── cross-round memory ────────────────────────────────────────────────
    history: Annotated[list, _append_bounded(_HISTORY_MAX)]
    facts: Annotated[list, _append_bounded(_FACTS_MAX)]
    decisions: Annotated[list, _append]  # audit trail — every decision + gate verdict

    # ── episodic memory snapshot (read from store at orient_node) ────────
    excluded_attempts: Any  # set[tuple[action, target_canonical]] — failed (action, target) combinations this run

    # ── control ──────────────────────────────────────────────────────────
    llm_call_count: Annotated[int, _increment]
    applied_count: Annotated[int, _increment]
    stop_reason: str
