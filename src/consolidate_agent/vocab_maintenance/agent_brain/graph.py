"""Phase F brain as a LangGraph StateGraph (ADR-0008).

Topology:

  START → orient ───── reason ────── tool ──┐
                         ↑                   │ (loop until decide or budget)
                         └───────────────────┘
                         ↓ decide
                       verify (fact-override decision)
                         ↓
                       gate (7-gate filter)
                         ↓
                       apply (if AUTO + cache hit) | skip
                         ↓
                       archive
                         ↓
                    more rounds? ───── yes → orient
                         │
                         no
                         ↓
                        END

State is the single source of truth (see state.py). Nodes return partial
updates; reducers merge. LLM-self-reported fields go through `verify`
which overrides them against state facts before they reach `gate`.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model
from consolidate_agent.vocab_maintenance.observability import (
    RunLogger,
    get_default_logger,
    invoke_with_retry,
)

from .decision import (
    AgentDecision,
    GateDecision,
    GateResult,
    apply_reversibility_default,
    gate_decision,
    make_decision_id,
)
from .memory import RoundSummary, ToolResult, _detect_critical_patterns, NetworkState, AgentMemory
from .state import BrainState
from .tools import TOOLS, TOOLS_BY_NAME, BrainContext, call_tool_with_cache
from .triage import compute_triage


# ── BrainStep schema (LLM output) ─────────────────────────────────────────


class BrainStep(BaseModel):
    mode: str
    tool_name: str = ""
    tool_args: dict = Field(default_factory=dict)
    reason: str = ""
    decision: AgentDecision | None = None


# ── System prompt (shared with old brain.py) ──────────────────────────────


def _render_tools_for_llm() -> str:
    lines: list[str] = []
    for t in TOOLS:
        lines.append(f"Tool: {t.name}")
        lines.append(f"  Description: {t.description}")
        props = t.input_schema.get("properties", {})
        required = t.input_schema.get("required", [])
        if props:
            for pname, pschema in props.items():
                req_mark = " (required)" if pname in required else " (optional)"
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                default = pschema.get("default")
                default_str = f", default={default}" if default is not None else ""
                desc_str = f" — {pdesc}" if pdesc else ""
                lines.append(f"    {pname}: {ptype}{req_mark}{default_str}{desc_str}")
        lines.append(f"  Output keys: {t.output_keys}")
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a vocabulary maintenance agent. You manage a controlled vocabulary
of tags applied to ~700 engineering knowledge records.

YOUR GOAL: improve the network's quality (coherence, low hallucination)
through targeted modifications: split, refine, merge, deprecate.

YOU HAVE TOOLS to explore and act. On each step, you decide:
- (mode="call_tool"): call one tool to gather information
- (mode="decide"): output an AgentDecision (action + reasoning + certainty + evidence)

OPERATING PRINCIPLES:
1. A `=== Network triage ===` section at the top of context lists action candidates
   already prioritized by deterministic signals (coherence, size, neighbor overlap,
   confidence ratio). USE IT as your starting worklist — pick the top candidate
   whose suggested_action you can verify, do not re-discover candidates from scratch.
2. The triage's `suggested_action` is a starting hypothesis, not a verdict. Always
   call `inspect_tag` (and `propose_*_preview` if applicable) to verify samples
   match the hypothesis before deciding. You CAN override the suggestion when
   samples reveal a different pattern.
3. Skip candidates flagged `recently_modified` unless evidence is decisively new.
4. For high-impact actions (>30 records affected), ALWAYS call propose_*_preview first.
5. Preview proposals (propose_*_preview) BEFORE apply.
6. ONE PROPOSAL PER ROUND. Once a `propose_*_preview` call SUCCEEDS (returns a
   proposal_id), that subject is LOCKED for the round — you MUST decide on it
   (split/refine/deprecate/merge) this round; previewing any other subject will
   be REJECTED. A FAILED preview (returns `proposals=[]`) does NOT lock — you
   may try a different action or pick a different target.
7. If a `propose_*_preview` returns `proposals=[]` or `"no ... proposed"`, the
   judge has REJECTED this action. Do NOT call the same preview again — pick a
   different action or end with action="inspect_more" / "stop".
8. Output action="stop" when remaining triage candidates are all `suggestion_confidence=low`
   OR you've already addressed all high/medium-confidence candidates.

TOOL LIST:
{tools_description}

DECISION SCHEMA (use when mode="decide"):
- action: one of split / merge / refine / deprecate / inspect_more / stop
- target: tag name (or "" for stop)
- reasoning: 80-2000 chars
- certainty: high / medium / low
- supporting_observations: cite specific tool outputs
- opposing_observations: any reasons this might not work
- preview_reviewed: True only if you called propose_*_preview
- affected_records_estimate: how many records you expect to change
- expected_outcome: qualitative semantic prediction

OUTPUT ONE BrainStep PER TURN. Set mode to either "call_tool" or "decide".

NOTE: You do NOT have access to ground truth labels. Decisions must be based
on observable data, not optimization against any held-out evaluation set."""


# ── Helpers ───────────────────────────────────────────────────────────────


def _state_to_context(state: BrainState) -> BrainContext:
    """Build a BrainContext view over current state for tool dispatch.

    Tool implementations mutate ctx.proposal_cache and ctx._proposed_subject_this_round;
    after the call, tool_node extracts the diff and writes it back to state via reducer.
    """
    return BrainContext(
        db_path=Path(state["db_path"]),
        vocab=list(state["vocab"]),
        assignments=list(state["assignments"]),
        golden=[],  # ADR-0007: agent has no golden access
        proposal_cache=dict(state.get("proposal_cache", {})),
        _tool_cache_this_round=dict(state.get("tool_cache_this_round", {})),
        _proposed_subject_this_round=state.get("proposed_subject_this_round", frozenset()),
    )


def _state_to_memory_view(state: BrainState) -> AgentMemory:
    """Build a transient AgentMemory view for prompt rendering and critical-pattern detection.

    This stays a view — we never write back to it. NetworkState carries the
    triage report and tag_sizes; AgentMemory carries history/facts/working.
    """
    tag_sizes: dict[str, int] = {}
    for a in state["assignments"]:
        if a.get("missing"):
            continue
        for t in a.get("selected_tags", []):
            tag_sizes[t["name"]] = tag_sizes.get(t["name"], 0) + 1

    net = NetworkState(
        tag_sizes=tag_sizes,
        recent_operations={},
        total_records=sum(1 for a in state["assignments"] if not a.get("missing")),
        total_edges=sum(
            len(a.get("selected_tags", []))
            for a in state["assignments"]
            if not a.get("missing")
        ),
        triage_report=state.get("triage_report"),
    )
    mem = AgentMemory(initial_state=net)
    mem.history = list(state.get("history", []))
    mem.facts = list(state.get("facts", []))
    mem.working = list(state.get("working", []))
    mem.current_round_idx = state.get("current_round", 0)
    return mem


def _llm_model():
    settings = Settings()
    return _chat_model(settings).with_structured_output(BrainStep)


def _proposal_subject(proposal) -> frozenset:
    """Return the canonical subject for a proposal (set of tag names involved)."""
    ptype = getattr(proposal, "type", "")
    if ptype == "split" or ptype == "refine" or ptype == "deprecate":
        tag = getattr(proposal, "tag", "")
        return frozenset({tag}) if tag else frozenset()
    if ptype == "merge":
        return frozenset({getattr(proposal, "keep_tag", ""), getattr(proposal, "discard_tag", "")})
    return frozenset()


def _find_proposal_for_target(state: BrainState, target: str, action: str) -> str | None:
    expected_type = {"split": "split", "refine": "refine", "merge": "merge", "deprecate": "deprecate"}.get(action)
    if not expected_type:
        return None
    for pid, proposal in state.get("proposal_cache", {}).items():
        ptype = getattr(proposal, "type", "")
        if ptype != expected_type:
            continue
        ptag = getattr(proposal, "tag", "") or getattr(proposal, "keep_tag", "")
        if ptag == target:
            return pid
    return None


def _compute_real_affected(proposal, state: BrainState) -> int:
    ptype = getattr(proposal, "type", "")
    if ptype == "split":
        return sum(len(st.get("record_ids", [])) for st in getattr(proposal, "sub_tags", []))
    if ptype == "refine":
        return len(getattr(proposal, "prune_record_ids", []))
    if ptype == "deprecate":
        tag = getattr(proposal, "tag", "")
        return sum(
            1 for a in state["assignments"]
            if not a.get("missing") and any(t["name"] == tag for t in a.get("selected_tags", []))
        )
    if ptype == "merge":
        tag = getattr(proposal, "discard_tag", "")
        return sum(
            1 for a in state["assignments"]
            if not a.get("missing") and any(t["name"] == tag for t in a.get("selected_tags", []))
        )
    return 0


# ── Nodes ─────────────────────────────────────────────────────────────────


def orient_node(state: BrainState) -> dict:
    """Round-start: recompute triage with recently_modified set, clear round-local state."""
    log = get_default_logger()
    round_idx = state.get("current_round", 0)
    log.event("brain.round_start", round_idx=round_idx)

    try:
        modified_tags = {
            r.target for r in state.get("history", [])
            if r.committed and r.target
        }
        triage = compute_triage(
            state["vocab"], state["assignments"], Path(state["db_path"]),
            recently_modified=modified_tags,
        )
    except Exception as e:
        log.event("brain.triage_failed", round_idx=round_idx, error=str(e)[:200])
        triage = state.get("triage_report")

    return {
        "triage_report": triage,
        "round_tool_count": 0,
        "proposed_subject_this_round": frozenset(),
        "tool_cache_this_round": {},
        "working": [],
        "pending_decision": None,
        "pending_tool": None,
    }


def reason_node(state: BrainState) -> dict:
    """LLM emits one BrainStep: either call_tool or decide."""
    log = get_default_logger()
    round_idx = state.get("current_round", 0)

    mem = _state_to_memory_view(state)
    tools_description = _render_tools_for_llm()
    system_prompt = SYSTEM_PROMPT.format(tools_description=tools_description)
    user_msg = mem.render_for_llm() + "\n\nWhat is your next step?"

    t0 = time.perf_counter()
    step: BrainStep | None = invoke_with_retry(
        _llm_model(),
        [("system", system_prompt), ("user", user_msg)],
        retries=3,
        caller=f"brain.step.r{round_idx}.{state.get('round_tool_count', 0)}",
        logger=log,
    )
    elapsed_s = round(time.perf_counter() - t0, 2)

    if step is None:
        log.event("brain.llm_failed", round_idx=round_idx, elapsed_s=elapsed_s)
        return {"llm_call_count": 1, "pending_decision": None, "pending_tool": None,
                "stop_reason": "llm_failure"}

    log.event(
        "brain.llm_call",
        round_idx=round_idx,
        mode=step.mode,
        input_chars=len(system_prompt) + len(user_msg),
        output_chars=len(str(step.model_dump())),
        elapsed_s=elapsed_s,
    )

    if step.mode == "decide" and step.decision is not None:
        return {
            "llm_call_count": 1,
            "pending_decision": apply_reversibility_default(step.decision),
            "pending_tool": None,
        }
    if step.mode == "call_tool" and step.tool_name:
        return {
            "llm_call_count": 1,
            "pending_decision": None,
            "pending_tool": (step.tool_name, dict(step.tool_args or {})),
        }
    log.event("brain.invalid_step", round_idx=round_idx, mode=step.mode)
    return {"llm_call_count": 1, "pending_decision": None, "pending_tool": None}


def tool_node(state: BrainState) -> dict:
    """Execute the pending tool, update working + proposal_cache + locks."""
    log = get_default_logger()
    round_idx = state.get("current_round", 0)
    pending = state.get("pending_tool")
    if not pending:
        return {"pending_tool": None}
    tool_name, tool_args = pending

    tool_spec = TOOLS_BY_NAME.get(tool_name)
    if tool_spec is None:
        result = {"ok": False, "result": None, "error": f"unknown tool: {tool_name}"}
        log.event("brain.tool_call", round_idx=round_idx, tool=tool_name, args=tool_args,
                  result_summary="unknown tool", elapsed_s=0.0)
    else:
        ctx = _state_to_context(state)
        t1 = time.perf_counter()
        result = call_tool_with_cache(tool_spec, tool_args, ctx)
        tool_elapsed = round(time.perf_counter() - t1, 2)
        summary = (
            str(result.get("result", result.get("error", "")))[:120]
            if isinstance(result, dict) else str(result)[:120]
        )
        log.event("brain.tool_call", round_idx=round_idx, tool=tool_name, args=tool_args,
                  result_summary=summary, elapsed_s=tool_elapsed)

    # Capture tool result in working memory
    tr = ToolResult(
        tool=tool_name,
        args=tool_args,
        result=result,
        round_idx=round_idx,
        timestamp="",  # not used by view
    )

    update: dict = {
        "working": list(state.get("working", [])) + [tr],
        "round_tool_count": state.get("round_tool_count", 0) + 1,
        "pending_tool": None,
    }

    # If propose_* succeeded, extract new proposals + subject lock from ctx
    if tool_spec is not None:
        new_proposals = {
            pid: proposal for pid, proposal in ctx.proposal_cache.items()
            if pid not in state.get("proposal_cache", {})
        }
        if new_proposals:
            update["proposal_cache"] = {"add": new_proposals}
        update["proposed_subject_this_round"] = ctx._proposed_subject_this_round
        update["tool_cache_this_round"] = dict(ctx._tool_cache_this_round)

    return update


def forced_decide_node(state: BrainState) -> dict:
    """Tool budget exhausted without a decision — force LLM to decide now."""
    log = get_default_logger()
    round_idx = state.get("current_round", 0)

    mem = _state_to_memory_view(state)
    tools_description = _render_tools_for_llm()
    system_prompt = SYSTEM_PROMPT.format(tools_description=tools_description)
    forced_msg = (
        mem.render_for_llm()
        + "\n\nYou have used your tool budget for this round. "
          "You MUST now output mode=\"decide\" with an AgentDecision. "
          "Choose one action: split / merge / refine / deprecate / inspect_more / stop."
    )

    t0 = time.perf_counter()
    step: BrainStep | None = invoke_with_retry(
        _llm_model(),
        [("system", system_prompt), ("user", forced_msg)],
        retries=3,
        caller=f"brain.forced.r{round_idx}",
        logger=log,
    )
    elapsed_s = round(time.perf_counter() - t0, 2)

    if step is None or step.mode != "decide" or step.decision is None:
        log.event("brain.forced_decide_failed", round_idx=round_idx, elapsed_s=elapsed_s)
        return {"llm_call_count": 1, "pending_decision": None}

    log.event("brain.llm_call", round_idx=round_idx, mode="decide_forced",
              input_chars=len(system_prompt) + len(forced_msg),
              output_chars=len(str(step.model_dump())), elapsed_s=elapsed_s)
    return {"llm_call_count": 1,
            "pending_decision": apply_reversibility_default(step.decision)}


def verify_node(state: BrainState) -> dict:
    """Fact-override LLM-self-reported fields against state.

    Currently overrides:
      - preview_reviewed: True if a matching proposal exists in cache
      - affected_records_estimate: computed from the proposal (not LLM's guess)
    """
    decision = state.get("pending_decision")
    if decision is None:
        return {}
    if decision.action not in {"split", "refine", "merge", "deprecate"}:
        return {"pending_decision": decision}

    pid = _find_proposal_for_target(state, decision.target, decision.action)
    if pid is None:
        return {"pending_decision": decision}

    proposal = state["proposal_cache"][pid]
    real_affected = _compute_real_affected(proposal, state)
    verified = decision.model_copy(update={
        "preview_reviewed": True,
        "affected_records_estimate": real_affected,
    })
    return {"pending_decision": verified}


def gate_node(state: BrainState) -> dict:
    """Run 7-gate decision filter. Append (decision, gate) to decisions audit trail."""
    log = get_default_logger()
    round_idx = state.get("current_round", 0)
    decision = state.get("pending_decision")
    if decision is None:
        return {"pending_gate": None}

    mem = _state_to_memory_view(state)
    gate = gate_decision(decision, mem)
    log.event("brain.decision", round_idx=round_idx,
              decision=decision.model_dump(),
              gate={"result": gate.result.value, "triggered": gate.triggered_gates, "reason": gate.reason})
    return {"pending_gate": gate}


def apply_node(state: BrainState) -> dict:
    """If gate=AUTO and a matching proposal exists, apply it and mutate vocab/assignments."""
    from consolidate_agent.vocab_maintenance.apply import apply_proposal

    log = get_default_logger()
    round_idx = state.get("current_round", 0)
    decision = state["pending_decision"]
    gate = state["pending_gate"]

    pid = _find_proposal_for_target(state, decision.target, decision.action)
    if pid is None:
        # decide_without_proposal gate should have caught this — but keep as safety net.
        return {"pending_apply_outcome": "no_proposal_in_cache"}

    proposal = state["proposal_cache"][pid]
    try:
        new_vocab, new_assignments = apply_proposal(
            list(state["vocab"]), list(state["assignments"]), proposal
        )
        t = round(time.perf_counter(), 2)
        log.event("brain.apply", round_idx=round_idx,
                  proposal_id=pid, action=decision.action, target=decision.target,
                  changes={"vocab_before": len(state["vocab"]),
                           "vocab_after": len(new_vocab)},
                  elapsed_s=0.0)
        return {
            "vocab": new_vocab,
            "assignments": new_assignments,
            "proposal_cache": {"remove": [pid]},
            "applied_count": 1,
            "pending_apply_outcome": f"applied: vocab {len(state['vocab'])}→{len(new_vocab)} tags",
        }
    except Exception as e:
        log.event("brain.apply_failed", round_idx=round_idx, proposal_id=pid, error=str(e)[:300])
        return {"pending_apply_outcome": f"apply failed: {e}"}


def archive_node(state: BrainState) -> dict:
    """Write round outcome to history + decisions audit trail; bump round counter."""
    log = get_default_logger()
    round_idx = state.get("current_round", 0)
    decision = state.get("pending_decision")
    gate = state.get("pending_gate")
    apply_outcome = state.get("pending_apply_outcome")

    tools_called = [tr.tool for tr in state.get("working", [])]

    committed = False
    outcome_str: str | None = None
    if decision is not None and gate is not None:
        if gate.result == GateResult.REVIEW:
            outcome_str = f"gate=REVIEW ({gate.reason}), not applied"
        elif gate.result == GateResult.AUTO:
            if apply_outcome and apply_outcome.startswith("applied"):
                outcome_str = apply_outcome
                committed = True
            elif apply_outcome:
                outcome_str = apply_outcome
            else:
                outcome_str = f"gate=AUTO, action={decision.action} (no apply needed)"

    round_summary = RoundSummary(
        round_idx=round_idx,
        tools_called=tools_called,
        target=(decision.target if decision else None),
        decision_action=(decision.action if decision else None),
        decision_certainty=(decision.certainty if decision else None),
        outcome=outcome_str,
        committed=committed,
        rolled_back=False,
    )

    decision_record = None
    if decision is not None and gate is not None:
        run_id = state.get("run_id", "unknown")
        decision_record = {
            "round": round_idx,
            "decision_id": make_decision_id(run_id, round_idx, decision.action, decision.target),
            "decision": decision.model_dump(),
            "gate": {"result": gate.result.value, "triggered": gate.triggered_gates, "reason": gate.reason},
            "committed": committed,
            "outcome": outcome_str,
        }

    log.event("brain.round_end", round_idx=round_idx, summary=round_summary.compact())

    stop_reason: str | None = None
    if decision and decision.action == "stop":
        stop_reason = decision.stop_reason or "agent stopped"
        log.event("brain.stop", round_idx=round_idx, reason=stop_reason)

    update: dict = {
        "history": [round_summary],
        "current_round": round_idx + 1,
        "pending_decision": None,
        "pending_gate": None,
        "pending_apply_outcome": None,
    }
    if decision_record is not None:
        update["decisions"] = [decision_record]
    if stop_reason is not None:
        update["stop_reason"] = stop_reason
    return update


# ── Routing ───────────────────────────────────────────────────────────────


def route_after_reason(state: BrainState) -> Literal["tool", "verify", "forced_decide", "end"]:
    if state.get("stop_reason"):
        return "end"
    if state.get("llm_call_count", 0) >= state.get("cost_cap_calls", 60):
        return "forced_decide"
    if state.get("pending_decision") is not None:
        return "verify"
    if state.get("pending_tool") is not None:
        if state.get("round_tool_count", 0) >= state.get("max_tools_per_round", 5):
            return "forced_decide"
        return "tool"
    # LLM returned neither — force decide
    return "forced_decide"


def route_after_tool(state: BrainState) -> Literal["reason", "forced_decide"]:
    if state.get("round_tool_count", 0) >= state.get("max_tools_per_round", 5):
        return "forced_decide"
    if state.get("llm_call_count", 0) >= state.get("cost_cap_calls", 60):
        return "forced_decide"
    return "reason"


def route_after_forced_decide(state: BrainState) -> Literal["verify", "archive"]:
    if state.get("pending_decision") is not None:
        return "verify"
    return "archive"


def route_after_gate(state: BrainState) -> Literal["apply", "archive"]:
    gate = state.get("pending_gate")
    if gate is None:
        return "archive"
    if gate.result == GateResult.AUTO and state["pending_decision"].action in {
        "split", "refine", "merge", "deprecate"
    }:
        return "apply"
    return "archive"


def route_after_archive(state: BrainState) -> Literal["orient", "end"]:
    if state.get("stop_reason"):
        return "end"
    if state.get("current_round", 0) >= state.get("max_rounds", 10):
        return "end"
    if state.get("llm_call_count", 0) >= state.get("cost_cap_calls", 60):
        return "end"
    return "orient"


# ── Graph construction ────────────────────────────────────────────────────


def build_graph(checkpointer=None):
    g = StateGraph(BrainState)

    g.add_node("orient", orient_node)
    g.add_node("reason", reason_node)
    g.add_node("tool", tool_node)
    g.add_node("forced_decide", forced_decide_node)
    g.add_node("verify", verify_node)
    g.add_node("gate", gate_node)
    g.add_node("apply", apply_node)
    g.add_node("archive", archive_node)

    g.add_edge(START, "orient")
    g.add_edge("orient", "reason")
    g.add_conditional_edges(
        "reason",
        route_after_reason,
        {"tool": "tool", "verify": "verify", "forced_decide": "forced_decide", "end": END},
    )
    g.add_conditional_edges(
        "tool",
        route_after_tool,
        {"reason": "reason", "forced_decide": "forced_decide"},
    )
    g.add_conditional_edges(
        "forced_decide",
        route_after_forced_decide,
        {"verify": "verify", "archive": "archive"},
    )
    g.add_edge("verify", "gate")
    g.add_conditional_edges(
        "gate",
        route_after_gate,
        {"apply": "apply", "archive": "archive"},
    )
    g.add_edge("apply", "archive")
    g.add_conditional_edges(
        "archive",
        route_after_archive,
        {"orient": "orient", "end": END},
    )

    return g.compile(checkpointer=checkpointer)


# ── Public API ────────────────────────────────────────────────────────────


def run_brain_graph(
    *,
    db_path: Path,
    vocab: list[dict],
    assignments: list[dict],
    max_rounds: int = 10,
    max_tools_per_round: int = 5,
    cost_cap_calls: int = 60,
    logger: RunLogger | None = None,
) -> dict:
    """Drop-in replacement for the old run_brain_loop.

    Returns a result dict compatible with scripts/run_brain_spike.py format.
    """
    if logger is not None:
        from consolidate_agent.vocab_maintenance.observability import set_default_logger
        set_default_logger(logger)

    run_id = uuid.uuid4().hex[:12]

    initial: BrainState = {
        "db_path": str(db_path),
        "max_rounds": max_rounds,
        "max_tools_per_round": max_tools_per_round,
        "cost_cap_calls": cost_cap_calls,
        "run_id": run_id,
        "vocab": list(vocab),
        "assignments": list(assignments),
        "triage_report": None,
        "current_round": 0,
        "round_tool_count": 0,
        "proposed_subject_this_round": frozenset(),
        "tool_cache_this_round": {},
        "working": [],
        "proposal_cache": {},
        "pending_decision": None,
        "history": [],
        "facts": [],
        "decisions": [],
        "llm_call_count": 0,
        "applied_count": 0,
        "stop_reason": "",
    }

    graph = build_graph()
    # Allow many node visits — orient→reason→tool→reason loop alone can hit 30+ per run.
    final = graph.invoke(initial, config={"recursion_limit": 200})

    return {
        "rounds": final.get("current_round", 0),
        "decisions": list(final.get("decisions", [])),
        "final_memory": {
            "tag_count": len({t["name"] for t in final.get("vocab", [])}),
            "total_records": sum(1 for a in final.get("assignments", []) if not a.get("missing")),
            "history_count": len(final.get("history", [])),
            "facts_count": len(final.get("facts", [])),
        },
        "applied_count": final.get("applied_count", 0),
        "stop_reason": final.get("stop_reason") or "max_rounds",
        "total_llm_calls": final.get("llm_call_count", 0),
    }
