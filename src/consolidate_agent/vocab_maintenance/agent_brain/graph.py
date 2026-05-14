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
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
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


# ── Episodic memory store (module-level; set by build_graph) ──────────────


_episodic_store: BaseStore | None = None


_PROPOSE_ACTION_MAP = {
    "propose_split_preview": "split",
    "propose_refine_preview": "refine",
    "propose_deprecate_preview": "deprecate",
    "propose_merge_preview": "merge",
}


def _attempt_target_canonical(tool_name: str, args: dict) -> str:
    """Canonical string for an attempt subject so equal subjects match in set lookup."""
    if tool_name == "propose_merge_preview":
        a, b = args.get("tag_a", ""), args.get("tag_b", "")
        return ",".join(sorted([a, b]))
    return args.get("tag_name", "")


def _record_failed_attempt(run_id: str, action: str, target: str, reason: str, round_idx: int) -> None:
    if _episodic_store is None:
        return
    _episodic_store.put(
        ("attempts", run_id),
        f"{action}|{target}",
        {"action": action, "target": target, "reason": reason, "round": round_idx},
    )


def _read_excluded_attempts(run_id: str) -> set[tuple[str, str]]:
    if _episodic_store is None:
        return set()
    try:
        items = _episodic_store.search(("attempts", run_id))
    except Exception:
        return set()
    excluded: set[tuple[str, str]] = set()
    for item in items:
        v = item.value
        excluded.add((v["action"], v["target"]))
    return excluded


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

ACTION OVER INVESTIGATION:
Your tool budget per round is TIGHT (typically 3 tools). Spending the whole budget
on inspect_tag of multiple tags without calling propose_*_preview wastes the round —
the gate will REVIEW any decision without a cached proposal, producing zero commits.

A healthy round costs at most: inspect_tag (1) → propose_*_preview (1) → decide.
If triage gives a high-confidence suggestion, you MAY skip inspect and go directly
to propose_*_preview. Aim for at least one commit per round when triage shows
medium-or-better candidates available.

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
   The triage report marks such combinations with `suggestion_confidence=exhausted`
   and signal `tried_and_rejected` — skip them; the tool will also short-circuit
   if you try again, costing you a wasted slot in your tool budget.
8. Output action="stop" when remaining triage candidates are all `suggestion_confidence=low`
   OR you've already addressed all high/medium-confidence candidates.

APPLY BEHAVIOR (what the system actually does):
  refine:    changes target tag's DEFINITION + REMOVES target tag from prune-list records.
             Does NOT add records to any other tag.
  deprecate: REMOVES target tag from all its records. Does NOT redirect them.
  split:     splits target into sub-tags, reassigns records to sub-tags by similarity.
  merge:     records of discard_tag get keep_tag added; discard_tag is removed.

PREVIEW SHOWS YOU REAL DIFFS, NOT YOUR GUESS:
  Each `propose_*_preview` returns concrete before/after data:
    - target tag size before/after
    - for each prune/affected record: its tag set before vs after
    - orphan count (records that would lose their last matter tag)
  Read these facts and base your `reasoning` on them. There is no need to
  guess or describe "expected outcome" — the preview shows you the actual outcome
  if you commit.

TOOL LIST:
{tools_description}

DECISION SCHEMA (use when mode="decide"):
- action: one of split / merge / refine / deprecate / inspect_more / stop
- target: tag name (or "" for stop)
- reasoning: 80-2000 chars — cite specific facts from preview's before/after diff
- certainty: high / medium / low
- supporting_observations: cite specific tool outputs (preview diffs, inspect samples)
- opposing_observations: any reasons this might not work
- preview_reviewed: True only if you called propose_*_preview
- affected_records_estimate: how many records you expect to change

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
    """Round-start: recompute triage with recently_modified + excluded_attempts, clear round-local."""
    log = get_default_logger()
    round_idx = state.get("current_round", 0)
    run_id = state.get("run_id", "default")
    log.event("brain.round_start", round_idx=round_idx)

    excluded = _read_excluded_attempts(run_id)
    if excluded:
        log.event("brain.excluded_attempts", round_idx=round_idx, count=len(excluded),
                  attempts=[f"{a}({t})" for a, t in sorted(excluded)])

    try:
        modified_tags = {
            r.target for r in state.get("history", [])
            if r.committed and r.target
        }
        triage = compute_triage(
            state["vocab"], state["assignments"], Path(state["db_path"]),
            recently_modified=modified_tags,
            excluded_attempts=excluded,
        )
    except Exception as e:
        log.event("brain.triage_failed", round_idx=round_idx, error=str(e)[:200])
        triage = state.get("triage_report")

    return {
        "triage_report": triage,
        "excluded_attempts": excluded,
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
    """Execute the pending tool, update working + proposal_cache + locks.

    Short-circuits propose_*_preview calls whose (action, target) was already
    rejected by the judge earlier this run — returns cached failure instead
    of spending another LLM call on the same dead path.
    """
    log = get_default_logger()
    round_idx = state.get("current_round", 0)
    run_id = state.get("run_id", "default")
    pending = state.get("pending_tool")
    if not pending:
        return {"pending_tool": None}
    tool_name, tool_args = pending

    tool_spec = TOOLS_BY_NAME.get(tool_name)
    ctx = None  # set below if we actually invoke the tool
    short_circuited = False

    if tool_spec is None:
        result = {"ok": False, "result": None, "error": f"unknown tool: {tool_name}"}
        log.event("brain.tool_call", round_idx=round_idx, tool=tool_name, args=tool_args,
                  result_summary="unknown tool", elapsed_s=0.0)
    else:
        # Episodic-memory short-circuit: skip propose_*_preview for (action, target) already rejected.
        action = _PROPOSE_ACTION_MAP.get(tool_name)
        target_canon = _attempt_target_canonical(tool_name, tool_args) if action else ""
        excluded: set = state.get("excluded_attempts") or set()
        if action and (action, target_canon) in excluded:
            short_circuited = True
            result = {
                "ok": True,
                "result": {
                    "tag_name": target_canon, "proposals": [],
                    "message": (
                        f"already tried {action} on `{target_canon}` this run — "
                        f"judge rejected; will not retry. Pick a different action or target."
                    ),
                },
                "error": None,
            }
            log.event("brain.tool_short_circuit", round_idx=round_idx, tool=tool_name,
                      args=tool_args, action=action, target=target_canon)
        else:
            ctx = _state_to_context(state)
            t1 = time.perf_counter()
            result = call_tool_with_cache(tool_spec, tool_args, ctx)
            tool_elapsed = round(time.perf_counter() - t1, 2)
            # propose_* tools produce the proposal artifact we need for downstream
            # judge review — log the full result, not a 120-char preview.
            is_propose = tool_name.startswith("propose_") and tool_name.endswith("_preview")
            truncate = 4000 if is_propose else 120
            summary = (
                str(result.get("result", result.get("error", "")))[:truncate]
                if isinstance(result, dict) else str(result)[:truncate]
            )
            log.event("brain.tool_call", round_idx=round_idx, tool=tool_name, args=tool_args,
                      result_summary=summary, elapsed_s=tool_elapsed)

            # If propose_* returned no proposal, record the failure to episodic store
            if action:
                res = result.get("result") if isinstance(result, dict) else None
                if isinstance(res, dict) and res.get("proposals") == []:
                    reason = res.get("message", "judge rejected")
                    _record_failed_attempt(run_id, action, target_canon, reason, round_idx)
                    # Update local excluded set so subsequent tool calls in the same round see it.
                    new_excluded = set(excluded) | {(action, target_canon)}
                    state_update_excluded: set = new_excluded
                else:
                    state_update_excluded = excluded
            else:
                state_update_excluded = excluded

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

    # If we actually invoked a tool (ctx is set), extract proposal_cache and lock updates
    if tool_spec is not None and ctx is not None:
        new_proposals = {
            pid: proposal for pid, proposal in ctx.proposal_cache.items()
            if pid not in state.get("proposal_cache", {})
        }
        if new_proposals:
            update["proposal_cache"] = {"add": new_proposals}
        update["proposed_subject_this_round"] = ctx._proposed_subject_this_round
        update["tool_cache_this_round"] = dict(ctx._tool_cache_this_round)
        # Propagate excluded set updates (if propose failed, this round's later tool calls should see it)
        update["excluded_attempts"] = state_update_excluded

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


def forced_preview_node(state: BrainState) -> dict:
    """Tool budget exhausted with NO proposal in cache — force LLM to call one
    propose_*_preview now so the round can produce an applyable decision.

    This avoids the "predictably fail" pattern where budget runs out, LLM is
    forced to decide without preview, and gate REVIEWs the decision.
    """
    log = get_default_logger()
    round_idx = state.get("current_round", 0)

    mem = _state_to_memory_view(state)
    tools_description = _render_tools_for_llm()
    system_prompt = SYSTEM_PROMPT.format(tools_description=tools_description)
    forced_msg = (
        mem.render_for_llm()
        + "\n\nYou have used your tool budget for this round, BUT you have no "
          "proposal cached yet — any decide right now would be REVIEW-gated. "
          "You MUST call exactly ONE propose_*_preview tool for a target you "
          "want to act on next. Output mode=\"call_tool\" with one of: "
          "propose_split_preview / propose_refine_preview / propose_deprecate_preview / "
          "propose_merge_preview."
    )

    t0 = time.perf_counter()
    step: BrainStep | None = invoke_with_retry(
        _llm_model(),
        [("system", system_prompt), ("user", forced_msg)],
        retries=3,
        caller=f"brain.forced_preview.r{round_idx}",
        logger=log,
    )
    elapsed_s = round(time.perf_counter() - t0, 2)

    if step is None or step.mode != "call_tool" or not step.tool_name:
        log.event("brain.forced_preview_failed", round_idx=round_idx, elapsed_s=elapsed_s)
        return {"llm_call_count": 1, "pending_tool": None}

    log.event("brain.llm_call", round_idx=round_idx, mode="preview_forced",
              input_chars=len(system_prompt) + len(forced_msg),
              output_chars=len(str(step.model_dump())), elapsed_s=elapsed_s)
    return {
        "llm_call_count": 1,
        "pending_tool": (step.tool_name, dict(step.tool_args or {})),
    }


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
            # Clear ALL cached proposals — vocab/assignments just changed, any
            # surviving proposal from an earlier REVIEW would now be stale
            # (its prune_record_ids may reference records whose tags have
            # shifted). Force the LLM to re-propose against the fresh state.
            "proposal_cache": {"replace": {}},
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


def route_after_tool(state: BrainState) -> Literal["reason", "forced_preview", "forced_decide"]:
    max_tools = state.get("max_tools_per_round", 5)
    used = state.get("round_tool_count", 0)
    cost_cap = state.get("llm_call_count", 0) >= state.get("cost_cap_calls", 60)
    # Allow at most ONE extra tool slot for the forced_preview rescue path.
    over_budget = used > max_tools

    if cost_cap or over_budget:
        return "forced_decide"
    if used >= max_tools:
        # Budget exhausted at exactly max: if cache is empty, divert one slot
        # to forced_preview so the round produces an applyable decision.
        if not state.get("proposal_cache"):
            return "forced_preview"
        return "forced_decide"
    return "reason"


def route_after_forced_preview(state: BrainState) -> Literal["tool", "forced_decide"]:
    if state.get("pending_tool") is not None:
        return "tool"
    return "forced_decide"


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


# ── Plan-Execute architecture (ADR-0009) ──────────────────────────────────


class SpecialistVerdict(BaseModel):
    """Output schema for the specialist reviewer."""
    verdict: Literal["approve", "reject", "dig_deeper"]
    reasoning: str = Field(min_length=40, max_length=2000)
    cited_facts: list[str] = Field(default_factory=list, max_length=8)
    # When verdict=dig_deeper, which read-only tool to call.
    dig_tool: Literal["inspect_tag", "compare_tags", "inspect_records", ""] = ""
    dig_args: dict = Field(default_factory=dict)


class ForcedCommitVerdict(BaseModel):
    """Final verdict when dig budget is exhausted — must choose approve/reject."""
    verdict: Literal["approve", "reject"]
    reasoning: str = Field(min_length=40, max_length=2000)
    cited_facts: list[str] = Field(default_factory=list, max_length=8)


_SHARED_HEADER = """You are a specialist reviewer for a vocabulary maintenance system.

ROLE: You do NOT choose what to do. The plan stage already gave you a specific
(action, target) ticket. Your job: look at the proposal + preview diff +
sample record contents, decide approve / reject / dig_deeper.

TICKET:
  action: {action}
  target: {target}

PROPOSAL + PREVIEW DIFF (this is what would actually happen if you approve):
{preview_block}

DIG_DEEPER TOOLS (when verdict=dig_deeper):

  inspect_records(record_ids=[...])  ← PREFERRED for auditing specific records
    → return full insight text + current matter tags for up to 10 record_ids.
    → use when: you want to verify the ACTUAL CONTENT of records named in
      the preview (e.g., prune list, suspicious sub-tag samples). The
      preview only shows 500-char insight snippets — call inspect_records
      with the suspect IDs to see full content and confirm/refute your doubt.

  inspect_tag(tag_name, n_samples=20)
    → return that tag's definition + N sample record titles/insights.
    → use when: you want to see what records a DIFFERENT tag holds, e.g.
      a neighbor tag to verify it could absorb pruned records, or a
      sub-tag of a split to audit cohesion. Do NOT use to re-inspect
      target — its samples are already in the preview.

  compare_tags(tag_a, tag_b)
    → return overlap stats + shared records between two tags.
    → use when: testing if a merge candidate pair really share semantics,
      or if pruned records might naturally migrate to a neighbor.

CRITICAL RULES:
  - Each dig must ask a DIFFERENT question. Re-querying same (tool, args)
    returns "ALREADY_INSPECTED" — counts toward your 2-dig budget but
    yields no new info.
  - Max 2 digs per ticket; after that, output approve or reject based on
    current evidence.

OUTPUT REQUIREMENTS:
- cited_facts MUST reference concrete data from the preview (numbers, record
  titles, tag transitions). Generic claims like "looks coherent" are not facts.
- If verdict=dig_deeper, set dig_tool to "inspect_tag" or "compare_tags" and
  fill dig_args (e.g., {{"tag_name": "config_env"}} or {{"tag_a": "X", "tag_b": "Y"}}).
- Reject means: the proposal has clear semantic problems (records mis-assigned,
  off-topic prune list, sub-tag definitions don't match their records, etc).
"""


_REFINE_PROMPT = _SHARED_HEADER + """
REFINE — pattern recognition:

GOOD refine signals:
- prune list has a coherent off-topic theme (e.g., 8 records all about
  auth/security, target is supposed to be HTTP protocol → prune is sharpening).
- orphan_count = 0 (records keep other matter tags after losing this one).
- target_size_after still >= 15 (tag stays viable).
- new_definition's exclusions explicitly call out the prune theme.

BAD refine signals:
- prune list semantically diverse — multiple distinct sub-themes. This means
  the tag should SPLIT, not refine. Verdict: reject.
- orphan_count > 0 — records lose their only matter tag.
- target_size_after < 10 — tag becomes too small.
- prune records' insights look ON-topic to the original definition — new_def
  is over-narrowing.

DIG WALKTHROUGH (use this Chain-of-Thought when you're uncertain):

  Suppose preview shows 5 records to prune, and 2 of them have titles
  that don't obviously match an off-topic theme.

  Step 1 — inspect those exact records' full content:
    verdict=dig_deeper, dig_tool="inspect_records",
    dig_args={{"record_ids": ["knowledge_<suspect1>", "knowledge_<suspect2>"]}}
    ← reveals full insight text + their other current matter tags

  After Step 1 reading the result:
  • If full text confirms records are off-topic → approve
    cite: "inspect_records knowledge_xxx insight confirms record is about
    CLI auth, off-topic for relational persistence — prune is correct"
  • If full text shows records are on-topic → reject
    cite: "inspect_records knowledge_xxx is core to relational persistence
    (table design, schema migration) — should NOT be pruned"
  • If still unclear → Step 2 with a DIFFERENT question:

  Step 2 — check the records' fallback tag:
    verdict=dig_deeper, dig_tool="inspect_tag",
    dig_args={{"tag_name": "<one of their other tags>", "n_samples": 20}}
    ← see if their other matter tag really houses similar concepts

  After Step 2: you MUST output approve or reject (dig budget exhausted).

  KEY: each dig asks a DIFFERENT question. Never repeat same dig_args.
"""


_SPLIT_PROMPT = _SHARED_HEADER + """
SPLIT — pattern recognition:

GOOD split signals:
- Each sub-tag has clear, distinct semantic anchor (visible in sub-tag samples).
- sub-tag sizes balanced (none < 15 records).
- sub-tag sample_records' insights consistently match their sub-tag's definition.
- orphan_count = 0.

BAD split signals:
- One sub-tag has < 10 records (pseudo-split — refine the dominant sub-tag).
- sub-tag samples don't match their definition (LLM judge fabricated a sub-tag).
- sub-tag definitions overlap heavily (split axis unclear).
- The original tag wasn't heterogeneous; samples could have stayed together.

DIG WALKTHROUGH:

  Suppose one sub-tag's samples look unclear — its records' titles span
  several themes.

  Step 1 — read full content of those records:
    verdict=dig_deeper, dig_tool="inspect_records",
    dig_args={{"record_ids": [<3-5 ids from the suspicious sub-tag>]}}

  After Step 1:
  • Records consistently match the sub-tag's definition → approve sub-tag valid
  • Records are heterogeneous within the sub-tag → reject (sub-tag itself
    needs splitting, the split axis is wrong)
  • Two sub-tags look like they might overlap → Step 2:

  Step 2 — compare two sub-tags that seem fuzzy:
    verdict=dig_deeper, dig_tool="compare_tags",
    dig_args={{"tag_a": "<sub_A>", "tag_b": "<sub_B>"}}

  After Step 2: MUST approve or reject.

  KEY: each dig asks a DIFFERENT question. Never repeat same dig_args.
"""


_DEPRECATE_PROMPT = _SHARED_HEADER + """
DEPRECATE — pattern recognition:

GOOD deprecate signals:
- target_size < 15 (marginal mass).
- orphan_count = 0 (records have other matter tags to absorb semantics).
- target_records_sample: records' semantics better served by other matter tags.

BAD deprecate signals:
- orphan_count > 0.
- target_records_sample shows distinctive semantics not captured elsewhere.
- target_size > 20 — not actually marginal.

DIG WALKTHROUGH:

  Suppose target_size=12 and you want to verify the records' other tags
  truly cover the semantics.

  Step 1 — see full content of target's records:
    verdict=dig_deeper, dig_tool="inspect_records",
    dig_args={{"record_ids": [<all/most target_records ids>]}}

  After Step 1:
  • Records clearly fit another matter tag (visible in current_matter_tags)
    → approve
  • Records contain distinctive concepts not captured elsewhere → reject
  • Need to check whether candidate fallback tag really fits → Step 2:

  Step 2 — inspect the proposed fallback tag:
    verdict=dig_deeper, dig_tool="inspect_tag",
    dig_args={{"tag_name": "<candidate fallback>", "n_samples": 20}}

  After Step 2: MUST approve or reject.
"""


_MERGE_PROMPT = _SHARED_HEADER + """
MERGE — pattern recognition:

GOOD merge signals:
- discard_records belong under keep_tag's definition (or already have keep_tag).
- Two tags near-synonymous.
- High pre-existing overlap.

BAD merge signals:
- discard_records' semantics NOT covered by keep_tag — merge forces wrong concept.
- Two tags address different facets of same area (different concerns).
- discard_tag is large — dilutes keep_tag's specificity.

DIG WALKTHROUGH:

  Suppose the two tag names sound similar but you're not sure they truly
  belong under one concept.

  Step 1 — see records of the discard_tag in detail:
    verdict=dig_deeper, dig_tool="inspect_records",
    dig_args={{"record_ids": [<3-5 ids from discard_records_sample>]}}

  After Step 1:
  • Records' content clearly fits keep_tag's definition → approve
  • Records have a distinct angle that keep_tag can't absorb → reject
  • Still unsure, want to see overlap pattern → Step 2:

  Step 2 — compare the two tags directly:
    verdict=dig_deeper, dig_tool="compare_tags",
    dig_args={{"tag_a": "<keep_tag>", "tag_b": "<discard_tag>"}}

  After Step 2: MUST approve or reject.
"""


_PROMPT_BY_ACTION = {
    "refine": _REFINE_PROMPT,
    "split": _SPLIT_PROMPT,
    "deprecate": _DEPRECATE_PROMPT,
    "merge": _MERGE_PROMPT,
}


def _format_preview_block(state: BrainState) -> str:
    """Render the current ticket's preview/diff for the specialist prompt."""
    ticket = state.get("current_ticket")
    if not ticket:
        return "(no ticket)"
    target = ticket.get("target", "")
    action = ticket.get("action", "")
    # Find the cached proposal for this ticket
    pid = _find_proposal_for_target(state, target, action)
    if pid is None:
        return f"(no proposal in cache for {action} {target} — preview may have failed)"
    proposal = state["proposal_cache"][pid]
    # Try to get the latest preview result from working memory
    preview_result = None
    for tr in reversed(state.get("working", [])):
        if tr.tool.startswith("propose_") and tr.tool.endswith("_preview"):
            res = tr.result.get("result") if isinstance(tr.result, dict) else None
            if isinstance(res, dict) and (res.get("proposal_id") == pid or res.get("tag_name") == target):
                preview_result = res
                break
    if preview_result is None:
        return f"(proposal {pid} cached but no preview result in working memory)"
    # Render the diff in a readable form
    lines = [f"proposal_id: {pid}"]
    if "new_definition" in preview_result:
        lines.append(f"new_definition: {preview_result['new_definition']}")
    if "sub_tags" in preview_result:
        lines.append(f"sub_tags: {preview_result['sub_tags']}")
    if "keep_tag" in preview_result:
        lines.append(f"keep_tag: {preview_result['keep_tag']}, discard_tag: {preview_result.get('discard_tag')}")
    diff = preview_result.get("diff", {})
    if diff:
        lines.append(f"target_size_before: {diff.get('target_size_before')}")
        lines.append(f"target_size_after: {diff.get('target_size_after')}")
        lines.append(f"orphan_count: {diff.get('orphan_count')}")
        lines.append(f"affected_record_count: {diff.get('affected_record_count')}")
        record_diffs = diff.get("record_diffs", [])
        if record_diffs:
            lines.append(f"\nAffected records ({len(record_diffs)} shown):")
            for rd in record_diffs:
                lines.append(f"  id={rd.get('record_id')}")
                if rd.get("title"):
                    lines.append(f"     title: {rd['title']}")
                if rd.get("insight_snippet"):
                    lines.append(f"     insight: {rd['insight_snippet']}")
                lines.append(
                    f"     tags: {rd.get('before_tags')} → {rd.get('after_tags')}"
                    f"{'  [ORPHAN]' if rd.get('becomes_orphan') else ''}"
                )

    # Action-specific samples
    sub_tags = preview_result.get("sub_tags")
    if sub_tags:
        lines.append("\nSub-tag samples (split):")
        for st in sub_tags:
            lines.append(f"  --- {st.get('name')} (size {st.get('record_count')}) ---")
            lines.append(f"     definition: {st.get('definition','')}")
            for s in st.get("sample_records", []):
                lines.append(f"     • {s.get('title')} — {s.get('insight_snippet','')[:120]}")

    target_records_sample = preview_result.get("target_records_sample")
    if target_records_sample:
        lines.append("\nTarget records sample (deprecate):")
        for s in target_records_sample:
            lines.append(f"  • {s.get('title')} — {s.get('insight_snippet','')[:120]}")

    discard_records_sample = preview_result.get("discard_records_sample")
    if discard_records_sample:
        lines.append("\nDiscard tag records sample (merge):")
        for s in discard_records_sample:
            lines.append(f"  • {s.get('title')} — {s.get('insight_snippet','')[:120]}")

    return "\n".join(lines)


def plan_node(state: BrainState) -> dict:
    """Initial plan stage — seeds the plan_queue once at run start.

    For now, plan is hardcoded for spike testing (ADR-0009 step C). Later this
    will be replaced by a triage-driven or LLM-driven planner.
    """
    log = get_default_logger()
    # Hardcoded ticket list for spike v19 — covers all 4 action types
    hardcoded_plan = [
        {"action": "refine", "target": "persistence_db"},
        {"action": "split", "target": "langgraph_state"},
        {"action": "deprecate", "target": "build_deployment"},
        {"action": "refine", "target": "http_api"},
        {"action": "merge", "target": "langgraph_state", "target_b": "llm_agent_runtime"},
    ]
    log.event("brain.plan_seeded", count=len(hardcoded_plan),
              tickets=[f"{t['action']}({t['target']})" for t in hardcoded_plan])
    return {"plan_queue": hardcoded_plan}


def take_next_ticket_node(state: BrainState) -> dict:
    """Pop next ticket from plan_queue, reset per-ticket state."""
    log = get_default_logger()
    queue = list(state.get("plan_queue", []))
    if not queue:
        log.event("brain.plan_exhausted")
        return {"current_ticket": None, "stop_reason": "plan_exhausted"}
    ticket = queue[0]
    remaining = queue[1:]
    log.event("brain.take_ticket", ticket=ticket, remaining=len(remaining))
    return {
        "current_ticket": ticket,
        "plan_queue": remaining,
        "dig_deeper_count": 0,
        "prior_dig_keys": set(),
        "working": [],
        "tool_cache_this_round": {},
        "proposed_subject_this_round": frozenset(),
        "pending_decision": None,
        "pending_tool": None,
        "current_round": state.get("current_round", 0) + 1,
    }


def auto_preview_node(state: BrainState) -> dict:
    """For the current ticket, automatically call the matching propose_*_preview."""
    log = get_default_logger()
    ticket = state.get("current_ticket") or {}
    action = ticket.get("action", "")
    target = ticket.get("target", "")

    tool_map = {
        "refine": "propose_refine_preview",
        "split": "propose_split_preview",
        "deprecate": "propose_deprecate_preview",
        "merge": "propose_merge_preview",
    }
    tool_name = tool_map.get(action)
    if not tool_name:
        log.event("brain.auto_preview_unknown_action", action=action)
        return {"pending_apply_outcome": f"unknown action: {action}"}

    if action == "merge":
        args = {"tag_a": target, "tag_b": ticket.get("target_b", "")}
    else:
        args = {"tag_name": target}

    ctx = _state_to_context(state)
    t0 = time.perf_counter()
    tool_spec = TOOLS_BY_NAME[tool_name]
    result = call_tool_with_cache(tool_spec, args, ctx)
    elapsed = round(time.perf_counter() - t0, 2)

    res_dict = result.get("result") if isinstance(result, dict) else None
    is_success = isinstance(res_dict, dict) and res_dict.get("proposal_id")
    log.event("brain.auto_preview", tool=tool_name, args=args,
              success=bool(is_success), elapsed_s=elapsed)

    # Record this tool call in working memory (so specialist prompt can render diff)
    tr = ToolResult(tool=tool_name, args=args, result=result, round_idx=state.get("current_round", 0), timestamp="")

    update: dict = {
        "working": list(state.get("working", [])) + [tr],
        "tool_cache_this_round": dict(ctx._tool_cache_this_round),
    }
    # Pull in any new proposals into state's cache
    new_proposals = {
        pid: p for pid, p in ctx.proposal_cache.items()
        if pid not in state.get("proposal_cache", {})
    }
    if new_proposals:
        update["proposal_cache"] = {"add": new_proposals}

    # If propose failed (proposals=[]), record in episodic store for future runs
    if not is_success and isinstance(res_dict, dict) and res_dict.get("proposals") == []:
        action_for_store = _PROPOSE_ACTION_MAP.get(tool_name)
        target_canon = _attempt_target_canonical(tool_name, args)
        if action_for_store:
            _record_failed_attempt(
                state.get("run_id", "default"), action_for_store,
                target_canon, res_dict.get("message", "judge rejected"),
                state.get("current_round", 0),
            )
            update["excluded_attempts"] = set(state.get("excluded_attempts") or set()) | {(action_for_store, target_canon)}

    return update


def specialist_reason_node(state: BrainState) -> dict:
    """LLM specialist reviews the (ticket, preview, diff) and outputs a verdict.

    Selects the action-specific prompt (refine/split/deprecate/merge) so the
    reviewer sees the right GOOD/BAD pattern recognition guidance.
    """
    log = get_default_logger()
    ticket = state.get("current_ticket") or {}
    action = ticket.get("action", "")
    target = ticket.get("target", "")
    if action == "merge":
        target_display = f"{target} ↔ {ticket.get('target_b', '')}"
    else:
        target_display = target

    preview_block = _format_preview_block(state)

    prompt_template = _PROMPT_BY_ACTION.get(action)
    if prompt_template is None:
        log.event("brain.specialist_unknown_action", action=action)
        return {"llm_call_count": 0, "pending_verdict": None}

    user_msg = prompt_template.format(
        action=action,
        target=target_display,
        preview_block=preview_block,
    )

    settings = Settings()
    model = _chat_model(settings).with_structured_output(SpecialistVerdict)

    t0 = time.perf_counter()
    state_update_calls = state.get("llm_call_count", 0)
    verdict: SpecialistVerdict | None = invoke_with_retry(
        model,
        [("system", "You are a careful, factual reviewer."), ("user", user_msg)],
        retries=3,
        caller=f"brain.specialist.t{state.get('current_round', 0)}",
        logger=log,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    if verdict is None:
        log.event("brain.specialist_failed", elapsed_s=elapsed)
        return {"llm_call_count": 1, "pending_verdict": None}

    log.event("brain.specialist_verdict",
              ticket=ticket, verdict=verdict.verdict,
              cited_facts=verdict.cited_facts,
              elapsed_s=elapsed)
    return {"llm_call_count": 1, "pending_verdict": verdict}


def specialist_dig_inspect_node(state: BrainState) -> dict:
    """Specialist requested dig_deeper — execute one read-only inspect tool.

    Allowed dig tools: inspect_tag(tag_name) / compare_tags(tag_a, tag_b).
    Debounces same (tool, args) combos within a ticket — returning a hint
    instead of repeating the same query, which prevents the v20 R1 pattern
    where specialist kept asking inspect_tag(persistence_db) and getting
    the same 5 samples every time.
    """
    import json as _json
    log = get_default_logger()
    verdict = state.get("pending_verdict")
    if verdict is None:
        return {}

    allowed = {"inspect_tag", "compare_tags", "inspect_records"}
    dig_tool = verdict.dig_tool if verdict.dig_tool in allowed else "inspect_tag"
    if not verdict.dig_args:
        if dig_tool == "inspect_tag":
            dig_args = {"tag_name": (state.get("current_ticket") or {}).get("target", ""), "n_samples": 20}
        else:
            dig_args = {}
    else:
        dig_args = dict(verdict.dig_args)

    # Debounce: same (tool, args) within this ticket → return a hint, don't re-query
    args_key = f"{dig_tool}|{_json.dumps(dig_args, sort_keys=True, default=str)}"
    prior_keys = state.get("prior_dig_keys") or set()
    if args_key in prior_keys:
        log.event("brain.specialist_dig_debounced", tool=dig_tool, args=dig_args)
        hint = ToolResult(
            tool=dig_tool,
            args=dig_args,
            result={
                "ok": False,
                "result": None,
                "error": "ALREADY_INSPECTED",
                "message": (
                    f"You already queried {dig_tool}({dig_args}). Same args produce "
                    f"the same result — re-asking won't yield new info. Either query "
                    f"with DIFFERENT args (different tag / different pair), or output "
                    f"a verdict (approve/reject) based on current evidence."
                ),
            },
            round_idx=state.get("current_round", 0),
            timestamp="",
        )
        return {
            "working": list(state.get("working", [])) + [hint],
            "dig_deeper_count": state.get("dig_deeper_count", 0) + 1,
            "pending_verdict": None,
        }

    ctx = _state_to_context(state)
    tool_spec = TOOLS_BY_NAME.get(dig_tool)
    if tool_spec is None or not dig_args:
        log.event("brain.specialist_dig_invalid", dig_tool=dig_tool, dig_args=dig_args)
        return {
            "dig_deeper_count": state.get("dig_deeper_count", 0) + 1,
            "pending_verdict": None,
        }

    t0 = time.perf_counter()
    result = call_tool_with_cache(tool_spec, dig_args, ctx)
    elapsed = round(time.perf_counter() - t0, 2)

    log.event("brain.specialist_dig", tool=dig_tool, args=dig_args, elapsed_s=elapsed)

    tr = ToolResult(
        tool=dig_tool, args=dig_args, result=result,
        round_idx=state.get("current_round", 0), timestamp="",
    )
    return {
        "working": list(state.get("working", [])) + [tr],
        "dig_deeper_count": state.get("dig_deeper_count", 0) + 1,
        "prior_dig_keys": prior_keys | {args_key},
        "pending_verdict": None,
    }


def route_after_take_ticket(state: BrainState) -> Literal["auto_preview", "end"]:
    if state.get("current_ticket") is None:
        return "end"
    if state.get("current_round", 0) > state.get("max_rounds", 10):
        return "end"
    if state.get("llm_call_count", 0) >= state.get("cost_cap_calls", 60):
        return "end"
    return "auto_preview"


def route_after_auto_preview(state: BrainState) -> Literal["specialist", "skip_ticket"]:
    """If preview produced a usable proposal_id, go review; otherwise skip ticket."""
    ticket = state.get("current_ticket") or {}
    pid = _find_proposal_for_target(state, ticket.get("target", ""), ticket.get("action", ""))
    if pid is None:
        return "skip_ticket"
    return "specialist"


def route_after_specialist(state: BrainState) -> Literal["approve_path", "dig", "forced_commit", "skip_ticket"]:
    verdict = state.get("pending_verdict")
    if verdict is None:
        return "skip_ticket"
    if verdict.verdict == "approve":
        return "approve_path"
    if verdict.verdict == "dig_deeper":
        if state.get("dig_deeper_count", 0) >= 2:
            # Out of dig budget — give the specialist one last forced commit
            # call (cannot pick dig_deeper) rather than auto-rejecting.
            return "forced_commit"
        return "dig"
    return "skip_ticket"  # reject (explicit reject from specialist)


def forced_commit_node(state: BrainState) -> dict:
    """Dig budget exhausted but specialist still wants more info. Force a final
    approve/reject verdict — uncertainty is OK in reasoning, but no more digs.
    """
    log = get_default_logger()
    ticket = state.get("current_ticket") or {}
    action = ticket.get("action", "")
    target = ticket.get("target", "")
    if action == "merge":
        target_display = f"{target} ↔ {ticket.get('target_b', '')}"
    else:
        target_display = target

    preview_block = _format_preview_block(state)
    # Show working memory's dig results so the model has all gathered evidence
    working = state.get("working", [])
    dig_log_lines = []
    for tr in working:
        if tr.tool in {"inspect_records", "inspect_tag", "compare_tags"}:
            res = tr.result.get("result") if isinstance(tr.result, dict) else None
            dig_log_lines.append(f"  {tr.tool}({tr.args}) → {str(res)[:600]}")
    dig_log = "\n".join(dig_log_lines) if dig_log_lines else "(none)"

    forced_msg = f"""You are a specialist reviewer. You have exhausted your dig budget for this
ticket. You MUST output a final verdict: approve OR reject. dig_deeper is NOT
allowed.

TICKET:
  action: {action}
  target: {target_display}

PREVIEW + DIFF:
{preview_block}

YOUR PRIOR DIG RESULTS:
{dig_log}

INSTRUCTIONS:
- Decide approve or reject based on the evidence above.
- It is OK to acknowledge uncertainty in `reasoning`. The system requires
  a binary verdict, not absolute certainty.
- General heuristic: if no clear deal-breaker (orphan_count>0, samples
  contradicting the proposed definition, etc.) was uncovered, lean approve.
  If you found at least one concrete problem the proposal does not address,
  reject and cite it.
- cited_facts MUST reference concrete data from preview OR your dig results.

Output ForcedCommitVerdict: verdict (approve|reject), reasoning, cited_facts.
"""

    settings = Settings()
    model = _chat_model(settings).with_structured_output(ForcedCommitVerdict)
    t0 = time.perf_counter()
    forced: ForcedCommitVerdict | None = invoke_with_retry(
        model,
        [("system", "You are forced to deliver a final binary verdict."), ("user", forced_msg)],
        retries=3,
        caller=f"brain.forced_commit.t{state.get('current_round', 0)}",
        logger=log,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    if forced is None:
        log.event("brain.forced_commit_failed", elapsed_s=elapsed)
        return {"llm_call_count": 1, "pending_verdict": None}

    log.event("brain.forced_commit", ticket=ticket, verdict=forced.verdict,
              cited_facts=forced.cited_facts, elapsed_s=elapsed)
    # Convert to SpecialistVerdict shape so downstream nodes work unchanged
    verdict = SpecialistVerdict(
        verdict=forced.verdict, reasoning=forced.reasoning,
        cited_facts=forced.cited_facts, dig_tool="", dig_args={},
    )
    return {"llm_call_count": 1, "pending_verdict": verdict}


def route_after_forced_commit(state: BrainState) -> Literal["approve_path", "skip_ticket"]:
    verdict = state.get("pending_verdict")
    if verdict is None:
        return "skip_ticket"
    if verdict.verdict == "approve":
        return "approve_path"
    return "skip_ticket"


def approve_to_decision_node(state: BrainState) -> dict:
    """Translate specialist's 'approve' verdict into an AgentDecision so the
    existing verify/gate/apply pipeline can run unchanged."""
    ticket = state.get("current_ticket") or {}
    verdict = state.get("pending_verdict")
    decision = AgentDecision(
        action=ticket.get("action", "inspect_more"),
        target=ticket.get("target", ""),
        reasoning=(verdict.reasoning if verdict else "approved by specialist"),
        certainty="high",
        supporting_observations=(verdict.cited_facts if verdict and verdict.cited_facts else ["specialist approved"]),
        preview_reviewed=True,
        affected_records_estimate=0,
        reversibility=REVERSIBILITY_DEFAULTS.get(ticket.get("action", ""), "clean_rollback"),
    )
    return {"pending_decision": decision}


def ticket_archive_node(state: BrainState) -> dict:
    """Archive ticket outcome (replaces archive_node for plan-execute)."""
    log = get_default_logger()
    ticket = state.get("current_ticket") or {}
    verdict = state.get("pending_verdict")
    decision = state.get("pending_decision")
    gate = state.get("pending_gate")
    apply_outcome = state.get("pending_apply_outcome")

    tools_called = [tr.tool for tr in state.get("working", [])]

    committed = False
    outcome_str: str | None = None
    if verdict and verdict.verdict == "reject":
        outcome_str = f"specialist rejected: {verdict.reasoning[:100]}"
    elif verdict and verdict.verdict == "dig_deeper" and state.get("dig_deeper_count", 0) >= 2:
        outcome_str = "dig_deeper budget exhausted, ticket skipped"
    elif apply_outcome and apply_outcome.startswith("applied"):
        outcome_str = apply_outcome
        committed = True
    elif gate and gate.result == GateResult.REVIEW:
        outcome_str = f"gate=REVIEW ({gate.reason})"
    elif apply_outcome:
        outcome_str = apply_outcome
    else:
        outcome_str = "skipped (no preview or other reason)"

    round_summary = RoundSummary(
        round_idx=state.get("current_round", 0),
        tools_called=tools_called,
        target=ticket.get("target", "") or None,
        decision_action=ticket.get("action", "") or None,
        decision_certainty=("approved" if verdict and verdict.verdict == "approve" else
                            ("rejected" if verdict and verdict.verdict == "reject" else None)),
        outcome=outcome_str,
        committed=committed,
        rolled_back=False,
    )

    decision_record = {
        "round": state.get("current_round", 0),
        "ticket": ticket,
        "verdict": (verdict.model_dump() if verdict else None),
        "decision": (decision.model_dump() if decision else None),
        "gate": ({"result": gate.result.value, "triggered": gate.triggered_gates, "reason": gate.reason} if gate else None),
        "committed": committed,
        "outcome": outcome_str,
    }

    log.event("brain.ticket_done",
              ticket=ticket, committed=committed, outcome=outcome_str)

    return {
        "history": [round_summary],
        "decisions": [decision_record],
        "current_ticket": None,
        "pending_decision": None,
        "pending_gate": None,
        "pending_apply_outcome": None,
        "pending_verdict": None,
    }


# Helper import for approve_to_decision_node
from .decision import REVERSIBILITY_DEFAULTS


# ── Graph construction ────────────────────────────────────────────────────


def build_graph(checkpointer=None, store: BaseStore | None = None):
    """Plan-execute graph (ADR-0009).

    START → plan → take_next ──(queue empty/budget)──→ END
                       │
                       └→ auto_preview → specialist
                                         (verdict)
                                          │
              ┌───────────────────────────┼──────────────────┐
              │                           │                  │
           approve                    dig_deeper          reject
              │                           │                  │
        approve_to_decision         dig_inspect         (skip)
              │                           │                  │
            verify                  back to specialist      │
              │                                              │
             gate                                            │
              │                                              │
            apply                                            │
              │                                              │
              └──────────────→ ticket_archive ←──────────────┘
                                       │
                                       └→ take_next (loop)
    """
    global _episodic_store
    _episodic_store = store if store is not None else InMemoryStore()

    g = StateGraph(BrainState)

    g.add_node("plan", plan_node)
    g.add_node("take_next", take_next_ticket_node)
    g.add_node("auto_preview", auto_preview_node)
    g.add_node("specialist", specialist_reason_node)
    g.add_node("dig_inspect", specialist_dig_inspect_node)
    g.add_node("forced_commit", forced_commit_node)
    g.add_node("approve_to_decision", approve_to_decision_node)
    g.add_node("verify", verify_node)
    g.add_node("gate", gate_node)
    g.add_node("apply", apply_node)
    g.add_node("ticket_archive", ticket_archive_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "take_next")
    g.add_conditional_edges(
        "take_next",
        route_after_take_ticket,
        {"auto_preview": "auto_preview", "end": END},
    )
    g.add_conditional_edges(
        "auto_preview",
        route_after_auto_preview,
        {"specialist": "specialist", "skip_ticket": "ticket_archive"},
    )
    g.add_conditional_edges(
        "specialist",
        route_after_specialist,
        {"approve_path": "approve_to_decision", "dig": "dig_inspect",
         "forced_commit": "forced_commit", "skip_ticket": "ticket_archive"},
    )
    g.add_edge("dig_inspect", "specialist")  # back to reasoning
    g.add_conditional_edges(
        "forced_commit",
        route_after_forced_commit,
        {"approve_path": "approve_to_decision", "skip_ticket": "ticket_archive"},
    )
    g.add_edge("approve_to_decision", "verify")
    g.add_edge("verify", "gate")
    g.add_conditional_edges(
        "gate",
        route_after_gate,
        {"apply": "apply", "archive": "ticket_archive"},
    )
    g.add_edge("apply", "ticket_archive")
    g.add_edge("ticket_archive", "take_next")

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
        "excluded_attempts": set(),
        "llm_call_count": 0,
        "applied_count": 0,
        "stop_reason": "",
        # plan-execute (ADR-0009)
        "plan_queue": [],
        "current_ticket": None,
        "dig_deeper_count": 0,
        "prior_dig_keys": set(),
        "pending_verdict": None,
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
        # Full final network state — caller decides whether to persist. Prefixed
        # with _ so it can be popped before serializing the summary JSON
        # (which shouldn't carry 696 records).
        "_final_vocab": list(final.get("vocab", [])),
        "_final_assignments": list(final.get("assignments", [])),
    }
