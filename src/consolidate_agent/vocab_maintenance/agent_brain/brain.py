"""LLM brain loop for vocabulary maintenance.

Implements a multi-round agent that uses tools to explore the tag network
and outputs structured AgentDecision objects.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

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
from .memory import AgentMemory, NetworkState, RoundSummary
from .tools import TOOLS, TOOLS_BY_NAME, BrainContext, call_tool_with_cache
from .triage import compute_triage


def render_tools_for_llm() -> str:
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


class BrainStep(BaseModel):
    mode: str  # "call_tool" or "decide"

    tool_name: str = ""
    tool_args: dict = Field(default_factory=dict)
    reason: str = ""

    decision: AgentDecision | None = None


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
   samples reveal a different pattern (e.g., triage says "refine" but samples
   show two distinct clusters → propose split instead).
3. Skip candidates flagged `recently_modified` unless evidence is decisively new.
4. For high-impact actions (>30 records affected), ALWAYS call propose_*_preview first.
5. Preview proposals (propose_*_preview) BEFORE apply.
6. ONE PROPOSAL PER ROUND. Once a `propose_*_preview` call SUCCEEDS (returns a
   proposal_id), that subject is LOCKED for the round — you MUST decide on it
   (split/refine/deprecate/merge) this round; previewing any other subject will
   be REJECTED. A FAILED preview (returns `proposals=[]` or `"no ... proposed"`)
   does NOT lock — you may try a different action or pick a different target.
   Do not batch-preview multiple candidates "to compare" — finish the
   inspect→propose→decide loop for one target per round.
7. If a `propose_*_preview` returns `proposals=[]` or `message: "no ... proposed"`,
   the judge has REJECTED this action for this tag. Do NOT call the same preview again —
   either pick a different action (e.g., merge instead of deprecate), or end with
   action="inspect_more" / "stop" and move to the next triage candidate next round.
8. Output action="stop" when remaining triage candidates are all `suggestion_confidence=low`
   OR you've already addressed all high/medium-confidence candidates.

If you see CRITICAL PATTERNS at the top of context, follow their guidance EXACTLY.
Repeating a blocked action will fail again — try the suggested alternative.

WORKED EXAMPLE 1 — follow triage suggestion:

Round X (triage suggests refine for http_api, confidence=medium):
  step 1: inspect_tag(http_api) → samples cluster around HTTP semantics, with
          ~8 stragglers about MySQL auth
  step 2: propose_refine_preview(http_api) → new_definition tightens scope, prune_count=8
  step 3: decide(action="refine", target="http_api", certainty="high",
                 preview_reviewed=True,
                 supporting_observations=["triage.signals=[low_coh,high_overlap]",
                                          "inspect_tag samples: ~8 off-topic stragglers",
                                          "propose_refine_preview.prune_count=8"])
  → gate AUTO

WORKED EXAMPLE 2 — override triage suggestion:

Round X (triage suggests refine for filesystem_path, confidence=medium):
  step 1: inspect_tag(filesystem_path) → 12 samples reveal TWO distinct clusters:
          one about path manipulation, another about directory-as-architecture
  step 2: propose_split_preview(filesystem_path) → split axis: "path_manipulation"
          vs "project_structure_inference", new tags well-balanced
  step 3: decide(action="split", target="filesystem_path", certainty="high",
                 preview_reviewed=True,
                 supporting_observations=["triage suggested refine, but samples reveal",
                                          "two distinct semantic clusters",
                                          "propose_split_preview shows balanced split"],
                 opposing_observations=["triage's high_overlap signal alone hinted at refine"])
  → gate AUTO; triage suggestion overridden with evidence.

NOTE: if you skip the propose_*_preview step, gate will BLOCK on
"high certainty without preview review". Always preview before high-certainty action.

CERTAINTY CALIBRATION TABLE:
  high: preview_reviewed=True AND >=3 supporting observations from tool outputs
  medium: preview_reviewed=True OR 2 supporting observations
  low: insufficient evidence (1 obs, no preview) — typically a sign you need inspect_more

TOOL LIST:
{tools_description}

DECISION SCHEMA (use when mode="decide"):
- action: one of split / merge / refine / deprecate / inspect_more / stop
- target: tag name (or "" for stop)
- reasoning: 80-2000 chars, explain WHY this action
- certainty: high / medium / low — be honest about uncertainty
- supporting_observations: cite specific tool outputs (e.g., "inspect_tag(http_api).coherence=0.39")
- opposing_observations: any reasons this might not work
- preview_reviewed: True only if you called propose_*_preview
- affected_records_estimate: how many records you expect to change
- reversibility: leave default (auto-filled) unless special case
- expected_outcome: qualitative semantic prediction (e.g., "http_api should split into 3 cleaner sub-concepts: auth, streaming, routing")

OUTPUT ONE BrainStep PER TURN. Set mode to either "call_tool" or "decide".
Memory will be shown to you so you remember past tool calls.

NOTE ON EVIDENCE: You have probing tools that show real network data
(record content, tag definitions, embeddings). You do NOT have access to
ground truth labels. Your decisions must be based on the data you can
inspect, not on optimization against any held-out evaluation set."""


def _build_initial_state(context: BrainContext) -> NetworkState:
    tag_sizes: dict[str, int] = {}
    for a in context.assignments:
        if a.get("missing"):
            continue
        for t in a.get("selected_tags", []):
            tag_sizes[t["name"]] = tag_sizes.get(t["name"], 0) + 1

    try:
        triage_report = compute_triage(context.vocab, context.assignments, context.db_path)
    except Exception:
        triage_report = None

    return NetworkState(
        tag_sizes=tag_sizes,
        recent_operations={},
        total_records=sum(1 for a in context.assignments if not a.get("missing")),
        total_edges=sum(
            len(a.get("selected_tags", []))
            for a in context.assignments
            if not a.get("missing")
        ),
        triage_report=triage_report,
    )


def run_brain_loop(
    context: BrainContext,
    *,
    max_rounds: int = 10,
    max_tools_per_round: int = 5,
    cost_cap_calls: int = 60,
    logger: RunLogger | None = None,
) -> dict:
    log = logger or get_default_logger()
    run_id = uuid.uuid4().hex[:12]

    memory = AgentMemory(initial_state=_build_initial_state(context))
    decisions_log: list[dict] = []
    llm_call_count = 0
    applied_count = 0
    stop_reason = "max_rounds"

    settings = Settings()
    model = _chat_model(settings).with_structured_output(BrainStep)
    tools_description = render_tools_for_llm()
    system_prompt = SYSTEM_PROMPT.format(tools_description=tools_description)

    while memory.current_round_idx < max_rounds:
        round_idx = memory.current_round_idx
        log.event("brain.round_start", round_idx=round_idx)

        round_tool_names: list[str] = []
        decision: AgentDecision | None = None
        gate: GateDecision | None = None
        round_tools = 0
        committed = False
        applied_changes: dict | None = None

        while round_tools < max_tools_per_round:
            if llm_call_count >= cost_cap_calls:
                log.event("brain.cost_cap_hit", count=llm_call_count)
                stop_reason = "cost cap"
                break

            user_msg = memory.render_for_llm() + "\n\nWhat is your next step?"
            input_chars = len(system_prompt) + len(user_msg)

            t0 = time.perf_counter()
            llm_call_count += 1
            step: BrainStep | None = invoke_with_retry(
                model,
                [("system", system_prompt), ("user", user_msg)],
                retries=3,
                caller=f"brain.step.r{round_idx}.{round_tools}",
                logger=log,
            )
            elapsed_s = round(time.perf_counter() - t0, 2)

            if step is None:
                log.event(
                    "brain.llm_failed",
                    round_idx=round_idx,
                    tool_call_idx=round_tools,
                    elapsed_s=elapsed_s,
                )
                break

            output_chars = len(str(step.model_dump()))
            log.event(
                "brain.llm_call",
                round_idx=round_idx,
                mode=step.mode,
                input_chars=input_chars,
                output_chars=output_chars,
                elapsed_s=elapsed_s,
            )

            if step.mode == "decide":
                if step.decision is None:
                    log.event("brain.llm_failed", round_idx=round_idx, reason="decide mode but decision=None")
                    break
                decision = apply_reversibility_default(step.decision)
                decision = _verify_decision_against_cache(decision, context)
                gate = gate_decision(decision, memory)
                log.event(
                    "brain.decision",
                    round_idx=round_idx,
                    decision=decision.model_dump(),
                    gate={"result": gate.result.value, "triggered": gate.triggered_gates, "reason": gate.reason},
                )
                break

            # mode == "call_tool"
            tool_name = step.tool_name
            tool_args = step.tool_args
            tool_spec = TOOLS_BY_NAME.get(tool_name)

            if tool_spec is None:
                result = {"ok": False, "result": None, "error": f"unknown tool: {tool_name}"}
                memory.record_tool_call(tool_name, tool_args, result)
                round_tool_names.append(tool_name)
                log.event(
                    "brain.tool_call",
                    round_idx=round_idx,
                    tool=tool_name,
                    args=tool_args,
                    result_summary="unknown tool",
                    elapsed_s=0.0,
                )
                round_tools += 1
                continue

            t1 = time.perf_counter()
            result = call_tool_with_cache(tool_spec, tool_args, context)
            tool_elapsed = round(time.perf_counter() - t1, 2)

            result_summary = (
                str(result.get("result", result.get("error", "")))[:120]
                if isinstance(result, dict)
                else str(result)[:120]
            )
            log.event(
                "brain.tool_call",
                round_idx=round_idx,
                tool=tool_name,
                args=tool_args,
                result_summary=result_summary,
                elapsed_s=tool_elapsed,
            )

            memory.record_tool_call(tool_name, tool_args, result)
            round_tool_names.append(tool_name)
            round_tools += 1

        # ── forced decide when tool budget exhausted without a decision ───────
        if decision is None and llm_call_count < cost_cap_calls:
            forced_user_msg = (
                memory.render_for_llm()
                + "\n\nYou have used your tool budget for this round. "
                  "You MUST now output mode=\"decide\" with an AgentDecision. "
                  "Choose one action: split / merge / refine / deprecate / inspect_more / stop."
            )
            input_chars = len(system_prompt) + len(forced_user_msg)
            t0 = time.perf_counter()
            llm_call_count += 1
            forced_step: BrainStep | None = invoke_with_retry(
                model,
                [("system", system_prompt), ("user", forced_user_msg)],
                retries=3,
                caller=f"brain.forced_decide.r{round_idx}",
                logger=log,
            )
            elapsed_s = round(time.perf_counter() - t0, 2)
            if forced_step is not None and forced_step.mode == "decide" and forced_step.decision is not None:
                decision = apply_reversibility_default(forced_step.decision)
                decision = _verify_decision_against_cache(decision, context)
                gate = gate_decision(decision, memory)
                log.event(
                    "brain.llm_call",
                    round_idx=round_idx,
                    mode="decide",
                    input_chars=input_chars,
                    output_chars=len(str(forced_step.model_dump())),
                    elapsed_s=elapsed_s,
                )
                log.event(
                    "brain.decision",
                    round_idx=round_idx,
                    decision=decision.model_dump(),
                    gate={"result": gate.result.value, "triggered": gate.triggered_gates, "reason": gate.reason},
                )
            else:
                log.event(
                    "brain.forced_decide_failed",
                    round_idx=round_idx,
                    elapsed_s=elapsed_s,
                )

        # ── post-round: handle decision and gate ──────────────────────────────
        outcome_str: str | None = None

        if decision is not None and gate is not None:
            decision_id = make_decision_id(run_id, round_idx, decision.action, decision.target)

            if gate.result == GateResult.REVIEW:
                log.event(
                    "brain.gate_review",
                    round_idx=round_idx,
                    decision_id=decision_id,
                    gates=gate.triggered_gates,
                    reason=gate.reason,
                    note="spike: would have escalated, skipping apply",
                )
                outcome_str = f"gate=REVIEW ({gate.reason}), not applied"

            elif gate.result == GateResult.AUTO and decision.action in {"split", "refine", "merge", "deprecate"}:
                # Find the proposal in cache matching target tag and action type
                proposal_id = _find_proposal_for_target(context, decision.target, decision.action)

                if proposal_id and proposal_id in context.proposal_cache:
                    t2 = time.perf_counter()
                    apply_result = _do_apply(proposal_id, context, log)
                    apply_elapsed = round(time.perf_counter() - t2, 2)

                    if apply_result["ok"]:
                        applied_count += 1
                        committed = True
                        r = apply_result["result"]
                        outcome_str = (
                            f"applied: vocab {r['vocab_before']}→{r['vocab_after']} tags"
                        )
                        applied_changes = {
                            "added_tags": _infer_added_tags(decision, r),
                            "removed_tags": _infer_removed_tags(decision, r),
                            "modified_tags": [],
                        }
                        log.event(
                            "brain.apply",
                            round_idx=round_idx,
                            proposal_id=proposal_id,
                            action=decision.action,
                            target=decision.target,
                            changes=apply_result["result"],
                            elapsed_s=apply_elapsed,
                        )
                    else:
                        outcome_str = f"apply failed: {apply_result['error']}"
                        log.event(
                            "brain.apply_failed",
                            round_idx=round_idx,
                            proposal_id=proposal_id,
                            error=apply_result["error"],
                        )
                else:
                    outcome_str = "gate=AUTO but no proposal in cache, skipping apply"
                    log.event(
                        "brain.no_proposal_to_apply",
                        round_idx=round_idx,
                        action=decision.action,
                        target=decision.target,
                    )

            elif gate.result == GateResult.AUTO:
                outcome_str = f"gate=AUTO, action={decision.action} (no apply needed)"

            decisions_log.append({
                "round": round_idx,
                "decision_id": decision_id,
                "decision": decision.model_dump(),
                "gate": {"result": gate.result.value, "triggered": gate.triggered_gates, "reason": gate.reason},
                "committed": committed,
                "outcome": outcome_str,
            })

        # ── archive round ─────────────────────────────────────────────────────
        round_summary = RoundSummary(
            round_idx=round_idx,
            tools_called=round_tool_names,
            target=(decision.target if decision else None),
            decision_action=(decision.action if decision else None),
            decision_certainty=(decision.certainty if decision else None),
            outcome=outcome_str,
            committed=committed,
            rolled_back=False,
        )
        context._tool_cache_this_round.clear()
        context._proposed_subject_this_round = frozenset()
        memory.archive_round(round_summary, applied_changes=applied_changes)

        # Recompute triage with updated recently_modified set, so next round
        # sees fresh signals and cooldown flags on tags we just touched.
        try:
            modified_tags = {
                r.target for r in memory.history
                if r.committed and r.target
            }
            memory.state.triage_report = compute_triage(
                context.vocab, context.assignments, context.db_path,
                recently_modified=modified_tags,
            )
        except Exception as e:
            log.event("brain.triage_recompute_failed", round_idx=round_idx, error=str(e)[:200])

        log.event(
            "brain.round_end",
            round_idx=round_idx,
            summary=round_summary.compact(),
        )

        if decision and decision.action == "stop":
            stop_reason = decision.stop_reason or "agent stopped"
            log.event("brain.stop", round_idx=round_idx, reason=stop_reason)
            break

        if stop_reason == "cost cap":
            break

    log.event("brain.stop", round_idx=memory.current_round_idx, reason=stop_reason)

    return {
        "rounds": memory.current_round_idx,
        "decisions": decisions_log,
        "final_memory": {
            "tag_count": len(memory.state.tag_sizes),
            "total_records": memory.state.total_records,
            "history_count": len(memory.history),
            "facts_count": len(memory.facts),
        },
        "applied_count": applied_count,
        "stop_reason": stop_reason,
        "total_llm_calls": llm_call_count,
    }


def _compute_real_affected(proposal, context: BrainContext) -> int:
    """Compute true affected_records from proposal data (overrides LLM's estimate)."""
    ptype = getattr(proposal, "type", "")
    if ptype == "split":
        return sum(len(st.get("record_ids", [])) for st in getattr(proposal, "sub_tags", []))
    if ptype == "refine":
        return len(getattr(proposal, "prune_record_ids", []))
    if ptype == "deprecate":
        tag = getattr(proposal, "tag", "")
        return sum(
            1 for a in context.assignments
            if not a.get("missing") and any(t["name"] == tag for t in a.get("selected_tags", []))
        )
    if ptype == "merge":
        tag = getattr(proposal, "discard_tag", "")
        return sum(
            1 for a in context.assignments
            if not a.get("missing") and any(t["name"] == tag for t in a.get("selected_tags", []))
        )
    return 0


def _verify_decision_against_cache(decision: AgentDecision, context: BrainContext) -> AgentDecision:
    """Override LLM self-reported preview_reviewed and affected_records_estimate
    with facts from proposal_cache. Closes the bug where LLM previews a tag,
    receives a proposal_id, then reports preview_reviewed=False in the decide
    schema and gets blocked by gate.
    """
    if decision.action not in {"split", "refine", "merge", "deprecate"}:
        return decision
    pid = _find_proposal_for_target(context, decision.target, decision.action)
    if pid is None:
        return decision
    proposal = context.proposal_cache.get(pid)
    if proposal is None:
        return decision
    real_affected = _compute_real_affected(proposal, context)
    return decision.model_copy(update={
        "preview_reviewed": True,
        "affected_records_estimate": real_affected,
    })


def _find_proposal_for_target(context: BrainContext, target: str, action: str) -> str | None:
    """Return a proposal_id from cache whose tag matches target and type matches action.

    No fallback: if no exact (target, action) match exists, return None. The
    decision will then be blocked by the decide_without_proposal gate rather
    than apply some stale proposal from a different tag.
    """
    action_type_map = {"split": "split", "refine": "refine", "merge": "merge", "deprecate": "deprecate"}
    expected_type = action_type_map.get(action)
    if not expected_type:
        return None
    for pid, proposal in list(context.proposal_cache.items()):
        ptype = getattr(proposal, "type", "")
        ptag = getattr(proposal, "tag", "") or getattr(proposal, "keep_tag", "")
        if ptype != expected_type:
            continue
        if ptag == target:
            return pid
    return None


def _do_apply(proposal_id: str, context: BrainContext, log: RunLogger) -> dict:
    from consolidate_agent.vocab_maintenance.apply import apply_proposal

    proposal = context.proposal_cache.get(proposal_id)
    if proposal is None:
        return {"ok": False, "result": None, "error": f"proposal {proposal_id} not in cache"}

    try:
        old_vocab_count = len(context.vocab)
        new_vocab, new_assignments = apply_proposal(context.vocab, context.assignments, proposal)
        context.vocab.clear()
        context.vocab.extend(new_vocab)
        context.assignments.clear()
        context.assignments.extend(new_assignments)
        del context.proposal_cache[proposal_id]
        return {"ok": True, "result": {"vocab_before": old_vocab_count, "vocab_after": len(context.vocab)}, "error": None}
    except Exception as e:
        log.event("brain.apply_exception", proposal_id=proposal_id, error=str(e)[:300])
        return {"ok": False, "result": None, "error": str(e)}


def _infer_added_tags(decision: AgentDecision, apply_result: dict) -> list[str]:
    if decision.action == "split":
        # sub-tag names aren't available here; return empty (state updated in memory via archive)
        return []
    return []


def _infer_removed_tags(decision: AgentDecision, apply_result: dict) -> list[str]:
    if decision.action in {"split", "deprecate"}:
        return [decision.target] if decision.target else []
    return []
