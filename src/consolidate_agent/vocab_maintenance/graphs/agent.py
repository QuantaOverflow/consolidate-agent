"""Agent loop as a LangGraph StateGraph.

Topology:
  START
    │
    ↓
  initial_measure
    │
    ↓
  termination_check ────────────── (max_iter / no_actions_remaining) ──→ END
    │
    ↓
  diagnose
    │
    ↓
  route_action ──── done ────→ finalize_completed ──→ END
                ──── blocked ──→ record_iter ──┐
                ──── unknown ──→ record_iter ──┤
                ──── propose_X ──→ propose ──→ route_proposals ─── empty ──→ record_iter ──┤
                                                              └─ has  ──→ apply ──→ route_apply
                                                                                          │
                                                                                          ├─ error ──→ record_iter ──┤
                                                                                          └─ ok ────→ post_apply ──→ llm_judge ──→ route_judge
                                                                                                                                              │
                                                                                                                                              ├─ rollback → record_iter ─┤
                                                                                                                                              └─ commit  → record_iter ─┤
                                                                                                                                                            │
                                                                                                                                                            ↓
                                                                                                                                          termination_check (next iter)

Persistence: every node boundary is a SqliteSaver checkpoint, so crash-then-
resume picks up at the last completed node.

Equivalence to the prior pure-Python Agent class: state transitions are
identical; only the orchestration substrate changed.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from langgraph.graph import StateGraph, START, END

from ..apply import (
    InvalidProposal,
    OrphanError,
    apply_proposal,
    check_invariants,
)
from ..network import emit_apply_event
from ..observability import get_default_logger
from .state import AgentLoopState


ACTIONS = ("propose_new", "propose_merge", "propose_deprecate")
_DIM_NAMES = ("coverage", "coherence", "distinctness", "granularity", "multi_axis")
_ZERO_DELTA = {k: 0.0 for k in _DIM_NAMES}


def _last_metrics(state: dict) -> dict[str, float]:
    """Most recent metrics snapshot from metrics_history; empty when disabled."""
    hist = state.get("metrics_history") or []
    return hist[-1]["metrics"] if hist else {}


def _diff_metrics(new: dict[str, float], prior: dict[str, float]) -> dict[str, float]:
    if not prior:
        return dict(_ZERO_DELTA)
    return {k: round(new.get(k, 0.0) - prior.get(k, 0.0), 4) for k in _DIM_NAMES}


def _hit_rate(diagnostics: dict) -> float:
    n = diagnostics.get("sample_size", 0)
    if n == 0:
        return 0.0
    return diagnostics.get("total_assigned", 0) / n


def _new_iter_record(iter_idx: int, hit_before: float) -> dict:
    return {
        "iter": iter_idx,
        "action": "",
        "decision": None,
        "proposals": [],
        "hit_rate_before": hit_before,
        "hit_rate_after": 0.0,
        "result": "",
        "blocked_actions_after": [],
        "error": None,
    }


def build_agent_graph(
    checkpointer: Any,
    *,
    measure_fn: Callable,
    diagnose_fn: Callable,
    propose_fns: dict[str, Callable],
    health_fn: Callable | None = None,
    judge_fn: Callable | None = None,
):
    """Compile the maintenance agent StateGraph with the given checkpointer.

    measure_fn(vocab, *, action=None, current_assignments=None,
               previous_assignments=None) -> (diag, assignments)
    diagnose_fn(vocab, diagnostics, assignments, *, blocked_actions=[],
                iter_deltas=[]) -> dict
        Returns either {"decision": {...}} or the decision dict directly.
        iter_deltas is rendered as a history table in PLAN_USER from iter 2+.
    propose_fns: dict mapping action name → fn(vocab, assignments, focus) -> list
    health_fn(vocab, assignments) -> HealthMetrics
        Computes the 5-dim metrics. Required when judge_fn is supplied.
        Disabled when None (legacy / unit tests) — falls back to the
        pre-Phase-A no-metrics behavior plus the old hit_rate gate.
    judge_fn(action, proposals, reasoning, before_metrics, after_metrics,
             before_vocab_size, after_vocab_size) -> JudgeVerdict (optional)
        When provided, replaces the hit_rate-threshold regression gate with
        LLM-as-judge. Verdict drives commit/rollback routing and is recorded
        in iter_deltas. Conservative bias on LLM failure → commit.

    Config values (max_iter, disabled_actions) come from the initial state.
    """

    # ── Nodes ────────────────────────────────────────────────────────────

    def initial_measure_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        vocab = state["initial_vocab"]
        diag, assignments = measure_fn(
            vocab,
            action=None,
            current_assignments=None,
            previous_assignments=None,
        )
        check_invariants(vocab, assignments)
        disabled = list(state.get("disabled_actions", []))
        logger.event(
            "agent.run.start",
            vocab_size=len(vocab),
            assignments=len(assignments),
            hit_rate=round(_hit_rate(diag), 4),
            disabled_actions=sorted(disabled),
            max_iter=state.get("max_iter", 5),
        )
        out: dict = {
            "vocab": list(vocab),
            "assignments": list(assignments),
            "diagnostics": diag,
            "iter": 0,
            "blocked_actions": disabled,  # start with disabled actions pre-blocked
            "history": [],
        }
        if health_fn is not None:
            health = health_fn(vocab, assignments)
            metrics = health.to_dict()
            logger.event("metrics.measured", iter=0, **metrics)
            out["metrics_history"] = [{"iter": 0, "metrics": metrics}]
            out["iter_deltas"] = []
        return out

    def termination_check_node(state: AgentLoopState) -> dict:
        """Set final_status if any termination condition holds; else no-op."""
        if state.get("final_status"):
            # Already set (e.g., fatal_error from diagnose) — keep.
            return {}
        if state["iter"] >= state.get("max_iter", 5):
            return {"final_status": "max_iter_exhausted"}
        blocked = set(state.get("blocked_actions", []))
        if blocked.issuperset(ACTIONS):
            return {"final_status": "no_actions_remaining"}
        return {}

    def diagnose_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        new_iter = state["iter"] + 1
        rec = _new_iter_record(new_iter, _hit_rate(state["diagnostics"]))
        logger.event(
            "agent.iter.start",
            iter=new_iter,
            hit_rate=round(rec["hit_rate_before"], 4),
            blocked=sorted(state.get("blocked_actions", [])),
        )
        try:
            diag_result = diagnose_fn(
                state["vocab"],
                state["diagnostics"],
                state["assignments"],
                blocked_actions=sorted(state.get("blocked_actions", [])),
                iter_deltas=list(state.get("iter_deltas", [])),
            )
        except Exception as e:
            # Fatal error — record + halt.
            import traceback as _tb
            rec["result"] = "fatal_error"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["blocked_actions_after"] = sorted(state.get("blocked_actions", []))
            logger.event("agent.iter.done", iter=new_iter, action="", result="fatal_error",
                         error=rec["error"], traceback=_tb.format_exc()[:1000])
            return {
                "iter": new_iter,
                "rec": rec,
                "iter_result": "fatal_error",
                "final_status": "fatal_error",
            }
        decision = (
            diag_result.get("decision")
            if isinstance(diag_result, dict) and "decision" in diag_result
            else diag_result
        )
        action = decision["next_action"]
        focus = decision.get("action_focus", "")
        rec["decision"] = decision
        logger.event(
            "agent.iter.diagnose.done",
            iter=new_iter, action=action, focus=focus[:120],
            confidence=decision.get("confidence"),
        )
        return {
            "iter": new_iter,
            "rec": rec,
            "decision": decision,
            "action": action,
            "focus": focus,
        }

    def propose_node(state: AgentLoopState) -> dict:
        action = state["action"]
        proposals = propose_fns[action](state["vocab"], state["assignments"], state.get("focus", ""))
        # Keep raw dataclass instances in rec for backward compat with
        # downstream code that does asdict(IterationRecord(...)).
        rec = dict(state["rec"])
        rec["action"] = action
        rec["proposals"] = list(proposals)
        return {"proposals": list(proposals), "rec": rec}

    def apply_node(state: AgentLoopState) -> dict:
        """Apply each proposal independently. Partial success is acceptable.

        Previously this was all-or-nothing: any failure rolled back the entire
        batch including innocent proposals. Now each proposal is tried in its
        own try/except. The action is only blocked if EVERY proposal failed.
        """
        logger = get_default_logger()
        cur_vocab, cur_assignments = state["vocab"], state["assignments"]
        successes: list = []
        failures: list[tuple[Any, Exception]] = []

        run_id = state.get("run_id", "")
        action = state.get("action", "")
        actor = f"agent:{action}" if action else "agent"
        reasoning = (state.get("decision") or {}).get("reasoning", "") or ""

        for p in state["proposals"]:
            before_vocab = list(cur_vocab)
            before_assignments = list(cur_assignments)
            try:
                cur_vocab, cur_assignments = apply_proposal(cur_vocab, cur_assignments, p)
                successes.append(p)
                emit_apply_event(
                    before_vocab, before_assignments,
                    cur_vocab, cur_assignments,
                    p, actor=actor, run_id=run_id, reasoning=reasoning,
                )
            except (InvalidProposal, OrphanError) as e:
                failures.append((p, e))
                logger.event("apply.proposal_skipped",
                             iter=state["iter"], action=state["action"],
                             error_type=type(e).__name__, error=str(e)[:200])

        # All failed → traditional apply_error path (block action for this run)
        if not successes:
            rec = dict(state["rec"])
            rec["result"] = "apply_error"
            first_err = failures[0][1] if failures else None
            rec["error"] = (
                f"all {len(failures)} proposals failed; first: "
                f"{type(first_err).__name__}: {first_err}"
            )
            rec["hit_rate_after"] = rec["hit_rate_before"]
            blocked = list(state.get("blocked_actions", []))
            if state["action"] not in blocked:
                blocked.append(state["action"])
            rec["blocked_actions_after"] = sorted(blocked)
            logger.event("agent.iter.done", iter=state["iter"], action=state["action"],
                         result="apply_error", error=rec["error"],
                         attempts=len(state["proposals"]))
            out: dict = {"rec": rec, "iter_result": "apply_error", "blocked_actions": blocked}
            if health_fn is not None:
                out["iter_deltas"] = [{
                    "iter": state["iter"], "action": state["action"],
                    "result": "apply_error", "delta": dict(_ZERO_DELTA),
                }]
            return out

        # At least one succeeded → commit the partial result.
        check_invariants(cur_vocab, cur_assignments)
        if failures:
            print(f"  [apply] {len(successes)}/{len(state['proposals'])} succeeded; "
                  f"{len(failures)} skipped (invariant violations)", flush=True)
            logger.event("apply.partial_success",
                         iter=state["iter"], action=state["action"],
                         succeeded=len(successes), failed=len(failures))
        return {"new_vocab": cur_vocab, "new_assignments": cur_assignments}

    def post_apply_measure_node(state: AgentLoopState) -> dict:
        """Re-compute diagnostics + 5-dim metrics after a successful apply.

        Metrics are computed here (not in commit_node) so the judge sees the
        post-apply state before deciding commit/rollback.
        """
        new_diag, new_assignments2 = measure_fn(
            state["new_vocab"],
            action=state["action"],
            current_assignments=state["new_assignments"],
            previous_assignments=state["assignments"],
        )
        rec = dict(state["rec"])
        rec["hit_rate_after"] = _hit_rate(new_diag)
        out: dict = {
            "new_diag": new_diag,
            "new_assignments": new_assignments2,
            "rec": rec,
        }
        if health_fn is not None:
            new_metrics = health_fn(state["new_vocab"], new_assignments2).to_dict()
            out["new_metrics"] = new_metrics
        return out

    def llm_judge_node(state: AgentLoopState) -> dict:
        """LLM-as-judge: decide commit/rollback/unsure based on metric deltas.

        Conservative: 'unsure' routes to commit; only 'rollback' reverts.
        Disabled (verdict="commit") when judge_fn is None — old hit_rate gate
        falls through via route_judge.
        """
        logger = get_default_logger()
        if judge_fn is None or health_fn is None:
            # No judge configured: act as a no-op that commits.
            return {
                "judge_verdict": "commit",
                "judge_reasoning": "(judge disabled)",
                "judge_confidence": "low",
            }
        before = _last_metrics(state)
        after = state.get("new_metrics") or {}
        try:
            verdict = judge_fn(
                action=state["action"],
                proposals=list(state.get("proposals", [])),
                reasoning=(state.get("decision") or {}).get("reasoning", ""),
                before_metrics=before,
                after_metrics=after,
                before_vocab_size=len(state["vocab"]),
                after_vocab_size=len(state["new_vocab"]),
            )
        except Exception as e:  # noqa: BLE001 — judge LLM failures must not crash the loop
            import traceback as _tb
            logger.event("judge.error", iter=state["iter"], action=state["action"],
                         error_type=type(e).__name__, error=str(e)[:200],
                         traceback=_tb.format_exc()[:1000])
            verdict = None
        if verdict is None:
            return {
                "judge_verdict": "commit",
                "judge_reasoning": "judge exception — defaulting to commit",
                "judge_confidence": "low",
            }
        logger.event(
            f"judge.{verdict.verdict}",
            iter=state["iter"], action=state["action"],
            confidence=verdict.confidence,
            primary_concern=verdict.primary_concern[:200],
            reasoning=verdict.reasoning[:400],
        )
        return {
            "judge_verdict": verdict.verdict,
            "judge_reasoning": verdict.reasoning,
            "judge_confidence": verdict.confidence,
        }

    def commit_node(state: AgentLoopState) -> dict:
        """Judge said commit (or unsure → conservative commit) → adopt new state."""
        logger = get_default_logger()
        rec = dict(state["rec"])
        rec["result"] = "applied"
        rec["blocked_actions_after"] = sorted(state.get("blocked_actions", []))
        logger.event(
            "agent.iter.done",
            iter=state["iter"], action=state["action"], result="applied",
            proposals=len(rec["proposals"]),
            vocab_size_after=len(state["new_vocab"]),
            hit_rate_before=round(rec["hit_rate_before"], 4),
            hit_rate_after=round(rec["hit_rate_after"], 4),
            judge_verdict=state.get("judge_verdict", ""),
        )
        out: dict = {
            "vocab": state["new_vocab"],
            "assignments": state["new_assignments"],
            "diagnostics": state["new_diag"],
            "rec": rec,
            "iter_result": "applied",
        }
        if health_fn is not None:
            new_metrics = state.get("new_metrics") or {}
            prior = _last_metrics(state)
            delta = _diff_metrics(new_metrics, prior)
            logger.event("metrics.measured", iter=state["iter"], **new_metrics)
            out["metrics_history"] = [{"iter": state["iter"], "metrics": new_metrics}]
            out["iter_deltas"] = [{
                "iter": state["iter"], "action": state["action"],
                "result": "applied", "delta": delta,
                "judge_verdict": state.get("judge_verdict", ""),
                "judge_reasoning": (state.get("judge_reasoning") or "")[:240],
            }]
        return out

    def rollback_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        rec = dict(state["rec"])
        rec["result"] = "rolled_back"
        blocked = list(state.get("blocked_actions", []))
        if state["action"] not in blocked:
            blocked.append(state["action"])
        rec["blocked_actions_after"] = sorted(blocked)
        logger.event(
            "agent.iter.done", iter=state["iter"], action=state["action"], result="rolled_back",
            hit_rate_before=round(rec["hit_rate_before"], 4),
            hit_rate_after=round(rec["hit_rate_after"], 4),
            judge_verdict=state.get("judge_verdict", ""),
        )
        out: dict = {"rec": rec, "iter_result": "rolled_back", "blocked_actions": blocked}
        if health_fn is not None:
            # state reverts → delta is zero against prior snapshot; judge fields recorded.
            out["iter_deltas"] = [{
                "iter": state["iter"], "action": state["action"],
                "result": "rolled_back", "delta": dict(_ZERO_DELTA),
                "judge_verdict": state.get("judge_verdict", ""),
                "judge_reasoning": (state.get("judge_reasoning") or "")[:240],
            }]
        return out

    def record_iter_node(state: AgentLoopState) -> dict:
        """Append the current iter record to history."""
        rec = dict(state["rec"])
        if rec.get("result") in ("blocked", "unknown_action", "blocked_empty"):
            # These rec.result values were set by upstream router/scratch nodes.
            pass
        history = list(state.get("history", []))
        history.append(rec)
        return {"history": history, "rec": {}}

    def _no_op_iter_delta(iter_idx: int, action: str, result: str) -> dict:
        return {"iter": iter_idx, "action": action, "result": result, "delta": dict(_ZERO_DELTA)}

    def handle_blocked_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        action = state["action"]
        rec = dict(state["rec"])
        rec["action"] = action
        rec["result"] = "blocked"
        rec["hit_rate_after"] = rec["hit_rate_before"]
        rec["blocked_actions_after"] = sorted(state.get("blocked_actions", []))
        logger.event("agent.iter.done", iter=state["iter"], action=action, result="blocked")
        out: dict = {"rec": rec, "iter_result": "blocked"}
        if health_fn is not None:
            out["iter_deltas"] = [_no_op_iter_delta(state["iter"], action, "blocked")]
        return out

    def handle_unknown_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        action = state["action"]
        rec = dict(state["rec"])
        rec["action"] = action
        rec["result"] = "unknown_action"
        rec["error"] = f"unknown action: {action}"
        blocked = list(state.get("blocked_actions", []))
        if action not in blocked:
            blocked.append(action)
        rec["blocked_actions_after"] = sorted(blocked)
        logger.event("agent.iter.done", iter=state["iter"], action=action,
                     result="unknown_action", error=rec["error"])
        out: dict = {"rec": rec, "iter_result": "unknown_action", "blocked_actions": blocked}
        if health_fn is not None:
            out["iter_deltas"] = [_no_op_iter_delta(state["iter"], action, "unknown_action")]
        return out

    def handle_blocked_empty_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        action = state["action"]
        rec = dict(state["rec"])
        rec["result"] = "blocked_empty"
        rec["hit_rate_after"] = rec["hit_rate_before"]
        blocked = list(state.get("blocked_actions", []))
        if action not in blocked:
            blocked.append(action)
        rec["blocked_actions_after"] = sorted(blocked)
        logger.event("agent.iter.done", iter=state["iter"], action=action, result="blocked_empty")
        out: dict = {"rec": rec, "iter_result": "blocked_empty", "blocked_actions": blocked}
        if health_fn is not None:
            out["iter_deltas"] = [_no_op_iter_delta(state["iter"], action, "blocked_empty")]
        return out

    def finalize_done_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        rec = dict(state["rec"])
        rec["action"] = "done"
        rec["result"] = "completed"
        rec["hit_rate_after"] = rec["hit_rate_before"]
        rec["blocked_actions_after"] = sorted(state.get("blocked_actions", []))
        history = list(state.get("history", []))
        history.append(rec)
        logger.event("agent.iter.done", iter=state["iter"], action="done", result="completed")
        logger.event("agent.run.done", status="completed",
                     iters=state["iter"], vocab_size=len(state["vocab"]))
        return {"history": history, "final_status": "completed"}

    def finalize_terminate_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        status = state.get("final_status") or "max_iter_exhausted"
        if status == "max_iter_exhausted":
            logger.event("agent.run.done", status="max_iter_exhausted",
                         iters=state["iter"], vocab_size=len(state["vocab"]))
        elif status == "no_actions_remaining":
            logger.event("agent.run.done", status="no_actions_remaining",
                         iters=state["iter"], vocab_size=len(state["vocab"]))
        else:
            logger.event("agent.run.done", status=status,
                         iters=state["iter"], vocab_size=len(state["vocab"]))
        return {"final_status": status}

    # ── Routing functions ────────────────────────────────────────────────

    def route_termination(state: AgentLoopState) -> str:
        return "terminate" if state.get("final_status") else "diagnose"

    def route_action(state: AgentLoopState) -> str:
        if state.get("iter_result") == "fatal_error":
            return "terminate"
        action = state["action"]
        if action == "done":
            return "done"
        if action in set(state.get("blocked_actions", [])):
            return "blocked"
        if action not in propose_fns:
            return "unknown"
        return "propose"

    def route_proposals(state: AgentLoopState) -> str:
        return "apply" if state.get("proposals") else "blocked_empty"

    def route_apply(state: AgentLoopState) -> str:
        return "post_apply" if state.get("iter_result") != "apply_error" else "record"

    def route_judge(state: AgentLoopState) -> str:
        """Route on the judge's verdict. 'unsure' is conservative → commit."""
        return "rollback" if state.get("judge_verdict") == "rollback" else "commit"

    # ── Wire ──────────────────────────────────────────────────────────────

    g = StateGraph(AgentLoopState)

    g.add_node("initial_measure", initial_measure_node)
    g.add_node("termination_check", termination_check_node)
    g.add_node("diagnose", diagnose_node)
    g.add_node("propose", propose_node)
    g.add_node("apply", apply_node)
    g.add_node("post_apply_measure", post_apply_measure_node)
    g.add_node("llm_judge", llm_judge_node)
    g.add_node("commit", commit_node)
    g.add_node("rollback", rollback_node)
    g.add_node("handle_blocked", handle_blocked_node)
    g.add_node("handle_unknown", handle_unknown_node)
    g.add_node("handle_blocked_empty", handle_blocked_empty_node)
    g.add_node("record_iter", record_iter_node)
    g.add_node("finalize_done", finalize_done_node)
    g.add_node("finalize_terminate", finalize_terminate_node)

    g.add_edge(START, "initial_measure")
    g.add_edge("initial_measure", "termination_check")

    g.add_conditional_edges(
        "termination_check",
        route_termination,
        {"diagnose": "diagnose", "terminate": "finalize_terminate"},
    )

    g.add_conditional_edges(
        "diagnose",
        route_action,
        {
            "done": "finalize_done",
            "blocked": "handle_blocked",
            "unknown": "handle_unknown",
            "propose": "propose",
            "terminate": "finalize_terminate",
        },
    )

    g.add_conditional_edges(
        "propose",
        route_proposals,
        {"apply": "apply", "blocked_empty": "handle_blocked_empty"},
    )

    g.add_conditional_edges(
        "apply",
        route_apply,
        {"post_apply": "post_apply_measure", "record": "record_iter"},
    )

    g.add_edge("post_apply_measure", "llm_judge")
    g.add_conditional_edges(
        "llm_judge",
        route_judge,
        {"commit": "commit", "rollback": "rollback"},
    )

    # Handlers that record then loop back
    g.add_edge("handle_blocked", "record_iter")
    g.add_edge("handle_unknown", "record_iter")
    g.add_edge("handle_blocked_empty", "record_iter")
    g.add_edge("commit", "record_iter")
    g.add_edge("rollback", "record_iter")
    g.add_edge("record_iter", "termination_check")  # ← the loop edge

    # Terminal nodes
    g.add_edge("finalize_done", END)
    g.add_edge("finalize_terminate", END)

    # Set max recursion limit higher than the per-iter node count × max_iter
    # to allow long loops without LangGraph's default 25 limit kicking in.
    return g.compile(checkpointer=checkpointer)
