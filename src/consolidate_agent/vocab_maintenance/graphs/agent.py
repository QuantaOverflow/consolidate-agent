"""Agent loop as a LangGraph StateGraph — plan-and-execute (Phase E).

Topology:
  START
    │
    ↓
  initial_measure
    │
    ↓
  plan_work_queue            (rules-based; no LLM)
    │
    ↓
  check_continue ──── queue_empty ──→ finalize_completed ──→ END
                ──── max_iter ──→ finalize_terminate ──→ END
                ──── continue ──┐
                                ↓
                              pop_item
                                │
                                ↓
                              propose
                                │
                                ↓
                          (proposals empty → record_iter)
                                │
                                ↓
                              apply
                                │
                          (apply_error → record_iter)
                                │
                                ↓
                              post_apply_measure
                                │
                                ↓
                              sanity_check       (pure fn; no LLM)
                                │
                              ┌─┴────────┐
                              ↓          ↓
                          commit     rollback
                              │          │
                              └────┬─────┘
                                   ↓
                              record_iter
                                   │
                                   ↓
                              check_continue (loop)

What Phase E removes from Phase A-D:
  - LLM diagnose: rules-based planner emits a full work queue at the start.
  - LLM-as-judge: pure `sanity_check_metrics` rolls back only on catastrophic
    regression (any monitored dim drops >= 10%). No LLM per-iter call.
  - Convergence rule: max_iter + queue exhaustion are the only termination
    signals (FATAL_ERROR still applies on exceptions).
  - HITL escape: with no LLM judgment loop there is no `confidence=low`
    signal to escalate on. Removed.

Per-iter LLM calls drop from ~4 (plan + decide + propose + judge) to 1
(propose). The vocab-maintenance loop becomes deterministic-modulo-propose.
"""
from __future__ import annotations

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
from ..planner import WorkItem, build_work_queue, sanity_check_metrics
from .state import AgentLoopState


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
        "focus": "",
        "work_item_reason": "",
        "proposals": [],
        "hit_rate_before": hit_before,
        "hit_rate_after": 0.0,
        "result": "",
        "sanity_verdict": "",
        "sanity_reason": "",
        "error": None,
    }


_DEGRADATION_THRESHOLD = 0.3
_FAILURE_RESULTS = frozenset({"propose_error", "apply_error", "unknown_action"})
_VALID_RESULTS = frozenset({
    "applied", "blocked_empty", "rolled_back",
    "propose_error", "apply_error", "unknown_action",
})


def compute_run_summary(history: list[dict]) -> dict:
    """Aggregate iter history into run-level statistics."""
    counts: dict[str, int] = {k: 0 for k in _VALID_RESULTS}
    for rec in history:
        r = rec.get("result", "")
        if r in _VALID_RESULTS:
            counts[r] += 1
    processed = sum(counts.values())
    failure_count = sum(counts[k] for k in _FAILURE_RESULTS)
    failure_rate = failure_count / processed if processed > 0 else 0.0
    return {
        "applied": counts["applied"],
        "blocked_empty": counts["blocked_empty"],
        "rolled_back": counts["rolled_back"],
        "apply_error": counts["apply_error"],
        "propose_error": counts["propose_error"],
        "unknown_action": counts["unknown_action"],
        "processed": processed,
        "failure_rate": failure_rate,
    }


def decide_final_status(
    termination_cause: str,
    summary: dict,
    threshold: float = _DEGRADATION_THRESHOLD,
) -> str:
    """Determine final_status from termination cause and run summary."""
    if summary["processed"] == 0:
        return "completed"
    if summary["failure_rate"] >= threshold:
        return "degraded"
    return termination_cause


def build_agent_graph(
    checkpointer: Any,
    *,
    measure_fn: Callable,
    propose_fns: dict[str, Callable],
    health_fn: Callable | None = None,
    db_path: Any = None,
    sanity_threshold: float = 0.10,
):
    """Compile the maintenance agent StateGraph (Phase E plan-and-execute).

    measure_fn(vocab, *, action=None, current_assignments=None,
               previous_assignments=None) -> (diag, assignments)
    propose_fns: dict mapping action name → fn(vocab, assignments, focus) -> list
    health_fn(vocab, assignments) -> HealthMetrics
        Computes 5-dim metrics for the planner's signals (forced_fit) and
        for sanity_check. Required.
    db_path: passed to the planner so it can run `compute_fit_signals`.
    sanity_threshold: catastrophic regression bound. Any monitored dim
        dropping by >= this triggers rollback. Default 0.10 (10%).
    """
    if health_fn is None:
        raise ValueError("Phase E requires health_fn — planner and sanity_check both need 5-dim metrics")
    if db_path is None:
        raise ValueError("Phase E requires db_path — planner runs compute_fit_signals which needs it")

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
        health = health_fn(vocab, assignments)
        metrics = health.to_dict()
        logger.event("metrics.measured", iter=0, **metrics)
        return {
            "vocab": list(vocab),
            "assignments": list(assignments),
            "diagnostics": diag,
            "iter": 0,
            "history": [],
            "metrics_history": [{"iter": 0, "metrics": metrics}],
            "iter_deltas": [],
        }

    def plan_work_queue_node(state: AgentLoopState) -> dict:
        """Rules-based planner: scan signals once, emit prioritized work queue."""
        logger = get_default_logger()
        disabled = set(state.get("disabled_actions", []))
        queue = build_work_queue(
            state["vocab"],
            state["assignments"],
            state["diagnostics"],
            db_path,
            disabled_actions=disabled,
        )
        logger.event(
            "planner.done",
            queue_size=len(queue),
            actions=[w.action for w in queue],
            items=[w.to_dict() for w in queue],
        )
        return {"work_queue": [w.to_dict() for w in queue]}

    def check_continue_node(state: AgentLoopState) -> dict:
        """Decide whether to keep looping. Sets final_status if not."""
        if state.get("final_status"):
            return {}
        if state["iter"] >= state.get("max_iter", 5):
            return {"final_status": "max_iter_exhausted"}
        if not state.get("work_queue"):
            return {"final_status": "completed"}
        return {}

    def pop_item_node(state: AgentLoopState) -> dict:
        """Pop the next WorkItem and bootstrap a fresh per-iter record."""
        logger = get_default_logger()
        queue = list(state.get("work_queue") or [])
        item = queue.pop(0)
        new_iter = state["iter"] + 1
        rec = _new_iter_record(new_iter, _hit_rate(state["diagnostics"]))
        rec["action"] = item["action"]
        rec["focus"] = item["focus"]
        rec["work_item_reason"] = item.get("reason", "")
        logger.event(
            "agent.iter.start",
            iter=new_iter,
            action=item["action"],
            focus=item["focus"],
            reason=item.get("reason", ""),
            queue_remaining=len(queue),
        )
        return {
            "iter": new_iter,
            "work_queue": queue,
            "current_item": item,
            "rec": rec,
            "action": item["action"],
            "focus": item["focus"],
        }

    def propose_node(state: AgentLoopState) -> dict:
        action = state["action"]
        if action not in propose_fns:
            # Unknown action in queue — record + skip.
            rec = dict(state["rec"])
            rec["result"] = "unknown_action"
            rec["error"] = f"unknown action: {action}"
            return {"proposals": [], "rec": rec}
        try:
            proposals = propose_fns[action](
                state["vocab"], state["assignments"], state.get("focus", ""),
            )
        except Exception as e:  # noqa: BLE001 — propose LLM may raise; isolate
            logger = get_default_logger()
            import traceback as _tb
            logger.event(
                "propose.error",
                iter=state["iter"], action=action,
                error_type=type(e).__name__, error=str(e)[:200],
                traceback=_tb.format_exc()[:1000],
            )
            rec = dict(state["rec"])
            rec["result"] = "propose_error"
            rec["error"] = f"{type(e).__name__}: {e}"
            return {"proposals": [], "rec": rec}
        rec = dict(state["rec"])
        rec["proposals"] = list(proposals)
        return {"proposals": list(proposals), "rec": rec}

    def apply_node(state: AgentLoopState) -> dict:
        """Apply each proposal independently. Partial success is acceptable."""
        logger = get_default_logger()
        cur_vocab, cur_assignments = state["vocab"], state["assignments"]
        successes: list = []
        failures: list[tuple[Any, Exception]] = []

        run_id = state.get("run_id", "")
        action = state.get("action", "")
        actor = f"agent:{action}" if action else "agent"
        reasoning = state.get("current_item", {}).get("reason", "")

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
                logger.event(
                    "apply.proposal_skipped",
                    iter=state["iter"], action=action,
                    error_type=type(e).__name__, error=str(e)[:200],
                )

        if not successes:
            rec = dict(state["rec"])
            rec["result"] = "apply_error"
            first_err = failures[0][1] if failures else None
            rec["error"] = (
                f"all {len(failures)} proposals failed; first: "
                f"{type(first_err).__name__}: {first_err}"
            )
            rec["hit_rate_after"] = rec["hit_rate_before"]
            logger.event(
                "agent.iter.done", iter=state["iter"], action=action,
                result="apply_error", error=rec["error"], attempts=len(state["proposals"]),
            )
            return {"rec": rec, "iter_result": "apply_error"}

        check_invariants(cur_vocab, cur_assignments)
        if failures:
            print(
                f"  [apply] {len(successes)}/{len(state['proposals'])} succeeded; "
                f"{len(failures)} skipped (invariant violations)",
                flush=True,
            )
            logger.event(
                "apply.partial_success",
                iter=state["iter"], action=action,
                succeeded=len(successes), failed=len(failures),
            )
        return {"new_vocab": cur_vocab, "new_assignments": cur_assignments}

    def post_apply_measure_node(state: AgentLoopState) -> dict:
        new_diag, new_assignments2 = measure_fn(
            state["new_vocab"],
            action=state["action"],
            current_assignments=state["new_assignments"],
            previous_assignments=state["assignments"],
        )
        new_metrics = health_fn(state["new_vocab"], new_assignments2).to_dict()
        rec = dict(state["rec"])
        rec["hit_rate_after"] = _hit_rate(new_diag)
        return {
            "new_diag": new_diag,
            "new_assignments": new_assignments2,
            "new_metrics": new_metrics,
            "rec": rec,
        }

    def sanity_check_node(state: AgentLoopState) -> dict:
        """Pure-fn gate: rollback only on catastrophic regression."""
        logger = get_default_logger()
        before = _last_metrics(state)
        after = state.get("new_metrics") or {}
        verdict, reason = sanity_check_metrics(
            before, after, catastrophic_threshold=sanity_threshold,
        )
        logger.event(
            f"sanity.{verdict}",
            iter=state["iter"], action=state.get("action", ""),
            reason=reason,
        )
        return {"sanity_verdict": verdict, "sanity_reason": reason}

    def commit_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        rec = dict(state["rec"])
        rec["result"] = "applied"
        rec["sanity_verdict"] = state.get("sanity_verdict", "commit")
        rec["sanity_reason"] = state.get("sanity_reason", "")
        logger.event(
            "agent.iter.done",
            iter=state["iter"], action=state["action"], result="applied",
            proposals=len(rec["proposals"]),
            vocab_size_after=len(state["new_vocab"]),
            hit_rate_before=round(rec["hit_rate_before"], 4),
            hit_rate_after=round(rec["hit_rate_after"], 4),
            sanity_reason=state.get("sanity_reason", ""),
        )
        new_metrics = state.get("new_metrics") or {}
        prior = _last_metrics(state)
        delta = _diff_metrics(new_metrics, prior)
        logger.event("metrics.measured", iter=state["iter"], **new_metrics)
        return {
            "vocab": state["new_vocab"],
            "assignments": state["new_assignments"],
            "diagnostics": state["new_diag"],
            "rec": rec,
            "iter_result": "applied",
            "metrics_history": [{"iter": state["iter"], "metrics": new_metrics}],
            "iter_deltas": [{
                "iter": state["iter"], "action": state["action"],
                "focus": state.get("focus", ""), "result": "applied",
                "delta": delta, "sanity_verdict": state.get("sanity_verdict", "commit"),
            }],
        }

    def rollback_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        rec = dict(state["rec"])
        rec["result"] = "rolled_back"
        rec["sanity_verdict"] = state.get("sanity_verdict", "rollback")
        rec["sanity_reason"] = state.get("sanity_reason", "")
        logger.event(
            "agent.iter.done", iter=state["iter"], action=state["action"],
            result="rolled_back",
            hit_rate_before=round(rec["hit_rate_before"], 4),
            hit_rate_after=round(rec["hit_rate_after"], 4),
            sanity_reason=state.get("sanity_reason", ""),
        )
        return {
            "rec": rec,
            "iter_result": "rolled_back",
            "iter_deltas": [{
                "iter": state["iter"], "action": state["action"],
                "focus": state.get("focus", ""), "result": "rolled_back",
                "delta": dict(_ZERO_DELTA),
                "sanity_verdict": "rollback",
            }],
        }

    def record_iter_node(state: AgentLoopState) -> dict:
        """Append iter record to history; clear per-iter scratch."""
        rec = dict(state["rec"])
        history = list(state.get("history", []))
        history.append(rec)
        return {"history": history, "rec": {}}

    def finalize_terminate_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        termination_cause = state.get("final_status") or "max_iter_exhausted"
        summary = compute_run_summary(state.get("history", []))
        status = decide_final_status(termination_cause, summary)
        logger.event(
            "agent.run.done",
            status=status,
            iters=state["iter"], vocab_size=len(state["vocab"]),
            queue_remaining=len(state.get("work_queue") or []),
            **{k: (round(v, 4) if k == "failure_rate" else v) for k, v in summary.items()},
        )
        return {"final_status": status, "run_summary": summary}

    # ── Routing functions ────────────────────────────────────────────────

    def route_continue(state: AgentLoopState) -> str:
        return "terminate" if state.get("final_status") else "pop"

    def route_after_propose(state: AgentLoopState) -> str:
        rec = state.get("rec") or {}
        if rec.get("result") in ("unknown_action", "propose_error"):
            return "record"
        return "apply" if state.get("proposals") else "blocked_empty"

    def route_after_apply(state: AgentLoopState) -> str:
        return "post_apply" if state.get("iter_result") != "apply_error" else "record"

    def route_after_sanity(state: AgentLoopState) -> str:
        return "rollback" if state.get("sanity_verdict") == "rollback" else "commit"

    def handle_blocked_empty_node(state: AgentLoopState) -> dict:
        logger = get_default_logger()
        rec = dict(state["rec"])
        rec["result"] = "blocked_empty"
        rec["hit_rate_after"] = rec["hit_rate_before"]
        logger.event(
            "agent.iter.done", iter=state["iter"], action=state.get("action", ""),
            result="blocked_empty",
        )
        return {
            "rec": rec,
            "iter_result": "blocked_empty",
            "iter_deltas": [{
                "iter": state["iter"], "action": state.get("action", ""),
                "focus": state.get("focus", ""), "result": "blocked_empty",
                "delta": dict(_ZERO_DELTA),
                "sanity_verdict": "",
            }],
        }

    # ── Wire ──────────────────────────────────────────────────────────────

    g = StateGraph(AgentLoopState)

    g.add_node("initial_measure", initial_measure_node)
    g.add_node("plan_work_queue", plan_work_queue_node)
    g.add_node("check_continue", check_continue_node)
    g.add_node("pop_item", pop_item_node)
    g.add_node("propose", propose_node)
    g.add_node("apply", apply_node)
    g.add_node("post_apply_measure", post_apply_measure_node)
    g.add_node("sanity_check", sanity_check_node)
    g.add_node("commit", commit_node)
    g.add_node("rollback", rollback_node)
    g.add_node("handle_blocked_empty", handle_blocked_empty_node)
    g.add_node("record_iter", record_iter_node)
    g.add_node("finalize_terminate", finalize_terminate_node)

    g.add_edge(START, "initial_measure")
    g.add_edge("initial_measure", "plan_work_queue")
    g.add_edge("plan_work_queue", "check_continue")
    g.add_conditional_edges(
        "check_continue",
        route_continue,
        {"pop": "pop_item", "terminate": "finalize_terminate"},
    )
    g.add_edge("pop_item", "propose")
    g.add_conditional_edges(
        "propose",
        route_after_propose,
        {"apply": "apply", "blocked_empty": "handle_blocked_empty", "record": "record_iter"},
    )
    g.add_conditional_edges(
        "apply",
        route_after_apply,
        {"post_apply": "post_apply_measure", "record": "record_iter"},
    )
    g.add_edge("post_apply_measure", "sanity_check")
    g.add_conditional_edges(
        "sanity_check",
        route_after_sanity,
        {"commit": "commit", "rollback": "rollback"},
    )
    g.add_edge("commit", "record_iter")
    g.add_edge("rollback", "record_iter")
    g.add_edge("handle_blocked_empty", "record_iter")
    g.add_edge("record_iter", "check_continue")
    g.add_edge("finalize_terminate", END)

    return g.compile(checkpointer=checkpointer)
