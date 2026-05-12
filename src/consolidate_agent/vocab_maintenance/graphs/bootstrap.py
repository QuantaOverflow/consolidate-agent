"""Bootstrap StateGraph: records → distill → synthesize → [HITL] → reverse_check → Network.

Each LLM-using node is parameterized via dependency injection so tests can
substitute fakes without monkey-patching. Defaults wire real LLM-backed
implementations from bootstrap.py + measure.py.

HITL: the `vocab_review` node uses `interrupt()` to pause execution after
synthesize. The runner resumes via `Command(resume="accept"|"regenerate"|"abort")`.
For unattended mode, set `state["auto_accept"] = True` and the gate
short-circuits without interrupting.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from .state import BootstrapState


# ── Default node implementations (LLM-backed) ────────────────────────────────


def _default_distill(records: list[dict], batch_size: int) -> list[dict]:
    """Real LLM distill on the given records (record loading happens in the node).

    Takes a records list rather than db_path so the node can pre-filter to
    just the uncached subset when a themes_seed is provided.
    """
    from consolidate_agent.config import Settings
    from consolidate_agent.consolidation._utils import _chat_model

    from ..bootstrap import distill_step

    settings = Settings()
    def model_factory():
        return _chat_model(settings)

    log_sink = io.StringIO()
    return distill_step(model_factory, records, batch_size, log_sink)


def _default_synthesize(themes: list[dict]) -> dict:
    """Real LLM synthesize: themes → {vocab, notes}."""
    from consolidate_agent.config import Settings
    from consolidate_agent.consolidation._utils import _chat_model

    from ..bootstrap import synthesize_step

    settings = Settings()
    def model_factory():
        return _chat_model(settings)

    log_sink = io.StringIO()
    return synthesize_step(model_factory, themes, log_sink)


def _default_reverse_check(db_path: str, vocab: list[dict], concurrency: int) -> list[dict]:
    """Real LLM reverse_check: records × vocab → assignments."""
    from ..measure import load_records, reverse_check_subset

    records = load_records(Path(db_path))
    return reverse_check_subset(records, vocab, batch_size=10, concurrency=concurrency)


def _filter_dangling_refs(assignments: list[dict], vocab: list[dict]) -> int:
    """In-place: drop tag refs not in vocab. Records w/ all dropped → missing."""
    vocab_names = {t["name"] for t in vocab}
    dropped = 0
    for a in assignments:
        original = a.get("selected_tags", []) or []
        filtered = [t for t in original if t["name"] in vocab_names]
        if len(filtered) < len(original):
            dropped += len(original) - len(filtered)
        a["selected_tags"] = filtered
        if not filtered and not a.get("missing"):
            a["missing"] = True
            a["missing_concept"] = a.get("missing_concept") or "all selected_tags absent from vocab"
    return dropped


# ── Graph builder ────────────────────────────────────────────────────────────


def _default_load_records(db_path: str) -> list[dict]:
    from ..measure import load_records
    return load_records(Path(db_path))


def build_bootstrap_graph(
    checkpointer: Any,
    *,
    distill_fn: Callable[[list[dict], int], list[dict]] | None = None,
    synthesize_fn: Callable[[list[dict]], dict] | None = None,
    reverse_check_fn: Callable[[str, list[dict], int], list[dict]] | None = None,
    records_loader_fn: Callable[[str], list[dict]] | None = None,
):
    """Compile a bootstrap StateGraph with the given checkpointer.

    Pass `*_fn` kwargs to inject fake LLM / DB functions for tests.

    distill_fn signature: (records: list[dict], batch_size: int) -> themes_list
        (changed from (db_path, batch_size) so the node can pre-filter records
        against the themes_seed cache before calling distill.)

    records_loader_fn signature: (db_path: str) -> list[dict]
        Returns records to distill. Defaults to load_records(Path(db_path)).
    """
    distill_impl = distill_fn or _default_distill
    synthesize_impl = synthesize_fn or _default_synthesize
    reverse_check_impl = reverse_check_fn or _default_reverse_check
    load_records_impl = records_loader_fn or _default_load_records

    # ── Nodes ────────────────────────────────────────────────────────────

    def distill_node(state: BootstrapState) -> dict:
        records = load_records_impl(state["db_path"])
        seed = state.get("themes_seed") or {}

        cached = []
        to_distill = []
        for r in records:
            if r["record_id"] in seed and seed[r["record_id"]]:
                cached.append({
                    "record_id": r["record_id"],
                    "title": r.get("title", ""),
                    "theme": seed[r["record_id"]],
                })
            else:
                to_distill.append(r)

        if cached:
            print(f"  [distill] {len(cached)}/{len(records)} themes from seed (zero LLM)", flush=True)

        if to_distill:
            print(f"  [distill] running LLM on {len(to_distill)} uncached records", flush=True)
            fresh = distill_impl(to_distill, state.get("batch_size", 30))
        else:
            fresh = []

        return {"themes": cached + fresh}

    def synthesize_node(state: BootstrapState) -> dict:
        result = synthesize_impl(state["themes"])
        attempts = state.get("synthesize_attempts", 0) + 1
        return {
            "vocab": result.get("vocab", []),
            "synthesize_notes": result.get("notes", ""),
            "synthesize_attempts": attempts,
            "review_decision": "",  # reset for re-review
        }

    def vocab_review_node(state: BootstrapState) -> dict:
        """HITL gate. Returns {review_decision}. Auto-accepts if state.auto_accept."""
        if state.get("auto_accept"):
            return {"review_decision": "accept"}
        decision = interrupt({
            "stage": "vocab_review",
            "vocab": state["vocab"],
            "notes": state.get("synthesize_notes", ""),
            "attempt": state.get("synthesize_attempts", 1),
        })
        # decision is whatever Command(resume=X) passes — string in our convention
        if decision not in ("accept", "regenerate", "abort"):
            decision = "accept"  # safe default for malformed input
        return {"review_decision": decision}

    def reverse_check_node(state: BootstrapState) -> dict:
        assignments = reverse_check_impl(
            state["db_path"], state["vocab"], state.get("concurrency", 10),
        )
        dropped = _filter_dangling_refs(assignments, state["vocab"])
        return {"assignments": assignments, "fake_tag_drops": dropped}

    def abort_node(state: BootstrapState) -> dict:
        return {"abort_reason": "user aborted at vocab review"}

    # ── Routing ──────────────────────────────────────────────────────────

    def route_after_review(state: BootstrapState) -> str:
        d = state.get("review_decision", "accept")
        if d == "regenerate":
            return "synthesize"
        if d == "abort":
            return "abort"
        return "reverse_check"  # accept (default)

    # ── Wire ──────────────────────────────────────────────────────────────

    g = StateGraph(BootstrapState)
    g.add_node("distill", distill_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("vocab_review", vocab_review_node)
    g.add_node("reverse_check", reverse_check_node)
    g.add_node("abort", abort_node)

    g.add_edge(START, "distill")
    g.add_edge("distill", "synthesize")
    g.add_edge("synthesize", "vocab_review")
    g.add_conditional_edges(
        "vocab_review",
        route_after_review,
        {"synthesize": "synthesize", "reverse_check": "reverse_check", "abort": "abort"},
    )
    g.add_edge("reverse_check", END)
    g.add_edge("abort", END)

    return g.compile(checkpointer=checkpointer)
