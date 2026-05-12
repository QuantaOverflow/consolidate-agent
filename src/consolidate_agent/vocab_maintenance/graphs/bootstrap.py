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

import json
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from ..observability import get_default_logger
from .state import BootstrapState


class _JsonlToLogger:
    """Adapter: `distill_step` / `synthesize_step` write jsonl lines here;
    each line is forwarded to the active `RunLogger` as a structured event.

    Replaces the legacy `io.StringIO()` sink (silently discarded the data).
    """
    def __init__(self):
        self._logger = get_default_logger()

    def write(self, s: str) -> None:
        s = s.strip()
        if not s:
            return
        try:
            data = json.loads(s)
            phase = data.pop("phase", "bootstrap")
            # batch-level vs phase-level: distill_step writes both
            if "batch" in data:
                self._logger.event(f"{phase}.batch", **data)
            else:
                self._logger.event(f"{phase}.summary", **data)
        except json.JSONDecodeError:
            self._logger.event("bootstrap.raw_log", raw=s[:500])

    def flush(self) -> None:
        pass


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

    log_sink = _JsonlToLogger()
    return distill_step(model_factory, records, batch_size, log_sink)


def _default_synthesize(themes: list[dict]) -> dict:
    """Real LLM synthesize via BERTopic-style clustering — embed → KMeans →
    per-cluster LLM-name.

    See docs/adr/0001-bootstrap-synthesize-via-embedding-cluster.md for the
    full rationale; in short, single-shot LLM synthesize at 500+ themes
    over-abstracts to umbrella categories (validated repeatedly with both
    qwen-plus and qwen-max). The cluster-first pipeline runs in ~80s and
    produces ~50 specific tags with healthy size distribution.

    `n_topics` defaults to len(themes) // 14, clamped to [5, 80] — i.e.
    one tag per ~14 themes, the empirically observed Goldilocks ratio for
    this corpus. For other corpora, adjust the divisor.
    """
    from consolidate_agent.config import Settings
    from consolidate_agent.consolidation._utils import _chat_model

    from ..bootstrap import synthesize_via_clustering

    settings = Settings()
    def model_factory():
        return _chat_model(settings)  # qwen-plus + timeout 600 + thinking off

    n_topics = max(5, min(80, len(themes) // 14))
    log_sink = _JsonlToLogger()
    return synthesize_via_clustering(
        model_factory, themes, log_sink,
        n_topics=n_topics, concurrency=10,
    )


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
        logger = get_default_logger()
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

        with logger.timed("phase.distill",
                          records_total=len(records),
                          from_seed=len(cached),
                          to_distill=len(to_distill)):
            if to_distill:
                print(f"  [distill] running LLM on {len(to_distill)} uncached records", flush=True)
                fresh = distill_impl(to_distill, state.get("batch_size", 30))
            else:
                fresh = []

        return {"themes": cached + fresh}

    def synthesize_node(state: BootstrapState) -> dict:
        logger = get_default_logger()
        attempts = state.get("synthesize_attempts", 0) + 1
        with logger.timed("phase.synthesize",
                          themes_in=len(state["themes"]),
                          attempt=attempts):
            result = synthesize_impl(state["themes"])
        logger.event("phase.synthesize.summary",
                     vocab_size=len(result.get("vocab", [])),
                     notes=str(result.get("notes", ""))[:300])
        return {
            "vocab": result.get("vocab", []),
            "synthesize_notes": result.get("notes", ""),
            "synthesize_attempts": attempts,
            "review_decision": "",  # reset for re-review
        }

    def vocab_review_node(state: BootstrapState) -> dict:
        """HITL gate. Returns {review_decision}. Auto-accepts if state.auto_accept."""
        logger = get_default_logger()
        if state.get("auto_accept"):
            logger.event("phase.review.auto_accept", vocab_size=len(state.get("vocab", [])))
            return {"review_decision": "accept"}
        logger.event("phase.review.interrupt", vocab_size=len(state.get("vocab", [])))
        decision = interrupt({
            "stage": "vocab_review",
            "vocab": state["vocab"],
            "notes": state.get("synthesize_notes", ""),
            "attempt": state.get("synthesize_attempts", 1),
        })
        # decision is whatever Command(resume=X) passes — string in our convention
        if decision not in ("accept", "regenerate", "abort"):
            decision = "accept"  # safe default for malformed input
        logger.event("phase.review.decision", decision=decision)
        return {"review_decision": decision}

    def reverse_check_node(state: BootstrapState) -> dict:
        logger = get_default_logger()
        with logger.timed("phase.reverse_check",
                          vocab_size=len(state["vocab"]),
                          concurrency=state.get("concurrency", 10)):
            assignments = reverse_check_impl(
                state["db_path"], state["vocab"], state.get("concurrency", 10),
            )
            dropped = _filter_dangling_refs(assignments, state["vocab"])
        missing = sum(1 for a in assignments if a.get("missing"))
        logger.event("phase.reverse_check.summary",
                     assignments=len(assignments),
                     missing=missing,
                     dangling_dropped=dropped,
                     hit_rate=round((len(assignments) - missing) / max(1, len(assignments)), 4))
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
