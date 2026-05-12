"""TypedDict state schemas for LangGraph StateGraphs.

TypedDict (vs Pydantic) chosen for:
  - Light merge semantics (each node returns partial dict)
  - No runtime validation overhead per state update
  - Direct JSON serialization for checkpointer
"""
from __future__ import annotations

from typing import TypedDict


class BootstrapState(TypedDict, total=False):
    """State for the bootstrap pipeline.

    Stages populate fields as they run; HITL gate may loop synthesize.
    """
    # Inputs (set at graph.invoke time)
    db_path: str                # Path serialized as str for checkpointer
    batch_size: int             # distill batch size (default 30)
    concurrency: int            # ThreadPool workers for reverse_check
    auto_accept: bool           # if True, skip vocab_review interrupt

    # Intermediate artifacts
    themes: list[dict]          # output of distill: [{record_id, title, theme}]
    vocab: list[dict]           # output of synthesize: [{name, definition}]
    synthesize_notes: str       # LLM's free-text rationale
    synthesize_attempts: int    # how many times synthesize ran (for retry tracking)

    # HITL gate
    review_decision: str        # "accept" / "regenerate" / "abort"

    # Final
    assignments: list[dict]     # output of reverse_check
    fake_tag_drops: int         # vocab-external tag refs filtered

    # Control flow
    abort_reason: str
