"""Faceted reverse-check pipeline (v2): Matter / Activity / Pattern + lesson_type.

Parallel to measure.py (flat single-space tagging). Both pipelines coexist.
This module adds structured 4-output generation for vocab_v2.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model

# ── Re-use the same TagSelection shape as measure.py ────────────────────────

from consolidate_agent.vocab_maintenance.measure import TagSelection  # noqa: F401


# ── Output models ────────────────────────────────────────────────────────────

MIN_MATTER_TAGS = 1
MAX_PATTERN_TAGS = 2


class FacetedRecordAssignment(BaseModel):
    record_idx: int = Field(description="1-based index within the batch")
    matter_tags: list[TagSelection] = Field(
        min_length=MIN_MATTER_TAGS,
        description="1-2 tags from Matter facet",
    )
    activity_tag: TagSelection = Field(description="exactly 1 tag from Activity facet")
    pattern_tags: list[TagSelection] = Field(
        max_length=MAX_PATTERN_TAGS,
        description="0-2 tags from Pattern facet (may be empty)",
    )
    lesson_type: Literal["anti_pattern", "discovery", "best_practice"]
    reason: str = Field(default="", description="1-sentence justification")


class FacetedBatchOutput(BaseModel):
    assignments: list[FacetedRecordAssignment]


# ── Vocab loader ─────────────────────────────────────────────────────────────


def load_faceted_vocab(vocab_path: Path) -> dict:
    """Load vocab_v2.json.

    Returns the full parsed dict with keys:
      facets.matter.tags / facets.activity.tags / facets.pattern.tags
      lesson_type.values
    """
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    return data


# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You classify engineering knowledge records using a FACETED vocabulary.
There are 4 independent fields per record. Fill ALL 4 — they do NOT influence each other.

═══ FACET RULES ═══

① MATTER (what it's about): choose 1-2 tags from the Matter list.
  - 1 tag for single-subject records; 2 tags only when the record genuinely spans two subjects at high confidence.
  - Matter tags describe technical subjects (e.g., persistence_db, llm_agent_runtime).

② ACTIVITY (what kind of work surfaced the lesson): choose EXACTLY 1 tag from the Activity list.
  - This is the engineering activity happening when the lesson was learned (designing, debugging, refactoring, etc.).
  - Do not leave it empty. There is always one dominant activity.

③ PATTERN (the abstract recurring lesson): choose 0-2 tags from the Pattern list.
  - Pattern is OPTIONAL. If the record is purely a tool-discovery with no general pattern, use [].
  - Use 1-2 only when the abstract lesson clearly matches the pattern definition.

④ LESSON_TYPE (lesson polarity): choose exactly 1 from: anti_pattern | discovery | best_practice.
  - anti_pattern: record tone is "must not / avoid / never / breaks / fragile / wrong approach"
  - discovery: record tone is "surprisingly / actually / silently / contrary to expectation / behavior X is Y (non-obvious)"
  - best_practice: record tone is "should / prefer / use X / recommend / correct way"
  - When ambiguous between anti_pattern and discovery, prefer anti_pattern if there is a clear mistake to avoid.

═══ CROSS-FACET INDEPENDENCE ═══
The tags in each facet are selected independently. Choosing `persistence_db` in Matter
does NOT restrict which Activity or Pattern you pick. Treat each facet as a separate question.

═══ IMPORTANT: DO NOT MIX FACETS ═══
- Only use Matter tags in `matter_tags`.
- Only use Activity tags in `activity_tag`.
- Only use Pattern tags in `pattern_tags`.
Putting a Matter tag in `activity_tag` or vice versa is a hard error.

═══ CONFIDENCE ═══
- high: the facet definition explicitly names the record's core concept.
- medium: the definition covers this concept as a documented variant.
- Do not use low confidence — if the best match is only "low", leave pattern_tags empty (for Pattern) or
  pick the closest tag at medium (for Matter/Activity where something must be chosen).

═══ WORKED EXAMPLES ═══

Example A — Record: "Using MemorySaver as checkpointer does not persist state across process restarts"
  matter_tags   : [{{"name": "langgraph_state", "confidence": "high"}}]
  activity_tag  : {{"name": "designing", "confidence": "high"}}
  pattern_tags  : [{{"name": "silent_failure", "confidence": "high"}}]
  lesson_type   : "anti_pattern"
  reason        : "MemorySaver silently loses state; the engineer was making a design-time mistake."

Example B — Record: "asyncio.to_thread wraps blocking DB calls; calling in async context without it causes event-loop stall"
  matter_tags   : [{{"name": "async_concurrency", "confidence": "high"}}, {{"name": "persistence_db", "confidence": "medium"}}]
  activity_tag  : {{"name": "integrating", "confidence": "high"}}
  pattern_tags  : [{{"name": "explicit_contract", "confidence": "medium"}}]
  lesson_type   : "best_practice"
  reason        : "Recommended way to bridge blocking IO into async; integrating two subsystems."

Example C — Record: "git fast-forward merge leaves no merge commit; git log looks like a straight line"
  matter_tags   : [{{"name": "git_vcs", "confidence": "high"}}]
  activity_tag  : {{"name": "debugging", "confidence": "medium"}}
  pattern_tags  : []
  lesson_type   : "discovery"
  reason        : "Non-obvious git behavior; no abstract pattern — pure tool-specific discovery."
"""

USER_PROMPT = """## Matter facet ({matter_count} tags)

{matter_vocab}

## Activity facet ({activity_count} tags)

{activity_vocab}

## Pattern facet ({pattern_count} tags)

{pattern_vocab}

## Records to classify ({batch_size} records)

{records}

For each record (record_idx 1..{batch_size}), output all 4 fields: matter_tags, activity_tag, pattern_tags, lesson_type."""


# ── Prompt formatters ────────────────────────────────────────────────────────


def _format_facet(tags: list[dict]) -> str:
    return "\n".join(f"- {t['name']}: {t['definition']}" for t in tags)


def _format_records(records: list[dict]) -> str:
    lines = []
    for idx, r in enumerate(records, 1):
        insight = r.get("insight", "")[:300]
        lines.append(f"[{idx}] {r['title']}\n    {insight}")
    return "\n\n".join(lines)


# ── Batch runner (thread-safe) ────────────────────────────────────────────────


def _run_faceted_batch(
    batch_idx: int,
    batch_records: list[dict],
    model,
    prompt: ChatPromptTemplate,
    vocab: dict,
    total_batches: int,
) -> tuple[int, list[dict], float]:
    """Run one batch of faceted tagging. Returns (batch_idx, assignments, elapsed)."""
    from .observability import invoke_with_retry

    matter_tags = vocab["facets"]["matter"]["tags"]
    activity_tags = vocab["facets"]["activity"]["tags"]
    pattern_tags = vocab["facets"]["pattern"]["tags"]

    t0 = time.perf_counter()
    messages = prompt.invoke({
        "matter_count": len(matter_tags),
        "matter_vocab": _format_facet(matter_tags),
        "activity_count": len(activity_tags),
        "activity_vocab": _format_facet(activity_tags),
        "pattern_count": len(pattern_tags),
        "pattern_vocab": _format_facet(pattern_tags),
        "batch_size": len(batch_records),
        "records": _format_records(batch_records),
    })

    result: FacetedBatchOutput | None = invoke_with_retry(
        model, messages, retries=3, caller=f"reverse_check_faceted.batch_{batch_idx}",
    )

    elapsed = time.perf_counter() - t0

    if result is None:
        out = [_fallback_assignment(rec) for rec in batch_records]
        return batch_idx, out, elapsed

    by_idx = {a.record_idx: a for a in result.assignments}
    out = []
    for rec_idx, rec in enumerate(batch_records, 1):
        a = by_idx.get(rec_idx)
        if a is None:
            out.append(_fallback_assignment(rec))
            continue
        out.append({
            "record_id": rec["record_id"],
            "title": rec["title"],
            "matter_tags": [t.model_dump() for t in a.matter_tags],
            "activity_tag": a.activity_tag.model_dump(),
            "pattern_tags": [t.model_dump() for t in a.pattern_tags],
            "lesson_type": a.lesson_type,
            "missing_matter": len(a.matter_tags) == 0,
            "reason": a.reason,
        })
    return batch_idx, out, elapsed


def _fallback_assignment(rec: dict) -> dict:
    import logging
    logging.getLogger(__name__).warning(
        "faceted batch failed for record %s; emitting empty assignment", rec["record_id"]
    )
    return {
        "record_id": rec["record_id"],
        "title": rec["title"],
        "matter_tags": [],
        "activity_tag": {},
        "pattern_tags": [],
        "lesson_type": None,
        "missing_matter": True,
        "reason": "batch failure: LLM returned None or validation error after 3 retries",
    }


# ── Public pipeline function ──────────────────────────────────────────────────


def reverse_check_faceted_subset(
    records: list[dict],
    vocab: dict,
    *,
    batch_size: int = 5,
    concurrency: int = 5,
) -> list[dict]:
    """Run faceted tagging on records. Returns list of assignment dicts."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not records:
        return []

    settings = Settings()
    model = _chat_model(settings).with_structured_output(FacetedBatchOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    total_batches = (len(records) + batch_size - 1) // batch_size
    batches = [
        (i, records[i * batch_size: (i + 1) * batch_size])
        for i in range(total_batches)
    ]

    results_by_idx: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(_run_faceted_batch, idx, recs, model, prompt, vocab, total_batches): idx
            for idx, recs in batches
        }
        for fut in as_completed(futures):
            idx, batch_out, _ = fut.result()
            results_by_idx[idx] = batch_out

    return [r for i in sorted(results_by_idx) for r in results_by_idx[i]]
