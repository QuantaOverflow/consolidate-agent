"""Stage 1 v3 — minimal tagging pipeline.

Per ADR-0004, Activity and Pattern facets were dropped. Each record gets:
  - matter_tags : 1-2 from Matter facet (16 tags)
  - lesson_type : 1 from enum {anti_pattern, discovery, best_practice}

Per-record LLM call outputs both fields in one structured generation.
Parallels measure_v2 but with the 2-field schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model


class TagSelection(BaseModel):
    name: str = Field(description="exact tag name from Matter vocab")
    confidence: Literal["high", "medium"] = Field(description="how well does this tag fit")


class MinimalRecordAssignment(BaseModel):
    record_idx: int = Field(description="1-based index within the batch")
    matter_tags: list[TagSelection] = Field(
        min_length=1, max_length=2,
        description="1-2 Matter facet tags",
    )
    lesson_type: Literal["anti_pattern", "discovery", "best_practice"]
    reason: str = Field(default="", description="brief justification, optional")


class MinimalBatchOutput(BaseModel):
    assignments: list[MinimalRecordAssignment]


def load_minimal_vocab(vocab_path: Path) -> dict:
    """Load vocab_v3.json (Matter + lesson_type only)."""
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    facets = data.get("facets", {})
    if "matter" not in facets:
        raise ValueError("vocab missing facets.matter")
    if "lesson_type" not in data:
        raise ValueError("vocab missing lesson_type")
    return data


SYSTEM_PROMPT = """You classify engineering knowledge records using a SINGLE-FACET vocabulary.

There are 2 fields to output per record:

① MATTER (what it's about) — choose 1-2 tags from the Matter list.
   - **Prefer 2 tags when the lesson genuinely touches two technical subjects.**
     Most engineering lessons involving integration, behavior across layers, or
     cross-domain bugs span 2 Matter subjects. Single-subject records are the
     minority.
   - Indicators that 2 tags fit:
     · Record names two distinct technical areas explicitly (e.g., "SQLite +
       concurrent writes", "FastAPI middleware + auth header", "LLM + JSON parsing")
     · Record describes one tech causing behavior in another (e.g., "env var
       changes affect HTTP client routing", "git index lock from filesystem perms")
     · Record's fix or workaround spans two layers
   - Use 1 tag only when the record is genuinely SINGLE-subject (a lesson about
     one tool's quirk with no interaction). Examples: "git fast-forward merge
     behavior", "TypedDict.get() returns None".
   - Output the EXACT tag name from the vocab; do not invent new names.

② LESSON_TYPE — choose exactly 1: anti_pattern | discovery | best_practice.
   - anti_pattern: record's core insight is that X fails, breaks, or is wrong.
     "must do Z to prevent failure Y" → anti_pattern (failure Y is the lesson, not the technique Z).
   - discovery: non-obvious behavior of a tool/system ("X surprisingly does Y", "X does NOT do Z").
     The insight is informational, not about a mistake.
   - best_practice: recommendation for a preferred technique where NOT following causes no failure.
     "prefer Y", "use Y for cleaner code", "asyncio.to_thread is the correct way".
   - When ambiguous between anti_pattern and discovery: prefer anti_pattern only if a clear failure mode is described.
   - When ambiguous between anti_pattern and best_practice: if NOT following the advice causes failures → anti_pattern; if it's just "cleaner" → best_practice.

CONFIDENCE for matter_tags:
   - high: the tag definition explicitly names the record's core concept.
   - medium: the definition covers this concept as a documented variant.

═══ WORKED EXAMPLES ═══

Example A (1 tag — single-subject):
  Record: "git fast-forward merge leaves no merge commit; git log looks like a straight line"
  matter_tags : [{{"name": "git_vcs", "confidence": "high"}}]
  lesson_type : "discovery"
  reason: Pure git behavior, no cross-domain interaction.

Example B (2 tags — cross-domain interaction):
  Record: "asyncio.to_thread wraps blocking DB calls; calling in async context without it causes event-loop stall"
  matter_tags : [{{"name": "async_concurrency", "confidence": "high"}}, {{"name": "persistence_db", "confidence": "medium"}}]
  lesson_type : "best_practice"
  reason: Both async runtime AND DB driver behavior matter.

Example C (2 tags — env affects HTTP):
  Record: "HTTP client environment variables can break local service calls"
  matter_tags : [{{"name": "http_api", "confidence": "high"}}, {{"name": "config_env", "confidence": "high"}}]
  lesson_type : "anti_pattern"
  reason: env var (config_env) affects HTTP routing — both subjects involved.

Example D (2 tags — testing + DB):
  Record: "Integration tests with real SQLite require explicit thread safety verification"
  matter_tags : [{{"name": "testing_framework", "confidence": "high"}}, {{"name": "persistence_db", "confidence": "high"}}]
  lesson_type : "best_practice"
  reason: Lesson is specifically about test setup for DB concurrency — both subjects equally central.

Example E (1 tag — pure framework behavior):
  Record: "TypedDict.get() does not provide safe fallbacks; returns None ignoring default"
  matter_tags : [{{"name": "langgraph_state", "confidence": "high"}}]
  lesson_type : "discovery"
  reason: Pure typing behavior, no second domain.
"""


USER_PROMPT = """## Matter facet ({matter_count} tags)

{matter_vocab}

## lesson_type values

- anti_pattern
- discovery
- best_practice

## Records to tag ({batch_size} records)

{records}

For each record (by record_idx 1..{batch_size}), select 1-2 matter tags + lesson_type.
Use ONLY tag names from the Matter facet list above. Do not invent new tag names."""


def _format_matter(vocab: dict) -> str:
    return "\n".join(
        f"- {t['name']}: {t['definition']}"
        for t in vocab["facets"]["matter"]["tags"]
    )


def _format_records(records: list[dict]) -> str:
    lines = []
    for idx, r in enumerate(records, 1):
        insight = r["insight"][:300]
        lines.append(f"[{idx}] {r['title']}\n    {insight}")
    return "\n\n".join(lines)


def _run_minimal_batch(
    batch_idx: int,
    batch_records: list[dict],
    model,
    prompt,
    vocab: dict,
) -> tuple[int, list[dict]]:
    """Run one LLM batch. Returns (batch_idx, list of assignments)."""
    from .observability import invoke_with_retry

    messages = prompt.invoke({
        "matter_count": len(vocab["facets"]["matter"]["tags"]),
        "matter_vocab": _format_matter(vocab),
        "batch_size": len(batch_records),
        "records": _format_records(batch_records),
    })
    result: MinimalBatchOutput | None = invoke_with_retry(
        model, messages, retries=3, caller=f"reverse_check_v3.batch_{batch_idx}",
    )
    if result is None:
        # Batch failed — emit empty assignments so downstream knows
        return batch_idx, [
            {
                "record_id": rec["record_id"],
                "title": rec["title"],
                "matter_tags": [],
                "lesson_type": None,
                "reason": "batch failure: LLM returned None after 3 retries",
            }
            for rec in batch_records
        ]

    by_idx = {a.record_idx: a for a in result.assignments}
    out = []
    for rec_idx, rec in enumerate(batch_records, 1):
        a = by_idx.get(rec_idx)
        if a is None:
            out.append({
                "record_id": rec["record_id"],
                "title": rec["title"],
                "matter_tags": [],
                "lesson_type": None,
                "reason": "missing in batch output",
            })
        else:
            out.append({
                "record_id": rec["record_id"],
                "title": rec["title"],
                "matter_tags": [t.model_dump() for t in a.matter_tags],
                "lesson_type": a.lesson_type,
                "reason": a.reason,
            })
    return batch_idx, out


def reverse_check_minimal_subset(
    records: list[dict],
    vocab: dict,
    *,
    batch_size: int = 5,
    concurrency: int = 5,
) -> list[dict]:
    """Run minimal tagging (matter + lesson_type) on records concurrently."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not records:
        return []

    settings = Settings()
    model = _chat_model(settings).with_structured_output(MinimalBatchOutput)
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
            ex.submit(_run_minimal_batch, idx, recs, model, prompt, vocab): idx
            for idx, recs in batches
        }
        for fut in as_completed(futures):
            idx, batch_out = fut.result()
            results_by_idx[idx] = batch_out

    return [r for i in sorted(results_by_idx) for r in results_by_idx[i]]
