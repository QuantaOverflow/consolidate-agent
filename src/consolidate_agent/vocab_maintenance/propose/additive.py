"""Additive new-tag coverage: attach a freshly-proposed tag to existing records.

Problem (asymmetry in propose_new): NewTagProposal only adds the tag to vocab.
apply.py does NOT touch any record's selected_tags. So already-assigned
records never receive the new tag, even when it's semantically a great fit
for them.

This module bridges that gap:
  1. Embedding-filter: find records whose themes are close to the new tag's
     definition (cheap; uses similarity.py).
  2. LLM binary check: for the filtered candidates, ask LLM "does the new tag
     ALSO apply to this record, given its current tags?".
  3. Record append: for "yes" decisions, add the new tag to selected_tags
     (does NOT remove or modify existing tags — additive only).

This preserves the invariant the user asked about: old tag-record
relationships never change. The new tag only PILES ON to records where the
LLM confirms semantic fit.
"""
from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model

from ..observability import get_default_logger, invoke_with_retry
from ..similarity import find_records_similar_to_tag


# ── LLM schema ───────────────────────────────────────────────────────────────


class AdditiveTagDecision(BaseModel):
    record_id: str = Field(description="exact record_id from input")
    applies: bool = Field(description="true if new tag also applies in addition to current tags")
    reason: str = Field(description="1 sentence justification")


class AdditiveTagBatch(BaseModel):
    decisions: list[AdditiveTagDecision]


# ── Prompts ──────────────────────────────────────────────────────────────────


ADDITIVE_SYSTEM = """You decide whether a NEW tag should ALSO apply to records that already have other tags.

This is ADDITIVE labeling — you cannot remove or modify existing tags. You only
say whether the new tag belongs ON TOP OF what's already there.

Criteria for `applies=true`:
  - The new tag captures a meaningful aspect of the record's lesson
  - That aspect is NOT already covered by the record's current tags
  - The new tag would be a natural label for this record if it were being
    classified fresh

Criteria for `applies=false`:
  - The new tag overlaps too much with the record's existing tags
  - The record's lesson is tangential to the new tag's concept
  - The match is generic/weak rather than substantive

Prefer false when uncertain. Conservative is better — adding noisy tags hurts
downstream search precision."""


ADDITIVE_USER = """## New tag
name: {tag_name}
definition: {tag_definition}

## Records to evaluate ({n_records} candidates, ordered by descending theme-vs-tag similarity)

{records_block}

For each record, decide whether the new tag applies in addition to its current tags."""


# ── Public API ───────────────────────────────────────────────────────────────


def expand_new_tag_coverage(
    network: Any,                                   # TagRecordNetwork (avoid circular import)
    new_tag_names: list[str],
    *,
    similarity_threshold: float = 0.6,
    top_n: int = 30,
    default_confidence: str = "medium",
) -> dict[str, int]:
    """For each new tag, find additionally-applicable records via embedding + LLM.

    Mutates network.assignments in-place — appends the new tag to selected_tags
    of records where LLM confirms the new tag applies.

    Returns {tag_name: number_of_records_added_to} for reporting.
    """
    logger = get_default_logger()
    settings = Settings()
    model = _chat_model(settings).with_structured_output(AdditiveTagBatch)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ADDITIVE_SYSTEM),
        ("user", ADDITIVE_USER),
    ])

    additions_by_tag: dict[str, int] = {}
    record_index = {a["record_id"]: a for a in network.assignments}

    for tag_name in new_tag_names:
        tag = next((t for t in network.vocab if t["name"] == tag_name), None)
        if tag is None:
            continue  # tag may have been merged/deprecated meanwhile

        # Step 1: embedding-filter candidates
        scored = find_records_similar_to_tag(
            themes=network.themes,
            assignments=network.assignments,
            tag_definition=tag["definition"],
            threshold=similarity_threshold,
            top_n=top_n,
        )
        if not scored:
            print(f"  [expand_coverage] '{tag_name}': 0 candidates above threshold {similarity_threshold}", flush=True)
            logger.event("expand_coverage.no_candidates", tag=tag_name, threshold=similarity_threshold)
            additions_by_tag[tag_name] = 0
            continue

        candidates = [a for _sim, a in scored]
        print(f"  [expand_coverage] '{tag_name}': {len(candidates)} candidates after embedding filter", flush=True)

        # Step 2: LLM binary check (batched in one call)
        records_block = "\n\n".join(
            f"[{i+1}] record_id: {c['record_id']}\n"
            f"    current_tags: {[t['name'] for t in c['selected_tags']]}\n"
            f"    theme: {network.themes.get(c['record_id'], '')[:200]}"
            for i, c in enumerate(candidates)
        )
        msg = prompt.invoke({
            "tag_name": tag_name,
            "tag_definition": tag["definition"],
            "n_records": len(candidates),
            "records_block": records_block,
        })
        result = invoke_with_retry(model, msg, retries=2, caller=f"expand_coverage.{tag_name}", logger=logger)
        if result is None:
            print(f"  [expand_coverage] '{tag_name}': LLM check failed after retries; skipping", flush=True)
            additions_by_tag[tag_name] = 0
            continue

        # Step 3: append tag to "yes" records
        yes_count = 0
        for decision in result.decisions:
            if not decision.applies:
                continue
            record = record_index.get(decision.record_id)
            if record is None:
                continue
            existing_tag_names = {t["name"] for t in record["selected_tags"]}
            if tag_name in existing_tag_names:
                continue  # already tagged (defensive)
            if len(record["selected_tags"]) >= 3:
                continue  # full
            record["selected_tags"].append({
                "name": tag_name,
                "confidence": default_confidence,
            })
            yes_count += 1

        additions_by_tag[tag_name] = yes_count
        print(f"  [expand_coverage] '{tag_name}': +{yes_count} additional records (of {len(candidates)} candidates)", flush=True)
        logger.event(
            "expand_coverage.done",
            tag=tag_name,
            candidates=len(candidates),
            added=yes_count,
            skipped_no=len(candidates) - yes_count,
        )

    return additions_by_tag
