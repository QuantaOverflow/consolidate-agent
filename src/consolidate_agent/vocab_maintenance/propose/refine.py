"""Propose a tag refinement: sharpened definition + small prune list.

Triggered when diagnose flags a tag in `forced_fit_candidates` (records under
the tag sit far from its definition on average). The LLM sees the tag's
current definition, low-fit outlier records (prune candidates), a few
high-fit records (reference for what the tag should keep), and produces:

  - a sharpened new definition tightening the boundary
  - a prune list (subset of the shown outliers, capped at 10)

`validate_refine_proposal` enforces the cosine-distance invariant: every
prune target must have cosine < threshold (default 0.7) to the NEW
definition; otherwise the record still fits the refined tag and prune is
invalid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model

from ..apply import InvalidProposal, RefineTagProposal
from ..observability import get_default_logger, invoke_with_retry
from ..probes import _load_record_details, inspect_outliers
from ..similarity import _embed, cosine


COSINE_KEEP_THRESHOLD = 0.7   # any prune target with sim >= this is rejected
MAX_PRUNE = 10                # mirrors apply._REFINE_PRUNE_HARD_CAP
N_OUTLIERS_SHOWN = 12         # LLM sees up to this many outlier candidates
N_HIGH_FIT_SHOWN = 4          # plus a few high-fit reference records


class RefineSuggestion(BaseModel):
    """Structured LLM output for one tag's refinement."""

    why_refine: str = Field(description="STEP 1: 1 sentence — what concept this tag should NARROW to.")
    boundary_check: str = Field(description="STEP 2: 1 sentence — what kinds of records should NO LONGER fall under the refined tag, with examples drawn from the outlier list.")
    new_definition: str = Field(description="STEP 3: refined definition. Tightens the boundary stated in step 1. 1-3 sentences.")
    prune_record_ids: list[str] = Field(
        description=f"STEP 4: record_ids (from the OUTLIER list shown) that should be detached under the refined definition. <= {MAX_PRUNE}. Empty list is acceptable if no record clearly falls outside the new boundary.",
        max_length=MAX_PRUNE,
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="high = clear semantic split between kept and pruned; medium = mostly clear with edge cases; low = boundary is hard to draw."
    )


REFINE_SYSTEM = """You are refining a tag in a knowledge vocabulary.

Inputs:
- Tag name + current definition (likely too vague — records have drifted in).
- Outlier records: assigned to this tag but with LOW cosine to the current definition. These are the prune candidates.
- High-fit reference records: assigned to this tag with HIGH cosine — representative of what the tag should preserve.
- Full vocab (so you don't accidentally drift the new definition into another tag's territory).

Your job:
1. Identify the CONCEPT the tag should narrow to, using the high-fit records as positive examples.
2. State which kinds of records should NO LONGER fall under the refined tag, citing 1-2 outlier records as examples.
3. Write the refined definition — sharper, less ambiguous than the current one, but still describing the same kept records.
4. From the outlier list, select records that clearly DON'T fit the refined definition. These become prune targets.

Constraints:
- prune_record_ids must be a subset of the OUTLIER record_ids shown.
- prune_record_ids length <= 10.
- If the outlier set looks legitimate under the current definition (LLM-tagger was right, embedding signal was noisy), output prune_record_ids=[] and a near-unchanged definition. Honest "no refinement needed" beats forced pruning.
- The new definition must NOT overlap meaningfully with any other vocab tag — refining should sharpen this tag's territory, not steal from a neighbor."""


REFINE_USER = """## Tag to refine
- Name: {tag_name}
- Current definition: {tag_def}
- Current usage: {usage} records

## Outlier records assigned to this tag (low cosine to current def — prune candidates)
## NOTE: these are pre-filtered to records that have at least one OTHER tag, so pruning is safe wrt orphans.

{outliers}

## High-fit reference records (preserve these under the refined def)

{high_fit}

## Other vocab (avoid drifting into these tags' territory)

{other_vocab}

Produce a refined definition + prune list. If outliers actually fit, emit prune_record_ids=[]."""


def _format_records(records: list[dict]) -> str:
    if not records:
        return "  (none)"
    return "\n\n".join(
        f"  [{r['record_id']}] {r.get('title','')}\n    {r.get('insight','')[:200]}"
        for r in records
    )


def _format_other_vocab(vocab: list[dict], focus_tag: str) -> str:
    return "\n".join(f"- {t['name']}: {t['definition'][:120]}" for t in vocab if t["name"] != focus_tag)


def validate_refine_proposal(
    proposal: RefineTagProposal,
    db_path: Path,
    *,
    cosine_threshold: float = COSINE_KEEP_THRESHOLD,
) -> None:
    """Enforce the cosine-distance invariant on prune targets.

    Loads each prune target's content (title+insight) from db, embeds, then
    compares to the proposal's new_definition embedding. If any target's
    cosine >= threshold, the record still fits the refined tag — raise
    InvalidProposal.

    Called by propose_refine_fn after the LLM emits a candidate proposal.
    apply.py's structural checks (orphan, size, tag-membership) are separate.
    """
    if not proposal.prune_record_ids:
        return

    details = _load_record_details(db_path, list(proposal.prune_record_ids))
    missing = [rid for rid in proposal.prune_record_ids if rid not in details]
    if missing:
        raise InvalidProposal(
            f"refine validate: {len(missing)} prune target(s) not found in db: {missing}"
        )

    new_def_emb = _embed(proposal.new_definition)
    violations: list[tuple[str, float]] = []
    for rid in proposal.prune_record_ids:
        d = details[rid]
        rec_emb = _embed(f"{d['title']}: {d['insight'][:200]}")
        sim = cosine(new_def_emb, rec_emb)
        if sim >= cosine_threshold:
            violations.append((rid, sim))
    if violations:
        sample = ", ".join(f"{rid} sim={s:.3f}" for rid, s in violations[:3])
        raise InvalidProposal(
            f"refine validate: {len(violations)} prune target(s) still fit new def "
            f"(cosine >= {cosine_threshold}): {sample}"
        )


def _multi_tag_record_ids(assignments: list[dict], tag_name: str) -> set[str]:
    """Record ids carrying `tag_name` AND at least one other tag.

    Pruning these is safe wrt the orphan invariant. Records tagged solely
    with `tag_name` are excluded as prune candidates upfront so the LLM
    can't propose orphan-inducing prunes.
    """
    out: set[str] = set()
    for a in assignments:
        if a.get("missing"):
            continue
        names = {t["name"] for t in a.get("selected_tags", [])}
        if tag_name in names and len(names) >= 2:
            out.add(a["record_id"])
    return out


def _split_outliers_and_high_fit(
    inspect_result: dict, total_n: int, n_outliers: int, n_high_fit: int,
    *, prunable_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """`inspect_outliers` returns bottom-N. We rerun with a larger n to get a
    fuller view, then split into low-fit prune candidates + high-fit reference.

    If `prunable_ids` is provided, outliers are filtered to records that
    survive a prune (≥2 tags currently). High-fit references are NOT filtered
    — they're shown as examples regardless of tag arity.
    """
    rows = inspect_result.get("bottom_n_by_fit", [])
    if prunable_ids is not None:
        low_candidates = [r for r in rows if r["record_id"] in prunable_ids]
    else:
        low_candidates = list(rows)
    low = low_candidates[:n_outliers]
    high = rows[-n_high_fit:] if len(rows) > n_outliers else []
    high = [r for r in high if r["record_id"] not in {x["record_id"] for x in low}]
    return low, high


def _extract_focus_tags(focus: str, vocab: list[dict]) -> list[str]:
    """Mirror of propose/deprecate.py:_extract_focus_tags."""
    if not focus:
        return []
    focus_lower = focus.lower()
    return [t["name"] for t in vocab if t["name"].lower() in focus_lower]


def propose_refine_fn(
    vocab: list[dict],
    assignments: list[dict],
    focus: str = "",
    *,
    db_path: Path,
    max_targets: int = 1,
) -> list[RefineTagProposal]:
    """In-memory propose_refine for the agent loop.

    Reads `focus` to determine which tag(s) to refine. Requires `db_path`
    for content embedding (no fallback). Returns at most `max_targets`
    validated proposals.
    """
    logger = get_default_logger()
    target_tags = _extract_focus_tags(focus, vocab)
    if not target_tags:
        logger.event("propose_refine.no_focus", focus=focus[:200])
        return []

    settings = Settings()
    model = _chat_model(settings).with_structured_output(RefineSuggestion)
    prompt = ChatPromptTemplate.from_messages([("system", REFINE_SYSTEM), ("user", REFINE_USER)])

    proposals: list[RefineTagProposal] = []
    for tag_name in target_tags[:max_targets]:
        tag = next((t for t in vocab if t["name"] == tag_name), None)
        if tag is None:
            continue

        outliers_dump = inspect_outliers(
            vocab, assignments, db_path, tag_name, n=N_OUTLIERS_SHOWN + N_HIGH_FIT_SHOWN + 4
        )
        if "error" in outliers_dump:
            logger.event("propose_refine.inspect_failed", tag=tag_name, error=outliers_dump["error"])
            continue
        usage = outliers_dump.get("record_count", 0)
        prunable = _multi_tag_record_ids(assignments, tag_name)
        low, high = _split_outliers_and_high_fit(
            outliers_dump, usage, N_OUTLIERS_SHOWN, N_HIGH_FIT_SHOWN,
            prunable_ids=prunable,
        )
        if not low:
            logger.event(
                "propose_refine.no_prunable_outliers", tag=tag_name,
                usage=usage, prunable_count=len(prunable),
            )
            continue

        msg = prompt.invoke({
            "tag_name": tag_name,
            "tag_def": tag["definition"],
            "usage": usage,
            "outliers": _format_records(low),
            "high_fit": _format_records(high),
            "other_vocab": _format_other_vocab(vocab, tag_name),
        })
        suggestion: RefineSuggestion | None = invoke_with_retry(
            model, msg, retries=3, caller=f"propose_refine.{tag_name}", logger=logger,
        )
        if suggestion is None:
            continue
        if not suggestion.prune_record_ids and suggestion.new_definition.strip() == tag["definition"].strip():
            logger.event("propose_refine.no_change", tag=tag_name)
            continue

        # Filter prune_record_ids to those actually shown as outliers (defensive).
        shown_ids = {r["record_id"] for r in low}
        cleaned_prune = [rid for rid in suggestion.prune_record_ids if rid in shown_ids][:MAX_PRUNE]

        proposal = RefineTagProposal(
            tag=tag_name,
            new_definition=suggestion.new_definition,
            prune_record_ids=tuple(cleaned_prune),
        )
        try:
            validate_refine_proposal(proposal, db_path)
        except InvalidProposal as e:
            logger.event("propose_refine.invalid", tag=tag_name, error=str(e)[:200])
            continue
        proposals.append(proposal)
        logger.event(
            "propose_refine.ok",
            tag=tag_name,
            prune_count=len(cleaned_prune),
            confidence=suggestion.confidence,
        )
    return proposals
