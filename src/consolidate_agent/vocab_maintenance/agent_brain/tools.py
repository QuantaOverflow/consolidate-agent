"""Brain-callable tools wrapping existing probe/propose functions.

Each tool:
  - accepts dict (from LLM tool call args)
  - returns {"ok": bool, "result": dict | None, "error": str | None}
  - never raises — errors are structured output
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class BrainContext:
    db_path: Path
    vocab: list[dict]
    assignments: list[dict]
    golden: list[dict]
    proposal_cache: dict[str, Any] = field(default_factory=dict)
    _tool_cache_this_round: dict = field(default_factory=dict)
    # tracks which subject (set of tag names) was previewed this round —
    # enforces "one proposal per round" so the LLM completes inspect→propose→decide
    # for one target before opening another preview.
    _proposed_subject_this_round: frozenset = field(default_factory=frozenset)


def _ok(result: dict) -> dict:
    return {"ok": True, "result": result, "error": None}


def _err(msg: str) -> dict:
    return {"ok": False, "result": None, "error": msg}


def _check_subject(ctx: BrainContext, subject: frozenset) -> dict | None:
    """Block previewing a NEW subject if a previous preview already produced a
    valid proposal this round.

    Subject is locked only when a preview returned a real proposal (caller does
    that via _lock_subject). A failed preview (judge said 'no') does NOT lock,
    so the LLM is free to try a different target. This avoids the R3-style
    batch-preview waste while still letting LLM recover from rejected proposals.
    """
    current = ctx._proposed_subject_this_round
    if not current:
        return None
    if current == subject:
        return None  # re-previewing same subject is allowed
    return _err(
        f"this round already has a pending proposal for {sorted(current)}. "
        f"You MUST decide on that subject (action=split/refine/deprecate/merge) "
        f"before previewing another. End this round with a decision first."
    )


def _lock_subject(ctx: BrainContext, subject: frozenset) -> None:
    ctx._proposed_subject_this_round = subject


def _simulate_diff(
    ctx: BrainContext, proposal, affected_tag_for_size: str | None,
    sample_size: int = 10,
) -> dict:
    """Sandbox-apply the proposal on a copy and compute a before/after diff.

    record_diffs are capped at `sample_size` and enriched with title + insight
    snippet from the DB so the specialist reviewer can see records' actual
    semantics — not just record_ids and tag lists.
    """
    from copy import deepcopy
    from consolidate_agent.vocab_maintenance.apply import apply_proposal
    from consolidate_agent.vocab_maintenance.probes import _load_record_details

    try:
        sandbox_vocab = deepcopy(ctx.vocab)
        sandbox_assignments = deepcopy(ctx.assignments)
        new_vocab, new_assignments = apply_proposal(
            sandbox_vocab, sandbox_assignments, proposal
        )
    except Exception as e:
        return {"sandbox_error": str(e)[:200]}

    new_by_id = {a["record_id"]: a for a in new_assignments}

    record_diffs: list[dict] = []
    orphan_count = 0
    for a in ctx.assignments:
        if a.get("missing"):
            continue
        rid = a["record_id"]
        before = [t["name"] for t in a.get("selected_tags", [])]
        new_a = new_by_id.get(rid)
        after = (
            [t["name"] for t in new_a.get("selected_tags", [])]
            if new_a else []
        )
        if set(before) != set(after):
            record_diffs.append({
                "record_id": rid,
                "before_tags": before,
                "after_tags": after,
                "becomes_orphan": len(after) == 0,
            })
            if len(after) == 0:
                orphan_count += 1

    target_before = target_after = None
    if affected_tag_for_size:
        target_before = sum(
            1 for a in ctx.assignments
            if not a.get("missing") and any(t["name"] == affected_tag_for_size for t in a.get("selected_tags", []))
        )
        target_after = sum(
            1 for a in new_assignments
            if not a.get("missing") and any(t["name"] == affected_tag_for_size for t in a.get("selected_tags", []))
        )

    # Enrich the sample with record content (title + insight snippet) so the
    # specialist can audit semantic appropriateness, not just structural fit.
    sample = record_diffs[:sample_size]
    if sample:
        details = _load_record_details(ctx.db_path, [r["record_id"] for r in sample])
        for r in sample:
            d = details.get(r["record_id"], {})
            r["title"] = (d.get("title") or "")[:120]
            r["insight_snippet"] = (d.get("insight") or "")[:500]

    return {
        "target_size_before": target_before,
        "target_size_after": target_after,
        "affected_record_count": len(record_diffs),
        "orphan_count": orphan_count,
        "record_diffs": sample,
    }


def _enrich_sample_records(db_path, record_ids: list[str], limit: int = 5) -> list[dict]:
    """Helper for action-specific sample blocks: fetch title + insight for records."""
    from consolidate_agent.vocab_maintenance.probes import _load_record_details
    ids = list(record_ids)[:limit]
    if not ids:
        return []
    details = _load_record_details(db_path, ids)
    return [
        {
            "record_id": rid,
            "title": (details.get(rid, {}).get("title") or "")[:120],
            "insight_snippet": (details.get(rid, {}).get("insight") or "")[:500],
        }
        for rid in ids
    ]


# ── impl functions ────────────────────────────────────────────────────────────


def _impl_assess_global_health(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.probes import compute_heterogeneous_tags

    try:
        tag_count = len(ctx.vocab)
        record_count = sum(1 for a in ctx.assignments if not a.get("missing"))
        tags_per_record: list[int] = [
            len(a.get("selected_tags", []))
            for a in ctx.assignments
            if not a.get("missing")
        ]
        mean_tags = round(sum(tags_per_record) / len(tags_per_record), 2) if tags_per_record else 0.0

        tag_sizes: dict[str, int] = {}
        for a in ctx.assignments:
            if a.get("missing"):
                continue
            for t in a.get("selected_tags", []):
                tag_sizes[t["name"]] = tag_sizes.get(t["name"], 0) + 1

        distribution_buckets = {"1-10": 0, "11-30": 0, "31-100": 0, "101+": 0}
        for sz in tag_sizes.values():
            if sz <= 10:
                distribution_buckets["1-10"] += 1
            elif sz <= 30:
                distribution_buckets["11-30"] += 1
            elif sz <= 100:
                distribution_buckets["31-100"] += 1
            else:
                distribution_buckets["101+"] += 1

        heterogeneous = compute_heterogeneous_tags(ctx.vocab, ctx.assignments, ctx.db_path)
        top_heterogeneous = [
            {"tag": h["tag"], "coherence": h["coherence"], "record_count": h["record_count"]}
            for h in heterogeneous[:8]
        ]

        return _ok({
            "tag_count": tag_count,
            "record_count": record_count,
            "mean_tags_per_record": mean_tags,
            "top_heterogeneous": top_heterogeneous,
            "tag_size_distribution": distribution_buckets,
            "tag_sizes": {k: v for k, v in sorted(tag_sizes.items(), key=lambda x: -x[1])},
        })
    except Exception as e:
        return _err(f"assess_global_health failed: {e}")


def _impl_inspect_tag(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.probes import (
        compute_heterogeneous_tags,
        inspect_tag,
    )

    tag_name = args.get("tag_name", "")
    n_samples = int(args.get("n_samples", 5))
    if not tag_name:
        return _err("tag_name is required")
    try:
        result = inspect_tag(ctx.vocab, ctx.assignments, ctx.db_path, tag_name, n=n_samples)

        # Try to get coherence from heterogeneous computation
        coherence: float | None = None
        try:
            heterogeneous = compute_heterogeneous_tags(ctx.vocab, ctx.assignments, ctx.db_path)
            for h in heterogeneous:
                if h["tag"] == tag_name:
                    coherence = h["coherence"]
                    break
        except Exception:
            pass

        result["coherence"] = coherence
        result["neighbor_tags"] = result.pop("top_co_occurring_tags", [])
        return _ok(result)
    except Exception as e:
        return _err(f"inspect_tag({tag_name}) failed: {e}")


def _impl_inspect_records(args: dict, ctx: BrainContext) -> dict:
    """Return full content + current tag membership for specific record_ids.

    Designed for the specialist reviewer to audit a small set of records
    (e.g., a proposal's prune list) when title+snippet in the preview
    aren't enough to judge semantic appropriateness.
    """
    from consolidate_agent.vocab_maintenance.probes import _load_record_details

    record_ids = args.get("record_ids") or []
    if isinstance(record_ids, str):
        # tolerate accidental scalar input
        record_ids = [record_ids]
    if not record_ids:
        return _err("record_ids is required (list of record_id strings)")
    ids = list(record_ids)[:10]  # cap to avoid prompt overflow

    try:
        details = _load_record_details(ctx.db_path, ids)
    except Exception as e:
        return _err(f"inspect_records failed: {e}")

    by_id = {a["record_id"]: a for a in ctx.assignments}
    out = []
    for rid in ids:
        d = details.get(rid, {})
        a = by_id.get(rid, {})
        out.append({
            "record_id": rid,
            "title": d.get("title", "")[:200],
            "insight": (d.get("insight") or "")[:1500],
            "current_matter_tags": [t["name"] for t in a.get("selected_tags", [])],
            "lesson_type": a.get("lesson_type", ""),
        })
    return _ok({
        "queried": len(ids),
        "records": out,
    })


def _impl_compare_tags(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.probes import compare_tag_records

    tag_a = args.get("tag_a", "")
    tag_b = args.get("tag_b", "")
    if not tag_a or not tag_b:
        return _err("tag_a and tag_b are required")
    try:
        result = compare_tag_records(ctx.vocab, ctx.assignments, ctx.db_path, tag_a, tag_b)
        return _ok(result)
    except Exception as e:
        return _err(f"compare_tags({tag_a}, {tag_b}) failed: {e}")


def _impl_propose_split_preview(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.propose.split import propose_split_fn

    tag_name = args.get("tag_name", "")
    if not tag_name:
        return _err("tag_name is required")
    block = _check_subject(ctx, frozenset({tag_name}))
    if block is not None:
        return block
    try:
        proposals = propose_split_fn(
            ctx.vocab, ctx.assignments, tag_name, db_path=ctx.db_path
        )
        if not proposals:
            return _ok({"tag_name": tag_name, "proposals": [], "message": "no split proposed"})

        proposal = proposals[0]
        proposal_id = str(uuid.uuid4())
        ctx.proposal_cache[proposal_id] = proposal
        _lock_subject(ctx, frozenset({tag_name}))

        sub_tags_preview = [
            {
                "name": st["name"],
                "definition": st["definition"],
                "record_count": len(st["record_ids"]),
                # 5 sample records per sub-tag so the reviewer can audit semantic
                # cohesion of each split partition.
                "sample_records": _enrich_sample_records(ctx.db_path, st["record_ids"], limit=5),
            }
            for st in proposal.sub_tags
        ]
        diff = _simulate_diff(ctx, proposal, tag_name)
        return _ok({
            "tag_name": tag_name,
            "proposal_id": proposal_id,
            "sub_tags": sub_tags_preview,
            "total_records": sum(len(st["record_ids"]) for st in proposal.sub_tags),
            "diff": diff,
        })
    except Exception as e:
        return _err(f"propose_split_preview({tag_name}) failed: {e}")


def _impl_propose_refine_preview(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.propose.refine import propose_refine_fn

    tag_name = args.get("tag_name", "")
    if not tag_name:
        return _err("tag_name is required")
    block = _check_subject(ctx, frozenset({tag_name}))
    if block is not None:
        return block
    try:
        proposals = propose_refine_fn(
            ctx.vocab, ctx.assignments, focus=tag_name, db_path=ctx.db_path
        )
        if not proposals:
            return _ok({"tag_name": tag_name, "proposals": [], "message": "no refine proposed"})

        proposal = proposals[0]
        proposal_id = str(uuid.uuid4())
        ctx.proposal_cache[proposal_id] = proposal
        _lock_subject(ctx, frozenset({tag_name}))

        diff = _simulate_diff(ctx, proposal, tag_name)
        return _ok({
            "tag_name": tag_name,
            "proposal_id": proposal_id,
            "new_definition": proposal.new_definition,
            "prune_count": len(proposal.prune_record_ids),
            "diff": diff,
        })
    except Exception as e:
        return _err(f"propose_refine_preview({tag_name}) failed: {e}")


def _impl_propose_deprecate_preview(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.apply import DeprecateProposal, MergeProposal
    from consolidate_agent.vocab_maintenance.propose.deprecate import propose_deprecate_fn

    tag_name = args.get("tag_name", "")
    if not tag_name:
        return _err("tag_name is required")
    block = _check_subject(ctx, frozenset({tag_name}))
    if block is not None:
        return block
    try:
        proposals = propose_deprecate_fn(
            ctx.vocab, ctx.assignments, focus=tag_name, db_path=ctx.db_path
        )
        if not proposals:
            return _ok({"tag_name": tag_name, "proposals": [], "message": "no deprecate proposed (judge said keep)"})

        proposal = proposals[0]
        proposal_id = str(uuid.uuid4())
        ctx.proposal_cache[proposal_id] = proposal
        _lock_subject(ctx, frozenset({tag_name}))

        # Compute affected record summary: records currently under tag_name that would lose it
        affected_records = [
            a["record_id"] for a in ctx.assignments
            if not a.get("missing") and any(t["name"] == tag_name for t in a.get("selected_tags", []))
        ]
        # Orphan count: records whose ONLY matter tag is the one being deprecated
        orphan_count = 0
        for a in ctx.assignments:
            if a.get("missing"):
                continue
            tag_names = [t["name"] for t in a.get("selected_tags", [])]
            if tag_names == [tag_name]:
                orphan_count += 1

        if isinstance(proposal, MergeProposal):
            diff = _simulate_diff(ctx, proposal, proposal.discard_tag)
            # Show what records currently sit under the soon-to-be-discarded tag.
            discard_records_ids = [
                a["record_id"] for a in ctx.assignments
                if not a.get("missing") and any(t["name"] == proposal.discard_tag for t in a.get("selected_tags", []))
            ]
            return _ok({
                "tag_name": tag_name,
                "proposal_id": proposal_id,
                "proposal_type": "merge",
                "keep_tag": proposal.keep_tag,
                "discard_tag": proposal.discard_tag,
                "affected_record_count": len(affected_records),
                "note": "judge recommended merge_to instead of deprecate",
                "discard_records_sample": _enrich_sample_records(ctx.db_path, discard_records_ids, limit=8),
                "diff": diff,
            })
        diff = _simulate_diff(ctx, proposal, tag_name)
        return _ok({
            "tag_name": tag_name,
            "proposal_id": proposal_id,
            "proposal_type": "deprecate",
            "affected_record_count": len(affected_records),
            "orphan_count": orphan_count,
            "note": f"{orphan_count} records would lose their only matter tag if deprecated",
            # Show what records this tag currently holds — to judge whether tag truly is dispensable.
            "target_records_sample": _enrich_sample_records(ctx.db_path, affected_records, limit=8),
            "diff": diff,
        })
    except Exception as e:
        return _err(f"propose_deprecate_preview({tag_name}) failed: {e}")


def _impl_propose_merge_preview(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.propose.merge import propose_merge_fn

    tag_a = args.get("tag_a", "")
    tag_b = args.get("tag_b", "")
    if not tag_a or not tag_b:
        return _err("tag_a and tag_b are required")
    block = _check_subject(ctx, frozenset({tag_a, tag_b}))
    if block is not None:
        return block
    try:
        proposals = propose_merge_fn(
            ctx.vocab, ctx.assignments, focus=f"{tag_a},{tag_b}", db_path=ctx.db_path
        )
        if not proposals:
            return _ok({
                "tag_a": tag_a, "tag_b": tag_b, "proposals": [],
                "message": "no merge proposed (judge said keep separate)",
            })

        proposal = proposals[0]
        proposal_id = str(uuid.uuid4())
        ctx.proposal_cache[proposal_id] = proposal
        _lock_subject(ctx, frozenset({tag_a, tag_b}))

        keep = proposal.keep_tag
        discard = proposal.discard_tag
        affected = [
            a["record_id"] for a in ctx.assignments
            if not a.get("missing") and any(t["name"] == discard for t in a.get("selected_tags", []))
        ]
        diff = _simulate_diff(ctx, proposal, discard)
        return _ok({
            "tag_a": tag_a,
            "tag_b": tag_b,
            "proposal_id": proposal_id,
            "keep_tag": keep,
            "discard_tag": discard,
            "affected_record_count": len(affected),
            "discard_records_sample": _enrich_sample_records(ctx.db_path, affected, limit=8),
            "diff": diff,
        })
    except Exception as e:
        return _err(f"propose_merge_preview({tag_a}, {tag_b}) failed: {e}")


def _impl_apply_proposal(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.apply import apply_proposal

    proposal_id = args.get("proposal_id", "")
    if not proposal_id:
        return _err("proposal_id is required")

    proposal = ctx.proposal_cache.get(proposal_id)
    if proposal is None:
        return _err(f"proposal_id not found in cache: {proposal_id}")

    try:
        new_vocab, new_assignments = apply_proposal(ctx.vocab, ctx.assignments, proposal)
        old_tag_count = len(ctx.vocab)
        old_assignment_count = len(ctx.assignments)

        ctx.vocab.clear()
        ctx.vocab.extend(new_vocab)
        ctx.assignments.clear()
        ctx.assignments.extend(new_assignments)

        # Remove from cache after apply
        del ctx.proposal_cache[proposal_id]

        return _ok({
            "proposal_id": proposal_id,
            "proposal_type": getattr(proposal, "type", "unknown"),
            "vocab_before": old_tag_count,
            "vocab_after": len(ctx.vocab),
            "assignments_count": old_assignment_count,
        })
    except Exception as e:
        return _err(f"apply_proposal({proposal_id}) failed: {e}")


# ── ToolSpec and TOOLS ────────────────────────────────────────────────────────


def call_tool_with_cache(tool: "ToolSpec", args: dict, ctx: BrainContext) -> dict:
    """Wrap tool.impl with per-round deduplication."""
    cache_key = (tool.name, json.dumps(args, sort_keys=True, default=str))
    if cache_key in ctx._tool_cache_this_round:
        cached = ctx._tool_cache_this_round[cache_key]
        return {
            "ok": False,
            "result": None,
            "error": (
                f"You already called `{tool.name}` with these exact args this round. "
                f"Previous result was: {str(cached)[:200]}... "
                f"Use a DIFFERENT tool or different args."
            ),
            "duplicate_call": True,
        }
    result = tool.impl(args, ctx)
    ctx._tool_cache_this_round[cache_key] = result
    return result


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    output_keys: list[str]
    impl: Callable[[dict, BrainContext], dict]


TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="assess_global_health",
        description=(
            "Get overall network statistics: tag count, record count, top heterogeneous tags, "
            "distribution. Call this first to orient yourself."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
        output_keys=["tag_count", "record_count", "mean_tags_per_record", "top_heterogeneous", "tag_size_distribution"],
        impl=_impl_assess_global_health,
    ),
    ToolSpec(
        name="inspect_tag",
        description=(
            "Inspect one specific tag: definition, sample records, coherence, size. "
            "Use when you need detail on a specific tag."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tag_name": {"type": "string"},
                "n_samples": {"type": "integer", "default": 5},
            },
            "required": ["tag_name"],
        },
        output_keys=["name", "definition", "usage_count", "coherence", "sample_records", "neighbor_tags"],
        impl=_impl_inspect_tag,
    ),
    ToolSpec(
        name="inspect_records",
        description=(
            "Return full title + insight + current matter tags for specific record_ids. "
            "Use when you need to audit a small set of records' actual semantics — "
            "e.g., the prune list of a refine proposal, to verify each record really "
            "is off-topic. Capped at 10 records per call."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "record_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["record_ids"],
        },
        output_keys=["queried", "records"],
        impl=_impl_inspect_records,
    ),
    ToolSpec(
        name="compare_tags",
        description=(
            "Compare two tags side by side: definition similarity, record overlap. "
            "Use before deciding if they should merge."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tag_a": {"type": "string"},
                "tag_b": {"type": "string"},
            },
            "required": ["tag_a", "tag_b"],
        },
        output_keys=["tag_a", "tag_b", "overlap_count", "overlap_ratio_of_smaller", "only_a_count", "only_b_count"],
        impl=_impl_compare_tags,
    ),
    ToolSpec(
        name="propose_split_preview",
        description=(
            "Generate a split proposal for a tag (uses propose_split_fn). "
            "DOES NOT apply. Returns the proposed sub-tags + record assignments so you can review."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tag_name": {"type": "string"},
            },
            "required": ["tag_name"],
        },
        output_keys=["tag_name", "proposal_id", "sub_tags", "total_records"],
        impl=_impl_propose_split_preview,
    ),
    ToolSpec(
        name="propose_refine_preview",
        description=(
            "Generate a refine proposal for a tag. "
            "DOES NOT apply. Returns proposed new definition + prune list for review."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tag_name": {"type": "string"},
            },
            "required": ["tag_name"],
        },
        output_keys=["tag_name", "proposal_id", "new_definition", "prune_count"],
        impl=_impl_propose_refine_preview,
    ),
    ToolSpec(
        name="propose_deprecate_preview",
        description=(
            "Generate a deprecate proposal for a tag (LLM judges whether it should be "
            "deprecated or merged into another). DOES NOT apply. Returns proposal_id + "
            "affected_record_count + orphan_count (records that would lose their only matter tag). "
            "If judge recommends merge instead, proposal_type='merge' with keep_tag/discard_tag."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tag_name": {"type": "string"},
            },
            "required": ["tag_name"],
        },
        output_keys=["tag_name", "proposal_id", "proposal_type", "affected_record_count", "orphan_count", "note"],
        impl=_impl_propose_deprecate_preview,
    ),
    ToolSpec(
        name="propose_merge_preview",
        description=(
            "Generate a merge proposal for two tags (LLM judges if they should merge "
            "and which to keep). DOES NOT apply. Returns proposal_id + keep_tag + discard_tag + "
            "affected_record_count. Use when two tags have high overlap and similar semantics."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tag_a": {"type": "string"},
                "tag_b": {"type": "string"},
            },
            "required": ["tag_a", "tag_b"],
        },
        output_keys=["tag_a", "tag_b", "proposal_id", "keep_tag", "discard_tag", "affected_record_count"],
        impl=_impl_propose_merge_preview,
    ),
    ToolSpec(
        name="apply_proposal",
        description=(
            "Apply a previously generated proposal (by id). "
            "USE WITH CAUTION — this modifies vocab."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
            },
            "required": ["proposal_id"],
        },
        output_keys=["proposal_id", "proposal_type", "vocab_before", "vocab_after"],
        impl=_impl_apply_proposal,
    ),
]

TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOLS}
