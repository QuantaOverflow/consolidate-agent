"""Brain-callable tools wrapping existing probe/propose functions.

Each tool:
  - accepts dict (from LLM tool call args)
  - returns {"ok": bool, "result": dict | None, "error": str | None}
  - never raises — errors are structured output
"""
from __future__ import annotations

import json
import random
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


def _ok(result: dict) -> dict:
    return {"ok": True, "result": result, "error": None}


def _err(msg: str) -> dict:
    return {"ok": False, "result": None, "error": msg}


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


def _impl_quick_golden_sample(args: dict, ctx: BrainContext) -> dict:
    n = int(args.get("n", 5))
    filter_tag: str | None = args.get("filter_tag") or None

    try:
        pool = ctx.golden
        if filter_tag:
            pool = [
                g for g in pool
                if filter_tag in g.get("expected", {}).get("matter_tags", [])
            ]
        if not pool:
            return _ok({
                "sampled_count": 0,
                "matter_exact_match_pct": None,
                "lesson_match_pct": None,
                "disagreements": [],
            })

        sample = random.sample(pool, min(n, len(pool)))

        assignments_by_id = {
            a["record_id"]: a
            for a in ctx.assignments
            if not a.get("missing")
        }

        matter_exact_matches = 0
        lesson_matches = 0
        disagreements: list[dict] = []

        for g in sample:
            rid = g["record_id"]
            expected_matter = set(g.get("expected", {}).get("matter_tags", []))
            expected_lesson = g.get("expected", {}).get("lesson_type", "")

            a = assignments_by_id.get(rid)
            if a is None:
                disagreements.append({
                    "record_id": rid,
                    "issue": "not found in assignments",
                    "expected_matter": sorted(expected_matter),
                    "actual_matter": [],
                })
                continue

            actual_matter = set(t["name"] for t in a.get("selected_tags", []))
            actual_lesson = a.get("lesson_type", "")

            if expected_matter == actual_matter:
                matter_exact_matches += 1
            else:
                disagreements.append({
                    "record_id": rid,
                    "expected_matter": sorted(expected_matter),
                    "actual_matter": sorted(actual_matter),
                    "missing": sorted(expected_matter - actual_matter),
                    "extra": sorted(actual_matter - expected_matter),
                })

            if expected_lesson and actual_lesson == expected_lesson:
                lesson_matches += 1

        total = len(sample)
        return _ok({
            "sampled_count": total,
            "matter_exact_match_pct": round(matter_exact_matches / total * 100, 1) if total else None,
            "lesson_match_pct": round(lesson_matches / total * 100, 1) if total else None,
            "disagreements": disagreements,
        })
    except Exception as e:
        return _err(f"quick_golden_sample failed: {e}")


def _impl_propose_split_preview(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.propose.split import propose_split_fn

    tag_name = args.get("tag_name", "")
    if not tag_name:
        return _err("tag_name is required")
    try:
        proposals = propose_split_fn(
            ctx.vocab, ctx.assignments, tag_name, db_path=ctx.db_path
        )
        if not proposals:
            return _ok({"tag_name": tag_name, "proposals": [], "message": "no split proposed"})

        proposal = proposals[0]
        proposal_id = str(uuid.uuid4())
        ctx.proposal_cache[proposal_id] = proposal

        sub_tags_preview = [
            {
                "name": st["name"],
                "definition": st["definition"],
                "record_count": len(st["record_ids"]),
            }
            for st in proposal.sub_tags
        ]
        return _ok({
            "tag_name": tag_name,
            "proposal_id": proposal_id,
            "sub_tags": sub_tags_preview,
            "total_records": sum(len(st["record_ids"]) for st in proposal.sub_tags),
        })
    except Exception as e:
        return _err(f"propose_split_preview({tag_name}) failed: {e}")


def _impl_propose_refine_preview(args: dict, ctx: BrainContext) -> dict:
    from consolidate_agent.vocab_maintenance.propose.refine import propose_refine_fn

    tag_name = args.get("tag_name", "")
    if not tag_name:
        return _err("tag_name is required")
    try:
        proposals = propose_refine_fn(
            ctx.vocab, ctx.assignments, focus=tag_name, db_path=ctx.db_path
        )
        if not proposals:
            return _ok({"tag_name": tag_name, "proposals": [], "message": "no refine proposed"})

        proposal = proposals[0]
        proposal_id = str(uuid.uuid4())
        ctx.proposal_cache[proposal_id] = proposal

        return _ok({
            "tag_name": tag_name,
            "proposal_id": proposal_id,
            "new_definition": proposal.new_definition,
            "prune_count": len(proposal.prune_record_ids),
            "prune_record_ids": list(proposal.prune_record_ids)[:10],
        })
    except Exception as e:
        return _err(f"propose_refine_preview({tag_name}) failed: {e}")


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
        name="quick_golden_sample",
        description=(
            "Sample N records from the 80 human-labeled golden set and check whether "
            "LLM's current matter_tags match. Use to calibrate yourself against ground truth."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 5},
                "filter_tag": {
                    "type": "string",
                    "description": "optional: only sample records that golden assigns to this tag",
                },
            },
            "required": [],
        },
        output_keys=["sampled_count", "matter_exact_match_pct", "lesson_match_pct", "disagreements"],
        impl=_impl_quick_golden_sample,
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
