"""Propose tag deprecations (or merge-to-existing).

Pipeline:
  1. From assignments, compute per-tag usage + records
  2. Take tags with usage < threshold
  3. For each tag: load its records as evidence + show all other vocab,
     ask LLM to judge keep / deprecate / merge_to:<existing_tag>
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model


class DeprecateJudgement(BaseModel):
    # CoT thinking steps — Pydantic field order forces LLM to output these BEFORE decision
    records_concept: str = Field(description="STEP 1: In 1 sentence, what core concept do this tag's records collectively represent?")
    independence_check: str = Field(description="STEP 2: What UNIQUE value does this tag provide that no other vocab tag captures with the same precision? Name the closest other tag and what's distinct.")
    candidate_targets: str = Field(description="STEP 3: If considering merge_to, list 1-3 plausible target tags and what precision would be lost by merging into each. If no plausible target, say 'none'.")
    size_prior: str = Field(description="STEP 4: Given usage_count, state the prior. usage>=15: strong keep prior, merge needs OVERWHELMING evidence. usage 5-14: balanced. usage<5: open to merge/deprecate.")

    decision: Literal["keep", "deprecate", "merge_to"] = Field(
        description="FINAL decision after analysis above. Default keep unless burden of proof clearly met."
    )
    merge_target: str = Field(default="", description="if decision=merge_to, the existing tag name to fold into")
    reasoning: str = Field(description="1-2 sentence final justification grounded in the steps above")


SYSTEM_PROMPT = """You judge whether a tag in a knowledge tag vocabulary should be kept, deprecated, or merged into another existing tag.

**CRITICAL DEFAULT: KEEP.** The burden of proof is on merge_to/deprecate, NOT on keep. Most tags should be kept unless clearly redundant.

Work through these analysis steps IN ORDER before deciding:

STEP 1 — records_concept: Read all records. In ONE sentence, state the core concept they collectively represent. Don't list features; identify the underlying lesson/pattern.

STEP 2 — independence_check: For each of the closest vocab tags, ask: "could THIS tag's definition cover these records WITH THE SAME PRECISION as the tag under review?" If no — what specific aspect would be lost? This step protects against superficial "this also relates to X" reasoning.

STEP 3 — candidate_targets: If considering merge_to, list 1-3 candidates and for EACH, articulate what precision/aspect is lost by merging. If no candidate truly covers the concept, write 'none'. Be honest: relatedness ≠ subsumption.

STEP 4 — size_prior: usage_count signals usage strength:
  - usage >= 15: strong KEEP prior. Many records have been tagged this way — merging requires OVERWHELMING evidence ALL records belong elsewhere, not just relatedness.
  - usage 5-14: balanced. Decide on merits.
  - usage < 5: more open to merge_to or deprecate.

Then DECIDE:
- merge_to: records OVERWHELMINGLY belong to another tag's scope, AND no precision is lost
- deprecate: records don't cohere or are too few/contextual to justify existence
- keep: in ALL other cases (this is the default)

Important: "tag A's records also relate to tag B" is NOT sufficient for merge_to(A, B). Both tags can capture different aspects of the same records. Merge ONLY when tag A's distinct contribution would not be missed."""


USER_PROMPT = """## Tag under review
name: {tag_name}
definition: {tag_def}
usage_count: {tag_count}

## Records currently assigned to this tag ({record_count})

{records}

## Other vocab tags ({other_vocab_count})

{other_vocab}

Decide: keep / deprecate / merge_to. If merge_to, specify exact target tag name from the vocab above."""


def load_vocab(vocab_path: Path) -> list[dict]:
    return json.loads(vocab_path.read_text(encoding="utf-8"))["vocab"]


def compute_tag_records(assignments: list[dict]) -> dict[str, list[str]]:
    """Returns tag_name → list of record_ids assigned to it."""
    tag_records: dict[str, list[str]] = defaultdict(list)
    for a in assignments:
        if a.get("missing"):
            continue
        for t in a.get("selected_tags", []):
            tag_records[t["name"]].append(a["record_id"])
    return dict(tag_records)


def load_record_details(db_path: Path, record_ids: list[str]) -> dict[str, dict]:
    if not record_ids:
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(record_ids))
    rows = conn.execute(
        f"SELECT record_id, title, insight FROM source_knowledge_records WHERE record_id IN ({placeholders})",
        record_ids,
    ).fetchall()
    conn.close()
    return {r["record_id"]: {"title": r["title"], "insight": r["insight"]} for r in rows}


def format_records(records: list[dict]) -> str:
    if not records:
        return "(no records — tag is unused)"
    parts = []
    for i, r in enumerate(records, 1):
        insight = r["insight"][:200]
        parts.append(f"[{i}] {r['title']}\n    {insight}")
    return "\n\n".join(parts)


def format_other_vocab(vocab: list[dict], exclude: str) -> str:
    return "\n".join(f"- {t['name']}: {t['definition']}" for t in vocab if t["name"] != exclude)


def judge_tag(
    model,
    prompt,
    tag: dict,
    other_vocab: list[dict],
    tag_records: list[dict],
    usage_count: int,
) -> DeprecateJudgement | None:
    messages = prompt.invoke({
        "tag_name": tag["name"],
        "tag_def": tag["definition"],
        "tag_count": usage_count,
        "record_count": len(tag_records),
        "records": format_records(tag_records),
        "other_vocab_count": len(other_vocab),
        "other_vocab": format_other_vocab(other_vocab, tag["name"]),
    })
    result: DeprecateJudgement | None = model.invoke(messages)
    if result is None:
        result = model.invoke(messages)
    return result


def _extract_focus_tags(focus: str, vocab: list[dict]) -> list[str]:
    """Find vocab tag names mentioned in the focus string.

    Agent's diagnose typically writes focus like "target tag X (0 usage)".
    Scan focus for any tag name substring match (case-insensitive).
    """
    if not focus:
        return []
    focus_lower = focus.lower()
    return [t["name"] for t in vocab if t["name"].lower() in focus_lower]


def propose_deprecate_fn(
    vocab: list[dict],
    assignments: list[dict],
    focus: str = "",
    *,
    db_path: Path | None = None,
    max_usage: int = 10,
    max_records_evidence: int = 8,
):
    """In-memory propose_deprecate for agent loop.

    Returns list of DeprecateProposal (or MergeProposal if LLM says merge_to).

    If `focus` names specific vocab tags, evaluation is restricted to those
    tags only — agent-directed targeting beats LLM's tendency to flag any
    niche tag for deprecation.
    """
    from ..apply import DeprecateProposal as _DeprecateProposal, MergeProposal as _MergeProposal

    settings = Settings()
    model = _chat_model(settings).with_structured_output(DeprecateJudgement)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + (f"\n\nFOCUS HINT: {focus}" if focus else "")),
        ("user", USER_PROMPT),
    ])

    vocab_names = {t["name"] for t in vocab}
    tag_records = compute_tag_records(assignments)

    # If focus names specific tags, restrict candidates to those. Otherwise
    # fall back to "all tags with usage <= max_usage".
    focus_tags = _extract_focus_tags(focus, vocab)
    if focus_tags:
        candidates = [(t, len(tag_records.get(t["name"], []))) for t in vocab if t["name"] in focus_tags]
        print(f"  [propose_deprecate] focus restricts evaluation to {focus_tags}", flush=True)
    else:
        candidates = []
        for t in vocab:
            usage = len(tag_records.get(t["name"], []))
            if usage <= max_usage:
                candidates.append((t, usage))
    candidates.sort(key=lambda x: x[1])

    proposals: list = []
    for tag, usage in candidates:
        record_ids = tag_records.get(tag["name"], [])[:max_records_evidence]
        details = load_record_details(db_path, record_ids) if db_path else {}
        evidence = [details[rid] for rid in record_ids if rid in details]

        result = judge_tag(model, prompt, tag, vocab, evidence, usage)
        if result is None:
            continue
        if result.decision == "keep":
            continue
        if result.decision == "deprecate":
            proposals.append(_DeprecateProposal(tag=tag["name"]))
        elif result.decision == "merge_to":
            target = result.merge_target
            if not target or target == tag["name"] or target not in vocab_names:
                continue
            proposals.append(_MergeProposal(keep_tag=target, discard_tag=tag["name"]))

    return proposals


def run(
    vocab_path: Path,
    assignments_path: Path,
    db_path: Path,
    output_dir: Path,
    max_usage: int,
    max_records_evidence: int,
) -> None:
    settings = Settings()
    model = _chat_model(settings).with_structured_output(DeprecateJudgement)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    vocab = load_vocab(vocab_path)
    vocab_names = {t["name"] for t in vocab}
    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))

    tag_records = compute_tag_records(assignments)

    # candidates: vocab tags with usage <= max_usage (only real vocab, not fake LLM-invented ones)
    candidates = []
    for t in vocab:
        usage = len(tag_records.get(t["name"], []))
        if usage <= max_usage:
            candidates.append((t, usage))
    candidates.sort(key=lambda x: x[1])

    print(f"Found {len(candidates)} candidate tags (usage <= {max_usage})\n", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "propose_deprecate_proposals.json"

    proposals = []
    for idx, (tag, usage) in enumerate(candidates, 1):
        record_ids = tag_records.get(tag["name"], [])[:max_records_evidence]
        details = load_record_details(db_path, record_ids)
        evidence = [details[rid] for rid in record_ids if rid in details]

        t0 = time.perf_counter()
        result = judge_tag(model, prompt, tag, vocab, evidence, usage)
        elapsed = time.perf_counter() - t0

        if result is None:
            print(f"  [{idx}/{len(candidates)}] {tag['name']} (size {usage}): ❌ LLM None ({elapsed:.1f}s)", flush=True)
            proposals.append({"tag": tag["name"], "usage": usage, "decision": "error"})
            continue

        if result.decision == "keep":
            symbol = "✅"
        elif result.decision == "deprecate":
            symbol = "🗑️"
        else:
            symbol = "🔀"

        target_str = f" → {result.merge_target}" if result.decision == "merge_to" else ""
        # validate merge_target is a real vocab tag
        target_valid = True
        if result.decision == "merge_to":
            if result.merge_target not in vocab_names or result.merge_target == tag["name"]:
                target_valid = False

        warn = "" if target_valid else " ⚠️(invalid target)"
        print(f"  [{idx}/{len(candidates)}] {tag['name']} (size {usage}): {symbol} {result.decision}{target_str}{warn} ({elapsed:.1f}s)", flush=True)
        print(f"      reasoning: {result.reasoning}", flush=True)

        proposals.append({
            "tag": tag["name"],
            "usage": usage,
            "decision": result.decision,
            "reasoning": result.reasoning,
            "merge_target": result.merge_target,
            "target_valid": target_valid,
        })

        output_path.write_text(json.dumps(proposals, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ {len(proposals)} proposals → {output_path}", flush=True)

    n_keep = sum(1 for p in proposals if p["decision"] == "keep")
    n_dep = sum(1 for p in proposals if p["decision"] == "deprecate")
    n_merge = sum(1 for p in proposals if p["decision"] == "merge_to")
    n_err = sum(1 for p in proposals if p["decision"] == "error")
    print(f"\n=== Summary ===")
    print(f"  keep:       {n_keep}")
    print(f"  deprecate:  {n_dep}")
    print(f"  merge_to:   {n_merge}")
    print(f"  error:      {n_err}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="outputs/tag_extraction_v2_vocab_v1.1.json", type=Path)
    parser.add_argument("--assignments", default="outputs/full_assignment/reverse_check_assignments.json", type=Path)
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--output-dir", default="outputs/full_assignment", type=Path)
    parser.add_argument("--max-usage", default=10, type=int, help="evaluate tags with usage <= this")
    parser.add_argument("--max-records-evidence", default=8, type=int)
    args = parser.parse_args()

    run(
        vocab_path=args.vocab,
        assignments_path=args.assignments,
        db_path=args.db,
        output_dir=args.output_dir,
        max_usage=args.max_usage,
        max_records_evidence=args.max_records_evidence,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
