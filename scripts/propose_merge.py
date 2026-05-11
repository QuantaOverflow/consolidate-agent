"""Propose tag merges from co-occurrence signal.

Pipeline:
  1. From assignments, compute pairwise co-occurrence
  2. Take top-K pairs with count >= threshold
  3. For each pair: load co-occurring records as evidence, ask LLM
     to judge merge vs keep_distinct
  4. Output proposals + reasoning
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model


class MergeJudgement(BaseModel):
    decision: Literal["merge", "keep_distinct"] = Field(description="merge: tags are synonymous; keep_distinct: different axes")
    reasoning: str = Field(description="1-2 sentences justifying decision")
    keep_tag: str = Field(default="", description="if merge, the tag to retain (broader/more-used)")
    discard_tag: str = Field(default="", description="if merge, the tag to fold into keep_tag")


SYSTEM_PROMPT = """You judge whether two co-occurring tags in a knowledge tag vocabulary should be merged.

Co-occurrence alone does NOT justify merge. Two tags appearing on the same record can be:
- Same concept from different angles → merge
- Different facets/axes of one record → keep_distinct

Decision rules:
- merge ONLY if the two tags' definitions describe the same underlying concept (one is a near-synonym, sub-concept indistinguishable in practice, or rephrasing of the other)
- keep_distinct if they capture different aspects: e.g. domain vs pattern, technology vs design principle, scenario vs cross-cutting concern

If merge: keep_tag = the broader/more-used one; discard_tag = the narrower/redundant one.
If keep_distinct: leave keep_tag and discard_tag empty.

Be conservative: when in doubt, keep_distinct. Better to leave a defensible pair than to over-merge."""


USER_PROMPT = """## Tag A
name: {tag_a_name}
definition: {tag_a_def}
usage_count: {tag_a_count}

## Tag B
name: {tag_b_name}
definition: {tag_b_def}
usage_count: {tag_b_count}

## Co-occurrence
These two tags appear together on {cooccur} records.

## Sample co-occurring records ({evidence_count} shown)

{evidence}

Decide: merge or keep_distinct."""


def load_vocab(vocab_path: Path) -> list[dict]:
    return json.loads(vocab_path.read_text(encoding="utf-8"))["vocab"]


def compute_cooccurrence(assignments: list[dict]) -> tuple[Counter, dict, dict]:
    """Returns (pair_counts, tag_record_ids, tag_usage)."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    tag_record_ids: dict[str, set[str]] = defaultdict(set)
    tag_usage: dict[str, int] = defaultdict(int)

    for a in assignments:
        if a.get("missing"):
            continue
        tags = [t["name"] for t in a.get("selected_tags", [])]
        for t in tags:
            tag_record_ids[t].add(a["record_id"])
            tag_usage[t] += 1
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                pair = tuple(sorted([tags[i], tags[j]]))
                pair_counts[pair] += 1

    return pair_counts, dict(tag_record_ids), dict(tag_usage)


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


def format_evidence(records: list[dict]) -> str:
    parts = []
    for i, r in enumerate(records, 1):
        insight = r["insight"][:240]
        parts.append(f"[{i}] {r['title']}\n    {insight}")
    return "\n\n".join(parts)


def judge_pair(
    model,
    prompt,
    tag_a: dict,
    tag_b: dict,
    tag_usage: dict,
    cooccur: int,
    evidence_records: list[dict],
) -> MergeJudgement | None:
    messages = prompt.invoke({
        "tag_a_name": tag_a["name"],
        "tag_a_def": tag_a["definition"],
        "tag_a_count": tag_usage.get(tag_a["name"], 0),
        "tag_b_name": tag_b["name"],
        "tag_b_def": tag_b["definition"],
        "tag_b_count": tag_usage.get(tag_b["name"], 0),
        "cooccur": cooccur,
        "evidence_count": len(evidence_records),
        "evidence": format_evidence(evidence_records),
    })
    result: MergeJudgement | None = model.invoke(messages)
    if result is None:
        result = model.invoke(messages)  # retry once
    return result


def run(
    vocab_path: Path,
    assignments_path: Path,
    db_path: Path,
    output_dir: Path,
    top_k: int,
    min_cooccur: int,
    max_evidence: int,
) -> None:
    settings = Settings()
    model = _chat_model(settings).with_structured_output(MergeJudgement)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    vocab = load_vocab(vocab_path)
    vocab_by_name = {t["name"]: t for t in vocab}
    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))

    pair_counts, tag_record_ids, tag_usage = compute_cooccurrence(assignments)
    candidates = [(p, c) for p, c in pair_counts.most_common(top_k * 2) if c >= min_cooccur][:top_k]
    print(f"Found {len(candidates)} candidate pairs (cooccur >= {min_cooccur})\n", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "propose_merge_proposals.json"

    proposals = []
    for idx, (pair, count) in enumerate(candidates, 1):
        a_name, b_name = pair
        if a_name not in vocab_by_name or b_name not in vocab_by_name:
            print(f"  [{idx}/{len(candidates)}] {a_name} + {b_name}: skip (not in vocab)", flush=True)
            continue
        # intersect record sets
        shared = list(tag_record_ids[a_name] & tag_record_ids[b_name])[:max_evidence]
        details = load_record_details(db_path, shared)
        evidence_records = [details[rid] for rid in shared if rid in details]

        t0 = time.perf_counter()
        result = judge_pair(
            model, prompt,
            vocab_by_name[a_name], vocab_by_name[b_name],
            tag_usage, count, evidence_records,
        )
        elapsed = time.perf_counter() - t0

        if result is None:
            print(f"  [{idx}/{len(candidates)}] {a_name} + {b_name} (co-occur {count}): ❌ LLM None ({elapsed:.1f}s)", flush=True)
            proposals.append({
                "pair": [a_name, b_name],
                "cooccur": count,
                "decision": "error",
                "reasoning": "LLM returned None twice",
            })
            continue

        symbol = "🔀" if result.decision == "merge" else "✅"
        print(f"  [{idx}/{len(candidates)}] {a_name} + {b_name} (co-occur {count}): {symbol} {result.decision} ({elapsed:.1f}s)", flush=True)
        if result.decision == "merge":
            print(f"      keep {result.keep_tag}, discard {result.discard_tag}", flush=True)
        print(f"      reasoning: {result.reasoning}", flush=True)

        proposals.append({
            "pair": [a_name, b_name],
            "cooccur": count,
            "decision": result.decision,
            "reasoning": result.reasoning,
            "keep_tag": result.keep_tag,
            "discard_tag": result.discard_tag,
        })

        # checkpoint after each
        output_path.write_text(json.dumps(proposals, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ {len(proposals)} proposals → {output_path}", flush=True)

    # Summary
    n_merge = sum(1 for p in proposals if p["decision"] == "merge")
    n_keep = sum(1 for p in proposals if p["decision"] == "keep_distinct")
    n_err = sum(1 for p in proposals if p["decision"] == "error")
    print(f"\n=== Summary ===")
    print(f"  merge:         {n_merge}")
    print(f"  keep_distinct: {n_keep}")
    print(f"  error:         {n_err}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="outputs/tag_extraction_v2_vocab_v1.1.json", type=Path)
    parser.add_argument("--assignments", default="outputs/full_assignment/reverse_check_assignments.json", type=Path)
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--output-dir", default="outputs/full_assignment", type=Path)
    parser.add_argument("--top-k", default=10, type=int, help="top-K co-occurring pairs to evaluate")
    parser.add_argument("--min-cooccur", default=5, type=int)
    parser.add_argument("--max-evidence", default=5, type=int, help="max sample records per pair")
    args = parser.parse_args()

    run(
        vocab_path=args.vocab,
        assignments_path=args.assignments,
        db_path=args.db,
        output_dir=args.output_dir,
        top_k=args.top_k,
        min_cooccur=args.min_cooccur,
        max_evidence=args.max_evidence,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
