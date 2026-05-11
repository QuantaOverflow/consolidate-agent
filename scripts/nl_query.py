"""Natural-language query: user question → relevant records via tag vocabulary.

Pipeline:
  1. LLM matches the query against vocab → 1-3 best tags + confidence
  2. Pull records assigned to those tags
  3. Rank by how many query-tags the record covers, return top-k
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model


class TagMatch(BaseModel):
    name: str = Field(description="exact tag name from vocab")
    relevance: Literal["high", "medium", "low"] = Field(description="how strongly the query maps to this tag")


class QueryRouting(BaseModel):
    matched_tags: list[TagMatch] = Field(description="1-3 best matching tags, ordered by relevance")
    interpretation: str = Field(description="brief interpretation of what the user is asking about")
    no_match: bool = Field(default=False, description="true if no vocab tag remotely matches the query")


ROUTING_SYSTEM = """You map user questions to a fixed knowledge tag vocabulary. Each tag has a name and definition.

Queries may be in Chinese, English, or mixed. Tag names are always English snake_case. Interpret the query semantics regardless of language.

Given a user query, output 1-3 tags from the vocab that best capture what the user is asking about. Order by relevance (most relevant first).

Relevance levels:
- high: the tag directly addresses what the user is asking
- medium: the tag covers part of the query or a related aspect
- low: tangentially relevant; only include if no better option

Use EXACT tag names from the vocab. If no tag fits, set no_match=true and leave matched_tags empty.

Keep your output concise — produce the structured JSON directly without lengthy reasoning."""


ROUTING_USER = """## Vocabulary ({tag_count} tags)

{vocab}

## User query

{query}

Map this query to 1-3 best-fitting tags. Briefly interpret what the user is asking."""


def load_vocab(vocab_path: Path) -> list[dict]:
    return json.loads(vocab_path.read_text(encoding="utf-8"))["vocab"]


def load_assignments(assignments_path: Path) -> dict[str, list[dict]]:
    """Returns map: tag_name → list of {record_id, title, confidence}."""
    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    tag_to_records: dict[str, list[dict]] = defaultdict(list)
    for a in assignments:
        if a.get("missing"):
            continue
        for t in a.get("selected_tags", []):
            tag_to_records[t["name"]].append({
                "record_id": a["record_id"],
                "title": a["title"],
                "confidence": t["confidence"],
            })
    return tag_to_records


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


def format_vocab(vocab: list[dict]) -> str:
    return "\n".join(f"- {t['name']}: {t['definition']}" for t in vocab)


_CONFIDENCE_WEIGHT = {"high": 3, "medium": 2, "low": 1}
_RELEVANCE_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def rank_records(
    routing: QueryRouting,
    tag_to_records: dict[str, list[dict]],
) -> list[dict]:
    """Score each record by: sum over matched tags of (tag relevance × record-tag confidence)."""
    record_scores: dict[str, float] = defaultdict(float)
    record_tag_hits: dict[str, list[tuple[str, str]]] = defaultdict(list)  # tag, relevance

    for tag in routing.matched_tags:
        relevance_w = _RELEVANCE_WEIGHT[tag.relevance]
        for rec in tag_to_records.get(tag.name, []):
            conf_w = _CONFIDENCE_WEIGHT[rec["confidence"]]
            record_scores[rec["record_id"]] += relevance_w * conf_w
            record_tag_hits[rec["record_id"]].append((tag.name, tag.relevance))

    ranked = sorted(record_scores.items(), key=lambda x: -x[1])
    return [
        {"record_id": rid, "score": score, "matched_tags": record_tag_hits[rid]}
        for rid, score in ranked
    ]


def query(
    user_query: str,
    vocab: list[dict],
    tag_to_records: dict[str, list[dict]],
    db_path: Path,
    top_k: int,
) -> dict:
    settings = Settings()
    model = _chat_model(settings).with_structured_output(QueryRouting)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTING_SYSTEM),
        ("user", ROUTING_USER),
    ])

    messages = prompt.invoke({
        "tag_count": len(vocab),
        "vocab": format_vocab(vocab),
        "query": user_query,
    })
    routing: QueryRouting | None = model.invoke(messages)
    # retry once if LLM returned unparseable output
    if routing is None:
        routing = model.invoke(messages)
    if routing is None:
        return {
            "query": user_query,
            "interpretation": "(LLM returned unparseable output after retry)",
            "matched_tags": [],
            "results": [],
            "no_match": True,
        }

    if routing.no_match or not routing.matched_tags:
        return {
            "query": user_query,
            "interpretation": routing.interpretation,
            "matched_tags": [],
            "results": [],
            "no_match": True,
        }

    ranked = rank_records(routing, tag_to_records)
    top = ranked[:top_k]
    details = load_record_details(db_path, [r["record_id"] for r in top])

    results = []
    for r in top:
        d = details.get(r["record_id"], {})
        results.append({
            "record_id": r["record_id"],
            "title": d.get("title", ""),
            "insight": d.get("insight", "")[:240],
            "score": r["score"],
            "matched_tags": r["matched_tags"],
        })

    return {
        "query": user_query,
        "interpretation": routing.interpretation,
        "matched_tags": [t.model_dump() for t in routing.matched_tags],
        "results": results,
        "no_match": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="natural-language query")
    parser.add_argument("--vocab", default="outputs/tag_extraction_v2_vocab_v1.1.json", type=Path)
    parser.add_argument("--assignments", default="outputs/full_assignment/reverse_check_assignments.json", type=Path)
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--top-k", default=8, type=int)
    parser.add_argument("--json", action="store_true", help="output raw JSON")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    tag_to_records = load_assignments(args.assignments)

    result = query(args.query, vocab, tag_to_records, args.db, args.top_k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # human-readable
    print(f"\n🔍 Query: {result['query']}")
    print(f"\n💡 Interpretation: {result['interpretation']}\n")

    if result["no_match"]:
        print("(No tag in the vocabulary matches this query.)")
        return 0

    print(f"📋 Matched tags ({len(result['matched_tags'])}):")
    for t in result["matched_tags"]:
        print(f"  • {t['name']}  [{t['relevance']}]")

    print(f"\n📚 Top {len(result['results'])} relevant records:\n")
    for i, r in enumerate(result["results"], 1):
        tags_str = ", ".join(f"{n} ({rel})" for n, rel in r["matched_tags"])
        print(f"{i}. [{r['score']:.0f}] {r['title']}")
        print(f"   tags: {tags_str}")
        print(f"   {r['insight']}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
