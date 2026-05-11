"""Reverse-check vocab quality: use the 42 tag vocab to tag sample records.

Outputs diagnostics:
  - low-confidence records (vocab doesn't cover them well)
  - boundary-blur tag pairs (records hesitating between 2-3 tags)
  - unused tags
  - co-occurrence pairs (strongly correlated tags → merge candidates)
"""
from __future__ import annotations

import argparse
import json
import random
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


class TagSelection(BaseModel):
    name: str = Field(description="exact tag name from vocab")
    confidence: Literal["high", "medium", "low"] = Field(description="how well does this tag fit")


class RecordAssignment(BaseModel):
    record_idx: int = Field(description="1-based index within the batch")
    selected_tags: list[TagSelection] = Field(description="1-3 tags from vocab, ordered by relevance. empty list ONLY if missing=true")
    missing: bool = Field(description="true ONLY if no vocab tag fits this record at any reasonable confidence")
    missing_concept: str = Field(default="", description="what concept the vocab is missing (only if missing=true)")
    reason: str = Field(default="", description="1 sentence justification for the selection")


class BatchAssignmentOutput(BaseModel):
    assignments: list[RecordAssignment]


SYSTEM_PROMPT = """You're given a tag vocabulary and a batch of engineering knowledge records. For each record, select 1-3 tags from the vocabulary that best capture its content.

Selection rules:
- Pick 1-3 tags ordered by relevance (most relevant first)
- Use exact tag names from the vocabulary (case-sensitive)
- Confidence levels:
    - high: the tag clearly captures the record's core lesson
    - medium: the tag captures part of the lesson or fits indirectly
    - low: weak fit, only included because no better option exists
- If NO tag in the vocab fits well at any confidence level → set missing=true, leave selected_tags=[], and describe in missing_concept what kind of tag is missing
- Don't force-fit: missing=true is acceptable if vocab doesn't cover the record"""


USER_PROMPT = """## Vocabulary ({tag_count} tags)

{vocab}

## Records to tag ({batch_size} records)

{records}

For each record (by record_idx 1..{batch_size}), select 1-3 best tags + justify."""


def load_records(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT record_id, title, insight FROM source_knowledge_records ORDER BY created_at, record_id"
    ).fetchall()
    conn.close()
    return [
        {"record_id": r["record_id"], "title": r["title"], "insight": r["insight"]}
        for r in rows
    ]


def load_vocab(vocab_path: Path) -> list[dict]:
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    return data["vocab"]


def format_vocab(vocab: list[dict]) -> str:
    return "\n".join(f"- {t['name']}: {t['definition']}" for t in vocab)


def format_records(records: list[dict]) -> str:
    lines = []
    for idx, r in enumerate(records, 1):
        insight = r["insight"][:300]
        lines.append(f"[{idx}] {r['title']}\n    {insight}")
    return "\n\n".join(lines)


def _run_single_batch(
    batch_idx: int,
    batch_records: list[dict],
    model,
    prompt,
    vocab: list[dict],
    total_batches: int,
) -> tuple[int, list[dict], float]:
    """Run one batch. Thread-safe (no shared state mutation).

    Returns (batch_idx, list_of_assignments, elapsed_seconds).
    On failure, returns assignments marked as missing with error reason.
    """
    t0 = time.perf_counter()
    try:
        messages = prompt.invoke({
            "tag_count": len(vocab),
            "vocab": format_vocab(vocab),
            "batch_size": len(batch_records),
            "records": format_records(batch_records),
        })
        result: BatchAssignmentOutput | None = model.invoke(messages)
        if result is None:
            raise ValueError("structured output returned None (LLM output unparseable)")
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        out = [{
            "record_id": rec["record_id"],
            "title": rec["title"],
            "selected_tags": [],
            "missing": True,
            "missing_concept": "",
            "reason": f"batch failure: {str(exc)[:200]}",
        } for rec in batch_records]
        return batch_idx, out, elapsed

    elapsed = time.perf_counter() - t0
    by_idx = {a.record_idx: a for a in result.assignments}
    out = []
    for rec_idx, rec in enumerate(batch_records, 1):
        a = by_idx.get(rec_idx)
        out.append({
            "record_id": rec["record_id"],
            "title": rec["title"],
            "selected_tags": [t.model_dump() for t in (a.selected_tags if a else [])],
            "missing": a.missing if a else True,
            "missing_concept": a.missing_concept if a else "",
            "reason": a.reason if a else "no assignment returned",
        })
    return batch_idx, out, elapsed


def run(
    db_path: Path,
    vocab_path: Path,
    output_dir: Path,
    sample_size: int,
    batch_size: int,
    seed: int,
    concurrency: int = 1,
) -> None:
    settings = Settings()
    model = _chat_model(settings).with_structured_output(BatchAssignmentOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    records = load_records(db_path)
    rng = random.Random(seed)
    rng.shuffle(records)
    sample = records[:sample_size]
    print(f"Sampled {len(sample)} records (seed={seed}) from {len(records)} total", flush=True)

    vocab = load_vocab(vocab_path)
    print(f"Loaded {len(vocab)} tags from {vocab_path}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "reverse_check_assignments.json"
    diagnostics_path = output_dir / "reverse_check_diagnostics.json"

    total_batches = (len(sample) + batch_size - 1) // batch_size
    print(f"Total batches: {total_batches}, concurrency: {concurrency}", flush=True)

    # Build batch list (batch_idx → records)
    batches: list[tuple[int, list[dict]]] = []
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(sample))
        batches.append((batch_idx, sample[start:end]))

    results_by_idx: dict[int, list[dict]] = {}
    wall_start = time.perf_counter()

    if concurrency <= 1:
        # serial path (original behavior)
        for batch_idx, batch_records in batches:
            _, batch_out, elapsed = _run_single_batch(
                batch_idx, batch_records, model, prompt, vocab, total_batches
            )
            results_by_idx[batch_idx] = batch_out
            print(f"  batch {batch_idx + 1}/{total_batches}  records={len(batch_records)}  ({elapsed:.1f}s)", flush=True)
            # checkpoint
            ordered = [r for i in sorted(results_by_idx) for r in results_by_idx[i]]
            assignments_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        # concurrent path
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_to_idx = {
                ex.submit(_run_single_batch, idx, recs, model, prompt, vocab, total_batches): idx
                for idx, recs in batches
            }
            completed = 0
            for future in as_completed(future_to_idx):
                batch_idx, batch_out, elapsed = future.result()
                results_by_idx[batch_idx] = batch_out
                completed += 1
                print(f"  batch {batch_idx + 1}/{total_batches}  records={len(batch_out)}  ({elapsed:.1f}s)  [{completed}/{total_batches}]", flush=True)
                # checkpoint after each completion
                ordered = [r for i in sorted(results_by_idx) for r in results_by_idx[i]]
                assignments_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")

    # Final ordered assignments
    all_assignments = [r for i in sorted(results_by_idx) for r in results_by_idx[i]]
    assignments_path.write_text(
        json.dumps(all_assignments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    wall_elapsed = time.perf_counter() - wall_start
    print(f"\n  ✅ {len(all_assignments)} assignments → {assignments_path}  (wall: {wall_elapsed:.1f}s)", flush=True)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    tag_usage: Counter[str] = Counter()
    confidence_by_tag: dict[str, list[str]] = defaultdict(list)
    cooccurrence: Counter[tuple[str, str]] = Counter()
    missing_records: list[dict] = []
    low_confidence_records: list[dict] = []
    boundary_blur_records: list[dict] = []  # records with 2+ tags at same confidence

    for a in all_assignments:
        if a["missing"]:
            missing_records.append({
                "record_id": a["record_id"],
                "title": a["title"],
                "missing_concept": a["missing_concept"],
            })
            continue

        tags = a["selected_tags"]
        if not tags:
            continue

        # tag usage
        for t in tags:
            tag_usage[t["name"]] += 1
            confidence_by_tag[t["name"]].append(t["confidence"])

        # pairwise co-occurrence
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                pair = tuple(sorted([tags[i]["name"], tags[j]["name"]]))
                cooccurrence[pair] += 1

        # low-confidence top pick = vocab fits poorly
        if tags[0]["confidence"] == "low":
            low_confidence_records.append({
                "record_id": a["record_id"],
                "title": a["title"],
                "selected": tags,
                "reason": a["reason"],
            })

        # boundary blur: 2+ tags at high or medium → tag边界模糊
        high_or_med = [t for t in tags if t["confidence"] in ("high", "medium")]
        if len(high_or_med) >= 2:
            confs = [t["confidence"] for t in high_or_med]
            if len(set(confs)) == 1:  # all same confidence
                boundary_blur_records.append({
                    "record_id": a["record_id"],
                    "title": a["title"],
                    "selected": high_or_med,
                    "reason": a["reason"],
                })

    vocab_names = {t["name"] for t in vocab}
    unused_tags = sorted(vocab_names - set(tag_usage.keys()))

    diagnostics = {
        "sample_size": len(all_assignments),
        "total_assigned": len([a for a in all_assignments if not a["missing"]]),
        "total_missing": len(missing_records),
        "tag_usage_count": dict(tag_usage.most_common()),
        "unused_tags": unused_tags,
        "missing_records": missing_records,
        "low_confidence_records": low_confidence_records,
        "boundary_blur_records": boundary_blur_records,
        "top_cooccurrence_pairs": [
            {"pair": list(p), "count": c}
            for p, c in cooccurrence.most_common(15)
        ],
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # console summary
    print(f"\n=== Diagnostics ===")
    print(f"Sample: {len(all_assignments)} records")
    print(f"  Assigned: {diagnostics['total_assigned']}")
    print(f"  Missing (no fit): {diagnostics['total_missing']}")
    print(f"  Low-confidence top pick: {len(low_confidence_records)}")
    print(f"  Boundary blur (2+ same-confidence tags): {len(boundary_blur_records)}")
    print(f"\nUnused tags ({len(unused_tags)}/{len(vocab)}): {unused_tags}")
    print(f"\nTop 10 most-used tags:")
    for name, count in tag_usage.most_common(10):
        confs = Counter(confidence_by_tag[name])
        print(f"  {name:30s} {count:3d}  (h:{confs.get('high', 0)} m:{confs.get('medium', 0)} l:{confs.get('low', 0)})")

    print(f"\nTop 10 co-occurrence pairs:")
    for p, c in cooccurrence.most_common(10):
        print(f"  {c:2d} × {p[0]} + {p[1]}")

    print(f"\nFull diagnostics → {diagnostics_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--vocab", default="outputs/tag_extraction_v2_vocab.json", type=Path)
    parser.add_argument("--output-dir", default="outputs", type=Path)
    parser.add_argument("--sample-size", default=30, type=int)
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--concurrency", default=1, type=int, help="parallel LLM workers (1=serial, 5-10 safe with DashScope)")
    args = parser.parse_args()

    run(
        db_path=args.db,
        vocab_path=args.vocab,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        seed=args.seed,
        concurrency=args.concurrency,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
