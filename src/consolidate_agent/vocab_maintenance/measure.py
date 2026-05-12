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


SYSTEM_PROMPT = """You're given a tag vocabulary and a batch of engineering knowledge records. Your job: either ASSIGN existing tags or MARK MISSING. The choice is a SIGNAL to the vocab maintenance system.

PRIMARY GOAL: surface vocab coverage gaps. Marking missing=true is the only way the system learns the vocab needs a new tag. Force-fitting a partially-related tag silences this signal and freezes vocab evolution.

When to ASSIGN (selected_tags=[...], missing=false):
  - A tag's DEFINITION explicitly names the concept the record teaches, OR
  - A tag's definition includes the concept as a documented variant; the record adds nuance but stays within the tag's scope.

When to MARK MISSING (selected_tags=[], missing=true):
  - The record introduces a SPECIFIC concept the vocab only covers GENERICALLY (umbrella tags).
  - The closest tag would only be at low/weak confidence.
  - The record's lesson is a precise sub-pattern of a broader vocab tag, but the broader tag doesn't name this sub-pattern.

Heuristic: if you can't write the record's lesson as a paraphrase of the tag's definition, prefer missing.

Confidence levels:
  - high: tag's definition explicitly names the record's core concept
  - medium: tag's definition lists this concept as a documented variant
  - DO NOT use low confidence — if your best match is only "low", that means the vocab is silent on this concept, so set missing=true instead.

Output format:
  - 1-3 tags ordered by relevance when assigning (single tag preferred when one fully covers the lesson)
  - When missing, populate `missing_concept` with a precise 5-15 word description of what tag would fit. This drives propose_new.

Examples:

  Vocab: [resource_management(def: "controlling memory, CPU, network usage"), fallback_strategy(def: "graceful degradation paths")]
  Record: "Token bucket algorithm caps API request rate at sustained level with burst allowance"
  → missing=true. resource_management's definition lists memory/CPU/network — not rate limiting per se. The "token bucket" mechanism is a distinct pattern.
  → missing_concept: "rate limiting mechanisms (token bucket, leaky bucket, distributed counters)"

  Vocab: [authentication(def: "user identity verification including OAuth flows, token refresh, session lifecycle")]
  Record: "OAuth refresh token rotation strategy"
  → assign [authentication] at high. Definition explicitly names "OAuth flows, token refresh".

  Vocab: [error_handling(def: "patterns for propagating, classifying, recovering from exceptions")]
  Record: "429 Too Many Requests should trigger exponential backoff with jitter"
  → missing=true. error_handling names "propagating exceptions"; backoff+jitter is a distinct concern (retry policy under rate limits).
  → missing_concept: "retry policy with exponential backoff and jitter"

  Vocab: [state_isolation(def: "preventing shared mutable state across execution contexts")]
  Record: "Distributed rate limiting needs shared counter state across instances"
  → missing=true. state_isolation is about PREVENTING shared state; this record is about COORDINATING shared state for rate limits. Opposite concern.
  → missing_concept: "distributed coordination for shared counters/limits"
"""


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


def build_diagnostics(vocab: list[dict], assignments: list[dict]) -> dict:
    """Pure local aggregation of assignments → diagnostics. No LLM.

    Single source of truth for vocab diagnostic signals. Used by both the
    agent loop (post-apply recompute) and the reverse_check CLI.
    """
    tag_usage: Counter[str] = Counter()
    cooccur: Counter[tuple[str, str]] = Counter()
    missing_records: list[dict] = []
    boundary_blur_records: list[dict] = []
    low_confidence_records: list[dict] = []

    for a in assignments:
        if a.get("missing"):
            missing_records.append({
                "record_id": a["record_id"],
                "title": a.get("title", ""),
                "missing_concept": a.get("missing_concept", ""),
            })
            continue
        tags = a.get("selected_tags", [])
        if not tags:
            continue

        for t in tags:
            tag_usage[t["name"]] += 1

        names = [t["name"] for t in tags]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cooccur[tuple(sorted([names[i], names[j]]))] += 1

        # low-confidence top pick = vocab fits poorly (ordered most-relevant first)
        if tags[0].get("confidence") == "low":
            low_confidence_records.append({
                "record_id": a["record_id"],
                "title": a.get("title", ""),
                "selected": tags,
                "reason": a.get("reason", ""),
            })

        # boundary blur: 2+ tags at high or medium with identical confidence
        # (excludes all-low which signals "vocab doesn't fit" rather than ambiguity).
        high_or_med = [t for t in tags if t.get("confidence") in ("high", "medium")]
        if len(high_or_med) >= 2 and len({t["confidence"] for t in high_or_med}) == 1:
            boundary_blur_records.append({
                "record_id": a["record_id"],
                "title": a.get("title", ""),
                "selected": high_or_med,
                "reason": a.get("reason", ""),
            })

    return {
        "sample_size": len(assignments),
        "total_assigned": sum(1 for a in assignments if not a.get("missing")),
        "total_missing": len(missing_records),
        "tag_usage_count": dict(tag_usage.most_common()),
        "unused_tags": sorted({t["name"] for t in vocab} - set(tag_usage)),
        "missing_records": missing_records,
        "boundary_blur_records": boundary_blur_records,
        "low_confidence_records": low_confidence_records,
        "top_cooccurrence_pairs": [{"pair": list(p), "count": c} for p, c in cooccur.most_common(15)],
    }


def reverse_check_subset(
    records: list[dict],
    vocab: list[dict],
    batch_size: int = 10,
    concurrency: int = 10,
) -> list[dict]:
    """Run LLM reverse_check on a record subset. Returns assignments list.

    Useful for incrementally re-checking a small group (e.g. previously-missing
    records after propose_new added new tag).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not records:
        return []

    settings = Settings()
    model = _chat_model(settings).with_structured_output(BatchAssignmentOutput)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])

    total_batches = (len(records) + batch_size - 1) // batch_size
    batches = [(i, records[i * batch_size : (i + 1) * batch_size]) for i in range(total_batches)]

    results_by_idx: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_run_single_batch, idx, recs, model, prompt, vocab, total_batches): idx
                   for idx, recs in batches}
        for fut in as_completed(futures):
            idx, batch_out, _ = fut.result()
            results_by_idx[idx] = batch_out

    return [r for i in sorted(results_by_idx) for r in results_by_idx[i]]


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

    diagnostics = build_diagnostics(vocab, all_assignments)
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # console summary (CLI-only — confidence breakdown not in build_diagnostics
    # because the agent loop doesn't need it; recompute here cheaply)
    confidence_by_tag: dict[str, list[str]] = defaultdict(list)
    for a in all_assignments:
        if a["missing"]:
            continue
        for t in a["selected_tags"]:
            confidence_by_tag[t["name"]].append(t["confidence"])

    unused = diagnostics["unused_tags"]
    print(f"\n=== Diagnostics ===")
    print(f"Sample: {len(all_assignments)} records")
    print(f"  Assigned: {diagnostics['total_assigned']}")
    print(f"  Missing (no fit): {diagnostics['total_missing']}")
    print(f"  Low-confidence top pick: {len(diagnostics['low_confidence_records'])}")
    print(f"  Boundary blur (2+ same-confidence tags): {len(diagnostics['boundary_blur_records'])}")
    print(f"\nUnused tags ({len(unused)}/{len(vocab)}): {unused}")
    print(f"\nTop 10 most-used tags:")
    for name, count in list(diagnostics["tag_usage_count"].items())[:10]:
        confs = Counter(confidence_by_tag[name])
        print(f"  {name:30s} {count:3d}  (h:{confs.get('high', 0)} m:{confs.get('medium', 0)} l:{confs.get('low', 0)})")

    print(f"\nTop 10 co-occurrence pairs:")
    for entry in diagnostics["top_cooccurrence_pairs"][:10]:
        p = entry["pair"]
        print(f"  {entry['count']:2d} × {p[0]} + {p[1]}")

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
