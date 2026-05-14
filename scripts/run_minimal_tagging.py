#!/usr/bin/env python3
"""Stage 1 v3: minimal tagging — Matter facet + lesson_type enum only.

Per ADR-0004, Activity and Pattern facets were dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def load_records(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT record_id, title, insight "
        "FROM source_knowledge_records "
        "WHERE insight IS NOT NULL "
        "ORDER BY created_at, record_id"
    ).fetchall()
    conn.close()
    return [
        {"record_id": r["record_id"], "title": r["title"], "insight": r["insight"]}
        for r in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default=Path("docs/plans/vocab_v3.json"), type=Path)
    parser.add_argument("--db", default=Path("outputs/knowledge.db"), type=Path)
    parser.add_argument("--out", default=Path("outputs/network_v3.json"), type=Path)
    parser.add_argument("--batch-size", default=5, type=int)
    parser.add_argument("--concurrency", default=10, type=int)
    parser.add_argument("--limit", default=0, type=int, help="0 = all")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY missing")

    from consolidate_agent.vocab_maintenance.measure_v3 import (
        load_minimal_vocab,
        reverse_check_minimal_subset,
    )

    vocab = load_minimal_vocab(args.vocab)
    matter_n = len(vocab["facets"]["matter"]["tags"])
    lesson_n = len(vocab["lesson_type"]["values"])
    print(f"loaded vocab: matter={matter_n}, lesson_type={lesson_n} values", flush=True)

    records = load_records(args.db)
    if args.limit:
        records = records[: args.limit]
    print(f"loaded {len(records)} records", flush=True)

    t0 = time.time()
    print(f"tagging: batch_size={args.batch_size} concurrency={args.concurrency}", flush=True)
    assignments = reverse_check_minimal_subset(
        records, vocab,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s ({elapsed/max(1,len(records))*1000:.0f}ms/record)", flush=True)

    output = {
        "version": "v3.0",
        "vocab_source": str(args.vocab),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "record_count": len(records),
        "vocab": vocab,
        "assignments": assignments,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    # Stats
    n = len(assignments)
    miss_matter = sum(1 for a in assignments if not a.get("matter_tags"))
    miss_lesson = sum(1 for a in assignments if not a.get("lesson_type"))
    total_matter = sum(len(a.get("matter_tags") or []) for a in assignments)
    print("\n=== Quick stats ===")
    print(f"  missing matter:  {miss_matter}/{n}")
    print(f"  missing lesson:  {miss_lesson}/{n}")
    print(f"  mean matter/record: {total_matter / max(1, n):.2f}")


if __name__ == "__main__":
    main()
