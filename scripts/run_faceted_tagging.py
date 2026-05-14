#!/usr/bin/env python3
"""Gate 3: run v2 faceted tagging on all knowledge records.

Loads vocab_v2.json + all records from knowledge.db, runs reverse_check_faceted
in concurrent batches, writes outputs/network_v2.json.

Usage:
  python scripts/run_faceted_tagging.py [--vocab PATH] [--db PATH] [--out PATH]
                                         [--batch-size N] [--concurrency N]
                                         [--limit N]   # 0 = all
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
    parser.add_argument("--vocab", default=Path("docs/plans/vocab_v2.json"), type=Path)
    parser.add_argument("--db", default=Path("outputs/knowledge.db"), type=Path)
    parser.add_argument("--out", default=Path("outputs/network_v2.json"), type=Path)
    parser.add_argument("--batch-size", default=5, type=int)
    parser.add_argument("--concurrency", default=5, type=int)
    parser.add_argument("--limit", default=0, type=int, help="0 = all records")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY missing (looked in .env)")

    from consolidate_agent.vocab_maintenance.measure_v2 import (
        load_faceted_vocab,
        reverse_check_faceted_subset,
    )

    vocab = load_faceted_vocab(args.vocab)
    matter_n = len(vocab["facets"]["matter"]["tags"])
    activity_n = len(vocab["facets"]["activity"]["tags"])
    pattern_n = len(vocab["facets"]["pattern"]["tags"])
    print(
        f"loaded vocab: matter={matter_n} activity={activity_n} pattern={pattern_n}",
        flush=True,
    )

    records = load_records(args.db)
    if args.limit:
        records = records[: args.limit]
    print(f"loaded {len(records)} records to tag", flush=True)

    t0 = time.time()
    print(
        f"starting tagging: batch_size={args.batch_size} concurrency={args.concurrency}",
        flush=True,
    )
    assignments = reverse_check_faceted_subset(
        records,
        vocab,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
    elapsed = time.time() - t0
    print(
        f"done in {elapsed:.1f}s ({elapsed / max(1, len(records)) * 1000:.0f}ms/record)",
        flush=True,
    )

    # Persist
    output = {
        "version": "v2.0",
        "vocab_source": str(args.vocab),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "record_count": len(records),
        "vocab": vocab,
        "assignments": assignments,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(assignments)} assignments)", flush=True)

    # Quick stats
    print("\n=== Quick stats ===")
    missing_matter = sum(1 for a in assignments if not a.get("matter_tags"))
    missing_activity = sum(1 for a in assignments if not a.get("activity_tag"))
    missing_pattern = sum(1 for a in assignments if not a.get("pattern_tags"))
    missing_lesson = sum(1 for a in assignments if not a.get("lesson_type"))
    print(f"  missing matter:  {missing_matter}/{len(assignments)}")
    print(f"  missing activity:{missing_activity}/{len(assignments)}")
    print(f"  empty pattern:   {missing_pattern}/{len(assignments)}")
    print(f"  missing lesson:  {missing_lesson}/{len(assignments)}")
    total_tags = sum(
        len(a.get("matter_tags") or [])
        + (1 if a.get("activity_tag") else 0)
        + len(a.get("pattern_tags") or [])
        for a in assignments
    )
    print(
        f"  mean tags/record: {total_tags / max(1, len(assignments)):.2f}",
    )


if __name__ == "__main__":
    main()
