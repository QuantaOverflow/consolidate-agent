#!/usr/bin/env python3
"""Backfill embeddings for source_knowledge_records rows where embedding IS NULL.

These are typically evidence_count=0 records that were intentionally skipped
by embed_knowledge_records (which filters on evidence_count > 0 as product policy).
This script provides a separate backfill path.

Usage:
    python scripts/backfill_record_embeddings.py [--db outputs/knowledge.db] [--dry-run] [--batch-size 50]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
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


def main() -> None:
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    if not args.db.exists():
        sys.exit(f"DB not found: {args.db}")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT record_id, session_id, title, insight, applicability, scope,
               evidence_count, processed_chars, created_at, updated_at,
               evidence_turns_json
        FROM source_knowledge_records
        WHERE embedding IS NULL
        ORDER BY record_id
        """
    ).fetchall()

    print(f"will embed {len(rows)} records")
    if args.dry_run or len(rows) == 0:
        conn.close()
        return

    from consolidate_agent.config import Settings
    from consolidate_agent.knowledge.store import KnowledgeStore
    from consolidate_agent.knowledge.vector import (
        _knowledge_record_text,
        create_dashscope_embeddings,
    )
    from consolidate_agent.types import KnowledgeRecord, KnowledgeScope

    settings = Settings()
    embeddings = create_dashscope_embeddings(settings)
    store = KnowledgeStore(args.db)

    batch_size = args.batch_size
    total = len(rows)
    processed = 0

    for batch_start in range(0, total, batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        records = [
            KnowledgeRecord(
                id=row["record_id"],
                session_id=row["session_id"],
                title=row["title"],
                insight=row["insight"],
                applicability=row["applicability"],
                scope=KnowledgeScope(row["scope"]),
                evidence_turns=json.loads(row["evidence_turns_json"]),
                evidence_count=row["evidence_count"],
                processed_chars=row["processed_chars"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in batch_rows
        ]
        texts = [_knowledge_record_text(r) for r in records]
        vectors = embeddings.embed_documents(texts)
        for record, vec in zip(records, vectors):
            store.save_knowledge_embedding(record.id, vec)
        processed += len(records)
        print(f"  embedded {processed}/{total}")

    remaining = conn.execute(
        "SELECT COUNT(*) FROM source_knowledge_records WHERE embedding IS NULL"
    ).fetchone()[0]
    conn.close()
    print(f"done. remaining NULL embeddings: {remaining}")


if __name__ == "__main__":
    main()
