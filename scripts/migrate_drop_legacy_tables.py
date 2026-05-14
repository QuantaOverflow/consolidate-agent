from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

LEGACY_TABLES = [
    "knowledge_tags",
    "knowledge_tag_assignments",
    "canonical_knowledge",
    "source_pitfall_records",
    "graph_nodes",
    "graph_edges",
    "concept_nodes",
    "concept_edges",
    "tool_domain_nodes",
    "tool_domain_edges",
    "graph_snapshots",
    "graph_evolution_log",
    "governance_proposals",
    "mechanism_tags",
    "agent_invocations",
    "consolidation_failures",
    "consolidation_runs",
    "knowledge_instance_links",
    "knowledge_relations",
    "rule_tag_assignments",
    "tag_proposals",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop legacy tables from knowledge.db")
    parser.add_argument("--db-path", default="outputs/knowledge.db", help="Path to the SQLite database")
    parser.add_argument("--dry-run", action="store_true", help="Print tables to drop without executing")
    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        existing_tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        to_drop = [t for t in LEGACY_TABLES if t in existing_tables]

        if not to_drop:
            print("No legacy tables found to drop.")
        elif args.dry_run:
            print("Dry run — would DROP the following tables:")
            for table in to_drop:
                print(f"  {table}")
        else:
            for table in to_drop:
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"Dropping {table} ({row_count} rows)")
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()

        remaining = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        print(f"\nRemaining tables: {remaining}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
