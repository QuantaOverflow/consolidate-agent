from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from consolidate_agent.types import (
    KnowledgeRecord,
    KnowledgeScope,
    MechanismTag,
    MechanismTagStatus,
    utc_now,
)


class KnowledgeStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.connection = conn
        return self._local.connection

    @connection.setter
    def connection(self, value: sqlite3.Connection) -> None:
        self._local.connection = value

    def close(self) -> None:
        self.connection.close()

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS source_knowledge_records (
                record_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                title TEXT NOT NULL,
                insight TEXT NOT NULL,
                applicability TEXT NOT NULL,
                scope TEXT NOT NULL,
                evidence_turns_json TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                processed_chars INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_tags (
                tag_id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                definition TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                merged_into_tag_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_tag_assignments (
                record_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.8,
                assignment_reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (record_id, tag_id)
            );

            CREATE TABLE IF NOT EXISTS processed_sessions (
                session_id TEXT PRIMARY KEY,
                xml TEXT NOT NULL,
                processed_chars INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._ensure_column("source_knowledge_records", "embedding", "TEXT")
        self._ensure_column("source_knowledge_records", "tags_json", "TEXT")
        self._ensure_column("source_knowledge_records", "consolidation_status", "TEXT DEFAULT 'pending'")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        if column in {row["name"] for row in rows}:
            return
        try:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except Exception:  # noqa: BLE001 - concurrent threads may race to add the same column
            pass

    def save_processed_session(self, session_id: str, xml: str, processed_chars: int) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO processed_sessions (session_id, xml, processed_chars, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, xml, processed_chars, utc_now().isoformat()),
            )
            self.connection.commit()

    def get_processed_session(self, session_id: str) -> str | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT xml FROM processed_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return str(row["xml"]) if row is not None else None

    def list_unembedded_sessions(self, embedded_session_ids: set[str]) -> list[tuple[str, str]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT session_id, xml
                FROM processed_sessions
                ORDER BY created_at, session_id
                """
            ).fetchall()
        return [
            (str(row["session_id"]), str(row["xml"]))
            for row in rows
            if str(row["session_id"]) not in embedded_session_ids
        ]

    def _get_embedding(self, table: str, key_column: str, key: str) -> list[float] | None:
        with self._lock:
            row = self.connection.execute(
                f"SELECT embedding FROM {table} WHERE {key_column} = ?",
                (key,),
            ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return [float(value) for value in json.loads(row["embedding"])]

    def _save_embedding(self, table: str, key_column: str, key: str, embedding: list[float]) -> None:
        with self._lock:
            cursor = self.connection.execute(
                f"UPDATE {table} SET embedding = ?, updated_at = ? WHERE {key_column} = ?",
                (json.dumps(embedding), utc_now().isoformat(), key),
            )
            if cursor.rowcount == 0:
                self.connection.rollback()
                raise ValueError(f"No record found for {key_column}={key!r} in {table}")
            self.connection.commit()

    def save_knowledge_embedding(self, record_id: str, embedding: list[float]) -> None:
        self._save_embedding("source_knowledge_records", "record_id", record_id, embedding)

    def get_knowledge_embedding(self, record_id: str) -> list[float] | None:
        return self._get_embedding("source_knowledge_records", "record_id", record_id)

    def load_all_knowledge_embeddings(self) -> dict[str, list[float]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT record_id, embedding
                FROM source_knowledge_records
                WHERE embedding IS NOT NULL
                  AND evidence_count > 0
                """
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def list_knowledge_records_without_embedding(self) -> list[KnowledgeRecord]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT record_id, session_id, title, insight, applicability, scope,
                       evidence_turns_json, evidence_count,
                       processed_chars, created_at, updated_at
                FROM source_knowledge_records
                WHERE embedding IS NULL
                  AND evidence_count > 0
                ORDER BY record_id
                """
            ).fetchall()
        return [_knowledge_record_from_row(row) for row in rows]

    def get_tag_embedding(self, tag_id: str) -> list[float] | None:
        return self._get_embedding("mechanism_tags", "tag_id", tag_id)

    def save_tag_embedding(self, tag_id: str, embedding: list[float]) -> None:
        self._save_embedding("mechanism_tags", "tag_id", tag_id, embedding)

    def upsert_knowledge_record(self, record: KnowledgeRecord) -> bool:
        existing = self.connection.execute(
            "SELECT record_id FROM source_knowledge_records WHERE record_id = ?",
            (record.id,),
        ).fetchone()
        self.connection.execute(
            """
            INSERT INTO source_knowledge_records (
                record_id, session_id, title, insight, applicability, scope,
                evidence_turns_json, evidence_count,
                processed_chars, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                session_id = excluded.session_id,
                title = excluded.title,
                insight = excluded.insight,
                applicability = excluded.applicability,
                scope = excluded.scope,
                evidence_turns_json = excluded.evidence_turns_json,
                evidence_count = excluded.evidence_count,
                processed_chars = excluded.processed_chars,
                updated_at = excluded.updated_at
            """,
            (
                record.id,
                record.session_id,
                record.title,
                record.insight,
                record.applicability,
                record.scope.value,
                json.dumps(record.evidence_turns, ensure_ascii=False),
                record.evidence_count,
                record.processed_chars,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
            ),
        )
        self.connection.commit()
        return existing is None

    def list_all_knowledge_records(self) -> list[KnowledgeRecord]:
        rows = self.connection.execute(
            """
            SELECT record_id, session_id, title, insight, applicability, scope,
                   evidence_turns_json, evidence_count,
                   processed_chars, created_at, updated_at
            FROM source_knowledge_records
            ORDER BY record_id
            """
        ).fetchall()
        return [_knowledge_record_from_row(row) for row in rows]

    def list_verified_knowledge_records(self) -> list[KnowledgeRecord]:
        rows = self.connection.execute(
            """
            SELECT record_id, session_id, title, insight, applicability, scope,
                   evidence_turns_json, evidence_count,
                   processed_chars, created_at, updated_at
            FROM source_knowledge_records
            WHERE evidence_count > 0
            ORDER BY record_id
            """
        ).fetchall()
        return [_knowledge_record_from_row(row) for row in rows]

    def list_admitted_knowledge_records(self) -> list[KnowledgeRecord]:
        return self.list_verified_knowledge_records()

    def upsert_knowledge_tag(self, tag: KnowledgeTag) -> bool:
        existing = self.connection.execute(
            "SELECT tag_id FROM knowledge_tags WHERE tag_id = ? OR name = ?",
            (tag.tag_id, tag.name),
        ).fetchone()
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO knowledge_tags (
                    tag_id, name, definition, status, merged_into_tag_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tag_id) DO UPDATE SET
                    name = excluded.name,
                    definition = excluded.definition,
                    status = excluded.status,
                    merged_into_tag_id = excluded.merged_into_tag_id,
                    updated_at = excluded.updated_at
                ON CONFLICT(name) DO UPDATE SET
                    definition = excluded.definition,
                    status = excluded.status,
                    merged_into_tag_id = excluded.merged_into_tag_id,
                    updated_at = excluded.updated_at
                """,
                (
                    tag.tag_id,
                    tag.name,
                    tag.definition,
                    tag.status,
                    tag.merged_into_tag_id,
                    tag.created_at.isoformat(),
                    tag.updated_at.isoformat(),
                ),
            )
            self.connection.commit()
        return existing is None

    def list_active_knowledge_tags(self) -> list[KnowledgeTag]:
        rows = self.connection.execute(
            """
            SELECT tag_id, name, definition, status, merged_into_tag_id, created_at, updated_at
            FROM knowledge_tags
            WHERE status = 'active'
            ORDER BY name
            """
        ).fetchall()
        return [_knowledge_tag_from_row(row) for row in rows]

    def merge_knowledge_tag(self, keep_id: str, merge_id: str) -> None:
        now = utc_now().isoformat()
        with self._lock:
            with self.connection:
                self.connection.execute(
                    """
                    UPDATE knowledge_tags
                    SET status = 'merged', merged_into_tag_id = ?, updated_at = ?
                    WHERE tag_id = ?
                    """,
                    (keep_id, now, merge_id),
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO knowledge_tag_assignments (
                        record_id, tag_id, confidence, assignment_reason, created_at
                    )
                    SELECT record_id, ?, confidence, assignment_reason, created_at
                    FROM knowledge_tag_assignments
                    WHERE tag_id = ?
                    """,
                    (keep_id, merge_id),
                )
                self.connection.execute(
                    "DELETE FROM knowledge_tag_assignments WHERE tag_id = ?",
                    (merge_id,),
                )

    def set_knowledge_consolidation_status(self, record_id: str, status: str) -> None:
        """Set consolidation_status for a knowledge record: pending / assigned / no_tag."""
        with self._lock:
            self.connection.execute(
                "UPDATE source_knowledge_records SET consolidation_status = ? WHERE record_id = ?",
                (status, record_id),
            )
            self.connection.commit()

    def list_no_tag_knowledge_records(self) -> list[KnowledgeRecord]:
        """Return admitted records with consolidation_status = 'no_tag'."""
        rows = self.connection.execute(
            """
            SELECT record_id, session_id, title, insight, applicability, scope,
                   evidence_turns_json, evidence_count, processed_chars, created_at, updated_at
            FROM source_knowledge_records
            WHERE evidence_count > 0 AND consolidation_status = 'no_tag'
            """
        ).fetchall()
        return [_knowledge_record_from_row(row) for row in rows]

    def upsert_knowledge_tag_assignment(self, record_id: str, tag_id: str, reason: str) -> None:
        now = utc_now().isoformat()
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO knowledge_tag_assignments (
                    record_id, tag_id, confidence, assignment_reason, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(record_id, tag_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    assignment_reason = excluded.assignment_reason
                """,
                (record_id, tag_id, 0.8, reason, now),
            )
            self.connection.commit()

    def list_knowledge_tag_assignments(self, record_id: str) -> list[tuple[str, str]]:
        rows = self.connection.execute(
            """
            SELECT t.name, a.assignment_reason
            FROM knowledge_tag_assignments a
            JOIN knowledge_tags t ON t.tag_id = a.tag_id
            WHERE a.record_id = ? AND t.status = 'active'
            ORDER BY t.name
            """,
            (record_id,),
        ).fetchall()
        return [(str(row["name"]), str(row["assignment_reason"])) for row in rows]

    def list_records_without_knowledge_tag(self) -> list[KnowledgeRecord]:
        rows = self.connection.execute(
            """
            SELECT r.record_id, r.session_id, r.title, r.insight, r.applicability, r.scope,
                   r.evidence_turns_json, r.evidence_count,
                   r.processed_chars, r.created_at, r.updated_at
            FROM source_knowledge_records r
            LEFT JOIN knowledge_tag_assignments a ON a.record_id = r.record_id
            WHERE r.evidence_count > 0 AND a.record_id IS NULL
            ORDER BY r.record_id
            """
        ).fetchall()
        return [_knowledge_record_from_row(row) for row in rows]

    def save_knowledge_tags(self, record_id: str, tags: list[str]) -> None:
        with self._lock:
            self.connection.execute(
                """
                UPDATE source_knowledge_records
                SET tags_json = ?, updated_at = ?
                WHERE record_id = ?
                """,
                (json.dumps(tags, ensure_ascii=False), utc_now().isoformat(), record_id),
            )
            self.connection.commit()

    def load_all_knowledge_tags(self) -> dict[str, list[str]]:
        """Return {record_id: tags} for all admitted records that have tags."""
        with self._lock:
            rows = self.connection.execute(
                "SELECT record_id, tags_json FROM source_knowledge_records "
                "WHERE evidence_count > 0 AND tags_json IS NOT NULL"
            ).fetchall()
        result = {}
        for row in rows:
            try:
                tags = json.loads(row["tags_json"])
                if isinstance(tags, list) and tags:
                    result[row["record_id"]] = [str(t) for t in tags]
            except json.JSONDecodeError:
                pass
        return result

    def apply_tag_merge_map(self, merge_map: dict[str, str]) -> int:
        """Apply a {old_tag: canonical_tag} merge map to all records. Returns updated record count."""
        if not merge_map:
            return 0
        all_tags = self.load_all_knowledge_tags()
        updated = 0
        for record_id, tags in all_tags.items():
            new_tags = list(dict.fromkeys(merge_map.get(t, t) for t in tags))
            if new_tags != tags:
                self.save_knowledge_tags(record_id, new_tags)
                updated += 1
        return updated

    def load_knowledge_tags(self, record_id: str) -> list[str]:
        with self._lock:
            row = self.connection.execute(
                "SELECT tags_json FROM source_knowledge_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        if row is None or row["tags_json"] is None:
            return []
        try:
            tags = json.loads(row["tags_json"])
        except json.JSONDecodeError:
            return []
        if not isinstance(tags, list):
            return []
        return [str(tag) for tag in tags]

    def count_knowledge_records(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM source_knowledge_records").fetchone()
        return int(row["count"])


def _dict_from_row(row: sqlite3.Row) -> dict[str, object]:
    return {key: row[key] for key in row.keys()}


def _tag_from_row(row: sqlite3.Row) -> MechanismTag:
    return MechanismTag(
        tag_id=row["tag_id"],
        name=row["name"],
        definition=row["definition"],
        status=MechanismTagStatus(row["status"]),
        positive_examples=json.loads(row["positive_examples_json"]),
        negative_examples=json.loads(row["negative_examples_json"]),
        merged_into_tag_id=row["merged_into_tag_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _knowledge_tag_from_row(row: sqlite3.Row):
    from consolidate_agent.knowledge.consolidation import KnowledgeTag

    return KnowledgeTag(
        tag_id=row["tag_id"],
        name=row["name"],
        definition=row["definition"],
        status=row["status"],
        merged_into_tag_id=row["merged_into_tag_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _knowledge_record_from_row(row: sqlite3.Row) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=row["record_id"],
        session_id=row["session_id"],
        title=row["title"],
        insight=row["insight"],
        applicability=row["applicability"],
        scope=KnowledgeScope(row["scope"]),
        evidence_turns=json.loads(row["evidence_turns_json"]),
        evidence_count=int(row["evidence_count"]),
        processed_chars=int(row["processed_chars"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
