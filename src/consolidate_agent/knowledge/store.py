from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from consolidate_agent.types import (
    KnowledgeRecord,
    KnowledgeScope,
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
        self._ensure_column("processed_sessions", "extract_status", "TEXT DEFAULT 'pending'")
        self._ensure_column("processed_sessions", "extract_error", "TEXT")
        self._ensure_column("processed_sessions", "normalized_hash", "TEXT")
        self._ensure_column("processed_sessions", "extracted_at", "TEXT")
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

    def is_session_extracted(self, session_id: str) -> bool:
        return self.get_session_extract_status(session_id) == "processed"

    def get_session_extract_status(self, session_id: str) -> str | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT extract_status FROM processed_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None or row["extract_status"] is None:
            return None
        return str(row["extract_status"])

    def mark_session_extract_status(
        self,
        session_id: str,
        status: str,
        error: str | None = None,
        normalized_hash: str | None = None,
    ) -> None:
        now = utc_now().isoformat()
        with self._lock:
            self.connection.execute(
                """
                UPDATE processed_sessions
                SET extract_status = ?,
                    extract_error = ?,
                    normalized_hash = COALESCE(?, normalized_hash),
                    extracted_at = CASE WHEN ? = 'processed' THEN ? ELSE extracted_at END
                WHERE session_id = ?
                """,
                (status, error, normalized_hash, status, now, session_id),
            )
            self.connection.commit()

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

    def list_pending_unverified_records(self) -> list[KnowledgeRecord]:
        rows = self.connection.execute(
            """
            SELECT record_id, session_id, title, insight, applicability, scope,
                   evidence_turns_json, evidence_count,
                   processed_chars, created_at, updated_at
            FROM source_knowledge_records
            WHERE evidence_count = 0 AND consolidation_status = 'pending'
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

    def count_knowledge_records(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM source_knowledge_records").fetchone()
        return int(row["count"])


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
