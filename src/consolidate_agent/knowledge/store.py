from __future__ import annotations

import json
import sqlite3
import hashlib
import threading
from pathlib import Path

from consolidate_agent.types import (
    CanonicalKnowledge,
    CanonicalKnowledgeStatus,
    KnowledgeInstanceLink,
    KnowledgeRelation,
    MechanismTag,
    MechanismTagStatus,
    PitfallCategory,
    PitfallRecord,
    PitfallScope,
    RuleTagAssignment,
    SourceConsolidationStatus,
    TagProposal,
    TagProposalDecision,
    utc_now,
)


class ConsolidationRunStore:
    def __init__(self, connection: sqlite3.Connection, lock: threading.Lock | None = None):
        self.connection = connection
        self._lock = lock or threading.Lock()

    def start_consolidation_run(self, run_id: str, input_path: Path, model: str | None, stats: object) -> None:
        now = utc_now().isoformat()
        self.connection.execute(
            """
            INSERT INTO consolidation_runs (
                run_id, input_path, knowledge_db_path, model, status, stats_json, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                run_id,
                str(input_path),
                _database_path(self.connection),
                model,
                "running",
                _json_payload(stats),
                now,
            ),
        )
        self.connection.commit()

    def finish_consolidation_run(self, run_id: str, status: str, stats: object) -> None:
        self.connection.execute(
            """
            UPDATE consolidation_runs
            SET status = ?, stats_json = ?, finished_at = ?
            WHERE run_id = ?
            """,
            (status, _json_payload(stats), utc_now().isoformat(), run_id),
        )
        self.connection.commit()

    def record_consolidation_failure(
        self,
        stage: str,
        error: str,
        payload: dict[str, object] | None = None,
        run_id: str | None = None,
    ) -> None:
        now = utc_now().isoformat()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
        failure_key = f"{stage}|{error}|{payload_json}|{now}"
        failure_id = f"failure_{hashlib.sha1(failure_key.encode('utf-8')).hexdigest()[:12]}"
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO consolidation_failures (failure_id, run_id, stage, error, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (failure_id, run_id, stage, error, payload_json, now),
            )
            self.connection.commit()

    def record_agent_invocation(
        self,
        *,
        run_id: str | None,
        stage: str,
        status: str,
        input_payload: dict[str, object],
        output_payload: dict[str, object] | None = None,
        repaired_output_payload: dict[str, object] | None = None,
        validation_error: str | None = None,
        latency_ms: int = 0,
    ) -> None:
        now = utc_now().isoformat()
        key = f"{run_id}|{stage}|{status}|{now}|{latency_ms}"
        invocation_id = f"invocation_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO agent_invocations (
                    invocation_id, run_id, stage, status, input_json, output_json,
                    repaired_output_json, validation_error, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    invocation_id,
                    run_id,
                    stage,
                    status,
                    json.dumps(input_payload, ensure_ascii=False, sort_keys=True),
                    json.dumps(output_payload, ensure_ascii=False, sort_keys=True) if output_payload is not None else None,
                    json.dumps(repaired_output_payload, ensure_ascii=False, sort_keys=True) if repaired_output_payload is not None else None,
                    validation_error,
                    latency_ms,
                    now,
                ),
            )
            self.connection.commit()

    def count_agent_invocations(self, run_id: str | None = None) -> int:
        if run_id is None:
            row = self.connection.execute("SELECT COUNT(*) AS count FROM agent_invocations").fetchone()
        else:
            row = self.connection.execute("SELECT COUNT(*) AS count FROM agent_invocations WHERE run_id = ?", (run_id,)).fetchone()
        return int(row["count"])

    def count_agent_failures(self, run_id: str | None = None) -> int:
        query = "SELECT COUNT(*) AS count FROM agent_invocations WHERE status = ?"
        params: tuple[object, ...] = ("failure",)
        if run_id is not None:
            query += " AND run_id = ?"
            params = ("failure", run_id)
        row = self.connection.execute(query, params).fetchone()
        return int(row["count"])

    def count_consolidation_failures(self, run_id: str | None = None) -> int:
        if run_id is None:
            row = self.connection.execute("SELECT COUNT(*) AS count FROM consolidation_failures").fetchone()
        else:
            row = self.connection.execute("SELECT COUNT(*) AS count FROM consolidation_failures WHERE run_id = ?", (run_id,)).fetchone()
        return int(row["count"])

    def latest_consolidation_run(self) -> dict[str, object] | None:
        row = self.connection.execute(
            """
            SELECT run_id, input_path, knowledge_db_path, model, status, stats_json, started_at, finished_at
            FROM consolidation_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
        return _dict_from_row(row) if row is not None else None

    def agent_invocation_summary(self, run_id: str | None = None) -> list[dict[str, object]]:
        query = """
            SELECT stage, status, COUNT(*) AS count,
                   SUM(CASE WHEN repaired_output_json IS NOT NULL THEN 1 ELSE 0 END) AS repaired_count,
                   SUM(CASE WHEN validation_error IS NOT NULL THEN 1 ELSE 0 END) AS validation_error_count,
                   SUM(latency_ms) AS latency_ms
            FROM agent_invocations
        """
        params: tuple[object, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        query += " GROUP BY stage, status ORDER BY stage, status"
        return [_dict_from_row(row) for row in self.connection.execute(query, params).fetchall()]

    def recent_consolidation_failures(self, run_id: str | None = None, limit: int = 5) -> list[dict[str, object]]:
        query = """
            SELECT failure_id, run_id, stage, error, created_at
            FROM consolidation_failures
        """
        params: tuple[object, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        query += " ORDER BY created_at DESC LIMIT ?"
        params = (*params, limit)
        return [_dict_from_row(row) for row in self.connection.execute(query, params).fetchall()]


class KnowledgeStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._run_store = ConsolidationRunStore(self.connection, self._lock)
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

    @property
    def runs(self) -> ConsolidationRunStore:
        return self._run_store

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS source_pitfall_records (
                source_record_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                category TEXT NOT NULL,
                scope TEXT NOT NULL,
                title TEXT NOT NULL,
                preventive_rule TEXT NOT NULL,
                consolidation_status TEXT NOT NULL,
                canonical_id TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS canonical_knowledge (
                canonical_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                summary TEXT NOT NULL,
                preventive_rule TEXT NOT NULL,
                scope TEXT NOT NULL,
                status TEXT NOT NULL,
                support_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_instance_links (
                source_record_id TEXT PRIMARY KEY,
                canonical_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                linked_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mechanism_tags (
                tag_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                definition TEXT NOT NULL,
                status TEXT NOT NULL,
                positive_examples_json TEXT NOT NULL,
                negative_examples_json TEXT NOT NULL,
                merged_into_tag_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rule_tag_assignments (
                canonical_id TEXT PRIMARY KEY,
                tag_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                assignment_reason TEXT NOT NULL,
                linked_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tag_proposals (
                proposal_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                definition TEXT NOT NULL,
                supporting_canonical_ids_json TEXT NOT NULL,
                nearest_existing_tag_ids_json TEXT NOT NULL,
                difference_from_existing TEXT NOT NULL,
                decision TEXT NOT NULL,
                target_tag_id TEXT,
                decision_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_record_id TEXT NOT NULL,
                canonical_id TEXT,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(source_record_id, canonical_id, relation)
            );

            CREATE TABLE IF NOT EXISTS consolidation_failures (
                failure_id TEXT PRIMARY KEY,
                run_id TEXT,
                stage TEXT NOT NULL,
                error TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consolidation_runs (
                run_id TEXT PRIMARY KEY,
                input_path TEXT NOT NULL,
                knowledge_db_path TEXT NOT NULL,
                model TEXT,
                status TEXT NOT NULL,
                stats_json TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_invocations (
                invocation_id TEXT PRIMARY KEY,
                run_id TEXT,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT,
                repaired_output_json TEXT,
                validation_error TEXT,
                latency_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._ensure_column("consolidation_failures", "run_id", "TEXT")
        self._ensure_column("canonical_knowledge", "embedding", "TEXT")
        self._ensure_column("mechanism_tags", "embedding", "TEXT")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        if column in {row["name"] for row in rows}:
            return
        try:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except Exception:  # noqa: BLE001 - concurrent threads may race to add the same column
            pass

    def upsert_source_record(self, record: PitfallRecord) -> None:
        now = utc_now().isoformat()
        payload_json = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        source_hash = _hash_payload(payload_json)
        existing = self.connection.execute(
            "SELECT consolidation_status, canonical_id, error, created_at FROM source_pitfall_records WHERE source_record_id = ?",
            (record.id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        consolidation_status = (
            existing["consolidation_status"] if existing else SourceConsolidationStatus.PENDING.value
        )
        canonical_id = existing["canonical_id"] if existing else None
        error = existing["error"] if existing else None
        self.connection.execute(
            """
            INSERT INTO source_pitfall_records (
                source_record_id, payload_json, source_hash, category, scope, title,
                preventive_rule, consolidation_status, canonical_id, error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_record_id) DO UPDATE SET
                payload_json = excluded.payload_json,
                source_hash = excluded.source_hash,
                category = excluded.category,
                scope = excluded.scope,
                title = excluded.title,
                preventive_rule = excluded.preventive_rule,
                updated_at = excluded.updated_at
            """,
            (
                record.id,
                payload_json,
                source_hash,
                record.category.value,
                record.scope.value,
                record.title,
                record.preventive_rule,
                consolidation_status,
                canonical_id,
                error,
                created_at,
                now,
            ),
        )
        self.connection.commit()

    def list_pending_records(self) -> list[PitfallRecord]:
        rows = self.connection.execute(
            """
            SELECT payload_json
            FROM source_pitfall_records
            WHERE consolidation_status IN (?, ?)
            ORDER BY source_record_id
            """,
            (SourceConsolidationStatus.PENDING.value, SourceConsolidationStatus.FAILED.value),
        ).fetchall()
        return [PitfallRecord.model_validate(json.loads(row["payload_json"])) for row in rows]

    def get_canonical_embedding(self, canonical_id: str) -> list[float] | None:
        with self._lock:
            self._ensure_column("canonical_knowledge", "embedding", "TEXT")
            row = self.connection.execute(
                "SELECT embedding FROM canonical_knowledge WHERE canonical_id = ?",
                (canonical_id,),
            ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return [float(value) for value in json.loads(row["embedding"])]

    def save_canonical_embedding(self, canonical_id: str, embedding: list[float]) -> None:
        with self._lock:
            self._ensure_column("canonical_knowledge", "embedding", "TEXT")
            self.connection.execute(
                "UPDATE canonical_knowledge SET embedding = ?, updated_at = ? WHERE canonical_id = ?",
                (json.dumps(embedding), utc_now().isoformat(), canonical_id),
            )
            self.connection.commit()

    def get_tag_embedding(self, tag_id: str) -> list[float] | None:
        with self._lock:
            self._ensure_column("mechanism_tags", "embedding", "TEXT")
            row = self.connection.execute(
                "SELECT embedding FROM mechanism_tags WHERE tag_id = ?",
                (tag_id,),
            ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return [float(value) for value in json.loads(row["embedding"])]

    def save_tag_embedding(self, tag_id: str, embedding: list[float]) -> None:
        with self._lock:
            self._ensure_column("mechanism_tags", "embedding", "TEXT")
            self.connection.execute(
                "UPDATE mechanism_tags SET embedding = ?, updated_at = ? WHERE tag_id = ?",
                (json.dumps(embedding), utc_now().isoformat(), tag_id),
            )
            self.connection.commit()

    def list_all_source_records(self) -> list[PitfallRecord]:
        rows = self.connection.execute(
            """
            SELECT *
            FROM source_pitfall_records
            ORDER BY source_record_id
            """
        ).fetchall()
        return [PitfallRecord.model_validate(json.loads(row["payload_json"])) for row in rows]

    def list_canonicals_by_category(self, category: PitfallCategory) -> list[CanonicalKnowledge]:
        rows = self.connection.execute(
            """
            SELECT canonical_id, title, category, summary, preventive_rule, scope, status,
                   support_count, created_at, updated_at
            FROM canonical_knowledge
            WHERE category = ? AND status = ?
            ORDER BY canonical_id
            """,
            (category.value, CanonicalKnowledgeStatus.ACTIVE.value),
        ).fetchall()
        canonicals = []
        for row in rows:
            source_record_ids = self._canonical_source_record_ids(row["canonical_id"])
            canonicals.append(
                CanonicalKnowledge(
                    canonical_id=row["canonical_id"],
                    title=row["title"],
                    category=PitfallCategory(row["category"]),
                    summary=row["summary"],
                    preventive_rule=row["preventive_rule"],
                    scope=PitfallScope(row["scope"]),
                    status=CanonicalKnowledgeStatus(row["status"]),
                    source_record_ids=source_record_ids,
                    support_count=row["support_count"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return canonicals

    def get_link(self, source_record_id: str) -> KnowledgeInstanceLink | None:
        row = self.connection.execute(
            """
            SELECT source_record_id, canonical_id, relation, linked_at
            FROM knowledge_instance_links
            WHERE source_record_id = ?
            """,
            (source_record_id,),
        ).fetchone()
        if row is None:
            return None
        return KnowledgeInstanceLink(
            source_record_id=row["source_record_id"],
            canonical_id=row["canonical_id"],
            relation=KnowledgeRelation(row["relation"]),
            linked_at=row["linked_at"],
        )

    def create_canonical(self, canonical: CanonicalKnowledge) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO canonical_knowledge (
                canonical_id, title, category, summary, preventive_rule,
                scope, status, support_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                canonical.canonical_id,
                canonical.title,
                canonical.category.value,
                canonical.summary,
                canonical.preventive_rule,
                canonical.scope.value,
                canonical.status.value,
                canonical.support_count,
                canonical.created_at.isoformat(),
                canonical.updated_at.isoformat(),
            ),
        )
        self.connection.commit()

    def upsert_mechanism_tag(self, tag: MechanismTag) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO mechanism_tags (
                tag_id, name, definition, status, positive_examples_json,
                negative_examples_json, merged_into_tag_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tag.tag_id,
                tag.name,
                tag.definition,
                tag.status.value,
                json.dumps(tag.positive_examples, ensure_ascii=False),
                json.dumps(tag.negative_examples, ensure_ascii=False),
                tag.merged_into_tag_id,
                tag.created_at.isoformat(),
                tag.updated_at.isoformat(),
            ),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def upsert_tag_proposal(self, proposal: TagProposal) -> bool:
        existing = self.connection.execute(
            "SELECT proposal_id FROM tag_proposals WHERE proposal_id = ?",
            (proposal.proposal_id,),
        ).fetchone()
        self.connection.execute(
            """
            INSERT INTO tag_proposals (
                proposal_id, name, definition, supporting_canonical_ids_json,
                nearest_existing_tag_ids_json, difference_from_existing, decision,
                target_tag_id, decision_reason, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(proposal_id) DO UPDATE SET
                decision = excluded.decision,
                target_tag_id = excluded.target_tag_id,
                decision_reason = excluded.decision_reason,
                updated_at = excluded.updated_at
            """,
            (
                proposal.proposal_id,
                proposal.name,
                proposal.definition,
                json.dumps(proposal.supporting_canonical_ids, ensure_ascii=False),
                json.dumps(proposal.nearest_existing_tag_ids, ensure_ascii=False),
                proposal.difference_from_existing,
                proposal.decision.value,
                proposal.target_tag_id,
                proposal.decision_reason,
                proposal.created_at.isoformat(),
                proposal.updated_at.isoformat(),
            ),
        )
        self.connection.commit()
        return existing is None

    def upsert_rule_tag_assignment(self, assignment: RuleTagAssignment) -> bool:
        existing = self.get_rule_tag_assignment(assignment.canonical_id)
        if existing is not None:
            return False
        self.connection.execute(
            """
            INSERT INTO rule_tag_assignments (canonical_id, tag_id, confidence, assignment_reason, linked_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                assignment.canonical_id,
                assignment.tag_id,
                assignment.confidence,
                assignment.assignment_reason,
                assignment.linked_at.isoformat(),
            ),
        )
        self.connection.commit()
        return True

    def increment_canonical_support(self, canonical_id: str) -> None:
        self.connection.execute(
            """
            UPDATE canonical_knowledge
            SET support_count = support_count + 1, updated_at = ?
            WHERE canonical_id = ?
            """,
            (utc_now().isoformat(), canonical_id),
        )
        self.connection.commit()

    def upsert_link(self, source_record_id: str, canonical_id: str, relation: KnowledgeRelation) -> bool:
        existing = self.get_link(source_record_id)
        if existing is not None:
            return False
        linked_at = utc_now().isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO knowledge_instance_links (source_record_id, canonical_id, relation, linked_at)
                VALUES (?, ?, ?, ?)
                """,
                (source_record_id, canonical_id, relation.value, linked_at),
            )
            self.connection.execute(
                """
                UPDATE source_pitfall_records
                SET consolidation_status = ?, canonical_id = ?, error = NULL, updated_at = ?
                WHERE source_record_id = ?
                """,
                (SourceConsolidationStatus.LINKED.value, canonical_id, linked_at, source_record_id),
            )
        return True

    def write_relation(
        self,
        source_record_id: str,
        canonical_id: str | None,
        relation: KnowledgeRelation,
    ) -> None:
        created_at = utc_now().isoformat()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO knowledge_relations (source_record_id, canonical_id, relation, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (source_record_id, canonical_id, relation.value, created_at),
        )
        self.connection.commit()

    def mark_classified(self, source_record_id: str) -> None:
        self.connection.execute(
            """
            UPDATE source_pitfall_records
            SET consolidation_status = ?, error = NULL, updated_at = ?
            WHERE source_record_id = ?
            """,
            (SourceConsolidationStatus.CLASSIFIED.value, utc_now().isoformat(), source_record_id),
        )
        self.connection.commit()

    def mark_failed(self, source_record_id: str, error: str) -> None:
        self.connection.execute(
            """
            UPDATE source_pitfall_records
            SET consolidation_status = ?, error = ?, updated_at = ?
            WHERE source_record_id = ?
            """,
            (SourceConsolidationStatus.FAILED.value, error, utc_now().isoformat(), source_record_id),
        )
        self.connection.commit()

    def get_source_status(self, source_record_id: str) -> SourceConsolidationStatus:
        row = self.connection.execute(
            "SELECT consolidation_status FROM source_pitfall_records WHERE source_record_id = ?",
            (source_record_id,),
        ).fetchone()
        return SourceConsolidationStatus(row["consolidation_status"])

    def count_canonicals(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM canonical_knowledge").fetchone()
        return int(row["count"])

    def count_tags(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM mechanism_tags").fetchone()
        return int(row["count"])

    def count_tag_proposals(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM tag_proposals").fetchone()
        return int(row["count"])

    def count_rule_tag_assignments(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM rule_tag_assignments").fetchone()
        return int(row["count"])

    def list_active_tags(self) -> list[MechanismTag]:
        rows = self.connection.execute(
            """
            SELECT tag_id, name, definition, status, positive_examples_json,
                   negative_examples_json, merged_into_tag_id, created_at, updated_at
            FROM mechanism_tags
            WHERE status = ?
            ORDER BY tag_id
            """,
            (MechanismTagStatus.ACTIVE.value,),
        ).fetchall()
        return [_tag_from_row(row) for row in rows]

    def list_active_canonicals(self) -> list[CanonicalKnowledge]:
        rows = self.connection.execute(
            """
            SELECT canonical_id, title, category, summary, preventive_rule, scope, status,
                   support_count, created_at, updated_at
            FROM canonical_knowledge
            WHERE status = ?
            ORDER BY canonical_id
            """,
            (CanonicalKnowledgeStatus.ACTIVE.value,),
        ).fetchall()
        return [
            CanonicalKnowledge(
                canonical_id=row["canonical_id"],
                title=row["title"],
                category=PitfallCategory(row["category"]),
                summary=row["summary"],
                preventive_rule=row["preventive_rule"],
                scope=PitfallScope(row["scope"]),
                status=CanonicalKnowledgeStatus(row["status"]),
                source_record_ids=self._canonical_source_record_ids(row["canonical_id"]),
                support_count=row["support_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_tag(self, tag_id: str) -> MechanismTag | None:
        row = self.connection.execute(
            """
            SELECT tag_id, name, definition, status, positive_examples_json,
                   negative_examples_json, merged_into_tag_id, created_at, updated_at
            FROM mechanism_tags
            WHERE tag_id = ?
            """,
            (tag_id,),
        ).fetchone()
        return _tag_from_row(row) if row is not None else None

    def get_rule_tag_assignment(self, canonical_id: str) -> RuleTagAssignment | None:
        row = self.connection.execute(
            """
            SELECT canonical_id, tag_id, confidence, assignment_reason, linked_at
            FROM rule_tag_assignments
            WHERE canonical_id = ?
            """,
            (canonical_id,),
        ).fetchone()
        if row is None:
            return None
        return RuleTagAssignment(
            canonical_id=row["canonical_id"],
            tag_id=row["tag_id"],
            confidence=row["confidence"],
            assignment_reason=row["assignment_reason"],
            linked_at=row["linked_at"],
        )

    def list_rule_tag_assignments(self, canonical_ids: list[str]) -> list[RuleTagAssignment]:
        assignments: list[RuleTagAssignment] = []
        for canonical_id in canonical_ids:
            assignment = self.get_rule_tag_assignment(canonical_id)
            if assignment is not None:
                assignments.append(assignment)
        return assignments

    def list_canonicals_without_tag(self) -> list[CanonicalKnowledge]:
        rows = self.connection.execute(
            """
            SELECT c.canonical_id, c.title, c.category, c.summary, c.preventive_rule, c.scope,
                   c.status, c.support_count, c.created_at, c.updated_at
            FROM canonical_knowledge c
            LEFT JOIN rule_tag_assignments a ON a.canonical_id = c.canonical_id
            WHERE c.status = ? AND a.canonical_id IS NULL
            ORDER BY c.canonical_id
            """,
            (CanonicalKnowledgeStatus.ACTIVE.value,),
        ).fetchall()
        return [
            CanonicalKnowledge(
                canonical_id=row["canonical_id"],
                title=row["title"],
                category=PitfallCategory(row["category"]),
                summary=row["summary"],
                preventive_rule=row["preventive_rule"],
                scope=PitfallScope(row["scope"]),
                status=CanonicalKnowledgeStatus(row["status"]),
                source_record_ids=self._canonical_source_record_ids(row["canonical_id"]),
                support_count=row["support_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_canonical(self, canonical_id: str) -> CanonicalKnowledge | None:
        row = self.connection.execute(
            """
            SELECT canonical_id, title, category, summary, preventive_rule, scope, status,
                   support_count, created_at, updated_at
            FROM canonical_knowledge
            WHERE canonical_id = ?
            """,
            (canonical_id,),
        ).fetchone()
        if row is None:
            return None
        return CanonicalKnowledge(
            canonical_id=row["canonical_id"],
            title=row["title"],
            category=PitfallCategory(row["category"]),
            summary=row["summary"],
            preventive_rule=row["preventive_rule"],
            scope=PitfallScope(row["scope"]),
            status=CanonicalKnowledgeStatus(row["status"]),
            source_record_ids=self._canonical_source_record_ids(row["canonical_id"]),
            support_count=row["support_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def count_links(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM knowledge_instance_links").fetchone()
        return int(row["count"])

    def count_relations(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM knowledge_relations").fetchone()
        return int(row["count"])

    def relation_values_for_source(self, source_record_id: str) -> list[str]:
        rows = self.connection.execute(
            "SELECT relation FROM knowledge_relations WHERE source_record_id = ? ORDER BY relation",
            (source_record_id,),
        ).fetchall()
        return [str(row["relation"]) for row in rows]

    def relation_values_for_canonical(self, canonical_id: str) -> list[str]:
        rows = self.connection.execute(
            "SELECT relation FROM knowledge_relations WHERE canonical_id = ? ORDER BY relation",
            (canonical_id,),
        ).fetchall()
        return [str(row["relation"]) for row in rows]

    def tag_rule_counts(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT m.name, COUNT(r.canonical_id) AS rule_count
            FROM mechanism_tags m
            LEFT JOIN rule_tag_assignments r ON r.tag_id = m.tag_id
            WHERE m.status = ?
            GROUP BY m.tag_id, m.name
            ORDER BY m.name
            """,
            (MechanismTagStatus.ACTIVE.value,),
        ).fetchall()
        return [_dict_from_row(row) for row in rows]

    def proposal_decision_counts(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT decision, COUNT(*) AS count
            FROM tag_proposals
            GROUP BY decision
            ORDER BY decision
            """
        ).fetchall()
        return [_dict_from_row(row) for row in rows]

    def _canonical_source_record_ids(self, canonical_id: str) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT source_record_id
            FROM knowledge_instance_links
            WHERE canonical_id = ?
            ORDER BY source_record_id
            """,
            (canonical_id,),
        ).fetchall()
        return [row["source_record_id"] for row in rows]


def _hash_payload(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _json_payload(value: object) -> str:
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _database_path(connection: sqlite3.Connection) -> str:
    row = connection.execute("PRAGMA database_list").fetchone()
    if row is None:
        return ""
    return str(row["file"])


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
