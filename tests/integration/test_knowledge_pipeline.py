from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

def _has_api_key() -> bool:
    return bool(os.getenv("DASHSCOPE_API_KEY"))

from consolidate_agent.config import Settings
from consolidate_agent.knowledge.consolidation import KnowledgeTag
from consolidate_agent.knowledge.obsidian_export import export_to_obsidian
from consolidate_agent.knowledge.session_turn_store import SessionTurnStore
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.knowledge.vector import (
    KnowledgeVectorStore,
    create_dashscope_embeddings,
    search_knowledge,
)
from consolidate_agent.types import KnowledgeRecord, KnowledgeScope


def _knowledge_record(
    record_id: str,
    *,
    session_id: str = "session-1",
    title: str | None = None,
    insight: str | None = None,
    applicability: str | None = None,
    evidence_turns: list[int] | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> KnowledgeRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    turns = evidence_turns if evidence_turns is not None else [1]
    return KnowledgeRecord(
        id=record_id,
        session_id=session_id,
        title=title or f"Knowledge {record_id}",
        insight=insight or f"Insight for {record_id}",
        applicability=applicability or f"Applicability for {record_id}",
        scope=KnowledgeScope.PROJECT_SPECIFIC,
        evidence_turns=turns,
        evidence_count=len(turns),
        processed_chars=1200,
        created_at=created_at or now,
        updated_at=updated_at or now,
    )


def _knowledge_tag(name: str) -> KnowledgeTag:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return KnowledgeTag(
        tag_id=f"tag-{name}",
        name=name,
        definition=f"{name} knowledge.",
        created_at=now,
        updated_at=now,
    )


def test_knowledge_record_upsert_is_idempotent_when_record_exists(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    updated_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    try:
        store.upsert_knowledge_record(
            _knowledge_record(
                "knowledge-idempotent",
                title="Initial title",
                insight="Initial insight",
                created_at=created_at,
                updated_at=created_at,
            )
        )
        store.upsert_knowledge_record(
            _knowledge_record(
                "knowledge-idempotent",
                title="Updated title",
                insight="Updated insight",
                created_at=created_at,
                updated_at=updated_at,
            )
        )

        records = store.list_all_knowledge_records()

        assert [(record.id, record.title, record.insight, record.updated_at) for record in records] == [
            ("knowledge-idempotent", "Updated title", "Updated insight", updated_at)
        ]
    finally:
        store.close()


def test_knowledge_record_writes_succeed_when_concurrent(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        records = [_knowledge_record(f"knowledge-{index:02d}") for index in range(32)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(store.upsert_knowledge_record, records))

        assert store.count_knowledge_records() == 32
    finally:
        store.close()


@pytest.mark.requires_api
@pytest.mark.skipif(not _has_api_key(), reason="DASHSCOPE_API_KEY required")
def test_knowledge_vector_search_returns_embedded_record_when_query_matches_insight(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        records = [
            _knowledge_record("knowledge-pytest", insight="Pytest fixtures should own setup and teardown boundaries."),
            _knowledge_record("knowledge-sqlite", insight="SQLite write contention needs a short critical section."),
            _knowledge_record("knowledge-chroma", insight="Chroma collections should use stable document ids."),
        ]
        for record in records:
            store.upsert_knowledge_record(record)

        embeddings = create_dashscope_embeddings(Settings())
        vector_store = KnowledgeVectorStore(store, embeddings)
        vector_store.embed_knowledge_records(records)

        results = search_knowledge(store, embeddings, "sqlite write contention", top_k=3, threshold=0.0)

        assert any(
            result["record"].id == "knowledge-sqlite" and result["similarity_score"] > 0
            for result in results
        )
    finally:
        store.close()


@pytest.mark.requires_api
@pytest.mark.skipif(not _has_api_key(), reason="DASHSCOPE_API_KEY required")
def test_session_turn_search_returns_embedded_turn_when_query_matches_turn(tmp_path: Path) -> None:
    session_xml = """<session>
<turn index="1" started_at="t1"><user>Plan the pytest fixture structure.</user></turn>
<turn index="2" started_at="t2"><assistant>SQLite write contention is handled with one store lock.</assistant></turn>
<turn index="3" started_at="t3"><user>Export markdown notes for Obsidian evidence review.</user></turn>
</session>"""
    embeddings = create_dashscope_embeddings(Settings())
    store = SessionTurnStore(tmp_path / "chroma", embeddings)

    store.embed_session("session-search", session_xml)
    results = store.search_turns("session-search", "sqlite write contention", top_k=3)

    assert 2 in {result["turn_index"] for result in results}


def test_knowledge_extraction_pipeline_reads_written_records_when_no_llm(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        records = [
            _knowledge_record("knowledge-pipeline-1"),
            _knowledge_record("knowledge-pipeline-2"),
            _knowledge_record("knowledge-pipeline-3"),
        ]
        for record in records:
            store.upsert_knowledge_record(record)

        loaded = store.list_admitted_knowledge_records()

        assert {record.id for record in loaded} == {
            "knowledge-pipeline-1",
            "knowledge-pipeline-2",
            "knowledge-pipeline-3",
        }
    finally:
        store.close()


def test_knowledge_tags_load_matches_saved_tags_when_record_exists(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        record = _knowledge_record("knowledge-tags")
        store.upsert_knowledge_record(record)

        store.save_knowledge_tags(record.id, ["tag_a", "tag_b"])

        assert store.load_knowledge_tags(record.id) == ["tag_a", "tag_b"]
    finally:
        store.close()


def test_knowledge_tag_assignments_include_confidence_when_assignment_exists(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        record = _knowledge_record("knowledge-assignment")
        store.upsert_knowledge_record(record)
        store.upsert_knowledge_tag(_knowledge_tag("tag_a"))
        store.upsert_knowledge_tag_assignment(record.id, "tag-tag_a", "Matches tag_a.")

        assignments = store.list_knowledge_tag_assignments(record.id)

        assert assignments == [("tag_a", "Matches tag_a.")]
    finally:
        store.close()


def test_obsidian_export_includes_evidence_turns_when_records_have_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "obsidian"
    records = [
        _knowledge_record("knowledge-export-1", session_id="session-export", evidence_turns=[2, 4]),
        _knowledge_record("knowledge-export-2", session_id="session-export", evidence_turns=[5]),
    ]

    export_to_obsidian(records, {"knowledge-export-1": ["tag_a"], "knowledge-export-2": ["tag_b"]}, output_dir)

    note = (output_dir / "knowledge-export-1.md").read_text(encoding="utf-8")

    assert all(
        expected in note
        for expected in [
            "evidence_turns: [2, 4]",
            "session_id: session-export",
            "## Insight",
            "## Applicability",
        ]
    )
