from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.knowledge.vector import KnowledgeVectorStore, find_related_knowledge, search_knowledge
from consolidate_agent.types import (
    CanonicalKnowledge,
    KnowledgeRecord,
    KnowledgeScope,
    PitfallCategory,
    PitfallScope,
)


class FakeEmbeddings(Embeddings):
    def __init__(self, vectors: list[list[float]] | None = None):
        self.vectors = vectors or []
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return self.vectors[: len(texts)]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.vectors[0]


def _store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "knowledge.db")


def _knowledge_record(record_id: str, title: str | None = None, *, evidence_count: int = 2) -> KnowledgeRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return KnowledgeRecord(
        id=record_id,
        session_id="session-1",
        title=title or f"Knowledge {record_id}",
        insight=f"Insight {record_id}",
        applicability=f"Applicability {record_id}",
        scope=KnowledgeScope.GLOBAL,
        evidence_turns=[1, 2] if evidence_count > 0 else [],
        evidence_count=evidence_count,
        processed_chars=1200,
        created_at=now,
        updated_at=now,
    )


def _canonical(canonical_id: str = "canonical-1") -> CanonicalKnowledge:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return CanonicalKnowledge(
        canonical_id=canonical_id,
        title="Canonical pitfall",
        category=PitfallCategory.EXECUTION_STRATEGY,
        summary="A pitfall that should retrieve related knowledge.",
        preventive_rule="Check related operating knowledge before acting.",
        scope=PitfallScope.GLOBAL,
        support_count=1,
        created_at=now,
        updated_at=now,
    )


def test_embed_knowledge_records_is_idempotent_for_existing_embeddings(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        existing = _knowledge_record("knowledge-1", "Already embedded")
        missing = _knowledge_record("knowledge-2", "Needs embedding")
        store.upsert_knowledge_record(existing)
        store.upsert_knowledge_record(missing)
        store.save_knowledge_embedding(existing.id, [1.0, 0.0])

        embeddings = FakeEmbeddings(vectors=[[0.0, 1.0]])
        vector_store = KnowledgeVectorStore(store, embeddings)

        vector_store.embed_knowledge_records([existing, missing])

        assert embeddings.document_calls == [
            [
                "Title: Needs embedding\n"
                "Insight: Insight knowledge-2\n"
                "Applicability: Applicability knowledge-2"
            ]
        ]
        assert store.get_knowledge_embedding(existing.id) == [1.0, 0.0]
        assert store.get_knowledge_embedding(missing.id) == [0.0, 1.0]
    finally:
        store.close()


def test_embed_knowledge_records_skips_soft_rejected_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        rejected = _knowledge_record("knowledge-0", "Soft rejected", evidence_count=0)
        verified = _knowledge_record("knowledge-1", "Verified")
        store.upsert_knowledge_record(rejected)
        store.upsert_knowledge_record(verified)

        embeddings = FakeEmbeddings(vectors=[[0.0, 1.0]])
        vector_store = KnowledgeVectorStore(store, embeddings)

        vector_store.embed_knowledge_records([rejected, verified])

        assert embeddings.document_calls == [
            [
                "Title: Verified\n"
                "Insight: Insight knowledge-1\n"
                "Applicability: Applicability knowledge-1"
            ]
        ]
        assert store.get_knowledge_embedding(rejected.id) is None
        assert store.get_knowledge_embedding(verified.id) == [0.0, 1.0]
    finally:
        store.close()


def test_unembedded_knowledge_records_excludes_soft_rejected_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        rejected = _knowledge_record("knowledge-0", "Soft rejected", evidence_count=0)
        verified = _knowledge_record("knowledge-1", "Verified")
        store.upsert_knowledge_record(rejected)
        store.upsert_knowledge_record(verified)

        assert [record.id for record in store.list_knowledge_records_without_embedding()] == ["knowledge-1"]
    finally:
        store.close()


def test_knowledge_vector_store_does_not_own_related_knowledge_search() -> None:
    assert not hasattr(KnowledgeVectorStore, "find_related_knowledge")


def test_save_embedding_raises_for_missing_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        with pytest.raises(ValueError, match="No record found for record_id='missing' in source_knowledge_records"):
            store.save_knowledge_embedding("missing", [1.0, 0.0])
    finally:
        store.close()


def test_find_related_knowledge_filters_by_threshold_and_orders_by_similarity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        canonical = _canonical()
        store.create_canonical(canonical)
        store.save_canonical_embedding(canonical.canonical_id, [1.0, 0.0])

        strong = _knowledge_record("knowledge-1", "Strong match")
        medium = _knowledge_record("knowledge-2", "Medium match")
        weak = _knowledge_record("knowledge-3", "Weak match")
        for record in [medium, weak, strong]:
            store.upsert_knowledge_record(record)
        store.save_knowledge_embedding(strong.id, [1.0, 0.0])
        store.save_knowledge_embedding(medium.id, [0.8, 0.6])
        store.save_knowledge_embedding(weak.id, [0.0, 1.0])

        results = find_related_knowledge(store, FakeEmbeddings(), canonical.canonical_id, threshold=0.75, top_k=3)

        assert [result["record_id"] for result in results] == ["knowledge-1", "knowledge-2"]
        assert results[0]["similarity_score"] > results[1]["similarity_score"]
        assert results[0] == {
            "record_id": "knowledge-1",
            "title": "Strong match",
            "insight": "Insight knowledge-1",
            "applicability": "Applicability knowledge-1",
            "scope": "global",
            "similarity_score": 1.0,
        }
    finally:
        store.close()


def test_find_related_knowledge_excludes_soft_rejected_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        canonical = _canonical()
        verified = _knowledge_record("knowledge-1", "Verified")
        rejected = _knowledge_record("knowledge-0", "Soft rejected", evidence_count=0)
        store.create_canonical(canonical)
        store.save_canonical_embedding(canonical.canonical_id, [1.0, 0.0])
        for record in [verified, rejected]:
            store.upsert_knowledge_record(record)
            store.save_knowledge_embedding(record.id, [1.0, 0.0])

        results = find_related_knowledge(store, FakeEmbeddings(), canonical.canonical_id, threshold=0.75, top_k=3)

        assert [result["record_id"] for result in results] == ["knowledge-1"]
    finally:
        store.close()


def test_search_knowledge_excludes_soft_rejected_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        verified = _knowledge_record("knowledge-1", "Verified")
        rejected = _knowledge_record("knowledge-0", "Soft rejected", evidence_count=0)
        for record in [verified, rejected]:
            store.upsert_knowledge_record(record)
            store.save_knowledge_embedding(record.id, [1.0, 0.0])

        results = search_knowledge(store, FakeEmbeddings(vectors=[[1.0, 0.0]]), "query", top_k=3)

        assert [result["record"].id for result in results] == ["knowledge-1"]
    finally:
        store.close()


def test_find_related_knowledge_returns_empty_when_pitfall_has_no_embedding(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        canonical = _canonical()
        record = _knowledge_record("knowledge-1")
        store.create_canonical(canonical)
        store.upsert_knowledge_record(record)
        store.save_knowledge_embedding(record.id, [1.0, 0.0])

        assert find_related_knowledge(store, FakeEmbeddings(), canonical.canonical_id) == []
    finally:
        store.close()


def test_find_related_knowledge_returns_empty_when_knowledge_records_are_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        canonical = _canonical()
        store.create_canonical(canonical)
        store.save_canonical_embedding(canonical.canonical_id, [1.0, 0.0])

        assert find_related_knowledge(store, FakeEmbeddings(), canonical.canonical_id) == []
    finally:
        store.close()
