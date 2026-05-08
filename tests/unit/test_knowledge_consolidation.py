from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from consolidate_agent.consolidation.taxonomy import normalize_tag_name
from consolidate_agent.knowledge.consolidation import (
    KnowledgeConsolidationService,
    KnowledgeTag,
    KnowledgeTagAssignmentDraft,
    KnowledgeTagProposalDraft,
    KnowledgeTaxonomyDraftResult,
    KnowledgeTaxonomyGovernanceDecision,
    KnowledgeTaxonomyGovernanceResult,
)
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.types import KnowledgeRecord, KnowledgeScope


def _record(record_id: str, title: str, insight: str | None = None) -> KnowledgeRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return KnowledgeRecord(
        id=record_id,
        session_id="session-1",
        title=title,
        insight=insight or f"{title} insight",
        applicability=f"{title} applicability",
        scope=KnowledgeScope.GLOBAL,
        evidence_turns=[1],
        evidence_count=1,
        processed_chars=100,
        created_at=now,
        updated_at=now,
    )


def _tag(name: str, definition: str | None = None) -> KnowledgeTag:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    normalized = normalize_tag_name(name)
    return KnowledgeTag(
        tag_id=f"tag-{normalized}",
        name=normalized,
        definition=definition or f"{normalized} knowledge.",
        created_at=now,
        updated_at=now,
    )


class FakeVectorStore:
    def __init__(self, records: list[KnowledgeRecord]):
        self.records = records
        self.queries: list[tuple[str, int]] = []

    def similarity_search_knowledge_records(self, query: str, k: int = 5) -> list[KnowledgeRecord]:
        self.queries.append((query, k))
        query_text = query.lower()
        matches = [record for record in self.records if any(token in query_text for token in record.title.lower().split())]
        return (matches or self.records)[:k]


class FakeDrafter:
    def __init__(self, *results: KnowledgeTaxonomyDraftResult):
        self.results = list(results)
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()

    def draft_batch(self, records: list[KnowledgeRecord], active_tags: list[KnowledgeTag]) -> KnowledgeTaxonomyDraftResult:
        del active_tags
        with self._lock:
            self.calls.append([record.id for record in records])
            if self.results:
                return self.results.pop(0)
        return KnowledgeTaxonomyDraftResult(proposals=[])


class FakeGovernor:
    def __init__(self, decisions: list[KnowledgeTaxonomyGovernanceDecision]):
        self.decisions = decisions
        self.calls: list[list[str]] = []

    def govern(
        self,
        active_tags: list[KnowledgeTag],
        proposals: list[KnowledgeTagProposalDraft],
    ) -> KnowledgeTaxonomyGovernanceResult:
        del active_tags
        self.calls.append([proposal.name for proposal in proposals])
        by_name = {normalize_tag_name(decision.proposal_name): decision for decision in self.decisions}
        return KnowledgeTaxonomyGovernanceResult(
            decisions=[by_name[normalize_tag_name(proposal.name)] for proposal in proposals]
        )


class FakeAssigner:
    def __init__(self, assignments_by_call: list[dict[str, list[str]]]):
        self.assignments_by_call = assignments_by_call
        self.calls = 0
        self._lock = threading.Lock()

    def assign_one(self, record: KnowledgeRecord, active_tags: list[KnowledgeTag]) -> KnowledgeTagAssignmentDraft | None:
        del active_tags
        with self._lock:
            call_index = min(self.calls, len(self.assignments_by_call) - 1)
            self.calls += 1
        tags = self.assignments_by_call[call_index].get(record.id)
        if tags is None:
            return None
        return KnowledgeTagAssignmentDraft(
            record_id=record.id,
            tag_names=tags,
            reasoning=f"{record.id} primarily matches {tags[0]}.",
        )


class CoverageAssigner:
    def assign_one(self, record: KnowledgeRecord, active_tags: list[KnowledgeTag]) -> KnowledgeTagAssignmentDraft | None:
        active_names = {tag.name for tag in active_tags}
        if record.id == "knowledge-1" and "pytest" in active_names:
            tags = ["pytest"]
        elif record.id == "knowledge-2" and "mysql" in active_names:
            tags = ["mysql"]
        else:
            return None
        return KnowledgeTagAssignmentDraft(
            record_id=record.id,
            tag_names=tags,
            reasoning=f"{record.id} primarily matches {tags[0]}.",
        )


def _store_with_records(tmp_path: Path, records: list[KnowledgeRecord]) -> KnowledgeStore:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    for record in records:
        store.upsert_knowledge_record(record)
    return store


def test_cold_start_draft_governance_accept_assignment_full_pipeline(tmp_path: Path) -> None:
    records = [_record("knowledge-1", "pytest fixtures"), _record("knowledge-2", "pytest assertions")]
    store = _store_with_records(tmp_path, records)
    try:
        proposal = KnowledgeTagProposalDraft(
            name="pytest",
            definition="Knowledge about pytest tests and fixtures.",
            supporting_record_ids=["knowledge-1"],
        )
        service = KnowledgeConsolidationService(
            store=store,
            vector_store=FakeVectorStore(records),
            taxonomy_drafter=FakeDrafter(KnowledgeTaxonomyDraftResult(proposals=[proposal])),
            taxonomy_governor=FakeGovernor([
                KnowledgeTaxonomyGovernanceDecision(
                    proposal_name="pytest",
                    decision="accept",
                    decision_reason="Shared pytest topic.",
                    accepted_tag=proposal,
                )
            ]),
            tag_assigner=FakeAssigner([{"knowledge-1": ["pytest"], "knowledge-2": ["pytest"]}]),
        )

        result = service.run(records)

        assert [tag.name for tag in result["active_tags"]] == ["pytest"]
        assert store.list_knowledge_tag_assignments("knowledge-1") == [("pytest", "knowledge-1 primarily matches pytest.")]
        assert store.list_knowledge_tag_assignments("knowledge-2") == [("pytest", "knowledge-2 primarily matches pytest.")]
    finally:
        store.close()


def test_governance_merge_keeps_existing_active_tag(tmp_path: Path) -> None:
    records = [_record("knowledge-1", "langgraph state")]
    store = _store_with_records(tmp_path, records)
    try:
        existing = _tag("langgraph")
        store.upsert_knowledge_tag(existing)
        proposal = KnowledgeTagProposalDraft(name="langgraph-routing", definition="Routing in LangGraph.", supporting_record_ids=["knowledge-1"])
        service = KnowledgeConsolidationService(
            store=store,
            vector_store=FakeVectorStore(records),
            taxonomy_drafter=FakeDrafter(),
            taxonomy_governor=FakeGovernor([
                KnowledgeTaxonomyGovernanceDecision(
                    proposal_name="langgraph-routing",
                    decision="merge",
                    target_tag_name="langgraph",
                    decision_reason="Covered by langgraph.",
                )
            ]),
            tag_assigner=FakeAssigner([{"knowledge-1": ["langgraph"]}]),
        )

        state = service._taxonomy_governance({"records": records, "active_tags": [existing], "batch_proposals": [proposal]})

        assert state["active_tags"] == [existing]
        assert store.list_active_knowledge_tags() == [existing]
    finally:
        store.close()


def test_governance_reject_does_not_create_tag(tmp_path: Path) -> None:
    records = [_record("knowledge-1", "abstract consistency")]
    store = _store_with_records(tmp_path, records)
    try:
        proposal = KnowledgeTagProposalDraft(name="abstraction", definition="Too abstract.", supporting_record_ids=["knowledge-1"])
        service = KnowledgeConsolidationService(
            store=store,
            vector_store=FakeVectorStore(records),
            taxonomy_drafter=FakeDrafter(),
            taxonomy_governor=FakeGovernor([
                KnowledgeTaxonomyGovernanceDecision(
                    proposal_name="abstraction",
                    decision="reject",
                    decision_reason="Too abstract.",
                )
            ]),
            tag_assigner=FakeAssigner([{}]),
        )

        state = service._taxonomy_governance({"records": records, "active_tags": [], "batch_proposals": [proposal]})

        assert state["active_tags"] == []
        assert store.list_active_knowledge_tags() == []
    finally:
        store.close()


def test_warm_start_skips_draft_and_assigns_existing_tags(tmp_path: Path) -> None:
    records = [_record("knowledge-1", "mysql timeout")]
    store = _store_with_records(tmp_path, records)
    try:
        store.upsert_knowledge_tag(_tag("mysql"))
        drafter = FakeDrafter(KnowledgeTaxonomyDraftResult(proposals=[]))
        service = KnowledgeConsolidationService(
            store=store,
            vector_store=FakeVectorStore(records),
            taxonomy_drafter=drafter,
            taxonomy_governor=FakeGovernor([]),
            tag_assigner=FakeAssigner([{"knowledge-1": ["mysql"]}]),
        )

        service.run(records)

        assert drafter.calls == []
        assert store.list_knowledge_tag_assignments("knowledge-1") == [("mysql", "knowledge-1 primarily matches mysql.")]
    finally:
        store.close()


def test_warm_coverage_check_triggers_supplemental_draft_for_uncovered_records(tmp_path: Path) -> None:
    records = [_record("knowledge-1", "pytest fixtures"), _record("knowledge-2", "mysql connections")]
    store = _store_with_records(tmp_path, records)
    try:
        store.upsert_knowledge_tag(_tag("pytest"))
        mysql_proposal = KnowledgeTagProposalDraft(
            name="mysql",
            definition="Knowledge about MySQL behavior and connection handling.",
            supporting_record_ids=["knowledge-2"],
        )
        service = KnowledgeConsolidationService(
            store=store,
            vector_store=FakeVectorStore(records),
            taxonomy_drafter=FakeDrafter(KnowledgeTaxonomyDraftResult(proposals=[mysql_proposal])),
            taxonomy_governor=FakeGovernor([
                KnowledgeTaxonomyGovernanceDecision(
                    proposal_name="mysql",
                    decision="accept",
                    decision_reason="Uncovered MySQL topic.",
                    accepted_tag=mysql_proposal,
                )
            ]),
            tag_assigner=CoverageAssigner(),
        )

        result = service.run(records)

        assert [record.id for record in result["uncovered_records"]] == ["knowledge-2"]
        assert {tag.name for tag in store.list_active_knowledge_tags()} == {"pytest", "mysql"}
        assert store.list_knowledge_tag_assignments("knowledge-2") == [("mysql", "knowledge-2 primarily matches mysql.")]
    finally:
        store.close()


def test_no_tag_marked_when_coverage_retries_exhausted(tmp_path: Path) -> None:
    """Records that cannot be assigned after MAX_COVERAGE_RETRIES are marked no_tag."""
    from consolidate_agent.knowledge.consolidation import MAX_COVERAGE_RETRIES
    records = [_record("knowledge-1", "pytest fixtures"), _record("knowledge-2", "unclassifiable edge case")]
    store = _store_with_records(tmp_path, records)
    try:
        proposal = KnowledgeTagProposalDraft(
            name="pytest",
            definition="pytest knowledge.",
            supporting_record_ids=["knowledge-1"],
        )
        # assigner always returns None for knowledge-2, triggering coverage retries
        assigner = FakeAssigner([
            {"knowledge-1": ["pytest"]},  # first assignment pass
            {},                            # supplemental passes all fail for knowledge-2
            {},
        ])
        service = KnowledgeConsolidationService(
            store=store,
            vector_store=FakeVectorStore(records),
            taxonomy_drafter=FakeDrafter(KnowledgeTaxonomyDraftResult(proposals=[proposal])),
            taxonomy_governor=FakeGovernor([
                KnowledgeTaxonomyGovernanceDecision(
                    proposal_name="pytest",
                    decision="accept",
                    decision_reason="Valid tag.",
                    accepted_tag=proposal,
                )
            ]),
            tag_assigner=assigner,
        )

        service.run(records)

        assert store.list_knowledge_tag_assignments("knowledge-1") != []
        assert store.list_knowledge_tag_assignments("knowledge-2") == []
        no_tag = store.list_no_tag_knowledge_records()
        assert any(r.id == "knowledge-2" for r in no_tag)
        assigned_status = store.connection.execute(
            "SELECT consolidation_status FROM source_knowledge_records WHERE record_id = ?",
            ("knowledge-1",)
        ).fetchone()["consolidation_status"]
        assert assigned_status == "assigned"
        no_tag_status = store.connection.execute(
            "SELECT consolidation_status FROM source_knowledge_records WHERE record_id = ?",
            ("knowledge-2",)
        ).fetchone()["consolidation_status"]
        assert no_tag_status == "no_tag"
    finally:
        store.close()
