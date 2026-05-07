from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from consolidate_agent.consolidation.pipeline import run_consolidation
from consolidate_agent.config import Settings
from consolidate_agent.consolidation.canonicalize import (
    CanonicalContent,
    CanonicalDraft,
    CanonicalizationDecision,
    CanonicalizationResult,
    build_canonicalization_result,
    deterministic_group,
    validate_canonicalization,
)
from consolidate_agent.consolidation.classify import (
    DeterministicRuleClassifier,
    LLMTagCoverageChecker,
    RuleClassificationResult,
    RuleTagAssignmentDraft,
    TagCoverageResult,
    validate_rule_classification,
)
from consolidate_agent.extraction.pipeline import ConsolidationGraph
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.extraction.normalize import SessionNormalizer
from consolidate_agent.consolidation.taxonomy import (
    LLMAgentTagGovernor,
    MechanismTagDraft,
    TagProposalDraft,
    TaxonomyDraftResult,
    TaxonomyGovernanceDecision,
    TaxonomyGovernanceResult,
    validate_taxonomy_draft,
)
from consolidate_agent.types import (
    KnowledgeRelation,
    ConsolidationStats,
    MechanismTag,
    MechanismTagStatus,
    PitfallCandidate,
    PitfallCategory,
    PitfallEvidence,
    PitfallRecord,
    PitfallScope,
    PipelineState,
    SourceConsolidationStatus,
    TagProposalDecision,
    normalize_text,
    utc_now,
)


def make_record(
    record_id: str,
    *,
    title: str = "Validate JSON before parsing",
    preventive_rule: str = "Validate JSON payloads against a schema before parsing model output.",
    category: PitfallCategory = PitfallCategory.EXECUTION_STRATEGY,
) -> PitfallRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return PitfallRecord(
        id=record_id,
        title=title,
        category=category,
        trigger="The model returns structured data.",
        failure_mode="Invalid structure reaches downstream code.",
        impact="The workflow fails late.",
        preventive_rule=preventive_rule,
        scope=PitfallScope.GLOBAL,
        evidence=PitfallEvidence(session_ids=["session-1"], message_refs=["m1"]),
        confidence=0.9,
        tags=[],
        created_at=now,
        updated_at=now,
    )


def seed_records(db_path: Path, records: list[PitfallRecord]) -> None:
    store = KnowledgeStore(db_path)
    try:
        for record in records:
            store.upsert_source_record(record)
    finally:
        store.close()


def active_tag(name: str = "structured_input_validation") -> MechanismTag:
    now = utc_now()
    return MechanismTag(
        tag_id=f"tag_{name}",
        name=name,
        definition="Validate structured inputs before using them downstream.",
        status=MechanismTagStatus.ACTIVE,
        positive_examples=[],
        negative_examples=[],
        created_at=now,
        updated_at=now,
    )


def canonical_draft(temporary_id: str, source_record_ids: list[str]) -> CanonicalDraft:
    return CanonicalDraft(
        temporary_id=temporary_id,
        title=f"Canonical {temporary_id}",
        category=PitfallCategory.EXECUTION_STRATEGY.value,
        summary="Stable canonical summary.",
        preventive_rule=f"Apply stable preventive rule {temporary_id}.",
        scope=PitfallScope.GLOBAL.value,
        source_record_ids=source_record_ids,
    )


def canonical_decision(source_record_id: str, relation: str, canonical_ref: str) -> CanonicalizationDecision:
    return CanonicalizationDecision(
        source_record_id=source_record_id,
        relation=relation,
        canonical_ref=canonical_ref,
        decision_reason="Test decision.",
    )


def test_taxonomy_draft_validation_rejects_duplicate_names() -> None:
    canonical = _canonical_fixture()
    draft = TaxonomyDraftResult(
        proposals=[
            TagProposalDraft(
                name="structured_input_validation",
                definition="Validate structured inputs.",
                supporting_canonical_ids=[canonical.canonical_id],
                difference_from_existing="New control.",
            ),
            TagProposalDraft(
                name="Structured Input Validation",
                definition="Validate JSON before parsing.",
                supporting_canonical_ids=[canonical.canonical_id],
                difference_from_existing="New control.",
            ),
        ]
    )

    with pytest.raises(ValueError, match="unique"):
        validate_taxonomy_draft(draft, [canonical], [])


def test_rule_classification_validation_rejects_unknown_tag() -> None:
    canonical = _canonical_fixture()
    result = RuleClassificationResult(
        assignments=[
            RuleTagAssignmentDraft(
                canonical_id=canonical.canonical_id,
                tag_name="new_unapproved_tag",
                confidence=0.8,
                assignment_reason="Looks close.",
            )
        ]
    )

    with pytest.raises(ValueError, match="unknown tag_name"):
        validate_rule_classification(result, [canonical], [active_tag()])


def test_rule_classification_validation_allows_partial_assignments() -> None:
    canonical = _canonical_fixture()
    other = _canonical_fixture_for("pitfall-2", title="Check timeout", preventive_rule="Configure timeout.")
    result = RuleClassificationResult(
        assignments=[
            RuleTagAssignmentDraft(
                canonical_id=canonical.canonical_id,
                tag_name="structured_input_validation",
                confidence=0.8,
                assignment_reason="Covered by approved tag.",
            )
        ]
    )

    validate_rule_classification(result, [canonical, other], [active_tag()])


def test_relation_values_for_canonical_uses_canonical_id(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        store.write_relation("source-1", "canonical-1", KnowledgeRelation.DISTINCT)
        store.write_relation("canonical-1", "other-canonical", KnowledgeRelation.DUPLICATE)

        assert store.relation_values_for_canonical("canonical-1") == [KnowledgeRelation.DISTINCT.value]
    finally:
        store.close()


def test_upsert_source_record_is_idempotent_for_same_record(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    record = make_record("pitfall-1")
    try:
        store.upsert_source_record(record)
        store.upsert_source_record(record)

        records = store.list_all_source_records()

        assert len(records) == 1
        assert records[0].id == "pitfall-1"
    finally:
        store.close()


def test_upsert_source_record_can_be_read_back_with_key_fields(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    record = make_record(
        "pitfall-1",
        title="Apply bounded retries",
        preventive_rule="Use bounded retries with timeout and fallback for flaky external services.",
        category=PitfallCategory.TOOLING_ENVIRONMENT,
    )
    try:
        store.upsert_source_record(record)

        [loaded] = store.list_all_source_records()

        assert loaded.title == record.title
        assert loaded.preventive_rule == record.preventive_rule
        assert loaded.category == record.category
    finally:
        store.close()


def test_upsert_link_rolls_back_when_status_update_fails(tmp_path: Path) -> None:
    class FailingConnection:
        def __init__(self, connection):
            self.connection = connection
            self.in_transaction_context = False

        def __enter__(self):
            self.in_transaction_context = True
            self.connection.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.in_transaction_context = False
            return self.connection.__exit__(exc_type, exc, traceback)

        def execute(self, sql, params=()):
            if self.in_transaction_context and "UPDATE source_pitfall_records" in sql:
                raise RuntimeError("simulated interruption")
            return self.connection.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        store.upsert_source_record(make_record("pitfall-1"))
        store.connection = FailingConnection(store.connection)

        with pytest.raises(RuntimeError, match="simulated interruption"):
            store.upsert_link("pitfall-1", "canonical-1", KnowledgeRelation.DISTINCT)

        assert store.get_link("pitfall-1") is None
        assert store.get_source_status("pitfall-1") == SourceConsolidationStatus.PENDING
    finally:
        store.close()


def test_vector_rule_classifier_uses_similarity_confidence() -> None:
    class FakeVectorStore:
        def find_best_tag(self, canonical, tags):
            return tags[0], 0.42

    canonical = _canonical_fixture()
    result = DeterministicRuleClassifier(vector_store=FakeVectorStore()).classify([canonical], [active_tag()])

    assert result.assignments[0].confidence == 0.42
    assert result.assignments[0].confidence != 0.7


def test_normalize_text_is_shared_by_dedupe_call_sites() -> None:
    from consolidate_agent.consolidation.pipeline import _canonical_id
    from consolidate_agent.extraction.merge import _dedupe_key as merge_dedupe_key
    from consolidate_agent.consolidation.canonicalize import _canonical_key

    title = "  Validate\nJSON\tBefore Parsing  "
    rule = "  Validate   JSON\nPayloads  "
    expected_key = f"{PitfallCategory.EXECUTION_STRATEGY.value}|{normalize_text(title)}|{normalize_text(rule)}"
    candidate = PitfallCandidate(
        candidate_id="candidate-1",
        session_id="session-1",
        title=title,
        category=PitfallCategory.EXECUTION_STRATEGY,
        trigger="trigger",
        failure_mode="failure",
        impact="impact",
        preventive_rule=rule,
        scope=PitfallScope.GLOBAL,
        evidence_refs=["msg_0001"],
        confidence=0.8,
    )
    messy_record = make_record("pitfall-1", title=title, preventive_rule=rule)
    equivalent_record = make_record("pitfall-2", title="validate json before parsing", preventive_rule="validate json payloads")

    assert merge_dedupe_key(candidate) == expected_key
    assert _canonical_key(PitfallCategory.EXECUTION_STRATEGY.value, title, rule) == expected_key
    assert _canonical_id(messy_record) == _canonical_id(equivalent_record)


def test_deterministic_group_groups_same_key_records() -> None:
    records = [make_record("pitfall-1"), make_record("pitfall-2")]
    groups, direct_duplicates = deterministic_group(records, [])

    assert direct_duplicates == []
    assert [[record.id for record in group] for group in groups] == [["pitfall-1", "pitfall-2"]]


def test_build_canonicalization_result_creates_complete_decisions_for_group() -> None:
    records = [make_record("pitfall-1"), make_record("pitfall-2")]
    groups, direct_duplicates = deterministic_group(records, [])
    content = CanonicalContent(
        title="Validate JSON before parsing",
        category=PitfallCategory.EXECUTION_STRATEGY.value,
        summary="Validate structured model output before downstream use.",
        preventive_rule="Validate JSON payloads against a schema before parsing model output.",
        scope=PitfallScope.GLOBAL.value,
    )
    result = build_canonicalization_result(groups, [content], direct_duplicates, [])

    assert len(result.canonicals) == 1
    assert {decision.source_record_id for decision in result.decisions} == {"pitfall-1", "pitfall-2"}
    validate_canonicalization(result, records, [])


def test_build_canonicalization_result_routes_rewritten_duplicate_to_existing() -> None:
    existing = _canonical_fixture()
    records = [
        make_record(
            "pitfall-2",
            title="Different title",
            preventive_rule="Different source wording.",
        )
    ]
    groups, direct_duplicates = deterministic_group(records, [existing])
    content = CanonicalContent(
        title=existing.title,
        category=existing.category.value,
        summary=existing.summary,
        preventive_rule=existing.preventive_rule,
        scope=existing.scope.value,
    )
    result = build_canonicalization_result(groups, [content], direct_duplicates, [existing])

    assert result.canonicals == []
    assert result.decisions[0].canonical_ref == existing.canonical_id
    assert result.decisions[0].relation == "duplicate"
    validate_canonicalization(result, records, [existing])


def test_deterministic_group_routes_exact_existing_duplicates() -> None:
    existing = _canonical_fixture()
    records = [make_record("pitfall-1")]
    groups, direct_duplicates = deterministic_group(records, [existing])

    assert groups == []
    assert direct_duplicates == [(records[0], existing.canonical_id)]


def test_build_canonicalization_result_allows_singletons() -> None:
    records = [
        make_record(f"pitfall-{index}", title=f"Rule {index}", preventive_rule=f"Prevent failure {index}.")
        for index in range(4)
    ]
    groups, direct_duplicates = deterministic_group(records, [])
    contents = [
        CanonicalContent(
            title=record.title,
            category=record.category.value,
            summary="Stable singleton summary.",
            preventive_rule=record.preventive_rule,
            scope=record.scope.value,
        )
        for record in records
    ]
    result = build_canonicalization_result(groups, contents, direct_duplicates, [])

    assert len(result.canonicals) == 4
    validate_canonicalization(result, records, [])


def test_run_consolidation_creates_tags_and_assignments_without_patterns(tmp_path: Path) -> None:
    db_path = tmp_path / "knowledge.db"
    seed_records(db_path, [make_record("pitfall-1")])

    stats = run_consolidation(db_path)

    store = KnowledgeStore(db_path)
    try:
        assert stats.canonical_created == 1
        assert stats.tags_created == 1
        assert stats.rules_classified == 1
        assert store.count_tags() == 1
        assert store.count_rule_tag_assignments() == 1
        assert not hasattr(store, "count_patterns")
    finally:
        store.close()


def test_duplicate_records_reuse_existing_canonical_and_assignment(tmp_path: Path) -> None:
    db_path = tmp_path / "knowledge.db"
    seed_records(db_path, [make_record("pitfall-1"), make_record("pitfall-2")])

    stats = run_consolidation(db_path)

    store = KnowledgeStore(db_path)
    try:
        assert stats.canonical_created == 1
        assert stats.source_records_linked == 2
        assert stats.canonical_reused == 0
        assert store.count_canonicals() == 1
        assert store.list_active_canonicals()[0].support_count == 2
        assert store.count_rule_tag_assignments() == 1
    finally:
        store.close()


def test_taxonomy_governance_merge_reuses_existing_tag(tmp_path: Path) -> None:
    class MergeGovernor:
        def govern(self, active_tags, proposals, canonicals):
            return TaxonomyGovernanceResult(
                decisions=[
                    TaxonomyGovernanceDecision(
                        proposal_name=proposals[0].name,
                        decision="merge",
                        target_tag_name=active_tags[0].name,
                        decision_reason="Covered by existing tag.",
                    )
                ]
            )

    db_path = tmp_path / "knowledge.db"
    seed_records(db_path, [make_record("pitfall-1")])
    run_consolidation(db_path)
    seed_records(db_path, [make_record(
        "pitfall-2",
        title="Apply bounded retries",
        preventive_rule="Use bounded retries with timeout and fallback for flaky external services.",
    )])
    stats = run_consolidation(db_path, taxonomy_governor=MergeGovernor())

    store = KnowledgeStore(db_path)
    try:
        assert stats.tags_created == 0
        assert store.count_tags() == 1
        assert store.count_rule_tag_assignments() == 2
        decisions = {row["decision"]: row["count"] for row in store.proposal_decision_counts()}
        assert decisions[TagProposalDecision.MERGED.value] == 1
    finally:
        store.close()


def test_governance_all_reject_keeps_rules_untagged_without_failure(tmp_path: Path) -> None:
    class RejectGovernor:
        def govern(self, active_tags, proposals, canonicals):
            return TaxonomyGovernanceResult(
                decisions=[
                    TaxonomyGovernanceDecision(
                        proposal_name=proposal.name,
                        decision="reject",
                        decision_reason="Too narrow.",
                    )
                    for proposal in proposals
                ]
            )

    db_path = tmp_path / "knowledge.db"
    seed_records(db_path, [make_record("pitfall-1")])

    stats = run_consolidation(db_path, taxonomy_governor=RejectGovernor())

    store = KnowledgeStore(db_path)
    try:
        assert stats.rule_classification_failures == 0
        assert stats.rules_classified == 0
        assert stats.rules_skipped_no_tag == 1
        assert store.count_rule_tag_assignments() == 0
        assert len(store.list_canonicals_without_tag()) == 1
    finally:
        store.close()


def test_governance_partial_reject_skips_unassigned_rules(tmp_path: Path) -> None:
    class PartialGovernor:
        def govern(self, active_tags, proposals, canonicals):
            decisions = []
            for proposal in proposals:
                if proposal.name == "structured_input_validation":
                    decisions.append(
                        TaxonomyGovernanceDecision(
                            proposal_name=proposal.name,
                            decision="accept",
                            decision_reason="Reusable control.",
                            accepted_tag=MechanismTagDraft(
                                name=proposal.name,
                                definition=proposal.definition,
                                positive_examples=proposal.positive_examples,
                                negative_examples=proposal.negative_examples,
                            ),
                        )
                    )
                else:
                    decisions.append(
                        TaxonomyGovernanceDecision(
                            proposal_name=proposal.name,
                            decision="reject",
                            decision_reason="Too narrow.",
                        )
                    )
            return TaxonomyGovernanceResult(decisions=decisions)

    class PartialClassifier:
        def classify(self, canonicals, tags):
            assignments = []
            for canonical in canonicals:
                if "JSON" in canonical.title:
                    assignments.append(
                        RuleTagAssignmentDraft(
                            canonical_id=canonical.canonical_id,
                            tag_name=tags[0].name,
                            confidence=0.9,
                            assignment_reason="Covered by accepted tag.",
                        )
                    )
            return RuleClassificationResult(assignments=assignments)

    db_path = tmp_path / "knowledge.db"
    seed_records(db_path, [
        make_record("pitfall-1"),
        make_record(
            "pitfall-2",
            title="Apply bounded retries",
            preventive_rule="Use bounded retries with timeout and fallback for flaky external services.",
        ),
    ])

    stats = run_consolidation(
        db_path,
        taxonomy_governor=PartialGovernor(),
        rule_classifier=PartialClassifier(),
    )

    store = KnowledgeStore(db_path)
    try:
        assert stats.rule_classification_failures == 0
        assert stats.rules_classified == 1
        assert stats.rules_skipped_no_tag == 1
        assert store.count_rule_tag_assignments() == 1
        assert len(store.list_canonicals_without_tag()) == 1
    finally:
        store.close()


def test_warm_start_existing_tag_coverage_skips_taxonomy_draft(tmp_path: Path) -> None:
    class AssignAllClassifier:
        def classify(self, canonicals, tags):
            return RuleClassificationResult(
                assignments=[
                    RuleTagAssignmentDraft(
                        canonical_id=canonical.canonical_id,
                        tag_name=tags[0].name,
                        confidence=0.9,
                        assignment_reason="Covered by existing tag.",
                    )
                    for canonical in canonicals
                ]
            )

    class FailingDrafter:
        called = False

        def draft(self, canonicals, active_tags):
            self.called = True
            raise AssertionError("taxonomy_draft should not be called")

    db_path = tmp_path / "knowledge.db"
    seed_records(db_path, [make_record("pitfall-1")])
    run_consolidation(db_path)
    seed_records(db_path, [make_record(
        "pitfall-2",
        title="Validate tool result JSON",
        preventive_rule="Validate structured tool result payloads before using them downstream.",
    )])
    drafter = FailingDrafter()
    stats = run_consolidation(db_path, taxonomy_drafter=drafter, rule_classifier=AssignAllClassifier())

    assert drafter.called is False
    assert stats.rules_covered_by_existing == 1


def test_warm_start_empty_uncovered_canonicals_skips_taxonomy_governance(tmp_path: Path) -> None:
    class AssignAllClassifier:
        def classify(self, canonicals, tags):
            return RuleClassificationResult(
                assignments=[
                    RuleTagAssignmentDraft(
                        canonical_id=canonical.canonical_id,
                        tag_name=tags[0].name,
                        confidence=0.9,
                        assignment_reason="Covered by existing tag.",
                    )
                    for canonical in canonicals
                ]
            )

    class FailingGovernor:
        called = False

        def govern(self, active_tags, proposals, canonicals):
            self.called = True
            raise AssertionError("taxonomy_governance should not be called")

    db_path = tmp_path / "knowledge.db"
    seed_records(db_path, [make_record("pitfall-1")])
    run_consolidation(db_path)
    seed_records(db_path, [make_record("pitfall-2", title="Validate tool result JSON")])
    governor = FailingGovernor()
    run_consolidation(db_path, taxonomy_governor=governor, rule_classifier=AssignAllClassifier())

    assert governor.called is False


def test_cold_start_runs_original_taxonomy_flow(tmp_path: Path) -> None:
    class RecordingDrafter:
        called = False

        def draft(self, canonicals, active_tags):
            self.called = True
            assert len(canonicals) == 1
            assert active_tags == []
            return TaxonomyDraftResult(
                proposals=[
                    TagProposalDraft(
                        name="structured_input_validation",
                        definition="Validate structured inputs before using them downstream.",
                        supporting_canonical_ids=[canonicals[0].canonical_id],
                        difference_from_existing="No active tag covers this mechanism.",
                    )
                ]
            )

    class RecordingGovernor:
        called = False

        def govern(self, active_tags, proposals, canonicals):
            self.called = True
            return TaxonomyGovernanceResult(
                decisions=[
                    TaxonomyGovernanceDecision(
                        proposal_name=proposals[0].name,
                        decision="accept",
                        decision_reason="Reusable mechanism.",
                        accepted_tag=MechanismTagDraft(
                            name=proposals[0].name,
                            definition=proposals[0].definition,
                            positive_examples=[],
                            negative_examples=[],
                        ),
                    )
                ]
            )

    db_path = tmp_path / "knowledge.db"
    seed_records(db_path, [make_record("pitfall-1")])

    drafter = RecordingDrafter()
    governor = RecordingGovernor()
    stats = run_consolidation(db_path, taxonomy_drafter=drafter, taxonomy_governor=governor)

    assert drafter.called is True
    assert governor.called is True
    assert stats.rules_classified == 1


def test_llm_tag_coverage_checker_rejects_unknown_canonical_id(monkeypatch) -> None:
    class FakeStructuredModel:
        def invoke(self, prompt):
            return TagCoverageResult(uncovered_ids=["unknown-canonical"])

    class FakeLLM:
        def with_structured_output(self, model):
            assert model is TagCoverageResult
            return FakeStructuredModel()

    monkeypatch.setattr("consolidate_agent.consolidation.classify._chat_model", lambda settings: FakeLLM())
    checker = LLMTagCoverageChecker(Settings(DASHSCOPE_API_KEY="dummy"))

    with pytest.raises(ValueError, match="unknown canonical ids"):
        checker.check([_canonical_fixture()], [active_tag()])


def test_llm_agent_tag_governor_decides_one_with_search_tool(monkeypatch) -> None:
    class FakeVectorStore:
        def __init__(self):
            self.queries = []

        def similarity_search_canonicals(self, query, k=5):
            self.queries.append((query, k))
            return [_canonical_fixture()]

    class FakeStructuredModel:
        def invoke(self, prompt_value):
            text = str(prompt_value)
            assert "Validate JSON before parsing" in text
            return TaxonomyGovernanceDecision(
                proposal_name="wrong_name_from_model",
                decision="accept",
                decision_reason="Evidence shows a reusable structured input validation mechanism.",
                accepted_tag=MechanismTagDraft(
                    name="structured_input_validation",
                    definition="Validate structured inputs before downstream use.",
                    positive_examples=["Validate JSON before parsing"],
                    negative_examples=[],
                ),
            )

    class FakeLLM:
        def with_structured_output(self, model, **kwargs):
            assert model is TaxonomyGovernanceDecision
            return FakeStructuredModel()

    class FakeAgent:
        def __init__(self, tools):
            self.tools = tools

        def invoke(self, payload):
            from langchain_core.messages import AIMessage
            evidence = self.tools[0].invoke({"query": "structured JSON validation"})
            assert "Validate JSON before parsing" in evidence
            return {"messages": payload["messages"] + [AIMessage(content=f"Evidence: {evidence}. Decision: accept.")]}

    def fake_create_react_agent(llm, tools, **kwargs):
        assert isinstance(llm, FakeLLM)
        return FakeAgent(tools)

    monkeypatch.setattr("consolidate_agent.consolidation.taxonomy._chat_model", lambda settings: FakeLLM())
    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", fake_create_react_agent)

    proposal = TagProposalDraft(
        name="structured_input_validation",
        definition="Validate structured inputs before downstream use.",
        supporting_canonical_ids=["canonical_1"],
        difference_from_existing="No active tag covers this mechanism.",
        positive_examples=["Validate JSON before parsing"],
    )
    vector_store = FakeVectorStore()
    governor = LLMAgentTagGovernor(Settings(DASHSCOPE_API_KEY="dummy"), vector_store)

    decision = governor._decide_one(proposal, [])

    assert decision.proposal_name == "structured_input_validation"
    assert decision.decision == "accept"
    assert decision.accepted_tag is not None
    assert vector_store.queries == [("structured JSON validation", 5)]


def test_llm_agent_tag_governor_preserves_proposal_order_when_concurrent(monkeypatch) -> None:
    class FakeGovernor:
        def _decide_one(self, proposal, active_tags):
            delays = {"proposal_0": 0.03, "proposal_1": 0.01, "proposal_2": 0.02}
            time.sleep(delays[proposal.name])
            return TaxonomyGovernanceDecision(
                proposal_name=proposal.name,
                decision="reject",
                decision_reason=f"Rejected {proposal.name}.",
            )

    proposals = [
        TagProposalDraft(
            name=f"proposal_{index}",
            definition=f"Definition {index}.",
            supporting_canonical_ids=[f"canonical_{index}"],
            difference_from_existing="Distinct test proposal.",
        )
        for index in range(3)
    ]
    governor = FakeGovernor()

    result = LLMAgentTagGovernor.govern(governor, [], proposals, [])

    assert [decision.proposal_name for decision in result.decisions] == [proposal.name for proposal in proposals]


def test_rule_classification_cannot_create_tags(tmp_path: Path) -> None:
    class UnknownTagClassifier:
        def classify(self, canonicals, tags):
            return RuleClassificationResult(
                assignments=[
                    RuleTagAssignmentDraft(
                        canonical_id=canonicals[0].canonical_id,
                        tag_name="unapproved_runtime_tag",
                        confidence=0.7,
                        assignment_reason="Invalid on purpose.",
                    )
                ]
            )

    db_path = tmp_path / "knowledge.db"
    seed_records(db_path, [make_record("pitfall-1")])

    with pytest.raises(ValueError, match="unknown tag_name"):
        run_consolidation(db_path, rule_classifier=UnknownTagClassifier())

    store = KnowledgeStore(db_path)
    try:
        assert store.count_rule_tag_assignments() == 0
        latest = store.runs.latest_consolidation_run()
        assert latest is not None
        assert latest["status"] == "failed"
        stats = json.loads(str(latest["stats_json"]))
        assert stats["rule_classification_failures"] == 1
        assert store.runs.count_consolidation_failures(str(latest["run_id"])) == 1
    finally:
        store.close()


def test_consolidation_run_store_start_finish_round_trip(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        stats = ConsolidationStats(run_id="run_test", source_records_seen=1)
        store.runs.start_consolidation_run("run_test", tmp_path / "input.json", "model-a", stats)
        store.runs.finish_consolidation_run("run_test", "completed", stats)

        run = store.runs.latest_consolidation_run()
        assert run is not None
        assert run["run_id"] == "run_test"
        assert run["status"] == "completed"
        assert run["model"] == "model-a"
    finally:
        store.close()


def test_agent_invocation_report_helpers_scope_to_run_id(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        store.runs.record_agent_invocation(
            run_id="run_a",
            stage="taxonomy_draft",
            status="success",
            input_payload={"x": 1},
            output_payload={"y": 2},
            latency_ms=5,
        )
        store.runs.record_agent_invocation(
            run_id="run_b",
            stage="rule_classification",
            status="failure",
            input_payload={"x": 1},
            validation_error="bad",
            latency_ms=7,
        )

        assert store.runs.count_agent_invocations("run_a") == 1
        assert store.runs.count_agent_failures("run_a") == 0
        assert store.runs.count_agent_failures("run_b") == 1
        assert store.runs.agent_invocation_summary("run_b")[0]["validation_error_count"] == 1
    finally:
        store.close()


def _canonical_fixture():
    from consolidate_agent.consolidation.pipeline import _canonical_from_record

    return _canonical_from_record(make_record("pitfall-1"))


def _canonical_fixture_for(record_id: str, *, title: str, preventive_rule: str):
    from consolidate_agent.consolidation.pipeline import _canonical_from_record

    return _canonical_from_record(make_record(record_id, title=title, preventive_rule=preventive_rule))


def test_session_normalizer_skips_dict_source_subagent_session(tmp_path: Path) -> None:
    session_path = tmp_path / "subagent-session.jsonl"
    session_path.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "session-subagent",
                    "source": {"subagent": {"depth": 1}},
                    "cwd": str(tmp_path),
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert SessionNormalizer().normalize_file(session_path) is None


def test_normalize_sessions_counts_skipped_normalized_none(monkeypatch, tmp_path: Path) -> None:
    skipped_session_path = tmp_path / "skipped.jsonl"
    skipped_session_path.write_text("", encoding="utf-8")
    state = PipelineState(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        cursor_path=str(tmp_path / "cursor.json"),
        processed_index_path=str(tmp_path / "processed.json"),
        sessions=[str(skipped_session_path)],
    )

    monkeypatch.setattr(SessionNormalizer, "normalize_file", lambda self, path: None)

    result = ConsolidationGraph(Settings(DASHSCOPE_API_KEY="dummy")).normalize_sessions(state)

    assert result.transcripts == []
    assert result.stats.skipped_sessions == 1
    assert str(skipped_session_path) not in result.transcript_paths.values()


def test_knowledge_store_saves_canonical_embeddings_concurrently(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        canonicals = [
            _canonical_fixture_for(
                f"pitfall-{index}",
                title=f"Concurrent rule {index}",
                preventive_rule=f"Prevent concurrent write issue {index}.",
            )
            for index in range(16)
        ]
        for canonical in canonicals:
            store.create_canonical(canonical)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    store.save_canonical_embedding,
                    canonical.canonical_id,
                    [float(index), float(index + 1)],
                )
                for index, canonical in enumerate(canonicals)
            ]

            for future in futures:
                future.result()
    finally:
        store.close()


def test_llm_agent_tag_governor_omits_empty_agent_messages_from_structured_call(monkeypatch) -> None:
    class FakeVectorStore:
        def similarity_search_canonicals(self, query, k=5):
            return []

    class FakeStructuredModel:
        def invoke(self, prompt_value):
            messages = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else prompt_value
            assert all(str(message.content).strip() for message in messages)
            assert "Useful search conclusion." in str(prompt_value)
            return TaxonomyGovernanceDecision(
                proposal_name="structured_input_validation",
                decision="reject",
                decision_reason="The evidence is too narrow.",
            )

    class FakeLLM:
        def with_structured_output(self, model, **kwargs):
            assert model is TaxonomyGovernanceDecision
            return FakeStructuredModel()

    class FakeAgent:
        def invoke(self, payload):
            from langchain_core.messages import AIMessage

            return {
                "messages": payload["messages"]
                + [
                    AIMessage(content=""),
                    AIMessage(content="Useful search conclusion."),
                ]
            }

    def fake_create_react_agent(llm, tools, **kwargs):
        assert isinstance(llm, FakeLLM)
        return FakeAgent()

    monkeypatch.setattr("consolidate_agent.consolidation.taxonomy._chat_model", lambda settings: FakeLLM())
    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", fake_create_react_agent)

    proposal = TagProposalDraft(
        name="structured_input_validation",
        definition="Validate structured inputs before downstream use.",
        supporting_canonical_ids=["canonical_1"],
        difference_from_existing="No active tag covers this mechanism.",
    )
    governor = LLMAgentTagGovernor(Settings(DASHSCOPE_API_KEY="dummy"), FakeVectorStore())

    decision = governor._decide_one(proposal, [])

    assert decision == TaxonomyGovernanceDecision(
        proposal_name="structured_input_validation",
        decision="reject",
        decision_reason="The evidence is too narrow.",
    )


def test_llm_agent_tag_governor_soft_rejects_after_empty_structured_retries(monkeypatch) -> None:
    class FakeVectorStore:
        def similarity_search_canonicals(self, query, k=5):
            return []

    class FakeStructuredModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            from langchain_core.messages import AIMessage

            self.calls += 1
            return {
                "raw": AIMessage(content="", additional_kwargs={"tool_calls": []}),
                "parsed": None,
                "parsing_error": None,
            }

    structured_model = FakeStructuredModel()

    class FakeLLM:
        def with_structured_output(self, model, **kwargs):
            assert model is TaxonomyGovernanceDecision
            return structured_model

    class FakeAgent:
        def invoke(self, payload):
            from langchain_core.messages import AIMessage

            return {"messages": payload["messages"] + [AIMessage(content="Evidence is too narrow.")]}

    monkeypatch.setattr("consolidate_agent.consolidation.taxonomy._chat_model", lambda settings: FakeLLM())
    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", lambda llm, tools, **kwargs: FakeAgent())

    proposal = TagProposalDraft(
        name="non_idempotent_execution_path",
        definition="Execution logic that produces different outcomes on retry.",
        supporting_canonical_ids=["canonical_1"],
        difference_from_existing="No active tag covers this mechanism.",
    )
    governor = LLMAgentTagGovernor(Settings(DASHSCOPE_API_KEY="dummy"), FakeVectorStore())

    decision = governor._decide_one(proposal, [])

    assert structured_model.calls == 3
    assert decision == TaxonomyGovernanceDecision(
        proposal_name="non_idempotent_execution_path",
        decision="reject",
        decision_reason="Structured governance decision failed after 3 attempts; proposal rejected to keep consolidation running.",
    )
