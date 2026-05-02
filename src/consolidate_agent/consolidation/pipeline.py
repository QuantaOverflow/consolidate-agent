from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from consolidate_agent.config import Settings
from consolidate_agent.consolidation.canonicalize import (
    DeterministicCanonicalizer,
    LLMCanonicalizer,
    canonical_from_draft,
    validate_canonicalization,
)
from consolidate_agent.consolidation.classify import (
    DeterministicRuleClassifier,
    DeterministicTagCoverageChecker,
    LLMRuleClassifier,
    LLMTagCoverageChecker,
    RuleClassificationResult,
    RuleTagAssignmentDraft,
    TagCoverageResult,
    assignment_from_draft,
    validate_rule_classification,
    validate_tag_coverage,
)
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.consolidation.taxonomy import (
    DeterministicTaxonomyDrafter,
    DeterministicTaxonomyGovernor,
    LLMAgentTagGovernor,
    LLMTaxonomyDrafter,
    TaxonomyDraftResult,
    TaxonomyGovernanceResult,
    normalize_tag_name,
    proposal_from_draft,
    tag_from_draft,
    validate_taxonomy_draft,
    validate_taxonomy_governance,
)
from consolidate_agent.types import (
    CanonicalKnowledge,
    CanonicalKnowledgeStatus,
    ConsolidationStats,
    KnowledgeRelation,
    MechanismTag,
    PitfallRecord,
    TagProposalDecision,
    normalize_text,
    utc_now,
)
from consolidate_agent.knowledge.vector import KnowledgeVectorStore, create_dashscope_embeddings

CANONICALIZATION_BATCH_SIZE = 500


class ConsolidationGraphState(TypedDict, total=False):
    records: list[PitfallRecord]
    stats: ConsolidationStats
    canonicals: list[CanonicalKnowledge]
    active_tags: list[MechanismTag]
    is_warm_start: bool
    draft: TaxonomyDraftResult
    governance: TaxonomyGovernanceResult
    governed_tags: list[MechanismTag]
    classification: RuleClassificationResult
    governance_assignments: list[RuleTagAssignmentDraft]
    coverage: TagCoverageResult
    uncovered_canonicals: list[CanonicalKnowledge]


class ConsolidationService:
    def __init__(
        self,
        store: KnowledgeStore,
        canonicalizer: DeterministicCanonicalizer | LLMCanonicalizer | None = None,
        taxonomy_drafter: DeterministicTaxonomyDrafter | LLMTaxonomyDrafter | None = None,
        tag_coverage_checker: DeterministicTagCoverageChecker | LLMTagCoverageChecker | None = None,
        taxonomy_governor: DeterministicTaxonomyGovernor | LLMAgentTagGovernor | None = None,
        rule_classifier: DeterministicRuleClassifier | LLMRuleClassifier | None = None,
        vector_store: KnowledgeVectorStore | None = None,
        run_id: str | None = None,
    ):
        self.store = store
        self.run_id = run_id
        self.last_stats: ConsolidationStats | None = None
        self.canonicalizer = canonicalizer or DeterministicCanonicalizer()
        self.taxonomy_drafter = taxonomy_drafter or DeterministicTaxonomyDrafter()
        self.tag_coverage_checker = tag_coverage_checker or DeterministicTagCoverageChecker()
        self.taxonomy_governor = taxonomy_governor or DeterministicTaxonomyGovernor()
        self.vector_store = vector_store
        self.rule_classifier = rule_classifier or DeterministicRuleClassifier(vector_store=vector_store)

    def run(self, records: list[PitfallRecord]) -> ConsolidationStats:
        graph = self._build_graph()
        stats = ConsolidationStats(run_id=self.run_id, source_records_seen=len(records))
        self.last_stats = stats
        initial: ConsolidationGraphState = {
            "records": records,
            "stats": stats,
        }
        result = graph.invoke(initial)
        self.last_stats = result["stats"]
        return result["stats"]

    def _build_graph(self):
        graph = StateGraph(ConsolidationGraphState)
        graph.add_node("canonicalize_sources", self._canonicalize_sources)
        graph.add_node("taxonomy_draft", self._taxonomy_draft)
        graph.add_node("taxonomy_governance", self._taxonomy_governance)
        graph.add_node("rule_classification", self._rule_classification)
        graph.add_node("tag_coverage_check", self._tag_coverage_check)
        graph.add_node("rule_classification_uncovered", self._rule_classification_uncovered)
        graph.add_node("persist_assignments", self._persist_assignments)
        graph.add_edge(START, "canonicalize_sources")
        graph.add_conditional_edges(
            "canonicalize_sources",
            self._route_after_canonicalization,
            {
                "cold_start": "taxonomy_draft",
                "warm_start": "rule_classification",
            },
        )
        graph.add_edge("taxonomy_draft", "taxonomy_governance")
        graph.add_conditional_edges(
            "taxonomy_governance",
            self._route_after_governance,
            {
                "cold_start": "rule_classification",
                "warm_start": "rule_classification_uncovered",
            },
        )
        graph.add_conditional_edges(
            "rule_classification",
            self._route_after_rule_classification,
            {
                "cold_start": "persist_assignments",
                "warm_start": "tag_coverage_check",
            },
        )
        graph.add_edge("tag_coverage_check", "taxonomy_draft")
        graph.add_edge("rule_classification_uncovered", "persist_assignments")
        graph.add_edge("persist_assignments", END)
        return graph.compile()

    def _route_after_canonicalization(self, state: ConsolidationGraphState) -> str:
        return "warm_start" if state["is_warm_start"] else "cold_start"

    def _route_after_governance(self, state: ConsolidationGraphState) -> str:
        return "warm_start" if state["is_warm_start"] else "cold_start"

    def _route_after_rule_classification(self, state: ConsolidationGraphState) -> str:
        return "warm_start" if state["is_warm_start"] else "cold_start"

    def _canonicalize_sources(self, state: ConsolidationGraphState) -> ConsolidationGraphState:
        records = state["records"]
        stats = state["stats"]
        _progress(f"canonicalization start source_records={len(records)} run_id={self.run_id}")
        for record in records:
            self.store.upsert_source_record(record)

        pending_records = self.store.list_pending_records()
        if pending_records:
            batches = [
                pending_records[index : index + CANONICALIZATION_BATCH_SIZE]
                for index in range(0, len(pending_records), CANONICALIZATION_BATCH_SIZE)
            ]
            for batch_index, batch_records in enumerate(batches, start=1):
                existing_canonicals = self.store.list_active_canonicals()
                _progress(
                    f"canonicalization batch={batch_index}/{len(batches)} "
                    f"source_records={len(batch_records)} existing_canonicals={len(existing_canonicals)}"
                )
                try:
                    result = self.canonicalizer.canonicalize(batch_records, existing_canonicals)
                    validate_canonicalization(result, batch_records, existing_canonicals)
                    draft_by_ref = {draft.temporary_id: canonical_from_draft(draft) for draft in result.canonicals}
                    existing_ids = {canonical.canonical_id for canonical in existing_canonicals}
                    for canonical in draft_by_ref.values():
                        self.store.create_canonical(canonical)
                        stats.canonical_created += 1
                    for decision in result.decisions:
                        canonical_id = decision.canonical_ref
                        if canonical_id not in existing_ids:
                            canonical_id = draft_by_ref[decision.canonical_ref].canonical_id
                        relation = KnowledgeRelation(decision.relation)
                        self.store.write_relation(decision.source_record_id, canonical_id, relation)
                        self.store.mark_classified(decision.source_record_id)
                        if self.store.upsert_link(decision.source_record_id, canonical_id, relation):
                            stats.source_records_linked += 1
                            if decision.canonical_ref in existing_ids:
                                self.store.increment_canonical_support(canonical_id)
                                stats.canonical_reused += 1
                except Exception as exc:
                    stats.classification_failures += len(batch_records)
                    self.store.runs.record_consolidation_failure(
                        "canonicalization",
                        _concise_error(exc),
                        payload={"batch_index": batch_index, "batch_size": len(batch_records)},
                        run_id=self.run_id,
                    )
                    for record in batch_records:
                        self.store.mark_failed(record.id, _concise_error(exc))
                    raise

        canonicals = self.store.list_canonicals_without_tag()
        _progress(
            f"canonicalization done linked={stats.source_records_linked} created={stats.canonical_created} "
            f"reused={stats.canonical_reused} failures={stats.classification_failures} untagged={len(canonicals)}"
        )
        active_tags = self.store.list_active_tags()
        return {**state, "stats": stats, "canonicals": canonicals, "active_tags": active_tags, "is_warm_start": bool(active_tags)}

    def _taxonomy_draft(self, state: ConsolidationGraphState) -> ConsolidationGraphState:
        canonicals = state["uncovered_canonicals"] if "uncovered_canonicals" in state else state["canonicals"]
        active_tags = state["active_tags"]
        if not canonicals:
            return {**state, "draft": TaxonomyDraftResult(proposals=[])}
        _progress(f"taxonomy_draft start unclassified={len(canonicals)} active_tags={len(active_tags)}")
        draft = self.taxonomy_drafter.draft(canonicals, active_tags)
        validate_taxonomy_draft(draft, canonicals, active_tags)
        _progress(f"taxonomy_draft done proposals={len(draft.proposals)}")
        return {**state, "draft": draft}

    def _taxonomy_governance(self, state: ConsolidationGraphState) -> ConsolidationGraphState:
        draft = state["draft"]
        active_tags = state["active_tags"]
        canonicals = state["uncovered_canonicals"] if "uncovered_canonicals" in state else state["canonicals"]
        stats = state["stats"]
        if not draft.proposals:
            return {**state, "governance": TaxonomyGovernanceResult(decisions=[]), "governed_tags": active_tags}

        for proposal in draft.proposals:
            if self.store.upsert_tag_proposal(proposal_from_draft(proposal)):
                stats.tag_proposals_created += 1

        _progress(f"taxonomy_governance start proposals={len(draft.proposals)} active_tags={len(active_tags)}")
        try:
            governance = self.taxonomy_governor.govern(active_tags, draft.proposals, canonicals)
            validate_taxonomy_governance(governance, draft.proposals, active_tags)
        except Exception as exc:
            stats.taxonomy_governance_failures += 1
            self.store.runs.record_consolidation_failure(
                "taxonomy_governance",
                _concise_error(exc),
                payload={"proposal_count": len(draft.proposals), "active_tag_count": len(active_tags)},
                run_id=self.run_id,
            )
            raise

        tag_by_name = {tag.name: tag for tag in active_tags}
        governed_by_proposal: dict[str, MechanismTag] = {}
        proposal_by_name = {normalize_tag_name(proposal.name): proposal for proposal in draft.proposals}

        for decision in governance.decisions:
            if decision.decision == "accept" and decision.accepted_tag is not None:
                proposal = proposal_by_name[normalize_tag_name(decision.proposal_name)]
                tag = tag_from_draft(decision.accepted_tag)
                if self.store.upsert_mechanism_tag(tag):
                    stats.tags_created += 1
                tag_by_name[tag.name] = tag
                governed_by_proposal[normalize_tag_name(proposal.name)] = tag
                self.store.upsert_tag_proposal(
                    proposal_from_draft(proposal, decision=TagProposalDecision.ACCEPTED, decision_reason=decision.decision_reason)
                )

        for decision in governance.decisions:
            proposal = proposal_by_name[normalize_tag_name(decision.proposal_name)]
            if decision.decision == "merge" and decision.target_tag_name:
                target = tag_by_name[normalize_tag_name(decision.target_tag_name)]
                governed_by_proposal[normalize_tag_name(proposal.name)] = target
                self.store.upsert_tag_proposal(
                    proposal_from_draft(
                        proposal,
                        decision=TagProposalDecision.MERGED,
                        target_tag_id=target.tag_id,
                        decision_reason=decision.decision_reason,
                    )
                )
                continue
            if decision.decision == "reject":
                self.store.upsert_tag_proposal(
                    proposal_from_draft(proposal, decision=TagProposalDecision.REJECTED, decision_reason=decision.decision_reason)
                )

        decision_counts: dict[str, int] = {}
        for decision in governance.decisions:
            decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1
        governed_tags = list(tag_by_name.values())
        if self.vector_store is not None:
            _progress(f"embedding precompute start canonicals={len(canonicals)} tags={len(governed_tags)}")
            self.vector_store.embed_canonicals(canonicals)
            self.vector_store.embed_tags(governed_tags)
            _progress("embedding precompute done")
        governance_assignments: list[RuleTagAssignmentDraft] = []
        for proposal_norm, tag in governed_by_proposal.items():
            proposal = proposal_by_name[proposal_norm]
            for canonical_id in proposal.supporting_canonical_ids:
                governance_assignments.append(RuleTagAssignmentDraft(
                    canonical_id=canonical_id,
                    tag_name=tag.name,
                    confidence=0.85,
                    assignment_reason=f"Assigned by governance decision for proposal '{proposal.name}'.",
                ))

        _progress(f"taxonomy_governance done decisions={decision_counts} governance_assignments={len(governance_assignments)}")
        return {**state, "stats": stats, "governance": governance, "governed_tags": governed_tags, "governance_assignments": governance_assignments}

    def _rule_classification(self, state: ConsolidationGraphState) -> ConsolidationGraphState:
        canonicals = state["canonicals"]
        governed_tags = state.get("governed_tags", state["active_tags"])
        stats = state["stats"]
        if not canonicals:
            return {**state, "classification": RuleClassificationResult(assignments=[])}
        _progress(f"rule_classification start rules={len(canonicals)} tags={len(governed_tags)}")
        try:
            classification = self.rule_classifier.classify(canonicals, governed_tags)
            validate_rule_classification(classification, canonicals, governed_tags)
        except Exception as exc:
            stats.rule_classification_failures += 1
            self.store.runs.record_consolidation_failure(
                "rule_classification",
                _concise_error(exc),
                payload={"canonical_count": len(canonicals), "tag_count": len(governed_tags)},
                run_id=self.run_id,
            )
            raise
        best_by_canonical = {a.canonical_id: a for a in state.get("governance_assignments", [])}
        for a in classification.assignments:
            best_by_canonical[a.canonical_id] = a
        classification = RuleClassificationResult(assignments=list(best_by_canonical.values()))
        _progress(f"rule_classification done assignments={len(classification.assignments)}")
        return {**state, "stats": stats, "classification": classification, "governed_tags": governed_tags}

    def _tag_coverage_check(self, state: ConsolidationGraphState) -> ConsolidationGraphState:
        canonicals = state["canonicals"]
        active_tags = state["active_tags"]
        assigned_ids = {assignment.canonical_id for assignment in state["classification"].assignments}
        omitted = [canonical for canonical in canonicals if canonical.canonical_id not in assigned_ids]
        if not omitted:
            coverage = TagCoverageResult(uncovered_ids=[])
            return {**state, "coverage": coverage, "uncovered_canonicals": []}

        _progress(f"tag_coverage_check start omitted={len(omitted)} active_tags={len(active_tags)}")
        coverage = self.tag_coverage_checker.check(omitted, active_tags)
        validate_tag_coverage(coverage, omitted)
        uncovered_ids = set(coverage.uncovered_ids)
        uncovered_canonicals = [canonical for canonical in omitted if canonical.canonical_id in uncovered_ids]
        _progress(f"tag_coverage_check done uncovered={len(uncovered_canonicals)}")
        return {**state, "coverage": coverage, "uncovered_canonicals": uncovered_canonicals}

    def _rule_classification_uncovered(self, state: ConsolidationGraphState) -> ConsolidationGraphState:
        governance_assigned_ids = {a.canonical_id for a in state.get("governance_assignments", [])}
        uncovered = [c for c in state["uncovered_canonicals"] if c.canonical_id not in governance_assigned_ids]
        governed_tags = state["governed_tags"]
        stats = state["stats"]

        uncovered_classification = RuleClassificationResult(assignments=[])
        if uncovered:
            _progress(f"rule_classification_uncovered start rules={len(uncovered)} tags={len(governed_tags)}")
            try:
                uncovered_classification = self.rule_classifier.classify(uncovered, governed_tags)
                validate_rule_classification(uncovered_classification, uncovered, governed_tags)
            except Exception as exc:
                stats.rule_classification_failures += 1
                self.store.runs.record_consolidation_failure(
                    "rule_classification_uncovered",
                    _concise_error(exc),
                    payload={"canonical_count": len(uncovered), "tag_count": len(governed_tags)},
                    run_id=self.run_id,
                )
                raise

        best_by_canonical = {a.canonical_id: a for a in state["classification"].assignments}
        for a in state.get("governance_assignments", []):
            best_by_canonical[a.canonical_id] = a
        for a in uncovered_classification.assignments:
            best_by_canonical[a.canonical_id] = a
        classification = RuleClassificationResult(assignments=list(best_by_canonical.values()))
        validate_rule_classification(classification, state["canonicals"], governed_tags)
        _progress(f"rule_classification_uncovered done assignments={len(classification.assignments)}")
        return {**state, "stats": stats, "classification": classification}

    def _persist_assignments(self, state: ConsolidationGraphState) -> ConsolidationGraphState:
        stats = state["stats"]
        governed_tags = state["governed_tags"]
        tag_by_name = {tag.name: tag for tag in governed_tags}
        assigned_ids = {draft.canonical_id for draft in state["classification"].assignments}
        if "coverage" in state:
            uncovered_ids = {canonical.canonical_id for canonical in state.get("uncovered_canonicals", [])}
            stats.rules_covered_by_existing = len(assigned_ids - uncovered_ids)
        for draft in state["classification"].assignments:
            tag = tag_by_name[normalize_tag_name(draft.tag_name)]
            if self.store.upsert_rule_tag_assignment(assignment_from_draft(draft, tag)):
                stats.rules_tagged += 1
                stats.rules_classified += 1
        stats.rules_skipped_no_tag += len({canonical.canonical_id for canonical in state["canonicals"]} - assigned_ids)
        _progress(
            f"persist_assignments done rules_classified={stats.rules_classified} "
            f"rules_skipped_no_tag={stats.rules_skipped_no_tag}"
        )
        return {**state, "stats": stats}


def run_consolidation(
    db_path: Path,
    canonicalizer: DeterministicCanonicalizer | LLMCanonicalizer | None = None,
    taxonomy_drafter: DeterministicTaxonomyDrafter | LLMTaxonomyDrafter | None = None,
    tag_coverage_checker: DeterministicTagCoverageChecker | LLMTagCoverageChecker | None = None,
    taxonomy_governor: DeterministicTaxonomyGovernor | LLMAgentTagGovernor | None = None,
    rule_classifier: DeterministicRuleClassifier | LLMRuleClassifier | None = None,
    settings: Settings | None = None,
) -> ConsolidationStats:
    store = KnowledgeStore(db_path)
    run_id = f"run_{uuid4().hex[:12]}"
    records = store.list_all_source_records()
    stats = ConsolidationStats(run_id=run_id, source_records_seen=len(records))
    _progress(f"run_consolidation start run_id={run_id} db_path={db_path}")
    store.runs.start_consolidation_run(run_id, db_path, settings.qwen_model if settings else None, stats)
    service: ConsolidationService | None = None
    try:
        has_api_key = bool(settings and settings.dashscope_api_key)
        if has_api_key:
            vector_store = KnowledgeVectorStore(store, create_dashscope_embeddings(settings))
            canonicalizer = DeterministicCanonicalizer()
            taxonomy_drafter = LLMTaxonomyDrafter(settings, recorder=store.runs, run_id=run_id)
            tag_coverage_checker = LLMTagCoverageChecker(settings, recorder=store.runs, run_id=run_id)
            taxonomy_governor = LLMAgentTagGovernor(settings, vector_store, recorder=store.runs, run_id=run_id)
            rule_classifier = DeterministicRuleClassifier(vector_store=vector_store)
        elif settings is not None:
            vector_store = None
            canonicalizer = DeterministicCanonicalizer()
            taxonomy_drafter = DeterministicTaxonomyDrafter()
            tag_coverage_checker = DeterministicTagCoverageChecker()
            taxonomy_governor = DeterministicTaxonomyGovernor()
            rule_classifier = DeterministicRuleClassifier()
        else:
            vector_store = None
            canonicalizer = canonicalizer or DeterministicCanonicalizer()
            taxonomy_drafter = taxonomy_drafter or DeterministicTaxonomyDrafter()
            tag_coverage_checker = tag_coverage_checker or DeterministicTagCoverageChecker()
            taxonomy_governor = taxonomy_governor or DeterministicTaxonomyGovernor()
            rule_classifier = rule_classifier or DeterministicRuleClassifier()
        service = ConsolidationService(
            store,
            canonicalizer=canonicalizer,
            taxonomy_drafter=taxonomy_drafter,
            tag_coverage_checker=tag_coverage_checker,
            taxonomy_governor=taxonomy_governor,
            rule_classifier=rule_classifier,
            vector_store=vector_store,
            run_id=run_id,
        )
        stats = service.run(records)
        stats.agent_invocations = store.runs.count_agent_invocations(run_id)
        stats.agent_failures = store.runs.count_agent_failures(run_id)
        status = (
            "completed"
            if stats.classification_failures == 0
            and stats.taxonomy_governance_failures == 0
            and stats.rule_classification_failures == 0
            else "completed_with_failures"
        )
        store.runs.finish_consolidation_run(run_id, status, stats)
        _progress(f"run_consolidation done run_id={run_id} status={status} rules_classified={stats.rules_classified}")
        return stats
    except Exception:
        if service is not None and service.last_stats is not None:
            stats = service.last_stats
        stats.agent_invocations = store.runs.count_agent_invocations(run_id)
        stats.agent_failures = store.runs.count_agent_failures(run_id)
        store.runs.finish_consolidation_run(run_id, "failed", stats)
        _progress(f"run_consolidation failed run_id={run_id}")
        raise
    finally:
        store.close()



def _canonical_from_record(record: PitfallRecord) -> CanonicalKnowledge:
    now = utc_now()
    return CanonicalKnowledge(
        canonical_id=_canonical_id(record),
        title=record.title,
        category=record.category,
        summary=f"{record.failure_mode} {record.impact}".strip(),
        preventive_rule=record.preventive_rule,
        scope=record.scope,
        status=CanonicalKnowledgeStatus.ACTIVE,
        source_record_ids=[record.id],
        support_count=1,
        created_at=now,
        updated_at=now,
    )


def _canonical_id(record: PitfallRecord) -> str:
    key = f"{record.category.value}|{normalize_text(record.title)}|{normalize_text(record.preventive_rule)}"
    return f"canonical_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def _concise_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:500]


def _progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[{timestamp}] {message}", flush=True)
