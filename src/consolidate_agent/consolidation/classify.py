from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.consolidation.canonicalize import canonical_rule_view
from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model, _invoke_with_retry
from consolidate_agent.observability import AgentInvocationRecorder, NullAgentInvocationRecorder
from consolidate_agent.prompt_loader import load_prompt
from consolidate_agent.consolidation.taxonomy import _best_existing_tag, _infer_tag_name, normalize_tag_name
from consolidate_agent.types import (
    CanonicalKnowledge,
    MechanismTag,
    RuleTagAssignment,
    utc_now,
)

if TYPE_CHECKING:
    from consolidate_agent.knowledge.vector import KnowledgeVectorStore

RULE_BATCH_SIZE = 20


class RuleTagAssignmentDraft(BaseModel):
    canonical_id: str
    tag_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    assignment_reason: str


class RuleClassificationResult(BaseModel):
    assignments: list[RuleTagAssignmentDraft] = Field(default_factory=list)


class TagCoverageResult(BaseModel):
    uncovered_ids: list[str] = Field(default_factory=list)


class RuleAssignment(BaseModel):
    canonical_id: str
    tag_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class RuleBatchAssignmentResult(BaseModel):
    assignments: list[RuleAssignment] = Field(default_factory=list)


class DeterministicTagCoverageChecker:
    def check(self, canonicals: list[CanonicalKnowledge], tags: list[MechanismTag]) -> TagCoverageResult:
        return TagCoverageResult(uncovered_ids=[canonical.canonical_id for canonical in canonicals])


class LLMTagCoverageChecker:
    def __init__(self, settings: Settings, recorder: AgentInvocationRecorder | None = None, run_id: str | None = None):
        self.recorder = recorder or NullAgentInvocationRecorder()
        self.run_id = run_id
        self.structured_model = _chat_model(settings).with_structured_output(TagCoverageResult)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                    "You decide whether each canonical rule can be semantically covered by any existing mechanism tag. "
                    "Covered means the failure mechanism is essentially the same as a tag definition, even if wording differs. "
                    "Return uncovered_ids containing only canonical_id values for rules that truly have no corresponding existing tag."),
                ("user", "Existing tags:\n{{ tags_json }}\n\nCanonical rules to check:\n{{ rules_json }}\n\nReturn structured output with uncovered_ids."),
            ],
            template_format="jinja2",
        )
        self.retry_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                    "You decide whether each canonical rule can be semantically covered by any existing mechanism tag. "
                    "Return uncovered_ids containing only canonical_id values for rules that truly have no corresponding existing tag."),
                ("user", "Existing tags:\n{{ tags_json }}\n\nCanonical rules to check:\n{{ rules_json }}\n\nPrevious attempt failed: {{ validation_error }}\nPrevious result: {{ invalid_result_json }}\n\nReturn corrected structured output with uncovered_ids."),
            ],
            template_format="jinja2",
        )

    def check(self, canonicals: list[CanonicalKnowledge], tags: list[MechanismTag]) -> TagCoverageResult:
        values = _tag_coverage_values(canonicals, tags)
        return _invoke_with_retry(
            stage="tag_coverage_check",
            structured_model=self.structured_model,
            prompt=self.prompt,
            retry_prompt=self.retry_prompt,
            values=values,
            recorder=self.recorder,
            run_id=self.run_id,
            validator=lambda result: validate_tag_coverage(result, canonicals),
            empty_error="Tag coverage check returned no structured result.",
        )


class DeterministicRuleClassifier:
    def __init__(self, vector_store: KnowledgeVectorStore | None = None):
        self.vector_store = vector_store

    def classify(self, canonicals: list[CanonicalKnowledge], tags: list[MechanismTag]) -> RuleClassificationResult:
        assignments: list[RuleTagAssignmentDraft] = []
        for canonical in canonicals:
            if self.vector_store is not None:
                tag, confidence = self.vector_store.find_best_tag(canonical, tags)
            else:
                tag = _best_existing_tag(canonical, tags)
                if tag is None:
                    inferred = _infer_tag_name(canonical)
                    tag = next((t for t in tags if normalize_tag_name(t.name) == normalize_tag_name(inferred)), None)
                confidence = 0.7
            if tag is None:
                continue
            assignments.append(
                RuleTagAssignmentDraft(
                    canonical_id=canonical.canonical_id,
                    tag_name=tag.name,
                    confidence=confidence,
                    assignment_reason=f"Rule is best covered by governed tag '{tag.name}'.",
                )
            )
        return RuleClassificationResult(assignments=assignments)


class LLMRuleClassifier:
    def __init__(self, settings: Settings, recorder: AgentInvocationRecorder | None = None, run_id: str | None = None):
        self.recorder = recorder or NullAgentInvocationRecorder()
        self.run_id = run_id
        self.structured_model = _chat_model(settings).with_structured_output(RuleBatchAssignmentResult)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("rule_batch_assignment_system.md")), ("user", load_prompt("rule_batch_assignment_user.md"))],
            template_format="jinja2",
        )
        self.retry_prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("rule_batch_assignment_retry_system.md")), ("user", load_prompt("rule_batch_assignment_retry_user.md"))],
            template_format="jinja2",
        )

    def classify(self, canonicals: list[CanonicalKnowledge], tags: list[MechanismTag]) -> RuleClassificationResult:
        best_by_canonical: dict[str, RuleTagAssignmentDraft] = {}
        for index in range(0, len(canonicals), RULE_BATCH_SIZE):
            batch = canonicals[index : index + RULE_BATCH_SIZE]
            values = _rule_batch_assignment_values(batch, tags)
            batch_result = _invoke_with_retry(
                stage="rule_batch_assignment",
                structured_model=self.structured_model,
                prompt=self.prompt,
                retry_prompt=self.retry_prompt,
                values=values,
                recorder=self.recorder,
                run_id=self.run_id,
                validator=lambda result, batch=batch: validate_rule_batch_assignment(result, batch, tags),
                empty_error="Rule batch assignment returned no structured result.",
            )
            for rule_assignment in batch_result.assignments:
                assignment = RuleTagAssignmentDraft(
                    canonical_id=rule_assignment.canonical_id,
                    tag_name=rule_assignment.tag_name,
                    confidence=rule_assignment.confidence,
                    assignment_reason=rule_assignment.reason,
                )
                existing = best_by_canonical.get(rule_assignment.canonical_id)
                if existing is None or assignment.confidence > existing.confidence:
                    best_by_canonical[rule_assignment.canonical_id] = assignment
        result = RuleClassificationResult(assignments=list(best_by_canonical.values()))
        validate_rule_classification(result, canonicals, tags)
        return result


def validate_rule_classification(result: RuleClassificationResult, canonicals: list[CanonicalKnowledge], tags: list[MechanismTag]) -> None:
    expected_ids = {canonical.canonical_id for canonical in canonicals}
    seen_ids = [assignment.canonical_id for assignment in result.assignments]
    if len(seen_ids) != len(set(seen_ids)):
        raise ValueError("Rule classification must assign each canonical_id at most once.")
    unknown_ids = set(seen_ids) - expected_ids
    if unknown_ids:
        raise ValueError(f"Rule classification references unknown canonical ids: {sorted(unknown_ids)}")
    tag_names = {normalize_tag_name(tag.name) for tag in tags}
    for assignment in result.assignments:
        if normalize_tag_name(assignment.tag_name) not in tag_names:
            raise ValueError(f"Rule classification references unknown tag_name: {assignment.tag_name}")


def validate_tag_coverage(result: TagCoverageResult, canonicals: list[CanonicalKnowledge]) -> None:
    expected_ids = {canonical.canonical_id for canonical in canonicals}
    unknown_ids = set(result.uncovered_ids) - expected_ids
    if unknown_ids:
        raise ValueError(f"Tag coverage references unknown canonical ids: {sorted(unknown_ids)}")


def validate_rule_batch_assignment(
    result: RuleBatchAssignmentResult,
    canonicals: list[CanonicalKnowledge],
    tags: list[MechanismTag],
) -> None:
    expected_ids = {canonical.canonical_id for canonical in canonicals}
    assignment_ids = [assignment.canonical_id for assignment in result.assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        duplicates = sorted({id for id in assignment_ids if assignment_ids.count(id) > 1})
        raise ValueError(
            f"Rule batch assignment contains duplicate canonical_ids: {duplicates}. "
            "Each canonical_id must appear at most once."
        )
    unknown_ids = sorted(set(assignment_ids) - expected_ids)
    if unknown_ids:
        raise ValueError(
            f"Rule batch assignment references unknown canonical_ids: {unknown_ids}. "
            "Only use canonical_ids from the provided rules list."
        )
    tag_names = {normalize_tag_name(tag.name) for tag in tags}
    unknown_tag_names = sorted(
        {assignment.tag_name for assignment in result.assignments if normalize_tag_name(assignment.tag_name) not in tag_names}
    )
    if unknown_tag_names:
        raise ValueError(
            f"Rule batch assignment references unknown tag_names: {unknown_tag_names}. "
            "Only use tag names from the provided tags list."
        )


def assignment_from_draft(draft: RuleTagAssignmentDraft, tag: MechanismTag) -> RuleTagAssignment:
    return RuleTagAssignment(
        canonical_id=draft.canonical_id,
        tag_id=tag.tag_id,
        confidence=draft.confidence,
        assignment_reason=draft.assignment_reason,
        linked_at=utc_now(),
    )


def _classification_values(canonicals: list[CanonicalKnowledge], tags: list[MechanismTag]) -> dict[str, str]:
    return {
        "tags_json": json.dumps([tag.model_dump(mode="json") for tag in tags], ensure_ascii=False, indent=2),
        "rules_json": json.dumps([canonical_rule_view(canonical).model_dump(mode="json") for canonical in canonicals], ensure_ascii=False, indent=2),
    }


def _rule_batch_assignment_values(canonicals: list[CanonicalKnowledge], tags: list[MechanismTag]) -> dict[str, str]:
    tag_views = [
        {
            "name": tag.name,
            "definition": tag.definition,
            "positive_examples": tag.positive_examples,
            "negative_examples": tag.negative_examples,
        }
        for tag in tags
    ]
    rule_views = [
        {
            "canonical_id": canonical.canonical_id,
            "title": canonical.title,
            "preventive_rule": canonical.preventive_rule,
        }
        for canonical in canonicals
    ]
    return {
        "tags_json": json.dumps(tag_views, ensure_ascii=False, indent=2),
        "rules_json": json.dumps(rule_views, ensure_ascii=False, indent=2),
    }


def _tag_coverage_values(canonicals: list[CanonicalKnowledge], tags: list[MechanismTag]) -> dict[str, str]:
    return {
        "tags_json": json.dumps(
            [
                {
                    "name": tag.name,
                    "definition": tag.definition,
                    "positive_examples": tag.positive_examples,
                    "negative_examples": tag.negative_examples,
                }
                for tag in tags
            ],
            ensure_ascii=False,
            indent=2,
        ),
        "rules_json": json.dumps(
            [
                {
                    "canonical_id": canonical.canonical_id,
                    "title": canonical.title,
                    "category": canonical.category.value,
                    "preventive_rule": canonical.preventive_rule,
                }
                for canonical in canonicals
            ],
            ensure_ascii=False,
            indent=2,
        ),
    }
