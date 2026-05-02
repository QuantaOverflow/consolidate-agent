from __future__ import annotations

import hashlib
import json

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model, _invoke_with_retry
from consolidate_agent.observability import AgentInvocationRecorder, NullAgentInvocationRecorder
from consolidate_agent.prompt_loader import load_prompt
from consolidate_agent.types import (
    CanonicalKnowledge,
    CanonicalKnowledgeStatus,
    KnowledgeRelation,
    PitfallCategory,
    PitfallRecord,
    PitfallScope,
    normalize_text,
    utc_now,
)

DETERMINISTIC_GROUP_MAX_SIZE = 5


class SourcePitfallView(BaseModel):
    source_record_id: str
    title: str
    category: str
    preventive_rule: str
    scope: str


class CanonicalRuleView(BaseModel):
    canonical_id: str
    title: str
    category: str
    preventive_rule: str


class CanonicalDraft(BaseModel):
    temporary_id: str
    title: str
    category: str
    summary: str
    preventive_rule: str
    scope: str
    source_record_ids: list[str] = Field(default_factory=list)


class CanonicalizationDecision(BaseModel):
    source_record_id: str
    relation: str
    canonical_ref: str
    decision_reason: str


class CanonicalizationResult(BaseModel):
    canonicals: list[CanonicalDraft] = Field(default_factory=list)
    decisions: list[CanonicalizationDecision] = Field(default_factory=list)


class CanonicalContent(BaseModel):
    title: str
    category: str
    summary: str
    preventive_rule: str
    scope: str


class DeterministicCanonicalizer:
    def canonicalize(
        self,
        records: list[PitfallRecord],
        existing_canonicals: list[CanonicalKnowledge],
    ) -> CanonicalizationResult:
        existing_by_key = {
            _canonical_key(canonical.category.value, canonical.title, canonical.preventive_rule): canonical
            for canonical in existing_canonicals
        }
        drafts_by_key: dict[str, CanonicalDraft] = {}
        decisions: list[CanonicalizationDecision] = []
        for record in records:
            key = _canonical_key(record.category.value, record.title, record.preventive_rule)
            existing = existing_by_key.get(key)
            if existing is not None:
                decisions.append(
                    CanonicalizationDecision(
                        source_record_id=record.id,
                        relation=KnowledgeRelation.DUPLICATE.value,
                        canonical_ref=existing.canonical_id,
                        decision_reason="Exact canonical key already exists.",
                    )
                )
                continue
            draft = drafts_by_key.get(key)
            if draft is None:
                draft = CanonicalDraft(
                    temporary_id=f"draft_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}",
                    title=record.title,
                    category=record.category.value,
                    summary=f"{record.failure_mode} {record.impact}".strip(),
                    preventive_rule=record.preventive_rule,
                    scope=record.scope.value,
                    source_record_ids=[],
                )
                drafts_by_key[key] = draft
            draft.source_record_ids.append(record.id)
            decisions.append(
                CanonicalizationDecision(
                    source_record_id=record.id,
                    relation=KnowledgeRelation.DISTINCT.value,
                    canonical_ref=draft.temporary_id,
                    decision_reason="Source maps to a new canonical draft.",
                )
            )
        return CanonicalizationResult(canonicals=list(drafts_by_key.values()), decisions=decisions)


class LLMCanonicalizer:
    def __init__(self, settings: Settings, recorder: AgentInvocationRecorder | None = None, run_id: str | None = None):
        self.recorder = recorder or NullAgentInvocationRecorder()
        self.run_id = run_id
        self.structured_model = _chat_model(settings).with_structured_output(CanonicalizationResult)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("canonicalization_system.md")), ("user", load_prompt("canonicalization_user.md"))],
            template_format="jinja2",
        )
        self.retry_prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("canonicalization_retry_system.md")), ("user", load_prompt("canonicalization_retry_user.md"))],
            template_format="jinja2",
        )

    def canonicalize(
        self,
        records: list[PitfallRecord],
        existing_canonicals: list[CanonicalKnowledge],
    ) -> CanonicalizationResult:
        alias_map: dict[str, str] = {}
        values = _canonicalization_values(records, existing_canonicals, alias_map)
        result = _invoke_with_retry(
            stage="canonicalization",
            structured_model=self.structured_model,
            prompt=self.prompt,
            retry_prompt=self.retry_prompt,
            values=values,
            recorder=self.recorder,
            run_id=self.run_id,
            validator=lambda result: validate_canonicalization(result, records, existing_canonicals, alias_map),
            empty_error="Canonicalization returned no structured result.",
        )
        return _resolve_canonical_aliases(result, alias_map)


class LLMGroupRewriter:
    def __init__(self, settings: Settings, recorder: AgentInvocationRecorder | None = None, run_id: str | None = None):
        self.recorder = recorder or NullAgentInvocationRecorder()
        self.run_id = run_id
        self.structured_model = _chat_model(settings).with_structured_output(CanonicalContent)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("group_rewrite_system.md")), ("user", load_prompt("group_rewrite_user.md"))],
            template_format="jinja2",
        )
        self.retry_prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("group_rewrite_retry_system.md")), ("user", load_prompt("group_rewrite_retry_user.md"))],
            template_format="jinja2",
        )

    def rewrite(self, group: list[PitfallRecord]) -> CanonicalContent:
        values = _group_rewrite_values(group)
        return _invoke_with_retry(
            stage="group_rewrite",
            structured_model=self.structured_model,
            prompt=self.prompt,
            retry_prompt=self.retry_prompt,
            values=values,
            recorder=self.recorder,
            run_id=self.run_id,
            validator=validate_canonical_content,
            empty_error="Group rewrite returned no structured result.",
        )


def validate_canonicalization(
    result: CanonicalizationResult,
    records: list[PitfallRecord],
    existing_canonicals: list[CanonicalKnowledge],
    alias_map: dict[str, str] | None = None,
) -> None:
    source_ids = {record.id for record in records}
    decision_ids = [decision.source_record_id for decision in result.decisions]
    if set(decision_ids) != source_ids or len(decision_ids) != len(set(decision_ids)):
        raise ValueError("Canonicalization must decide every source_record_id exactly once.")

    existing_ids = {canonical.canonical_id for canonical in existing_canonicals}
    if alias_map:
        existing_ids = existing_ids | set(alias_map.keys())
    draft_ids = [draft.temporary_id for draft in result.canonicals]
    if len(draft_ids) != len(set(draft_ids)):
        raise ValueError("Canonicalization draft temporary_id values must be unique.")

    draft_by_id = {draft.temporary_id: draft for draft in result.canonicals}
    valid_relations = {relation.value for relation in KnowledgeRelation}
    for draft in result.canonicals:
        if not draft.temporary_id:
            raise ValueError("Canonical draft temporary_id is required.")
        if not draft.title.strip() or not draft.preventive_rule.strip() or not draft.summary.strip():
            raise ValueError(f"Canonical draft is missing stable knowledge fields: {draft.temporary_id}")
        try:
            PitfallCategory(draft.category)
            PitfallScope(draft.scope)
        except ValueError as exc:
            raise ValueError(f"Canonical draft has invalid category or scope: {draft.temporary_id}") from exc
        draft_source_ids = set(draft.source_record_ids)
        if not draft_source_ids:
            raise ValueError(f"Canonical draft must cite source records: {draft.temporary_id}")
        unknown = draft_source_ids - source_ids
        if unknown:
            raise ValueError(f"Canonical draft references unknown source records: {sorted(unknown)}")

    for decision in result.decisions:
        if decision.relation not in valid_relations:
            raise ValueError(f"Canonicalization decision has invalid relation: {decision.relation}")
        if decision.canonical_ref not in existing_ids and decision.canonical_ref not in draft_by_id:
            raise ValueError(f"Canonicalization decision references unknown canonical_ref: {decision.canonical_ref}")
        if decision.canonical_ref in draft_by_id and decision.source_record_id not in draft_by_id[decision.canonical_ref].source_record_ids:
            raise ValueError(
                f"Canonicalization decision source is not cited by draft: {decision.source_record_id} -> {decision.canonical_ref}"
            )

    for draft_id in draft_by_id:
        decisions_for_draft = [d for d in result.decisions if d.canonical_ref == draft_id]
        if not decisions_for_draft:
            raise ValueError(f"New canonical draft has no decisions pointing to it: {draft_id}")


def validate_canonical_content(content: CanonicalContent) -> None:
    if not content.title.strip() or not content.preventive_rule.strip() or not content.summary.strip():
        raise ValueError("Canonical content must include title, summary, and preventive_rule.")
    try:
        PitfallCategory(content.category)
        PitfallScope(content.scope)
    except ValueError as exc:
        raise ValueError("Canonical content has invalid category or scope.") from exc


def canonical_from_draft(draft: CanonicalDraft) -> CanonicalKnowledge:
    now = utc_now()
    return CanonicalKnowledge(
        canonical_id=canonical_id(draft.category, draft.title, draft.preventive_rule),
        title=draft.title.strip(),
        category=PitfallCategory(draft.category),
        summary=draft.summary.strip(),
        preventive_rule=draft.preventive_rule.strip(),
        scope=PitfallScope(draft.scope),
        status=CanonicalKnowledgeStatus.ACTIVE,
        source_record_ids=sorted(draft.source_record_ids),
        support_count=len(set(draft.source_record_ids)),
        created_at=now,
        updated_at=now,
    )


def canonical_id(category: str, title: str, preventive_rule: str) -> str:
    return f"canonical_{hashlib.sha1(_canonical_key(category, title, preventive_rule).encode('utf-8')).hexdigest()[:12]}"


def canonical_rule_view(canonical: CanonicalKnowledge) -> CanonicalRuleView:
    return CanonicalRuleView(
        canonical_id=canonical.canonical_id,
        title=canonical.title,
        category=canonical.category.value,
        preventive_rule=canonical.preventive_rule,
    )


def source_pitfall_view(record: PitfallRecord) -> SourcePitfallView:
    return SourcePitfallView(
        source_record_id=record.id,
        title=record.title,
        category=record.category.value,
        preventive_rule=record.preventive_rule,
        scope=record.scope.value,
    )


def canonical_content_from_record(record: PitfallRecord) -> CanonicalContent:
    return CanonicalContent(
        title=record.title,
        category=record.category.value,
        summary=f"{record.failure_mode} {record.impact}".strip(),
        preventive_rule=record.preventive_rule,
        scope=record.scope.value,
    )


def build_canonicalization_result(
    groups: list[list[PitfallRecord]],
    contents: list[CanonicalContent],
    direct_duplicates: list[tuple[PitfallRecord, str]],
    existing_canonicals: list[CanonicalKnowledge],
) -> CanonicalizationResult:
    if len(groups) != len(contents):
        raise ValueError("Canonicalization groups and contents must have the same length.")

    existing_by_key = {
        _canonical_key(canonical.category.value, canonical.title, canonical.preventive_rule): canonical.canonical_id
        for canonical in existing_canonicals
    }
    drafts_by_ref: dict[str, CanonicalDraft] = {}
    decisions: list[CanonicalizationDecision] = []

    for record, canonical_id_value in direct_duplicates:
        decisions.append(
            CanonicalizationDecision(
                source_record_id=record.id,
                relation=KnowledgeRelation.DUPLICATE.value,
                canonical_ref=canonical_id_value,
                decision_reason="Source exactly matches an existing canonical key.",
            )
        )

    for group, content in zip(groups, contents, strict=True):
        validate_canonical_content(content)
        key = _canonical_key(content.category, content.title, content.preventive_rule)
        existing_id = existing_by_key.get(key)
        if existing_id is not None:
            for record in group:
                decisions.append(
                    CanonicalizationDecision(
                        source_record_id=record.id,
                        relation=KnowledgeRelation.DUPLICATE.value,
                        canonical_ref=existing_id,
                        decision_reason="Rewritten canonical content matches an existing canonical key.",
                    )
                )
            continue

        temporary_id = f"draft_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"
        draft = drafts_by_ref.get(temporary_id)
        if draft is None:
            draft = CanonicalDraft(
                temporary_id=temporary_id,
                title=content.title,
                category=content.category,
                summary=content.summary,
                preventive_rule=content.preventive_rule,
                scope=content.scope,
                source_record_ids=[],
            )
            drafts_by_ref[temporary_id] = draft
        for record in group:
            if record.id not in draft.source_record_ids:
                draft.source_record_ids.append(record.id)
            decisions.append(
                CanonicalizationDecision(
                    source_record_id=record.id,
                    relation=KnowledgeRelation.DISTINCT.value,
                    canonical_ref=temporary_id,
                    decision_reason="Source group produced a new canonical draft.",
                )
            )

    return CanonicalizationResult(canonicals=list(drafts_by_ref.values()), decisions=decisions)


def deterministic_group(
    records: list[PitfallRecord],
    existing_canonicals: list[CanonicalKnowledge],
) -> tuple[list[list[PitfallRecord]], list[tuple[PitfallRecord, str]]]:
    existing_by_key = {
        _canonical_key(canonical.category.value, canonical.title, canonical.preventive_rule): canonical.canonical_id
        for canonical in existing_canonicals
    }
    grouped_by_key: dict[str, list[PitfallRecord]] = {}
    direct_duplicates: list[tuple[PitfallRecord, str]] = []
    for record in records:
        key = _canonical_key(record.category.value, record.title, record.preventive_rule)
        existing_id = existing_by_key.get(key)
        if existing_id is not None:
            direct_duplicates.append((record, existing_id))
            continue
        grouped_by_key.setdefault(key, []).append(record)

    groups: list[list[PitfallRecord]] = []
    for group in grouped_by_key.values():
        for index in range(0, len(group), DETERMINISTIC_GROUP_MAX_SIZE):
            groups.append(group[index : index + DETERMINISTIC_GROUP_MAX_SIZE])
    return groups, direct_duplicates


def _resolve_canonical_aliases(result: CanonicalizationResult, alias_map: dict[str, str]) -> CanonicalizationResult:
    if not alias_map:
        return result
    resolved_decisions = []
    for decision in result.decisions:
        canonical_ref = alias_map.get(decision.canonical_ref, decision.canonical_ref)
        resolved_decisions.append(CanonicalizationDecision(
            source_record_id=decision.source_record_id,
            relation=decision.relation,
            canonical_ref=canonical_ref,
            decision_reason=decision.decision_reason,
        ))
    return CanonicalizationResult(canonicals=result.canonicals, decisions=resolved_decisions)


def _canonicalization_values(
    records: list[PitfallRecord],
    existing_canonicals: list[CanonicalKnowledge],
    alias_map: dict[str, str] | None = None,
) -> dict[str, str]:
    batch_categories = {record.category for record in records}
    filtered_canonicals = [c for c in existing_canonicals if c.category in batch_categories]
    canonical_views = []
    for i, canonical in enumerate(filtered_canonicals):
        view = canonical_rule_view(canonical).model_dump(mode="json")
        if alias_map is not None:
            alias = f"EC_{i + 1}"
            alias_map[alias] = canonical.canonical_id
            view["canonical_id"] = alias
        canonical_views.append(view)
    return {
        "source_records_json": json.dumps(
            [source_pitfall_view(record).model_dump(mode="json") for record in records],
            ensure_ascii=False,
            indent=2,
        ),
        "existing_canonicals_json": json.dumps(canonical_views, ensure_ascii=False, indent=2),
    }


def _group_rewrite_values(group: list[PitfallRecord]) -> dict[str, str]:
    return {
        "records_json": json.dumps(
            [
                {
                    "source_record_id": record.id,
                    "title": record.title,
                    "category": record.category.value,
                    "preventive_rule": record.preventive_rule,
                }
                for record in group
            ],
            ensure_ascii=False,
            indent=2,
        )
    }


def _canonical_key(category: str, title: str, preventive_rule: str) -> str:
    return f"{category}|{normalize_text(title)}|{normalize_text(preventive_rule)}"
