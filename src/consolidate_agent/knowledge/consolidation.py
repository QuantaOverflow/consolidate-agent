from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Protocol, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model, _invoke_structured
from consolidate_agent.consolidation.taxonomy import (
    TagDeduplicationResult,
    normalize_tag_name,
    validate_tag_deduplication,
)
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.knowledge.vector import KnowledgeVectorStore
from consolidate_agent.prompt_loader import load_prompt
from consolidate_agent.types import KnowledgeRecord, MechanismTag, MechanismTagStatus, normalize_text, utc_now

KNOWLEDGE_TAXONOMY_BATCH_SIZE = 50
MAX_DRAFT_WORKERS = 5
MAX_GOVERNANCE_WORKERS = 10
MAX_ASSIGNMENT_WORKERS = 10
MAX_COVERAGE_RETRIES = 2


class KnowledgeConsolidationState(TypedDict, total=False):
    records: list[KnowledgeRecord]
    is_warm_start: bool
    coverage_retry_count: int
    batch_proposals: list[KnowledgeTagProposalDraft]
    active_tags: list[KnowledgeTag]
    assignments: list[KnowledgeTagAssignmentDraft]
    uncovered_records: list[KnowledgeRecord]
    coverage_checked: bool


class KnowledgeTag(BaseModel):
    tag_id: str
    name: str
    definition: str
    status: str = "active"
    merged_into_tag_id: str | None = None
    created_at: datetime
    updated_at: datetime


class KnowledgeTagProposalDraft(BaseModel):
    name: str
    definition: str
    supporting_record_ids: list[str] = Field(default_factory=list)


class KnowledgeTagAssignmentDraft(BaseModel):
    record_id: str
    tag_names: list[str] = Field(default_factory=list, max_length=3)
    reasoning: str


class KnowledgeTaxonomyDraftResult(BaseModel):
    proposals: list[KnowledgeTagProposalDraft] = Field(default_factory=list)


class KnowledgeTaxonomyGovernanceDecision(BaseModel):
    proposal_name: str
    decision: str
    target_tag_name: str | None = None
    decision_reason: str
    accepted_tag: KnowledgeTagProposalDraft | None = None


class KnowledgeTaxonomyGovernanceResult(BaseModel):
    decisions: list[KnowledgeTaxonomyGovernanceDecision] = Field(default_factory=list)


class KnowledgeTaxonomyDrafter(Protocol):
    def draft_batch(self, records: list[KnowledgeRecord], active_tags: list[KnowledgeTag]) -> KnowledgeTaxonomyDraftResult:
        ...


class KnowledgeTagAssigner(Protocol):
    def assign_one(self, record: KnowledgeRecord, active_tags: list[KnowledgeTag]) -> KnowledgeTagAssignmentDraft | None:
        ...


class LLMKnowledgeTaxonomyDrafter:
    def __init__(self, settings: Settings):
        self.structured_model = _chat_model(settings).with_structured_output(KnowledgeTaxonomyDraftResult, include_raw=True)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("knowledge_taxonomy_draft_system.md")),
                ("user", load_prompt("knowledge_taxonomy_draft_user.md")),
            ],
            template_format="jinja2",
        )

    def draft_batch(self, records: list[KnowledgeRecord], active_tags: list[KnowledgeTag]) -> KnowledgeTaxonomyDraftResult:
        values = {
            "active_tags_json": json.dumps([_tag_view(tag) for tag in active_tags], ensure_ascii=False, indent=2),
            "records_json": json.dumps([_record_view(record) for record in records], ensure_ascii=False, indent=2),
        }
        base_messages = self.prompt.invoke(values).to_messages()
        messages = base_messages
        for attempt in range(1, 4):
            try:
                raw_result = self.structured_model.invoke(messages)
                parsed = raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result
                if parsed is not None:
                    return parsed
                messages = base_messages + [_retry_message(raw_result, attempt)]
            except Exception as exc:  # noqa: BLE001
                _progress(f"knowledge_taxonomy_draft batch attempt={attempt} error={exc}")
                messages = base_messages + [_error_retry_message(str(exc), attempt)]
        _progress("knowledge_taxonomy_draft batch exhausted retries, returning empty proposals")
        return KnowledgeTaxonomyDraftResult(proposals=[])


class LLMKnowledgeTagAssigner:
    def __init__(self, settings: Settings):
        self._llm = _chat_model(settings)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("knowledge_tag_assignment_system.md")),
                ("user", load_prompt("knowledge_tag_assignment_user.md")),
            ],
            template_format="jinja2",
        )

    def _build_structured_model(self, active_tags: list[KnowledgeTag]):
        """Build a structured model with Literal tag names constrained to active_tags."""
        from typing import Literal
        from pydantic import create_model
        tag_names_tuple = tuple(tag.name for tag in active_tags)
        TagNameLiteral = Literal[tag_names_tuple]  # type: ignore[valid-type]
        DynamicAssignment = create_model(
            "DynamicKnowledgeTagAssignment",
            record_id=(str, ""),
            tag_names=(list[TagNameLiteral], []),  # type: ignore[valid-type]
            reasoning=(str, ""),
        )
        return self._llm.with_structured_output(DynamicAssignment, include_raw=True)

    def assign_one(self, record: KnowledgeRecord, active_tags: list[KnowledgeTag]) -> KnowledgeTagAssignmentDraft | None:
        if not active_tags:
            return None
        structured_model = self._build_structured_model(active_tags)
        values = {
            "tags_json": json.dumps([_tag_view(tag) for tag in active_tags], ensure_ascii=False, indent=2),
            "title": record.title,
            "insight": record.insight,
            "applicability": record.applicability,
        }
        base_messages = self.prompt.invoke(values).to_messages()
        messages = base_messages
        for attempt in range(1, 4):
            try:
                raw_result = structured_model.invoke(messages)
                parsed = raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result
                if parsed is not None:
                    return KnowledgeTagAssignmentDraft(
                        record_id=record.id,
                        tag_names=list(parsed.tag_names),
                        reasoning=parsed.reasoning,
                    )
                messages = base_messages + [_retry_message(raw_result, attempt)]
            except Exception as exc:  # noqa: BLE001
                _progress(f"knowledge_tag_assignment record={record.id} attempt={attempt} error={exc}")
                messages = base_messages + [_error_retry_message(str(exc), attempt)]
        _progress(f"knowledge_tag_assignment record={record.id} exhausted retries, skipping")
        return None


class LLMKnowledgeTagGovernor:
    def __init__(self, settings: Settings, vector_store: KnowledgeVectorStore):
        self.llm = _chat_model(settings)
        self.vector_store = vector_store
        self._structured_model = self.llm.with_structured_output(KnowledgeTaxonomyGovernanceDecision, include_raw=True)

    def govern(
        self,
        active_tags: list[KnowledgeTag],
        proposals: list[KnowledgeTagProposalDraft],
    ) -> KnowledgeTaxonomyGovernanceResult:
        decisions: list[KnowledgeTaxonomyGovernanceDecision | None] = [None] * len(proposals)
        with ThreadPoolExecutor(max_workers=MAX_GOVERNANCE_WORKERS) as executor:
            future_to_index = {
                executor.submit(self._decide_one, proposal, active_tags): index
                for index, proposal in enumerate(proposals)
            }
            for future in as_completed(future_to_index):
                decisions[future_to_index[future]] = future.result()
        result = KnowledgeTaxonomyGovernanceResult(decisions=[decision for decision in decisions if decision is not None])
        _validate_governance(result, proposals, active_tags)
        return result

    def _decide_one(
        self,
        proposal: KnowledgeTagProposalDraft,
        active_tags: list[KnowledgeTag],
    ) -> KnowledgeTaxonomyGovernanceDecision:
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        @tool
        def search_knowledge_records(query: str) -> str:
            """Search knowledge records semantically relevant to a query."""
            results = self.vector_store.similarity_search_knowledge_records(query, k=5)
            if not results:
                return "No relevant knowledge records found."
            return "\n\n".join(f"- {record.title}: {record.insight}" for record in results)

        active_tags_text = "\n".join(f"- {tag.name}: {tag.definition}" for tag in active_tags) or "None yet."
        user_message = f"""Decide whether to accept, merge, or reject this knowledge tag proposal.

Proposal:
  name: {proposal.name}
  definition: {proposal.definition}

Active tags:
{active_tags_text}

Use search_knowledge_records to find evidence if needed.
Summarize your reasoning and final decision (accept, merge, or reject)."""

        agent = create_react_agent(self.llm, [search_knowledge_records])
        agent_result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
        agent_summary = "\n".join(
            message.content
            for message in agent_result["messages"]
            if isinstance(message, AIMessage) and getattr(message, "content", None)
        ) or "No conclusion from search."

        extract_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extract a structured governance decision. "
                    f"proposal_name must be exactly \"{proposal.name}\". "
                    "decision must be one of: accept, merge, reject. "
                    "merge requires target_tag_name set to an existing active tag name. "
                    "accept requires accepted_tag with name, definition, and supporting_record_ids.",
                ),
                (
                    "human",
                    f"Proposal name: {proposal.name}\n"
                    f"Proposal definition: {proposal.definition}\n"
                    f"Active tags: {active_tags_text}\n\n"
                    f"Research summary:\n{agent_summary}",
                ),
            ],
            template_format="jinja2",
        )
        prompt_value = extract_prompt.invoke({})
        raw_result = self._structured_model.invoke(prompt_value)
        decision = raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result
        if decision is None:
            decision = KnowledgeTaxonomyGovernanceDecision(
                proposal_name=proposal.name,
                decision="reject",
                decision_reason="Structured governance decision failed; proposal rejected to keep consolidation running.",
            )
        decision.proposal_name = proposal.name
        return decision


class LLMKnowledgeTagDeduplicator:
    def __init__(self, settings: Settings):
        self.structured_model = _chat_model(settings).with_structured_output(TagDeduplicationResult)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You identify knowledge tags whose technical topics overlap. "
                    "The keep tag should be the more precise or canonical tag. "
                    "Each tag name in merges must appear exactly once across all merge decisions.",
                ),
                (
                    "user",
                    "Review the full active knowledge tag list below.\n\n"
                    "{{ tags_json }}\n\n"
                    "Return a merges list where each entry has keep, merge, and reason. "
                    "If no tags overlap, return an empty merges list.",
                ),
            ],
            template_format="jinja2",
        )

    def deduplicate(self, tags: list[KnowledgeTag]) -> TagDeduplicationResult:
        result = _invoke_structured(
            self.structured_model,
            self.prompt,
            {"tags_json": json.dumps([_tag_view(tag) for tag in tags], ensure_ascii=False, indent=2)},
            "Knowledge tag deduplication returned no structured result.",
        )
        validate_tag_deduplication(result, [_mechanism_tag_view(tag) for tag in tags])
        return result


class KnowledgeConsolidationService:
    def __init__(
        self,
        store: KnowledgeStore,
        vector_store: KnowledgeVectorStore,
        settings: Settings | None = None,
        taxonomy_drafter: KnowledgeTaxonomyDrafter | None = None,
        taxonomy_governor: LLMKnowledgeTagGovernor | None = None,
        tag_deduplicator: LLMKnowledgeTagDeduplicator | None = None,
        tag_assigner: KnowledgeTagAssigner | None = None,
    ):
        self.store = store
        self.vector_store = vector_store
        self.settings = settings
        self.taxonomy_drafter = taxonomy_drafter or LLMKnowledgeTaxonomyDrafter(_require_settings(settings))
        self.taxonomy_governor = taxonomy_governor or LLMKnowledgeTagGovernor(_require_settings(settings), vector_store)
        self.tag_deduplicator = tag_deduplicator
        self.tag_assigner = tag_assigner or LLMKnowledgeTagAssigner(_require_settings(settings))

    def run(self, records: list[KnowledgeRecord]) -> KnowledgeConsolidationState:
        return self._build_graph().invoke({"records": records})

    def _build_graph(self):
        graph = StateGraph(KnowledgeConsolidationState)
        graph.add_node("init", self._init)
        graph.add_node("taxonomy_draft", self._taxonomy_draft)
        graph.add_node("taxonomy_governance", self._taxonomy_governance)
        graph.add_node("tag_dedup", self._tag_dedup)
        graph.add_node("tag_assignment", self._tag_assignment)
        graph.add_node("tag_coverage_check", self._tag_coverage_check)
        graph.add_node("persist", self._persist)

        graph.add_edge(START, "init")
        graph.add_conditional_edges(
            "init",
            self._route_after_init,
            {"cold_start": "taxonomy_draft", "warm_start": "tag_assignment"},
        )
        graph.add_edge("taxonomy_draft", "taxonomy_governance")
        graph.add_edge("taxonomy_governance", "tag_dedup")
        graph.add_edge("tag_dedup", "tag_assignment")
        graph.add_conditional_edges(
            "tag_assignment",
            self._route_after_assignment,
            {"coverage": "tag_coverage_check", "persist": "persist"},
        )
        graph.add_conditional_edges(
            "tag_coverage_check",
            self._route_after_coverage,
            {"taxonomy_draft": "taxonomy_draft", "persist": "persist"},
        )
        graph.add_edge("persist", END)
        return graph.compile()

    def _route_after_init(self, state: KnowledgeConsolidationState) -> str:
        return "warm_start" if state["active_tags"] else "cold_start"

    def _route_after_assignment(self, state: KnowledgeConsolidationState) -> str:
        # always run coverage check on first assignment pass (cold or warm start)
        return "coverage" if not state.get("coverage_checked") else "persist"

    def _route_after_coverage(self, state: KnowledgeConsolidationState) -> str:
        retry_count = state.get("coverage_retry_count", 0)
        if state.get("uncovered_records") and retry_count < MAX_COVERAGE_RETRIES:
            return "taxonomy_draft"
        if state.get("uncovered_records"):
            _progress(f"coverage_check max retries reached, {len(state['uncovered_records'])} records marked no_tag")
        return "persist"

    def _init(self, state: KnowledgeConsolidationState) -> KnowledgeConsolidationState:
        active_tags = self.store.list_active_knowledge_tags()
        return {**state, "active_tags": active_tags, "is_warm_start": bool(active_tags), "coverage_checked": False, "coverage_retry_count": 0}

    def _taxonomy_draft(self, state: KnowledgeConsolidationState) -> KnowledgeConsolidationState:
        records = state.get("uncovered_records") or state["records"]
        active_tags = state["active_tags"]
        if not records:
            return {**state, "batch_proposals": []}
        batches = [records[index : index + KNOWLEDGE_TAXONOMY_BATCH_SIZE] for index in range(0, len(records), KNOWLEDGE_TAXONOMY_BATCH_SIZE)]
        draft_results: list[KnowledgeTaxonomyDraftResult | None] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=MAX_DRAFT_WORKERS) as executor:
            future_to_index = {
                executor.submit(self.taxonomy_drafter.draft_batch, batch, active_tags): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(future_to_index):
                draft_results[future_to_index[future]] = future.result()

        proposals_by_name: dict[str, KnowledgeTagProposalDraft] = {}
        for result in draft_results:
            if result is None:
                continue
            for proposal in result.proposals:
                name = normalize_tag_name(proposal.name)
                if name in {tag.name for tag in active_tags}:
                    continue
                existing = proposals_by_name.get(name)
                if existing is None:
                    existing = KnowledgeTagProposalDraft(name=name, definition=proposal.definition)
                    proposals_by_name[name] = existing
                existing.supporting_record_ids.extend(proposal.supporting_record_ids)

        record_ids = {record.id for record in state["records"]}
        proposals: list[KnowledgeTagProposalDraft] = []
        for proposal in proposals_by_name.values():
            query = f"{proposal.name}\n{proposal.definition}"
            similar_records = self.vector_store.similarity_search_knowledge_records(query, k=5)
            supporting_ids = list(dict.fromkeys(
                [record_id for record_id in proposal.supporting_record_ids if record_id in record_ids]
                + [record.id for record in similar_records if record.id in record_ids]
            ))
            if supporting_ids:
                proposals.append(proposal.model_copy(update={"supporting_record_ids": supporting_ids}))
        return {**state, "batch_proposals": proposals}

    def _taxonomy_governance(self, state: KnowledgeConsolidationState) -> KnowledgeConsolidationState:
        proposals = state.get("batch_proposals", [])
        active_tags = state["active_tags"]
        if not proposals:
            return {**state, "active_tags": active_tags}
        governance = self.taxonomy_governor.govern(active_tags, proposals)
        _validate_governance(governance, proposals, active_tags)
        tag_by_name = {tag.name: tag for tag in active_tags}
        proposal_by_name = {normalize_tag_name(proposal.name): proposal for proposal in proposals}
        for decision in governance.decisions:
            if decision.decision == "accept":
                draft = decision.accepted_tag or proposal_by_name[normalize_tag_name(decision.proposal_name)]
                tag = _tag_from_draft(draft)
                self.store.upsert_knowledge_tag(tag)
                tag_by_name[tag.name] = tag
        active_tags = list(tag_by_name.values())
        return {**state, "active_tags": active_tags}

    def _tag_dedup(self, state: KnowledgeConsolidationState) -> KnowledgeConsolidationState:
        if self.tag_deduplicator is None or len(state["active_tags"]) < 2:
            return state
        result = self.tag_deduplicator.deduplicate(state["active_tags"])
        tag_by_name = {tag.name: tag for tag in state["active_tags"]}
        merged_names: set[str] = set()
        for decision in result.merges:
            keep = tag_by_name[normalize_tag_name(decision.keep)]
            merge = tag_by_name[normalize_tag_name(decision.merge)]
            self.store.merge_knowledge_tag(keep.tag_id, merge.tag_id)
            merged_names.add(merge.name)
        active_tags = [tag for tag in state["active_tags"] if tag.name not in merged_names]
        return {**state, "active_tags": active_tags}

    def _tag_assignment(self, state: KnowledgeConsolidationState) -> KnowledgeConsolidationState:
        # supplemental pass: only assign uncovered records; otherwise assign all records
        records = state.get("uncovered_records") or state["records"]
        active_tags = state["active_tags"]
        if not records or not active_tags:
            return {**state}
        new_assignments: list[KnowledgeTagAssignmentDraft | None] = [None] * len(records)
        with ThreadPoolExecutor(max_workers=MAX_ASSIGNMENT_WORKERS) as executor:
            future_to_index = {
                executor.submit(self.tag_assigner.assign_one, record, active_tags): index
                for index, record in enumerate(records)
            }
            for future in as_completed(future_to_index):
                new_assignments[future_to_index[future]] = future.result()
        valid_new = [
            _validated_assignment(assignment, records, active_tags)
            for assignment in new_assignments
            if assignment is not None
        ]
        # accumulate assignments across passes
        existing = {a.record_id: a for a in state.get("assignments", [])}
        for a in valid_new:
            existing[a.record_id] = a
        return {**state, "assignments": list(existing.values())}

    def _tag_coverage_check(self, state: KnowledgeConsolidationState) -> KnowledgeConsolidationState:
        # uncovered = no assignment OR assignment with empty tag_names
        assigned_with_tags = {
            a.record_id for a in state.get("assignments", []) if a.tag_names
        }
        uncovered = [r for r in state["records"] if r.id not in assigned_with_tags]
        retry_count = state.get("coverage_retry_count", 0) + 1
        if uncovered:
            _progress(f"tag_coverage_check uncovered={len(uncovered)}/{len(state['records'])} retry={retry_count}/{MAX_COVERAGE_RETRIES}")
        return {
            **state,
            "uncovered_records": uncovered,
            "coverage_checked": True,
            "coverage_retry_count": retry_count,
            "is_warm_start": False if uncovered else state.get("is_warm_start", False),
        }

    def _persist(self, state: KnowledgeConsolidationState) -> KnowledgeConsolidationState:
        tag_by_name = {tag.name: tag for tag in state["active_tags"]}
        assigned_ids: set[str] = set()
        for assignment in state.get("assignments", []):
            if assignment.tag_names:
                for tag_name in assignment.tag_names:
                    tag = tag_by_name[normalize_tag_name(tag_name)]
                    self.store.upsert_knowledge_tag_assignment(assignment.record_id, tag.tag_id, assignment.reasoning)
                self.store.set_knowledge_consolidation_status(assignment.record_id, "assigned")
                assigned_ids.add(assignment.record_id)
        # mark uncovered records as no_tag
        for record in state["records"]:
            if record.id not in assigned_ids:
                self.store.set_knowledge_consolidation_status(record.id, "no_tag")
        no_tag_count = len(state["records"]) - len(assigned_ids)
        if no_tag_count:
            _progress(f"persist no_tag={no_tag_count} records marked for future coverage")
        return state


def run_knowledge_consolidation(
    records: list[KnowledgeRecord],
    store: KnowledgeStore,
    vector_store: KnowledgeVectorStore,
    settings: Settings,
) -> KnowledgeConsolidationState:
    service = KnowledgeConsolidationService(
        store=store,
        vector_store=vector_store,
        settings=settings,
        tag_deduplicator=LLMKnowledgeTagDeduplicator(settings),
    )
    return service.run(records)


def _validate_governance(
    result: KnowledgeTaxonomyGovernanceResult,
    proposals: list[KnowledgeTagProposalDraft],
    active_tags: list[KnowledgeTag],
) -> None:
    proposal_names = {normalize_tag_name(proposal.name) for proposal in proposals}
    decision_names = [normalize_tag_name(decision.proposal_name) for decision in result.decisions]
    if set(decision_names) != proposal_names or len(decision_names) != len(set(decision_names)):
        raise ValueError("Knowledge taxonomy governance must decide every proposal exactly once.")
    active_names = {tag.name for tag in active_tags}
    accepted_names = {
        normalize_tag_name(decision.accepted_tag.name)
        for decision in result.decisions
        if decision.decision == "accept" and decision.accepted_tag is not None
    }
    for decision in result.decisions:
        if decision.decision not in {"accept", "merge", "reject"}:
            raise ValueError(f"Unknown knowledge taxonomy governance decision: {decision.decision}")
        if decision.decision == "accept" and decision.accepted_tag is None:
            raise ValueError(f"Accept decision missing accepted_tag: {decision.proposal_name}")
        if decision.decision == "merge" and normalize_tag_name(decision.target_tag_name or "") not in active_names | accepted_names:
            raise ValueError(f"Merge target is not active or accepted: {decision.target_tag_name}")


def _validated_assignment(
    assignment: KnowledgeTagAssignmentDraft,
    records: list[KnowledgeRecord],
    active_tags: list[KnowledgeTag],
) -> KnowledgeTagAssignmentDraft:
    record_ids = {record.id for record in records}
    if assignment.record_id not in record_ids:
        raise ValueError(f"Knowledge tag assignment references unknown record_id: {assignment.record_id}")
    active_names = {tag.name for tag in active_tags}
    normalized_names = [normalize_tag_name(name) for name in assignment.tag_names]
    valid_names = [name for name in normalized_names if name in active_names]
    if not valid_names:
        _progress(f"knowledge_tag_assignment record={assignment.record_id} all tag_names unknown={normalized_names}, skipping")
        return assignment.model_copy(update={"tag_names": []})
    return assignment.model_copy(update={"tag_names": list(dict.fromkeys(valid_names))})


def _tag_from_draft(draft: KnowledgeTagProposalDraft) -> KnowledgeTag:
    now = utc_now()
    name = normalize_tag_name(draft.name)
    definition = draft.definition.strip()
    return KnowledgeTag(
        tag_id=_knowledge_tag_id(name, definition),
        name=name,
        definition=definition,
        status="active",
        created_at=now,
        updated_at=now,
    )


def _knowledge_tag_id(name: str, definition: str) -> str:
    key = f"{normalize_tag_name(name)}|{normalize_text(definition)}"
    return f"knowledge_tag_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def _record_view(record: KnowledgeRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "title": record.title,
        "insight": record.insight,
        "applicability": record.applicability,
        "scope": record.scope.value,
    }


def _tag_view(tag: KnowledgeTag) -> dict[str, str]:
    return {"name": tag.name, "definition": tag.definition}


def _mechanism_tag_view(tag: KnowledgeTag) -> MechanismTag:
    return MechanismTag(
        tag_id=tag.tag_id,
        name=tag.name,
        definition=tag.definition,
        status=MechanismTagStatus.ACTIVE,
        created_at=tag.created_at,
        updated_at=tag.updated_at,
    )


def _require_settings(settings: Settings | None) -> Settings:
    if settings is None:
        raise ValueError("settings are required when no fake knowledge consolidation component is provided.")
    return settings


def _progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[{timestamp}] {message}", flush=True)


def _retry_message(raw_result: object, attempt: int) -> "HumanMessage":
    from langchain_core.messages import HumanMessage
    parts = [f"Attempt {attempt} did not produce valid structured output."]
    if isinstance(raw_result, dict):
        raw = raw_result.get("raw")
        parsing_error = raw_result.get("parsing_error")
        if raw is not None:
            content = getattr(raw, "content", None)
            if content:
                parts.append(f"Your output was: {str(content)[:300]}")
            invalid = getattr(raw, "invalid_tool_calls", None)
            if invalid:
                import json as _json
                parts.append(f"Invalid tool-call args: {_json.dumps(invalid, default=str)[:400]}")
        if parsing_error is not None:
            parts.append(f"Parse error: {str(parsing_error)[:200]}")
    parts.append("Please retry with a valid structured response.")
    return HumanMessage(content="\n".join(parts))


def _error_retry_message(error: str, attempt: int) -> "HumanMessage":
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=f"Attempt {attempt} raised an error: {error[:300]}\nPlease retry with a valid structured response.")
