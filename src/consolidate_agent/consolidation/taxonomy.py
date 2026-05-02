from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model, _invoke_structured, _invoke_with_retry, _token_overlap
from consolidate_agent.observability import AgentInvocationRecorder, NullAgentInvocationRecorder
from consolidate_agent.prompt_loader import load_prompt
from consolidate_agent.types import (
    CanonicalKnowledge,
    MechanismTag,
    MechanismTagStatus,
    TagProposal,
    TagProposalDecision,
    normalize_text,
    utc_now,
)

if TYPE_CHECKING:
    from consolidate_agent.knowledge.vector import KnowledgeVectorStore

MAX_CONCURRENT_PROPOSALS = 5
TOKEN_OVERLAP_TAG_MIN_SCORE = 2
TOKEN_OVERLAP_PROPOSAL_MIN_SCORE = 3


class TaxonomyRuleView(BaseModel):
    title: str
    preventive_rule: str


class MechanismTagDraft(BaseModel):
    name: str
    definition: str
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)


class TagProposalDraft(BaseModel):
    name: str
    definition: str
    supporting_canonical_ids: list[str] = Field(default_factory=list)
    nearest_existing_tag_names: list[str] = Field(default_factory=list)
    difference_from_existing: str
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)


class TaxonomyDraftResult(BaseModel):
    proposals: list[TagProposalDraft] = Field(default_factory=list)


class TaxonomyGovernanceDecision(BaseModel):
    proposal_name: str
    decision: str
    target_tag_name: str | None = None
    decision_reason: str
    accepted_tag: MechanismTagDraft | None = None


class TaxonomyGovernanceResult(BaseModel):
    decisions: list[TaxonomyGovernanceDecision] = Field(default_factory=list)


class TagMergeDecision(BaseModel):
    keep: str
    merge: str
    reason: str


class TagDeduplicationResult(BaseModel):
    merges: list[TagMergeDecision] = Field(default_factory=list)


class TaxonomyValidationError(ValueError):
    def __init__(self, message: str, *, validation_error: str, raw_result: BaseModel | None = None, retry_result: BaseModel | None = None):
        super().__init__(message)
        self.validation_error = validation_error
        self.raw_result = raw_result
        self.retry_result = retry_result

    def payload(self) -> dict[str, object]:
        return {
            "validation_error": self.validation_error,
            "raw_result": self.raw_result.model_dump(mode="json") if self.raw_result else None,
            "retry_result": self.retry_result.model_dump(mode="json") if self.retry_result else None,
        }


class DeterministicTaxonomyDrafter:
    def draft(self, canonicals: list[CanonicalKnowledge], active_tags: list[MechanismTag]) -> TaxonomyDraftResult:
        active_names = {tag.name for tag in active_tags}
        proposals: dict[str, TagProposalDraft] = {}
        for canonical in canonicals:
            name = _infer_tag_name(canonical)
            if name in active_names:
                continue
            proposal = proposals.get(name)
            if proposal is None:
                proposal = TagProposalDraft(
                    name=name,
                    definition=f"Failures prevented by controlling {name.replace('_', ' ')}.",
                    supporting_canonical_ids=[],
                    nearest_existing_tag_names=[tag.name for tag in active_tags[:3]],
                    difference_from_existing="No active tag covers this prevention control.",
                    positive_examples=[canonical.title],
                    negative_examples=[],
                )
                proposals[name] = proposal
            proposal.supporting_canonical_ids.append(canonical.canonical_id)
        return TaxonomyDraftResult(proposals=list(proposals.values()))


class LLMTaxonomyDrafter:
    def __init__(self, settings: Settings, recorder: AgentInvocationRecorder | None = None, run_id: str | None = None):
        self.recorder = recorder or NullAgentInvocationRecorder()
        self.run_id = run_id
        self.structured_model = _chat_model(settings).with_structured_output(TaxonomyDraftResult)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("taxonomy_draft_system.md")), ("user", load_prompt("taxonomy_draft_user.md"))],
            template_format="jinja2",
        )
        self.retry_prompt = ChatPromptTemplate.from_messages(
            [("system", load_prompt("taxonomy_draft_retry_system.md")), ("user", load_prompt("taxonomy_draft_retry_user.md"))],
            template_format="jinja2",
        )

    def draft(self, canonicals: list[CanonicalKnowledge], active_tags: list[MechanismTag]) -> TaxonomyDraftResult:
        values = _taxonomy_values(canonicals, active_tags)
        return _invoke_with_retry(
            stage="taxonomy_draft",
            structured_model=self.structured_model,
            prompt=self.prompt,
            retry_prompt=self.retry_prompt,
            values=values,
            recorder=self.recorder,
            run_id=self.run_id,
            validator=lambda result: validate_taxonomy_draft(result, canonicals, active_tags),
            empty_error="Taxonomy draft returned no structured result.",
        )


class LLMTagDeduplicator:
    def __init__(self, settings: Settings, recorder: AgentInvocationRecorder | None = None, run_id: str | None = None):
        self.recorder = recorder or NullAgentInvocationRecorder()
        self.run_id = run_id
        self.structured_model = _chat_model(settings).with_structured_output(TagDeduplicationResult)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You identify mechanism tags whose failure patterns functionally overlap. "
                    "Two tags overlap when a developer making the same mistake could be described by both — "
                    "even if their framing or scope differs. "
                    "The keep tag should be the more general or precise one. "
                    "Each tag name in merges must appear exactly once across all merge decisions.",
                ),
                (
                    "user",
                    "Review the full active tag list below. Each item has a name and definition.\n\n"
                    "{{ tags_json }}\n\n"
                    "Return a merges list where each entry has keep, merge, and reason. "
                    "A tag may appear as merge in at most one decision. "
                    "If no tags functionally overlap, return an empty merges list.",
                ),
            ],
            template_format="jinja2",
        )
        self.retry_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You identify mechanism tags whose failure patterns functionally overlap. "
                    "Two tags overlap when a developer making the same mistake could be described by both. "
                    "Each tag name in merges must appear exactly once across all merge decisions.",
                ),
                (
                    "user",
                    "Review the full active tag list below.\n\n"
                    "{{ tags_json }}\n\n"
                    "Your previous attempt failed validation:\n{{ validation_error }}\n"
                    "Previous result:\n{{ invalid_result_json }}\n\n"
                    "Fix the conflicts and return a corrected merges list. "
                    "A tag may appear as merge in at most one decision.",
                ),
            ],
            template_format="jinja2",
        )

    def deduplicate(self, tags: list[MechanismTag]) -> TagDeduplicationResult:
        values = {
            "tags_json": json.dumps(
                [{"name": tag.name, "definition": tag.definition} for tag in tags],
                ensure_ascii=False,
                indent=2,
            )
        }
        return _invoke_with_retry(
            stage="tag_dedup",
            structured_model=self.structured_model,
            prompt=self.prompt,
            retry_prompt=self.retry_prompt,
            values=values,
            recorder=self.recorder,
            run_id=self.run_id,
            validator=lambda result: validate_tag_deduplication(result, tags),
            empty_error="Tag deduplication returned no structured result.",
        )


class DeterministicTaxonomyGovernor:
    def govern(
        self,
        active_tags: list[MechanismTag],
        proposals: list[TagProposalDraft],
        canonicals: list[CanonicalKnowledge],
    ) -> TaxonomyGovernanceResult:
        decisions = []
        for proposal in proposals:
            target = _best_tag_for_proposal(proposal, active_tags)
            if target is not None:
                decisions.append(
                    TaxonomyGovernanceDecision(
                        proposal_name=proposal.name,
                        decision="merge",
                        target_tag_name=target.name,
                        decision_reason=f"Proposal overlaps active tag '{target.name}'.",
                    )
                )
                continue
            decisions.append(
                TaxonomyGovernanceDecision(
                    proposal_name=proposal.name,
                    decision="accept",
                    target_tag_name=None,
                    decision_reason="Proposal represents a distinct prevention control.",
                    accepted_tag=MechanismTagDraft(
                        name=proposal.name,
                        definition=proposal.definition,
                        positive_examples=proposal.positive_examples,
                        negative_examples=proposal.negative_examples,
                    ),
                )
            )
        return TaxonomyGovernanceResult(decisions=decisions)


class LLMAgentTagGovernor:
    def __init__(
        self,
        settings: Settings,
        vector_store: KnowledgeVectorStore,
        recorder: AgentInvocationRecorder | None = None,
        run_id: str | None = None,
    ):
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        self.llm = _chat_model(settings)
        self.vector_store = vector_store
        self.recorder = recorder or NullAgentInvocationRecorder()
        self.run_id = run_id

        self._structured_model = self.llm.with_structured_output(TaxonomyGovernanceDecision)

    def govern(
        self,
        active_tags: list[MechanismTag],
        proposals: list[TagProposalDraft],
        canonicals: list[CanonicalKnowledge],
    ) -> TaxonomyGovernanceResult:
        decisions: list[TaxonomyGovernanceDecision | None] = [None] * len(proposals)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROPOSALS) as executor:
            future_to_index = {
                executor.submit(self._decide_one, proposal, active_tags): index
                for index, proposal in enumerate(proposals)
            }
            for future in as_completed(future_to_index):
                decisions[future_to_index[future]] = future.result()
        ordered_decisions = [decision for decision in decisions if decision is not None]
        result = TaxonomyGovernanceResult(decisions=ordered_decisions)
        validate_taxonomy_governance(result, proposals, active_tags)
        return result

    def _decide_one(self, proposal: TagProposalDraft, active_tags: list[MechanismTag]) -> TaxonomyGovernanceDecision:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        @tool
        def search_canonical_rules(query: str) -> str:
            """Search canonical rules semantically relevant to a query."""
            results = self.vector_store.similarity_search_canonicals(query, k=5)
            if not results:
                return "No relevant canonical rules found."
            return "\n\n".join(f"- {canonical.title}: {canonical.preventive_rule}" for canonical in results)

        agent = create_react_agent(self.llm, [search_canonical_rules])
        active_tags_text = "\n".join(f"- {tag.name}: {tag.definition}" for tag in active_tags) or "None yet."
        user_message = f"""Decide whether to accept, merge, or reject this tag proposal.

Proposal:
  name: {proposal.name}
  definition: {proposal.definition}
  difference_from_existing: {proposal.difference_from_existing}

Active tags:
{active_tags_text}

Use search_canonical_rules to find evidence if needed.
Summarize your reasoning and final decision (accept, merge, or reject)."""

        agent_result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
        from langchain_core.messages import AIMessage as _AIMessage
        agent_conclusions = [
            m.content for m in agent_result["messages"]
            if isinstance(m, _AIMessage) and getattr(m, "content", None)
        ]
        agent_summary = "\n".join(agent_conclusions) or "No conclusion from search."

        extract_prompt = ChatPromptTemplate.from_messages([
            ("system",
                "Extract a structured governance decision based on the proposal and research summary below. "
                f"proposal_name must be exactly \"{proposal.name}\". "
                "decision must be one of: accept, merge, reject. "
                "merge requires target_tag_name set to an existing active tag name. "
                "accept requires accepted_tag with name, definition, positive_examples, negative_examples."),
            ("human",
                f"Proposal name: {proposal.name}\n"
                f"Proposal definition: {proposal.definition}\n"
                f"Active tags: {active_tags_text}\n\n"
                f"Research summary:\n{agent_summary}"),
        ], template_format="jinja2")
        decision = None
        for _ in range(3):
            decision = self._structured_model.invoke(extract_prompt.invoke({}))
            if decision is not None:
                break
        if decision is None:
            raise ValueError(f"LLMAgentTagGovernor failed to extract structured decision for proposal '{proposal.name}'")
        decision.proposal_name = proposal.name
        return decision


def validate_taxonomy_draft(result: TaxonomyDraftResult, canonicals: list[CanonicalKnowledge], active_tags: list[MechanismTag]) -> None:
    canonical_ids = {canonical.canonical_id for canonical in canonicals}
    active_names = {normalize_tag_name(tag.name) for tag in active_tags}
    names = [normalize_tag_name(proposal.name) for proposal in result.proposals]
    if len(names) != len(set(names)):
        raise ValueError("Taxonomy draft proposal names must be unique.")
    for proposal in result.proposals:
        name = normalize_tag_name(proposal.name)
        if name in active_names:
            raise ValueError(f"Taxonomy draft duplicates active tag: {proposal.name}")
        unknown_ids = set(proposal.supporting_canonical_ids) - canonical_ids
        if unknown_ids:
            proposal.supporting_canonical_ids = [i for i in proposal.supporting_canonical_ids if i in canonical_ids]


def validate_taxonomy_governance(result: TaxonomyGovernanceResult, proposals: list[TagProposalDraft], active_tags: list[MechanismTag]) -> None:
    proposal_names = {normalize_tag_name(proposal.name) for proposal in proposals}
    decision_names = [normalize_tag_name(decision.proposal_name) for decision in result.decisions]
    if set(decision_names) != proposal_names or len(decision_names) != len(set(decision_names)):
        raise ValueError("Taxonomy governance must decide every proposal exactly once.")
    active_names = {normalize_tag_name(tag.name) for tag in active_tags}
    accepted_names = {
        normalize_tag_name(decision.accepted_tag.name)
        for decision in result.decisions
        if decision.decision == "accept" and decision.accepted_tag
    }
    for decision in result.decisions:
        if decision.decision not in {"accept", "merge", "reject"}:
            raise ValueError(f"Unknown taxonomy governance decision: {decision.decision}")
        if decision.decision == "accept" and decision.accepted_tag is None:
            raise ValueError(f"Accept decision missing accepted_tag: {decision.proposal_name}")
        if decision.decision == "merge" and normalize_tag_name(decision.target_tag_name or "") not in active_names | accepted_names:
            raise ValueError(f"Merge target is not active or accepted: {decision.target_tag_name}")


def validate_tag_deduplication(result: TagDeduplicationResult, tags: list[MechanismTag]) -> None:
    tag_names = {normalize_tag_name(tag.name) for tag in tags}
    merge_targets: dict[str, str] = {}
    for decision in result.merges:
        keep = normalize_tag_name(decision.keep)
        merge = normalize_tag_name(decision.merge)
        if keep not in tag_names:
            raise ValueError(f"Tag deduplication keep tag is not active: {decision.keep}")
        if merge not in tag_names:
            raise ValueError(f"Tag deduplication merge tag is not active: {decision.merge}")
        if keep == merge:
            raise ValueError(f"Tag deduplication cannot merge a tag into itself: {decision.merge}")
        existing = merge_targets.get(merge)
        if existing is not None and existing != keep:
            raise ValueError(f"Tag deduplication merge target is ambiguous: {decision.merge}")
        merge_targets[merge] = keep

    for merge in merge_targets:
        seen: set[str] = set()
        current = merge
        while current in merge_targets:
            if current in seen:
                raise ValueError("Tag deduplication contains a merge cycle.")
            seen.add(current)
            current = merge_targets[current]


def tag_from_draft(draft: MechanismTagDraft) -> MechanismTag:
    now = utc_now()
    name = normalize_tag_name(draft.name)
    definition = draft.definition.strip()
    return MechanismTag(
        tag_id=tag_id(name, definition),
        name=name,
        definition=definition,
        status=MechanismTagStatus.ACTIVE,
        positive_examples=draft.positive_examples,
        negative_examples=draft.negative_examples,
        created_at=now,
        updated_at=now,
    )


def proposal_from_draft(
    draft: TagProposalDraft,
    decision: TagProposalDecision = TagProposalDecision.PROPOSED,
    target_tag_id: str | None = None,
    decision_reason: str | None = None,
) -> TagProposal:
    now = utc_now()
    name = normalize_tag_name(draft.name)
    definition = draft.definition.strip()
    return TagProposal(
        proposal_id=proposal_id(name, definition, draft.supporting_canonical_ids),
        name=name,
        definition=definition,
        supporting_canonical_ids=sorted(draft.supporting_canonical_ids),
        nearest_existing_tag_ids=[],
        difference_from_existing=draft.difference_from_existing,
        decision=decision,
        target_tag_id=target_tag_id,
        decision_reason=decision_reason,
        created_at=now,
        updated_at=now,
    )


def tag_id(name: str, definition: str) -> str:
    key = f"{normalize_tag_name(name)}|{normalize_text(definition)}"
    return f"tag_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def proposal_id(name: str, definition: str, supporting_ids: list[str]) -> str:
    key = f"{normalize_tag_name(name)}|{normalize_text(definition)}|{'|'.join(sorted(supporting_ids))}"
    return f"proposal_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def normalize_tag_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", value).strip("_") or "general_failure_mechanism"


def _taxonomy_values(canonicals: list[CanonicalKnowledge], active_tags: list[MechanismTag]) -> dict[str, str]:
    return {
        "tags_json": json.dumps([tag.model_dump(mode="json") for tag in active_tags], ensure_ascii=False, indent=2),
        "rules_json": json.dumps(
            [TaxonomyRuleView(title=c.title, preventive_rule=c.preventive_rule).model_dump(mode="json") for c in canonicals],
            ensure_ascii=False,
            indent=2,
        ),
    }


def _governance_values(canonicals: list[CanonicalKnowledge], active_tags: list[MechanismTag], proposals: list[TagProposalDraft]) -> dict[str, str]:
    values = _taxonomy_values(canonicals, active_tags)
    values["proposals_json"] = json.dumps([proposal.model_dump(mode="json") for proposal in proposals], ensure_ascii=False, indent=2)
    return values


def _best_existing_tag(canonical: CanonicalKnowledge, tags: list[MechanismTag]) -> MechanismTag | None:
    title = normalize_text(canonical.title)
    rule = normalize_text(canonical.preventive_rule)
    best_tag = None
    best_score = 0
    for tag in tags:
        tag_text = normalize_text(f"{tag.name} {tag.definition}")
        score = _token_overlap(title, tag_text) + _token_overlap(rule, tag_text)
        if score > best_score:
            best_score = score
            best_tag = tag
    return best_tag if best_score >= TOKEN_OVERLAP_TAG_MIN_SCORE else None


def _best_tag_for_proposal(proposal: TagProposalDraft, tags: list[MechanismTag]) -> MechanismTag | None:
    proposal_text = normalize_text(f"{proposal.name} {proposal.definition}")
    best_tag = None
    best_score = 0
    for tag in tags:
        tag_text = normalize_text(f"{tag.name} {tag.definition}")
        score = _token_overlap(proposal_text, tag_text)
        if score > best_score:
            best_score = score
            best_tag = tag
    return best_tag if best_score >= TOKEN_OVERLAP_PROPOSAL_MIN_SCORE else None


def _infer_tag_name(canonical: CanonicalKnowledge) -> str:
    text = normalize_text(f"{canonical.title} {canonical.preventive_rule}")
    if any(token in text for token in ["secret", "password", "credential", "api key", "token"]):
        return "secret_handling"
    if any(token in text for token in ["json", "regex", "schema", "parse", "structured", "payload"]):
        return "structured_input_validation"
    if any(token in text for token in ["config", "env", "path", "dependency", "pytest", "prompt"]):
        return "configuration_loading"
    if any(token in text for token in ["timeout", "fallback", "retry"]):
        return "timeout_fallback"
    if any(token in text for token in ["state", "continue", "route", "workflow", "memory"]):
        return "state_propagation"
    if any(token in text for token in ["api", "contract", "schema", "response", "protocol", "client"]):
        return "interface_contract_validation"
    if any(token in text for token in ["import", "startup", "initialization", "network side effect"]):
        return "runtime_dependency_isolation"
    if any(token in text for token in ["sandbox", "permission", "filesystem", "network access"]):
        return "sandbox_constraint"
    return "runtime_readiness_validation"
