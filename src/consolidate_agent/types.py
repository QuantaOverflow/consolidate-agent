from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageKind(str, Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    EVENT = "event"


class PitfallCategory(str, Enum):
    EXECUTION_STRATEGY = "execution_strategy"
    TOOLING_ENVIRONMENT = "tooling_environment"


class PitfallScope(str, Enum):
    GLOBAL = "global"
    PROJECT_SPECIFIC = "project_specific"
    SESSION_SPECIFIC = "session_specific"


class AdmissionStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ProcessedStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class KnowledgeRelation(str, Enum):
    DUPLICATE = "duplicate"
    OVERLAP = "overlap"
    PARENT_CHILD = "parent_child"
    DISTINCT = "distinct"


class CanonicalKnowledgeStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class SourceConsolidationStatus(str, Enum):
    PENDING = "pending"
    CLASSIFIED = "classified"
    LINKED = "linked"
    FAILED = "failed"


class MechanismTagStatus(str, Enum):
    ACTIVE = "active"
    PROPOSED = "proposed"
    REJECTED = "rejected"
    MERGED = "merged"


class TagProposalDecision(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    MERGED = "merged"
    REJECTED = "rejected"


class TranscriptMessage(BaseModel):
    ref: str
    timestamp: datetime
    role: MessageRole
    kind: MessageKind
    text: str
    tool_name: str | None = None


class Transcript(BaseModel):
    session_id: str
    thread_name: str | None = None
    source: str | None = None
    cwd: str | None = None
    started_at: datetime | None = None
    messages: list[TranscriptMessage] = Field(default_factory=list)


class TranscriptChunk(BaseModel):
    session_id: str
    chunk_id: str
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(ge=1)
    messages: list[TranscriptMessage] = Field(default_factory=list)
    char_count: int = Field(ge=0)


class PitfallCandidate(BaseModel):
    candidate_id: str
    session_id: str
    title: str
    category: PitfallCategory
    trigger: str
    failure_mode: str
    impact: str
    preventive_rule: str
    scope: PitfallScope
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    admission_status: AdmissionStatus = AdmissionStatus.PENDING
    chunk_id: str | None = None


class PitfallEvidence(BaseModel):
    session_ids: list[str] = Field(default_factory=list)
    message_refs: list[str] = Field(default_factory=list)


class PitfallRecord(BaseModel):
    id: str
    title: str
    category: PitfallCategory
    trigger: str
    failure_mode: str
    impact: str
    preventive_rule: str
    scope: PitfallScope
    evidence: PitfallEvidence
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime


class CanonicalKnowledge(BaseModel):
    canonical_id: str
    title: str
    category: PitfallCategory
    summary: str
    preventive_rule: str
    scope: PitfallScope
    status: CanonicalKnowledgeStatus = CanonicalKnowledgeStatus.ACTIVE
    source_record_ids: list[str] = Field(default_factory=list)
    support_count: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime


class KnowledgeInstanceLink(BaseModel):
    source_record_id: str
    canonical_id: str
    relation: KnowledgeRelation
    linked_at: datetime


class MechanismTag(BaseModel):
    tag_id: str
    name: str
    definition: str
    status: MechanismTagStatus = MechanismTagStatus.ACTIVE
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    merged_into_tag_id: str | None = None
    created_at: datetime
    updated_at: datetime


class RuleTagAssignment(BaseModel):
    canonical_id: str
    tag_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    assignment_reason: str
    linked_at: datetime


class TagProposal(BaseModel):
    proposal_id: str
    name: str
    definition: str
    supporting_canonical_ids: list[str] = Field(default_factory=list)
    nearest_existing_tag_ids: list[str] = Field(default_factory=list)
    difference_from_existing: str
    decision: TagProposalDecision = TagProposalDecision.PROPOSED
    target_tag_id: str | None = None
    decision_reason: str | None = None
    created_at: datetime
    updated_at: datetime


class ConsolidationStats(BaseModel):
    run_id: str | None = None
    source_records_seen: int = 0
    source_records_linked: int = 0
    canonical_created: int = 0
    canonical_reused: int = 0
    classification_failures: int = 0
    tags_created: int = 0
    tags_merged: int = 0
    tag_proposals_created: int = 0
    rules_tagged: int = 0
    rules_classified: int = 0
    rules_covered_by_existing: int = 0
    rules_skipped_no_tag: int = 0
    taxonomy_governance_failures: int = 0
    rule_classification_failures: int = 0
    agent_invocations: int = 0
    agent_failures: int = 0


class CursorState(BaseModel):
    last_session_path: str | None = None
    updated_at: datetime | None = None


class ProcessedSessionState(BaseModel):
    session_id: str
    path: str
    normalized_hash: str
    status: ProcessedStatus = ProcessedStatus.PENDING
    chunk_count: int = 0
    processed_at: datetime | None = None
    error: str | None = None


class ProcessedIndex(BaseModel):
    sessions: dict[str, ProcessedSessionState] = Field(default_factory=dict)
    updated_at: datetime | None = None


class RunStats(BaseModel):
    discovered_sessions: int = 0
    skipped_sessions: int = 0
    processed_sessions: int = 0
    failed_sessions: int = 0
    chunk_count: int = 0
    candidate_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0


class ExtractionOutput(BaseModel):
    pitfalls: list[PitfallCandidate] = Field(default_factory=list)


class PipelineState(BaseModel):
    input_dir: str
    output_dir: str
    cursor_path: str
    processed_index_path: str
    knowledge_db_path: str = ""
    session_index_path: str | None = None
    sample_limit: int | None = None
    max_chunk_chars: int = 30000
    overlap_messages: int = 5
    sessions: list[str] = Field(default_factory=list)
    transcript_paths: dict[str, str] = Field(default_factory=dict)
    transcript_hashes: dict[str, str] = Field(default_factory=dict)
    transcripts: list[Transcript] = Field(default_factory=list)
    chunks: list[TranscriptChunk] = Field(default_factory=list)
    candidates: list[PitfallCandidate] = Field(default_factory=list)
    accepted_records: list[PitfallRecord] = Field(default_factory=list)
    rejected_candidates: list[PitfallCandidate] = Field(default_factory=list)
    successful_session_ids: list[str] = Field(default_factory=list)
    failed_session_ids: list[str] = Field(default_factory=list)
    processed_index: ProcessedIndex = Field(default_factory=ProcessedIndex)
    stats: RunStats = Field(default_factory=RunStats)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
