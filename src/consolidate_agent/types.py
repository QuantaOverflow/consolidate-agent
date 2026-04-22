from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

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
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class CursorState(BaseModel):
    last_session_path: str | None = None
    updated_at: datetime | None = None


class RunStats(BaseModel):
    discovered_sessions: int = 0
    processed_sessions: int = 0
    candidate_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    library_size: int = 0


class ExtractionOutput(BaseModel):
    pitfalls: list[PitfallCandidate] = Field(default_factory=list)


class PipelineState(BaseModel):
    input_dir: str
    output_dir: str
    cursor_path: str
    session_index_path: str | None = None
    sample_limit: int | None = None
    sessions: list[str] = Field(default_factory=list)
    transcripts: list[Transcript] = Field(default_factory=list)
    candidates: list[PitfallCandidate] = Field(default_factory=list)
    accepted_records: list[PitfallRecord] = Field(default_factory=list)
    rejected_candidates: list[PitfallCandidate] = Field(default_factory=list)
    stats: RunStats = Field(default_factory=RunStats)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


GraphState = dict[str, object]
