from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class KnowledgeScope(str, Enum):
    GLOBAL = "global"
    PROJECT_SPECIFIC = "project_specific"
    SESSION_SPECIFIC = "session_specific"


class ProcessedStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class MechanismTagStatus(str, Enum):
    ACTIVE = "active"
    PROPOSED = "proposed"
    REJECTED = "rejected"
    MERGED = "merged"


class KnowledgeRecord(BaseModel):
    id: str
    session_id: str
    title: str
    insight: str
    applicability: str
    scope: KnowledgeScope
    evidence_turns: list[int] = Field(default_factory=list)
    evidence_count: int = 0
    processed_chars: int
    created_at: datetime
    updated_at: datetime


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


class KnowledgeExtractionStats(BaseModel):
    discovered_sessions: int = 0
    skipped_sessions: int = 0
    skip_reasons: dict[str, int] = Field(default_factory=dict)
    processed_sessions: int = 0
    failed_sessions: int = 0
    extracted_count: int = 0
    admitted_count: int = 0
    rejected_count: int = 0
    evidence_admitted_count: int = 0
    evidence_rejected_count: int = 0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
