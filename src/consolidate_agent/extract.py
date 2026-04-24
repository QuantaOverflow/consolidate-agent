from __future__ import annotations

import hashlib
import json
from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.types import (
    AdmissionStatus,
    PitfallCandidate,
    PitfallCategory,
    PitfallScope,
    Transcript,
    TranscriptChunk,
)


class PitfallCandidateInput(BaseModel):
    title: str = Field(min_length=3)
    category: PitfallCategory
    trigger: str = Field(min_length=8)
    failure_mode: str = Field(min_length=8)
    impact: str = Field(min_length=3)
    preventive_rule: str = Field(min_length=8)
    scope: PitfallScope
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class StructuredExtractionOutput(BaseModel):
    pitfalls: list[PitfallCandidateInput] = Field(default_factory=list)


class PitfallExtractor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self._build_model()
        self.structured_model = self.model.with_structured_output(StructuredExtractionOutput)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    dedent(
                        """
                        You extract reusable pitfall candidates from a bounded chunk of a normalized Codex transcript.
                        Only extract pitfalls in these categories: execution_strategy, tooling_environment.
                        Do not summarize the transcript.
                        Only return reusable pitfalls supported by evidence refs from this chunk.
                        Evidence refs must use the original message ref values exactly as provided.
                        Every pitfall must include trigger, failure_mode, impact, preventive_rule, scope, evidence_refs, confidence.
                        Scope must be one of: global, project_specific, session_specific.
                        If unsure, return fewer pitfalls.
                        Ignore developer instructions, general summaries, and one-off noise.
                        """
                    ).strip(),
                ),
                (
                    "user",
                    "Transcript chunk JSON:\n{chunk_json}\n\nReturn pitfall candidates for this bounded chunk only.",
                ),
            ]
        )

    def _build_model(self) -> ChatQwen:
        if not self.settings.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is not configured. Fill .env before running real extraction."
            )
        return ChatQwen(
            model=self.settings.qwen_model,
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_api_base,
            temperature=0,
        )

    def extract(self, transcript: Transcript) -> list[PitfallCandidate]:
        chunk = TranscriptChunk(
            session_id=transcript.session_id,
            chunk_id=f"{transcript.session_id}:chunk:0000",
            chunk_index=0,
            total_chunks=1,
            messages=transcript.messages,
            char_count=len(json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False)),
        )
        return self.extract_chunk(chunk)

    def extract_chunk(self, chunk: TranscriptChunk) -> list[PitfallCandidate]:
        prompt_value = self.prompt.invoke(
            {
                "chunk_json": json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, indent=2)
            }
        )
        result = self.structured_model.invoke(prompt_value)
        candidates: list[PitfallCandidate] = []
        for item in result.pitfalls:
            candidate_id = self._candidate_id(chunk.session_id, chunk.chunk_id, item.title, item.category)
            candidates.append(
                PitfallCandidate(
                    candidate_id=candidate_id,
                    session_id=chunk.session_id,
                    title=item.title,
                    category=item.category,
                    trigger=item.trigger,
                    failure_mode=item.failure_mode,
                    impact=item.impact,
                    preventive_rule=item.preventive_rule,
                    scope=item.scope,
                    evidence_refs=item.evidence_refs,
                    confidence=item.confidence,
                    admission_status=AdmissionStatus.PENDING,
                    chunk_id=chunk.chunk_id,
                )
            )
        return candidates

    def _candidate_id(self, session_id: str, chunk_id: str, title: str, category: PitfallCategory) -> str:
        raw = f"{session_id}:{chunk_id}:{category.value}:{title}".encode("utf-8")
        digest = hashlib.sha1(raw).hexdigest()[:12]
        return f"candidate_{digest}"
