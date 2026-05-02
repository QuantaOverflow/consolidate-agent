from __future__ import annotations

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from consolidate_agent.config import Settings
from consolidate_agent.context import ContextStats, ProcessedSession
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.knowledge_extraction import pipeline
from consolidate_agent.knowledge_extraction.extractor import (
    KnowledgeExtractionOutput,
    KnowledgeExtractor,
    KnowledgeItemInput,
)
from consolidate_agent.types import KnowledgeRecord, KnowledgeScope, ProcessedIndex, ProcessedStatus, utc_now


def write_jsonl(path: Path, events: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in events), encoding="utf-8")
    return path


def session_events(session_id: str = "session-1", *, body: str = "x" * 260, source: object = "cli") -> list[dict]:
    return [
        {
            "timestamp": "2026-04-17T09:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "timestamp": "2026-04-17T09:00:00Z",
                "cwd": "/repo",
                "source": source,
            },
        },
        {
            "timestamp": "2026-04-17T09:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"Please analyze this session. {body}"}],
            },
        },
        {
            "timestamp": "2026-04-17T09:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"The answer contains reusable reasoning. {body}"}],
            },
        },
    ]


class FakeStructuredModel:
    def invoke(self, prompt_value: object) -> KnowledgeExtractionOutput:
        return KnowledgeExtractionOutput(
            items=[
                KnowledgeItemInput(
                    title="Structured errors",
                    insight="Tool wrappers should return structured errors instead of raising raw exceptions.",
                    applicability="Applies when designing callable tools in agent pipelines.",
                    evidence_turns=[1, 2],
                    scope=KnowledgeScope.GLOBAL,
                )
            ]
        )


def test_knowledge_extractor_extract_returns_structured_items() -> None:
    extractor = object.__new__(KnowledgeExtractor)
    extractor.prompt = ChatPromptTemplate.from_messages(
        [("user", "Session:\n{{ session_xml }}")],
        template_format="jinja2",
    )
    extractor.structured_model = FakeStructuredModel()
    session = ProcessedSession(
        session_id="session-1",
        cwd="/repo",
        thread_name=None,
        started_at="2026-04-17T09:00:00Z",
        is_sub_agent=False,
        xml="<session><turn index=\"1\"><user>u</user><assistant>a</assistant></turn></session>",
        stats=ContextStats(raw_chars=500, processed_chars=300, compression_ratio=0.4),
    )

    items = extractor.extract(session)

    assert len(items) == 1
    assert items[0].title == "Structured errors"
    assert items[0].scope == KnowledgeScope.GLOBAL
    assert items[0].evidence_turns == [1, 2]


def test_admission_rejects_session_specific_and_empty_evidence() -> None:
    admitted, rejected = pipeline._admit_items(
        [
            KnowledgeItemInput(
                title="Good item",
                insight="A reusable engineering insight with enough substance.",
                applicability="Applies to reusable extraction pipelines.",
                evidence_turns=[1],
                scope=KnowledgeScope.GLOBAL,
            ),
            KnowledgeItemInput(
                title="Local fix",
                insight="This exact repository needs a local one-off fix.",
                applicability="Only this current session and repository need it.",
                evidence_turns=[1],
                scope=KnowledgeScope.SESSION_SPECIFIC,
            ),
            KnowledgeItemInput(
                title="No evidence",
                insight="A plausible insight without traceable evidence turns.",
                applicability="Applies nowhere without evidence support.",
                evidence_turns=[],
                scope=KnowledgeScope.PROJECT_SPECIFIC,
            ),
        ]
    )

    assert [item.title for item in admitted] == ["Good item"]
    assert [item.title for item in rejected] == ["Local fix", "No evidence"]


def test_upsert_knowledge_record_writes_and_is_idempotent(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    now = utc_now()
    record = KnowledgeRecord(
        id="knowledge_abc123",
        session_id="session-1",
        title="Evidence spread",
        insight="Evidence spread helps estimate whether an insight is supported across a session.",
        applicability="Applies when admitting session-level extracted knowledge.",
        scope=KnowledgeScope.PROJECT_SPECIFIC,
        evidence_turns=[1, 3],
        evidence_count=2,
        evidence_spread=1.0,
        processed_chars=500,
        created_at=now,
        updated_at=now,
    )
    try:
        assert store.upsert_knowledge_record(record) is True
        assert store.upsert_knowledge_record(record) is False
        assert store.count_knowledge_records() == 1
        records = store.list_all_knowledge_records()
    finally:
        store.close()

    assert len(records) == 1
    assert records[0].id == "knowledge_abc123"
    assert records[0].evidence_turns == [1, 3]
    assert records[0].scope == KnowledgeScope.PROJECT_SPECIFIC


def test_evidence_spread_calculation_including_single_turn_boundary() -> None:
    assert pipeline._evidence_spread([1, 3], total_turns=3) == 1.0
    assert pipeline._evidence_spread([2], total_turns=3) == 0.0
    assert pipeline._evidence_spread([1, 2], total_turns=1) == 0.0


def test_pipeline_admission_and_persistence_with_fake_extractor(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "session.jsonl", session_events())

    class FakeExtractor(KnowledgeExtractor):
        def __init__(self, settings: Settings):
            self.settings = settings

        def extract(self, session: ProcessedSession) -> list[KnowledgeItemInput]:
            return [
                KnowledgeItemInput(
                    title="Admitted item",
                    insight="Use structured records when storing extracted transferable knowledge.",
                    applicability="Applies to extraction pipelines that write durable knowledge records.",
                    evidence_turns=[1],
                    scope=KnowledgeScope.GLOBAL,
                ),
                KnowledgeItemInput(
                    title="Rejected local",
                    insight="This is a one-off decision tied to a specific file in this repository.",
                    applicability="Only useful in this current repository context.",
                    evidence_turns=[1],
                    scope=KnowledgeScope.SESSION_SPECIFIC,
                ),
                KnowledgeItemInput(
                    title="Rejected empty evidence",
                    insight="This insight has no evidence turn references and should be rejected.",
                    applicability="Applies to any admission rule requiring traceable support.",
                    evidence_turns=[],
                    scope=KnowledgeScope.GLOBAL,
                ),
            ]

    stats = pipeline.run_knowledge_extraction(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        processed_index_path=tmp_path / "knowledge-processed-index.json",
        knowledge_db_path=tmp_path / "knowledge.db",
        settings=Settings(dashscope_api_key="test"),
        extractor=FakeExtractor(Settings(dashscope_api_key="test")),
    )

    assert stats.discovered_sessions == 1
    assert stats.processed_sessions == 1
    assert stats.extracted_count == 3
    assert stats.admitted_count == 1
    assert stats.rejected_count == 2
    assert (tmp_path / "knowledge-processed-index.json").exists()
    index = ProcessedIndex.model_validate_json((tmp_path / "knowledge-processed-index.json").read_text())
    assert index.sessions["session-1"].status == ProcessedStatus.PROCESSED

    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        records = store.list_all_knowledge_records()
    finally:
        store.close()
    assert len(records) == 1
    assert records[0].title == "Admitted item"


def test_max_session_chars_filter_skips_before_extraction(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "session.jsonl", session_events(body="x" * 800))

    class FailingExtractor(KnowledgeExtractor):
        def __init__(self, settings: Settings):
            self.settings = settings

        def extract(self, session: ProcessedSession) -> list[KnowledgeItemInput]:
            raise AssertionError("extractor should not run for oversized sessions")

    stats = pipeline.run_knowledge_extraction(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        processed_index_path=tmp_path / "knowledge-processed-index.json",
        knowledge_db_path=tmp_path / "knowledge.db",
        settings=Settings(dashscope_api_key="test"),
        max_session_chars=200,
        extractor=FailingExtractor(Settings(dashscope_api_key="test")),
    )

    assert stats.discovered_sessions == 1
    assert stats.skipped_sessions == 1
    assert stats.processed_sessions == 0
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        assert store.count_knowledge_records() == 0
    finally:
        store.close()
