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
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call-1",
                "arguments": json.dumps({"cmd": "pytest tests/"}),
            },
        },
        {
            "timestamp": "2026-04-17T09:00:03Z",
            "type": "event_msg",
            "payload": {
                "type": "exec_command_end",
                "call_id": "call-1",
                "exit_code": 0,
                "aggregated_output": "1 passed",
            },
        },
        {
            "timestamp": "2026-04-17T09:00:04Z",
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


def test_admission_rejects_session_specific_only() -> None:
    admitted, rejected = pipeline._admit_items(
        [
            KnowledgeItemInput(
                title="Good item",
                insight="A reusable engineering insight with enough substance.",
                applicability="Applies to reusable extraction pipelines.",
                scope=KnowledgeScope.GLOBAL,
            ),
            KnowledgeItemInput(
                title="Local fix",
                insight="This exact repository needs a local one-off fix.",
                applicability="Only this current session and repository need it.",
                scope=KnowledgeScope.SESSION_SPECIFIC,
            ),
        ]
    )

    assert [item.title for item in admitted] == ["Good item"]
    assert [item.title for item in rejected] == ["Local fix"]


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
                    scope=KnowledgeScope.GLOBAL,
                ),
                KnowledgeItemInput(
                    title="Rejected local",
                    insight="This is a one-off decision tied to a specific file in this repository.",
                    applicability="Only useful in this current repository context.",
                    scope=KnowledgeScope.SESSION_SPECIFIC,
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
    assert stats.extracted_count == 2
    assert stats.admitted_count == 1
    assert stats.rejected_count == 1
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


def test_pipeline_record_ids_include_source_path_for_duplicate_session_ids(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    write_jsonl(tmp_path / "session-a.jsonl", session_events(session_id="duplicate-session", body="a" * 260))
    write_jsonl(nested / "session-b.jsonl", session_events(session_id="duplicate-session", body="b" * 260))

    class FakeExtractor(KnowledgeExtractor):
        def __init__(self, settings: Settings):
            self.settings = settings

        def extract(self, session: ProcessedSession) -> list[KnowledgeItemInput]:
            return [
                KnowledgeItemInput(
                    title="Same title",
                    insight=f"Same title should not collide across source files. {session.stats.processed_chars}",
                    applicability="Applies when multiple rollout files share a session id.",
                    scope=KnowledgeScope.GLOBAL,
                )
            ]

    stats = pipeline.run_knowledge_extraction(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        processed_index_path=tmp_path / "knowledge-processed-index.json",
        knowledge_db_path=tmp_path / "knowledge.db",
        settings=Settings(dashscope_api_key="test"),
        extractor=FakeExtractor(Settings(dashscope_api_key="test")),
    )

    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        records = store.list_all_knowledge_records()
    finally:
        store.close()

    assert stats.extracted_count == 2
    assert stats.admitted_count == 2
    assert len(records) == 2
    assert len({record.id for record in records}) == 2
    assert {record.session_id for record in records} == {"duplicate-session"}


def test_pipeline_embeds_missing_turns_before_evidence_verification(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "session.jsonl", session_events())

    class FakeExtractor(KnowledgeExtractor):
        def __init__(self, settings: Settings):
            self.settings = settings

        def extract(self, session: ProcessedSession) -> list[KnowledgeItemInput]:
            return [
                KnowledgeItemInput(
                    title="Needs evidence",
                    insight="Evidence verification should have turn embeddings available.",
                    applicability="Applies when running evidence verification after extraction.",
                    scope=KnowledgeScope.GLOBAL,
                )
            ]

    class FakeEvidenceAgent:
        def __init__(self) -> None:
            self.embedded: set[str] = set()
            self.embed_calls: list[dict[str, str]] = []

        def embedded_session_ids(self) -> set[str]:
            return set(self.embedded)

        def embed_missing_sessions(
            self,
            session_xmls: dict[str, str],
            embedded_session_ids: set[str] | None = None,
        ) -> int:
            missing = set(session_xmls) - set(embedded_session_ids or set())
            self.embed_calls.append({session_id: session_xmls[session_id] for session_id in sorted(missing)})
            self.embedded.update(missing)
            return len(missing)

        def verify_batch(
            self,
            records: list[KnowledgeRecord],
            session_xmls: dict[str, str],
        ) -> list[KnowledgeRecord]:
            assert self.embedded == set(session_xmls)
            return [
                record.model_copy(update={"evidence_turns": [1], "evidence_count": 1})
                for record in records
            ]

    evidence_agent = FakeEvidenceAgent()
    stats = pipeline.run_knowledge_extraction(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        processed_index_path=tmp_path / "knowledge-processed-index.json",
        knowledge_db_path=tmp_path / "knowledge.db",
        settings=Settings(dashscope_api_key="test"),
        extractor=FakeExtractor(Settings(dashscope_api_key="test")),
        evidence_agent=evidence_agent,
    )

    assert stats.evidence_admitted_count == 1
    assert len(evidence_agent.embed_calls) == 1
    assert set(evidence_agent.embed_calls[0]) == {"session-1"}


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
    assert stats.skip_reasons == {"too_large": 1}
    assert stats.processed_sessions == 0
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        assert store.count_knowledge_records() == 0
    finally:
        store.close()


def test_no_action_filter_records_skip_reason(tmp_path: Path) -> None:
    events = [
        event
        for event in session_events()
        if not (
            event["type"] == "response_item"
            and event["payload"].get("type") == "function_call"
        )
        and not (
            event["type"] == "event_msg"
            and event["payload"].get("type") == "exec_command_end"
        )
    ]
    write_jsonl(tmp_path / "session.jsonl", events)

    class FailingExtractor(KnowledgeExtractor):
        def __init__(self, settings: Settings):
            self.settings = settings

        def extract(self, session: ProcessedSession) -> list[KnowledgeItemInput]:
            raise AssertionError("extractor should not run for no-action sessions")

    stats = pipeline.run_knowledge_extraction(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        processed_index_path=tmp_path / "knowledge-processed-index.json",
        knowledge_db_path=tmp_path / "knowledge.db",
        settings=Settings(dashscope_api_key="test"),
        extractor=FailingExtractor(Settings(dashscope_api_key="test")),
    )

    assert stats.skipped_sessions == 1
    assert stats.skip_reasons == {"no_action": 1}


def test_already_processed_filter_records_skip_reason(tmp_path: Path) -> None:
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
                    scope=KnowledgeScope.GLOBAL,
                )
            ]

    first = pipeline.run_knowledge_extraction(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        processed_index_path=tmp_path / "knowledge-processed-index.json",
        knowledge_db_path=tmp_path / "knowledge.db",
        settings=Settings(dashscope_api_key="test"),
        extractor=FakeExtractor(Settings(dashscope_api_key="test")),
    )
    second = pipeline.run_knowledge_extraction(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        processed_index_path=tmp_path / "knowledge-processed-index.json",
        knowledge_db_path=tmp_path / "knowledge.db",
        settings=Settings(dashscope_api_key="test"),
        extractor=FakeExtractor(Settings(dashscope_api_key="test")),
    )

    assert first.processed_sessions == 1
    assert second.processed_sessions == 0
    assert second.skipped_sessions == 1
    assert second.skip_reasons == {"already_processed": 1}
