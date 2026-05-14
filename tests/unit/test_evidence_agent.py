from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from consolidate_agent.config import Settings
from consolidate_agent.knowledge.evidence_agent import EvidenceAgent, EvidencePlan, JudgmentOutput, SearchAction
from consolidate_agent.knowledge.session_turn_store import SessionTurnStore
from consolidate_agent.types import KnowledgeRecord, KnowledgeScope, utc_now

SESSION_XML = """<session>
<turn index="1" started_at="t1"><user>hello</user><assistant>world</assistant></turn>
<turn index="3" started_at="t3"><user>run tests</user><bash>uv run pytest</bash><bash_result>passed</bash_result></turn>
<turn index="7" started_at="t7"><assistant>Use evidence_turns after verification.</assistant></turn>
</session>"""


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 10


class FakeStructuredLLM:
    def __init__(self, outputs: list[Any]):
        self.outputs = outputs
        self.messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.messages.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class FakeChatQwen:
    structured_llms: dict[type, FakeStructuredLLM] = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def with_structured_output(self, schema: type, **kwargs) -> FakeStructuredLLM:
        return self.structured_llms[schema]


def test_search_text_finds_matching_turns() -> None:
    results = SessionTurnStore.search_text(SESSION_XML, "pytest")
    assert len(results) == 1
    assert results[0]["turn_index"] == 3
    assert "pytest" in results[0]["snippet"]


def test_search_text_returns_empty_for_no_match() -> None:
    assert SessionTurnStore.search_text(SESSION_XML, "not-present") == []


def test_search_text_snippet_truncates_correctly() -> None:
    xml = '<session><turn index="4" started_at="t">aaaaaaaaaaaaaaaa pytest bbbbbbbbbbbbbbbb</turn></session>'
    results = SessionTurnStore.search_text(xml, "pytest", snippet_chars=20)
    assert len(results) == 1
    assert len(results[0]["snippet"]) <= 40
    assert "pytest" in results[0]["snippet"]


def test_evidence_agent_verify_admit(tmp_path: Path, monkeypatch) -> None:
    plan_llm = FakeStructuredLLM([EvidencePlan(searches=[SearchAction(type="text", query="pytest")])])
    judge_llm = FakeStructuredLLM(
        [JudgmentOutput(reasoning="turn 3 directly supports it", verdict="admit", evidence_turns=[3, 7])]
    )
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm)

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_turns == [3, 7]
    assert verified.evidence_count == 2
    assert len(plan_llm.messages) == 1
    assert len(judge_llm.messages) == 1


def test_evidence_agent_verify_reject(tmp_path: Path, monkeypatch) -> None:
    plan_llm = FakeStructuredLLM([EvidencePlan(searches=[SearchAction(type="text", query="not-present")])])
    judge_llm = FakeStructuredLLM([JudgmentOutput(reasoning="no direct evidence", verdict="reject")])
    reject_trace_path = tmp_path / "rejects.jsonl"
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, reject_trace_path=reject_trace_path)

    verified = agent.verify(_record(evidence_turns=[3], evidence_count=1), SESSION_XML)

    assert verified.evidence_turns == []
    assert verified.evidence_count == 0
    assert len(judge_llm.messages) == 1
    trace = _read_trace(reject_trace_path)
    assert len(trace) == 1
    assert trace[0]["stage"] == "evidence_reject"
    assert trace[0]["record_id"] == "knowledge_test"
    assert trace[0]["final"] is False
    assert trace[0]["reject_reason"] == "judge_reject"
    assert trace[0]["judge_verdict"] == "reject"
    assert trace[0]["judge_reasoning"] == "no direct evidence"
    assert trace[0]["judge_evidence_turns"] == []
    assert trace[0]["search_actions"] == [{"query": "not-present", "type": "text"}]
    assert trace[0]["search_results_chars"] > 0
    assert len(trace[0]["search_results_hash"]) == 40


def test_evidence_agent_empty_admit_is_rejected(tmp_path: Path, monkeypatch) -> None:
    plan_llm = FakeStructuredLLM([EvidencePlan(searches=[SearchAction(type="text", query="pytest")])])
    judge_llm = FakeStructuredLLM([JudgmentOutput(reasoning="looks relevant but no specific turn", verdict="admit")])
    reject_trace_path = tmp_path / "rejects.jsonl"
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, reject_trace_path=reject_trace_path)

    verified = agent.verify(_record(evidence_turns=[3], evidence_count=1), SESSION_XML)

    assert verified.evidence_turns == []
    assert verified.evidence_count == 0
    trace = _read_trace(reject_trace_path)
    assert len(trace) == 1
    assert trace[0]["stage"] == "evidence_reject"
    assert trace[0]["reject_reason"] == "empty_admit_evidence"
    assert trace[0]["judge_verdict"] == "admit"
    assert trace[0]["judge_evidence_turns"] == []


def test_evidence_agent_verify_replan(tmp_path: Path, monkeypatch) -> None:
    plan_llm = FakeStructuredLLM([EvidencePlan(searches=[SearchAction(type="text", query="not-present")])])
    judge_llm = FakeStructuredLLM(
        [
            JudgmentOutput(
                reasoning="need a more specific search",
                verdict="need_more",
                additional_searches=[SearchAction(type="text", query="evidence_turns")],
            ),
            JudgmentOutput(reasoning="turn 7 directly supports it", verdict="admit", evidence_turns=[7]),
        ]
    )
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm)

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_turns == [7]
    assert verified.evidence_count == 1
    assert len(judge_llm.messages) == 2
    assert "=== 补充搜索 ===" in judge_llm.messages[1][1].content
    assert "不允许 need_more" in judge_llm.messages[1][0].content


def test_evidence_agent_verify_replan_still_reject(tmp_path: Path, monkeypatch) -> None:
    plan_llm = FakeStructuredLLM([EvidencePlan(searches=[SearchAction(type="text", query="not-present")])])
    judge_llm = FakeStructuredLLM(
        [
            JudgmentOutput(
                reasoning="need another angle",
                verdict="need_more",
                additional_searches=[SearchAction(type="text", query="evidence_turns")],
            ),
            JudgmentOutput(reasoning="still no direct evidence", verdict="reject"),
        ]
    )
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm)

    verified = agent.verify(_record(evidence_turns=[3], evidence_count=1), SESSION_XML)

    assert verified.evidence_turns == []
    assert verified.evidence_count == 0
    assert len(judge_llm.messages) == 2


def test_evidence_agent_traces_plan_none_and_uses_semantic_fallback(tmp_path: Path, monkeypatch) -> None:
    plan_llm = FakeStructuredLLM([None, None, None])
    judge_llm = FakeStructuredLLM([JudgmentOutput(reasoning="no direct evidence", verdict="reject")])
    failure_trace_path = tmp_path / "failures.jsonl"
    reject_trace_path = tmp_path / "rejects.jsonl"
    agent = _build_agent(
        tmp_path,
        monkeypatch,
        plan_llm,
        judge_llm,
        failure_trace_path=failure_trace_path,
        reject_trace_path=reject_trace_path,
    )

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_count == 0
    plan_trace = _read_trace(failure_trace_path)
    reject_trace = _read_trace(reject_trace_path)
    assert len(plan_trace) == 1
    assert len(reject_trace) == 1
    assert plan_trace[0]["record_id"] == "knowledge_test"
    assert plan_trace[0]["session_id"] == "session-1"
    assert plan_trace[0]["event_type"] == "recovery"
    assert plan_trace[0]["note"] == "semantic_fallback_after_exhausted_retries"
    assert "error" not in plan_trace[0]
    assert plan_trace[0]["attempt"] == 3
    assert plan_trace[0]["max_attempts"] == 3


def test_evidence_agent_traces_plan_raw_response_when_parsed_is_none(tmp_path: Path, monkeypatch) -> None:
    raw = AIMessage(content="not valid json", additional_kwargs={"tool_calls": []})
    plan_llm = FakeStructuredLLM(
        [
            {"raw": raw, "parsed": None, "parsing_error": ValueError("bad json")},
            {"raw": raw, "parsed": None, "parsing_error": ValueError("bad json")},
            {"raw": raw, "parsed": None, "parsing_error": ValueError("bad json")},
        ]
    )
    judge_llm = FakeStructuredLLM([JudgmentOutput(reasoning="no direct evidence", verdict="reject")])
    failure_trace_path = tmp_path / "failures.jsonl"
    reject_trace_path = tmp_path / "rejects.jsonl"
    agent = _build_agent(
        tmp_path,
        monkeypatch,
        plan_llm,
        judge_llm,
        failure_trace_path=failure_trace_path,
        reject_trace_path=reject_trace_path,
    )

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_count == 0
    plan_trace = _read_trace(failure_trace_path)
    reject_trace = _read_trace(reject_trace_path)
    assert len(plan_trace) == 1
    assert len(reject_trace) == 1
    assert plan_trace[0]["event_type"] == "recovery"
    assert plan_trace[0]["note"] == "semantic_fallback_after_exhausted_retries"
    assert "error" not in plan_trace[0]
    assert plan_trace[0]["raw_content_preview"] == "not valid json"
    assert plan_trace[0]["raw_additional_kwargs_preview"] == '{"tool_calls": []}'
    assert plan_trace[0]["parsing_error"] == "bad json"
    assert plan_trace[0]["attempt"] == 3


def test_evidence_agent_traces_plan_invalid_tool_calls(tmp_path: Path, monkeypatch) -> None:
    raw = AIMessage(
        content="",
        additional_kwargs={"refusal": None},
        invalid_tool_calls=[
            {
                "name": "EvidencePlan",
                "args": '{"searches": [{"type": "semantic", "query: "broken"}]}',
                "error": "invalid JSON",
            }
        ],
    )
    plan_llm = FakeStructuredLLM(
        [
            {"raw": raw, "parsed": None, "parsing_error": None},
            {"raw": raw, "parsed": None, "parsing_error": None},
            {"raw": raw, "parsed": None, "parsing_error": None},
        ]
    )
    judge_llm = FakeStructuredLLM([JudgmentOutput(reasoning="no direct evidence", verdict="reject")])
    trace_path = tmp_path / "failures.jsonl"
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, failure_trace_path=trace_path)

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_count == 0
    trace = _read_trace(trace_path)
    plan_trace = [item for item in trace if item["stage"] == "evidence_plan"]
    assert len(plan_trace) == 1
    assert plan_trace[0]["event_type"] == "recovery"
    assert plan_trace[0]["note"] == "semantic_fallback_after_exhausted_retries"
    assert "error" not in plan_trace[0]
    assert plan_trace[0]["raw_additional_kwargs_preview"] == '{"refusal": null}'
    assert "raw_invalid_tool_calls_hash" in plan_trace[0]
    assert "EvidencePlan" in plan_trace[0]["raw_invalid_tool_calls_preview"]
    assert "invalid JSON" in plan_trace[0]["raw_invalid_tool_calls_preview"]
    retry_input = plan_llm.messages[1][2].content
    assert "Invalid tool-call args excerpt" in retry_input
    assert "Use the key \"query\"" in retry_input
    assert plan_trace[0]["attempt"] == 3


def test_evidence_agent_plan_retry_recovers_from_invalid_tool_call(tmp_path: Path, monkeypatch) -> None:
    raw = AIMessage(
        content="",
        additional_kwargs={"refusal": None},
        invalid_tool_calls=[
            {
                "name": "EvidencePlan",
                "args": '{"searches": [{"type": "semantic", "query: "broken"}]}',
                "error": "invalid JSON",
            }
        ],
    )
    plan_llm = FakeStructuredLLM(
        [
            {"raw": raw, "parsed": None, "parsing_error": None},
            EvidencePlan(searches=[SearchAction(type="text", query="pytest")]),
        ]
    )
    judge_llm = FakeStructuredLLM([JudgmentOutput(reasoning="turn 3 supports it", verdict="admit", evidence_turns=[3])])
    trace_path = tmp_path / "failures.jsonl"
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, failure_trace_path=trace_path)

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_turns == [3]
    assert len(plan_llm.messages) == 2
    assert "Invalid tool-call args excerpt" in plan_llm.messages[1][2].content
    trace = _read_trace(trace_path)
    assert len(trace) == 1
    assert trace[0]["stage"] == "evidence_plan"
    assert trace[0]["event_type"] == "recovery"
    assert "error" not in trace[0]
    assert trace[0]["recovery_method"] == "retry_with_plan_error_feedback"
    assert trace[0]["attempt"] == 2


def test_evidence_agent_truncates_plan_searches_when_model_returns_too_many(tmp_path: Path, monkeypatch) -> None:
    too_many = [
        {"type": "text", "query": "pytest"},
        {"type": "text", "query": "evidence_turns"},
        {"type": "semantic", "query": "structured evidence"},
        {"type": "semantic", "query": "verification"},
        {"type": "text", "query": "ignored"},
    ]
    error: ValidationError | None = None
    try:
        EvidencePlan(searches=too_many)
    except ValidationError as exc:
        error = exc
    assert error is not None
    plan_llm = FakeStructuredLLM([error])
    judge_llm = FakeStructuredLLM([JudgmentOutput(reasoning="turn 3 supports it", verdict="admit", evidence_turns=[3])])
    trace_path = tmp_path / "failures.jsonl"
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, failure_trace_path=trace_path)

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_turns == [3]
    judge_input = judge_llm.messages[0][1].content
    assert "ignored" not in judge_input
    trace = _read_trace(trace_path)
    assert len(trace) == 1
    assert trace[0]["stage"] == "evidence_plan"
    assert trace[0]["event_type"] == "recovery"
    assert trace[0]["exception_class"] == "ValidationError"
    assert trace[0]["recovered_search_count"] == 4


def test_evidence_agent_truncates_plan_searches_from_include_raw_parsing_error(tmp_path: Path, monkeypatch) -> None:
    too_many = [
        {"type": "text", "query": "pytest"},
        {"type": "text", "query": "evidence_turns"},
        {"type": "semantic", "query": "structured evidence"},
        {"type": "semantic", "query": "verification"},
        {"type": "text", "query": "ignored"},
    ]
    error: ValidationError | None = None
    try:
        EvidencePlan(searches=too_many)
    except ValidationError as exc:
        error = exc
    assert error is not None
    raw = AIMessage(content="", additional_kwargs={"tool_calls": [{"args": {"searches": too_many}}]})
    plan_llm = FakeStructuredLLM([{"raw": raw, "parsed": None, "parsing_error": error}])
    judge_llm = FakeStructuredLLM([JudgmentOutput(reasoning="turn 3 supports it", verdict="admit", evidence_turns=[3])])
    trace_path = tmp_path / "failures.jsonl"
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, failure_trace_path=trace_path)

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_turns == [3]
    judge_input = judge_llm.messages[0][1].content
    assert "ignored" not in judge_input
    trace = _read_trace(trace_path)
    assert len(trace) == 1
    assert trace[0]["stage"] == "evidence_plan"
    assert trace[0]["event_type"] == "recovery"
    assert "error" not in trace[0]
    assert trace[0]["recovered_search_count"] == 4
    assert "parsing_error" in trace[0]
    assert "raw_additional_kwargs_hash" in trace[0]


def test_evidence_agent_traces_judge_none_with_search_context(tmp_path: Path, monkeypatch) -> None:
    plan_llm = FakeStructuredLLM([EvidencePlan(searches=[SearchAction(type="text", query="pytest")])])
    judge_llm = FakeStructuredLLM([None, None, None])
    trace_path = tmp_path / "failures.jsonl"
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, failure_trace_path=trace_path)

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_turns == []
    assert verified.evidence_count == 0
    assert len(judge_llm.messages) == 3
    assert "Previous JudgmentOutput attempt failed" in judge_llm.messages[1][2].content
    assert "short natural-language sentence" in judge_llm.messages[1][2].content
    trace = _read_trace(trace_path)
    assert len(trace) == 1
    assert trace[0]["stage"] == "evidence_judge"
    assert trace[0]["record_id"] == "knowledge_test"
    assert trace[0]["final"] is False
    assert trace[0]["error"] == "structured judge returned None"
    assert trace[0]["attempt"] == 3
    assert trace[0]["max_attempts"] == 3
    assert trace[0]["search_actions"] == [{"query": "pytest", "type": "text"}]
    assert trace[0]["search_results_chars"] >= len("uv run pytest")
    assert "uv run pytest" in trace[0]["search_results_preview"]
    assert len(trace[0]["search_results_hash"]) == 40


def test_evidence_agent_traces_judge_raw_response_when_parsed_is_none(tmp_path: Path, monkeypatch) -> None:
    raw = AIMessage(content="I cannot decide", additional_kwargs={"tool_calls": []})
    plan_llm = FakeStructuredLLM([EvidencePlan(searches=[SearchAction(type="text", query="pytest")])])
    judge_llm = FakeStructuredLLM(
        [
            {"raw": raw, "parsed": None, "parsing_error": ValueError("missing verdict")},
            {"raw": raw, "parsed": None, "parsing_error": ValueError("missing verdict")},
            {"raw": raw, "parsed": None, "parsing_error": ValueError("missing verdict")},
        ]
    )
    trace_path = tmp_path / "failures.jsonl"
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, failure_trace_path=trace_path)

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_count == 0
    trace = _read_trace(trace_path)
    assert len(trace) == 1
    assert trace[0]["stage"] == "evidence_judge"
    assert trace[0]["raw_content_preview"] == "I cannot decide"
    assert trace[0]["parsing_error"] == "missing verdict"
    assert trace[0]["attempt"] == 3
    assert "Parsing error: missing verdict" in judge_llm.messages[1][2].content


def test_evidence_agent_judge_retry_recovers_from_invalid_tool_call(tmp_path: Path, monkeypatch) -> None:
    raw = AIMessage(
        content="",
        additional_kwargs={"refusal": None},
        invalid_tool_calls=[
            {
                "name": "JudgmentOutput",
                "args": '{"reasoning": "bad regex \\[", "verdict": "admit", "evidence_turns": [3]}',
                "error": "invalid JSON",
            }
        ],
    )
    plan_llm = FakeStructuredLLM([EvidencePlan(searches=[SearchAction(type="text", query="pytest")])])
    judge_llm = FakeStructuredLLM(
        [
            {"raw": raw, "parsed": None, "parsing_error": None},
            JudgmentOutput(reasoning="turn 3 directly supports it", verdict="admit", evidence_turns=[3]),
        ]
    )
    trace_path = tmp_path / "failures.jsonl"
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, failure_trace_path=trace_path)

    verified = agent.verify(_record(), SESSION_XML)

    assert verified.evidence_turns == [3]
    assert verified.evidence_count == 1
    assert len(judge_llm.messages) == 2
    retry_input = judge_llm.messages[1][2].content
    assert "Invalid tool-call args excerpt" in retry_input
    assert "Do not manually copy raw JSON, code, regex" in retry_input
    trace = _read_trace(trace_path)
    assert len(trace) == 1
    assert trace[0]["stage"] == "evidence_judge"
    assert trace[0]["event_type"] == "recovery"
    assert "error" not in trace[0]
    assert trace[0]["recovery_method"] == "retry_with_judge_error_feedback"
    assert trace[0]["attempt"] == 2
    assert "raw_invalid_tool_calls_preview" in trace[0]
    assert "JudgmentOutput" in trace[0]["raw_invalid_tool_calls_preview"]


def test_evidence_agent_verify_batch_reuses_session_text_search_cache(tmp_path: Path, monkeypatch) -> None:
    plan_llm = FakeStructuredLLM(
        [
            EvidencePlan(searches=[SearchAction(type="text", query="pytest")]),
            EvidencePlan(searches=[SearchAction(type="text", query="pytest")]),
        ]
    )
    judge_llm = FakeStructuredLLM(
        [
            JudgmentOutput(reasoning="turn 3 supports it", verdict="admit", evidence_turns=[3]),
            JudgmentOutput(reasoning="turn 3 supports it", verdict="admit", evidence_turns=[3]),
        ]
    )
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm, workers=1)
    calls = 0

    def fake_search_text(turns, pattern):
        nonlocal calls
        calls += 1
        return [{"turn_index": 3, "snippet": "pytest"}]

    monkeypatch.setattr("consolidate_agent.knowledge.evidence_agent._search_text_turns", fake_search_text)

    verified = agent.verify_batch([_record(record_id="knowledge_1"), _record(record_id="knowledge_2")], {"session-1": SESSION_XML})

    assert calls == 1
    assert [record.evidence_turns for record in verified] == [[3], [3]]


def _build_agent(
    tmp_path: Path,
    monkeypatch,
    plan_llm: FakeStructuredLLM,
    judge_llm: FakeStructuredLLM,
    *,
    workers: int = 10,
    failure_trace_path: Path | None = None,
    reject_trace_path: Path | None = None,
) -> EvidenceAgent:
    FakeChatQwen.structured_llms = {
        EvidencePlan: plan_llm,
        JudgmentOutput: judge_llm,
    }
    monkeypatch.setattr("consolidate_agent.knowledge.evidence_agent.ChatQwen", FakeChatQwen)
    turn_store = SessionTurnStore(tmp_path / "chroma", FakeEmbeddings())
    return EvidenceAgent(
        Settings(dashscope_api_key="test"),
        turn_store,
        workers=workers,
        failure_trace_path=failure_trace_path,
        reject_trace_path=reject_trace_path,
    )


def _record(
    evidence_turns: list[int] | None = None,
    evidence_count: int = 0,
    *,
    record_id: str = "knowledge_test",
) -> KnowledgeRecord:
    now = utc_now()
    return KnowledgeRecord(
        id=record_id,
        session_id="session-1",
        title="Evidence agent",
        insight="Use evidence turns to verify extracted knowledge records.",
        applicability="Applies to extracted records that need support from source sessions.",
        scope=KnowledgeScope.GLOBAL,
        evidence_turns=evidence_turns or [],
        evidence_count=evidence_count,
        processed_chars=500,
        created_at=now,
        updated_at=now,
    )


def _read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
