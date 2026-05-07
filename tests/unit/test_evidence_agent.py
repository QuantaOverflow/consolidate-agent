from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

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
        return self.outputs.pop(0)


class FakeChatQwen:
    structured_llms: dict[type, FakeStructuredLLM] = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def with_structured_output(self, schema: type) -> FakeStructuredLLM:
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
    agent = _build_agent(tmp_path, monkeypatch, plan_llm, judge_llm)

    verified = agent.verify(_record(evidence_turns=[3], evidence_count=1), SESSION_XML)

    assert verified.evidence_turns == []
    assert verified.evidence_count == 0
    assert len(judge_llm.messages) == 1


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


def _build_agent(
    tmp_path: Path,
    monkeypatch,
    plan_llm: FakeStructuredLLM,
    judge_llm: FakeStructuredLLM,
) -> EvidenceAgent:
    FakeChatQwen.structured_llms = {
        EvidencePlan: plan_llm,
        JudgmentOutput: judge_llm,
    }
    monkeypatch.setattr("consolidate_agent.knowledge.evidence_agent.ChatQwen", FakeChatQwen)
    turn_store = SessionTurnStore(tmp_path / "chroma", FakeEmbeddings())
    return EvidenceAgent(Settings(dashscope_api_key="test"), turn_store)


def _record(evidence_turns: list[int] | None = None, evidence_count: int = 0) -> KnowledgeRecord:
    now = utc_now()
    return KnowledgeRecord(
        id="knowledge_test",
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
