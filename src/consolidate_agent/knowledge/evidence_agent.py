from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.knowledge.session_turn_store import SessionTurnStore
from consolidate_agent.prompt_loader import load_prompt
from consolidate_agent.types import KnowledgeRecord

_EVIDENCE_WORKERS = 10
_TEXT_SEARCH_LIMIT = 5
_SEMANTIC_SEARCH_LIMIT = 5
_FULL_TURN_LIMIT = 3000


class SearchAction(BaseModel):
    type: Literal["text", "semantic"]
    query: str


class EvidencePlan(BaseModel):
    searches: list[SearchAction] = Field(min_length=1, max_length=4)


class JudgmentOutput(BaseModel):
    reasoning: str
    verdict: Literal["admit", "reject", "need_more"]
    evidence_turns: list[int] = Field(default_factory=list)
    additional_searches: list[SearchAction] = Field(default_factory=list)


class EvidenceOutput(BaseModel):
    verdict: Literal["admit", "reject"]
    evidence_turns: list[int]


class EvidenceAgent:
    def __init__(self, settings: Settings, turn_store: SessionTurnStore):
        self._turn_store = turn_store
        self._plan_llm = ChatQwen(
            model="qwen-plus",
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_api_base,
            temperature=0,
        ).with_structured_output(EvidencePlan)
        self._judge_llm = ChatQwen(
            model="qwen-plus",
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_api_base,
            temperature=0,
        ).with_structured_output(JudgmentOutput)
        self._plan_prompt = load_prompt("evidence_plan_system.md")
        self._judge_prompt = load_prompt("evidence_judge_system.md")

    def _plan(self, insight: str) -> EvidencePlan:
        messages = [
            SystemMessage(content=self._plan_prompt),
            HumanMessage(content=f"Insight: {insight}"),
        ]
        result = self._plan_llm.invoke(messages)
        if result is None:
            return EvidencePlan(searches=[SearchAction(type="semantic", query=insight)])
        return result

    def _execute(self, searches: list[SearchAction], session_id: str, session_xml: str) -> str:
        parts: list[str] = []
        for i, action in enumerate(searches, 1):
            if action.type == "text":
                hits = SessionTurnStore.search_text(session_xml, action.query)
                part = f"=== 搜索 {i}: text({action.query!r}) ===\n"
                if hits:
                    for hit in hits[:_TEXT_SEARCH_LIMIT]:
                        content = self._turn_store.get_turn(session_xml, hit["turn_index"]) or ""
                        part += f"Turn {hit['turn_index']}:\n{content[:_FULL_TURN_LIMIT]}\n\n"
                else:
                    part += "(无匹配结果)\n"
            else:
                hits = self._turn_store.search_turns(session_id, action.query, top_k=_SEMANTIC_SEARCH_LIMIT)
                part = f"=== 搜索 {i}: semantic({action.query!r}) ===\n"
                if hits:
                    for hit in hits[:3]:
                        content = self._turn_store.get_turn(session_xml, hit["turn_index"]) or ""
                        part += (
                            f"Turn {hit['turn_index']} (score={hit['score']:.2f}):\n"
                            f"{content[:_FULL_TURN_LIMIT]}\n\n"
                        )
                else:
                    part += "(无语义相关结果)\n"
            parts.append(part)
        return "\n".join(parts)

    def _judge(self, insight: str, search_results: str, final: bool = False) -> JudgmentOutput:
        system = self._judge_prompt
        if final:
            system += "\n\n这是最终判断，必须输出 admit 或 reject，不允许 need_more。"
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Insight: {insight}\n\n搜索结果:\n{search_results}"),
        ]
        return self._judge_llm.invoke(messages)

    def verify(self, record: KnowledgeRecord, session_xml: str) -> KnowledgeRecord:
        try:
            plan = self._plan(record.insight)
            results = self._execute(plan.searches, record.session_id, session_xml)
            judgment = self._judge(record.insight, results, final=False)

            if judgment.verdict == "need_more" and judgment.additional_searches:
                additional = self._execute(judgment.additional_searches, record.session_id, session_xml)
                all_results = results + "\n\n=== 补充搜索 ===\n" + additional
                judgment = self._judge(record.insight, all_results, final=True)

            if judgment.verdict == "admit":
                return record.model_copy(
                    update={
                        "evidence_turns": judgment.evidence_turns,
                        "evidence_count": len(judgment.evidence_turns),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - failed verification should keep the original record.
            _progress(f"evidence_agent verify failed record_id={record.id} error={exc}")
            return record

        return record.model_copy(
            update={
                "evidence_turns": [],
                "evidence_count": 0,
            }
        )

    def verify_batch(
        self,
        records: list[KnowledgeRecord],
        session_xmls: dict[str, str],
    ) -> list[KnowledgeRecord]:
        eligible = [(r, session_xmls[r.session_id]) for r in records if r.session_id in session_xmls]
        ineligible = [r for r in records if r.session_id not in session_xmls]

        results: list[KnowledgeRecord] = list(ineligible)
        with ThreadPoolExecutor(max_workers=_EVIDENCE_WORKERS) as executor:
            futures = {executor.submit(self.verify, r, xml): r for r, xml in eligible}
            for future in as_completed(futures):
                original = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - one failed record should not stop the batch.
                    _progress(f"evidence_agent batch failed record_id={original.id} error={exc}")
                    results.append(original)
        return results


def _progress(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)
