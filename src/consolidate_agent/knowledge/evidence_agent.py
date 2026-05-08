from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field, ValidationError

from consolidate_agent.config import Settings
from consolidate_agent.knowledge.session_turn_store import SessionTurnStore, TURN_PATTERN
from consolidate_agent.prompt_loader import load_prompt
from consolidate_agent.types import KnowledgeRecord

_EVIDENCE_WORKERS = 10
_TEXT_SEARCH_LIMIT = 5
_SEMANTIC_SEARCH_LIMIT = 5
_FULL_TURN_LIMIT = 3000
_TRACE_PREVIEW_CHARS = 2000
_RAW_PREVIEW_CHARS = 4000
_DEFAULT_FAILURE_TRACE_PATH = Path("outputs/evidence-agent-failures.jsonl")
_DEFAULT_REJECT_TRACE_PATH = Path("outputs/evidence-agent-rejects.jsonl")
_PLAN_RETRY_LIMIT = 3
_JUDGE_RETRY_LIMIT = 3
_RETRY_ARGS_PREVIEW_CHARS = 800
_RETRY_ERROR_PREVIEW_CHARS = 300


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


@dataclass(frozen=True)
class _StructuredResult:
    parsed: BaseModel | None
    raw: object | None = None
    parsing_error: object | None = None


@dataclass
class _SessionEvidenceContext:
    session_id: str
    xml: str
    turns: dict[int, str]
    text_search_cache: dict[str, list[dict]] = field(default_factory=dict)


class EvidenceAgent:
    def __init__(
        self,
        settings: Settings,
        turn_store: SessionTurnStore,
        *,
        workers: int = _EVIDENCE_WORKERS,
        failure_trace_path: Path | None = _DEFAULT_FAILURE_TRACE_PATH,
        reject_trace_path: Path | None = _DEFAULT_REJECT_TRACE_PATH,
    ):
        self._turn_store = turn_store
        self._workers = max(1, workers)
        self._failure_trace_path = failure_trace_path
        self._reject_trace_path = reject_trace_path
        self._trace_lock = threading.Lock()
        for p in [self._failure_trace_path, self._reject_trace_path]:
            if p is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
        self._plan_llm = ChatQwen(
            model="qwen-plus",
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_api_base,
            temperature=0,
        ).with_structured_output(EvidencePlan, include_raw=True)
        self._judge_llm = ChatQwen(
            model="qwen-plus",
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_api_base,
            temperature=0,
        ).with_structured_output(JudgmentOutput, include_raw=True)
        self._plan_prompt = load_prompt("evidence_plan_system.md")
        self._judge_prompt = load_prompt("evidence_judge_system.md")

    def _plan(self, record: KnowledgeRecord) -> EvidencePlan:
        base_messages = [
            SystemMessage(content=self._plan_prompt),
            HumanMessage(content=f"Insight: {record.insight}"),
        ]
        messages = base_messages
        for attempt in range(1, _PLAN_RETRY_LIMIT + 1):
            try:
                result = _parse_structured_result(self._plan_llm.invoke(messages))
            except Exception as exc:
                recovered = _truncate_plan_from_validation_error(exc)
                if recovered is not None:
                    self._trace_recovery(
                        stage="evidence_plan",
                        record=record,
                        note=str(exc) or exc.__class__.__name__,
                        exception_class=exc.__class__.__name__,
                        recovered_search_count=len(recovered.searches),
                        recovery_method="truncate_too_many_searches",
                        attempt=attempt,
                        max_attempts=_PLAN_RETRY_LIMIT,
                    )
                    return recovered
                if attempt < _PLAN_RETRY_LIMIT:
                    messages = _plan_retry_messages(base_messages, error=str(exc) or exc.__class__.__name__)
                    continue
                self._trace_failure(
                    stage="evidence_plan",
                    record=record,
                    error=str(exc) or exc.__class__.__name__,
                    exception_class=exc.__class__.__name__,
                    attempt=attempt,
                    max_attempts=_PLAN_RETRY_LIMIT,
                )
                raise
            if result is None:
                if attempt < _PLAN_RETRY_LIMIT:
                    messages = _plan_retry_messages(base_messages, error="structured plan returned None")
                    continue
                self._trace_recovery(
                    stage="evidence_plan",
                    record=record,
                    note="semantic_fallback_after_exhausted_retries",
                    recovery_method="semantic_fallback",
                    attempt=attempt,
                    max_attempts=_PLAN_RETRY_LIMIT,
                )
                return EvidencePlan(searches=[SearchAction(type="semantic", query=record.insight)])
            if result.parsed is None:
                recovered = _truncate_plan_from_validation_error(result.parsing_error)
                if recovered is not None:
                    self._trace_recovery(
                        stage="evidence_plan",
                        record=record,
                        note=str(result.parsing_error) or "structured plan returned None",
                        raw=result.raw,
                        parsing_error=result.parsing_error,
                        recovered_search_count=len(recovered.searches),
                        recovery_method="truncate_too_many_searches",
                        attempt=attempt,
                        max_attempts=_PLAN_RETRY_LIMIT,
                    )
                    return recovered
                if attempt < _PLAN_RETRY_LIMIT:
                    messages = _plan_retry_messages(
                        base_messages,
                        error="structured plan returned None",
                        raw=result.raw,
                        parsing_error=result.parsing_error,
                    )
                    continue
                self._trace_recovery(
                    stage="evidence_plan",
                    record=record,
                    note="semantic_fallback_after_exhausted_retries",
                    recovery_method="semantic_fallback",
                    raw=result.raw,
                    parsing_error=result.parsing_error,
                    attempt=attempt,
                    max_attempts=_PLAN_RETRY_LIMIT,
                )
                return EvidencePlan(searches=[SearchAction(type="semantic", query=record.insight)])
            if attempt > 1:
                self._trace_recovery(
                    stage="evidence_plan",
                    record=record,
                    note="structured plan recovered after retry",
                    raw=result.raw,
                    recovery_method="retry_with_plan_error_feedback",
                    attempt=attempt,
                    max_attempts=_PLAN_RETRY_LIMIT,
                )
            return result.parsed
        return EvidencePlan(searches=[SearchAction(type="semantic", query=record.insight)])

    def _execute(self, searches: list[SearchAction], context: _SessionEvidenceContext) -> str:
        parts: list[str] = []
        for i, action in enumerate(searches, 1):
            if action.type == "text":
                hits = self._search_text(context, action.query)
                part = f"=== 搜索 {i}: text({action.query!r}) ===\n"
                if hits:
                    for hit in hits[:_TEXT_SEARCH_LIMIT]:
                        content = context.turns.get(hit["turn_index"], "")
                        part += f"Turn {hit['turn_index']}:\n{content[:_FULL_TURN_LIMIT]}\n\n"
                else:
                    part += "(无匹配结果)\n"
            else:
                hits = self._turn_store.search_turns(context.session_id, action.query, top_k=_SEMANTIC_SEARCH_LIMIT)
                part = f"=== 搜索 {i}: semantic({action.query!r}) ===\n"
                if hits:
                    for hit in hits[:3]:
                        content = context.turns.get(hit["turn_index"], "")
                        part += (
                            f"Turn {hit['turn_index']} (score={hit['score']:.2f}):\n"
                            f"{content[:_FULL_TURN_LIMIT]}\n\n"
                        )
                else:
                    part += "(无语义相关结果)\n"
            parts.append(part)
        return "\n".join(parts)

    def _judge(
        self,
        record: KnowledgeRecord,
        search_results: str,
        *,
        final: bool = False,
        search_actions: list[SearchAction] | None = None,
    ) -> JudgmentOutput | None:
        system = self._judge_prompt
        if final:
            system += "\n\n这是最终判断，必须输出 admit 或 reject，不允许 need_more。"
        base_messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Insight: {record.insight}\n\n搜索结果:\n{search_results}"),
        ]
        messages = base_messages
        last_failed_raw: object | None = None
        last_failed_parsing_error: object | None = None
        for attempt in range(1, _JUDGE_RETRY_LIMIT + 1):
            try:
                result = _parse_structured_result(self._judge_llm.invoke(messages))
            except Exception as exc:
                if attempt < _JUDGE_RETRY_LIMIT:
                    messages = _judge_retry_messages(
                        base_messages,
                        error=str(exc) or exc.__class__.__name__,
                        final=final,
                    )
                    continue
                self._trace_failure(
                    stage="evidence_judge",
                    record=record,
                    error=str(exc) or exc.__class__.__name__,
                    exception_class=exc.__class__.__name__,
                    final=final,
                    search_actions=search_actions,
                    search_results=search_results,
                    attempt=attempt,
                    max_attempts=_JUDGE_RETRY_LIMIT,
                )
                return None
            if result is None:
                if attempt < _JUDGE_RETRY_LIMIT:
                    messages = _judge_retry_messages(
                        base_messages,
                        error="structured judge returned None",
                        final=final,
                    )
                    continue
                self._trace_failure(
                    stage="evidence_judge",
                    record=record,
                    error="structured judge returned None",
                    final=final,
                    search_actions=search_actions,
                    search_results=search_results,
                    attempt=attempt,
                    max_attempts=_JUDGE_RETRY_LIMIT,
                )
                return None
            if result.parsed is None:
                last_failed_raw = result.raw
                last_failed_parsing_error = result.parsing_error
                if attempt < _JUDGE_RETRY_LIMIT:
                    messages = _judge_retry_messages(
                        base_messages,
                        error="structured judge returned None",
                        raw=result.raw,
                        parsing_error=result.parsing_error,
                        final=final,
                    )
                    continue
                self._trace_failure(
                    stage="evidence_judge",
                    record=record,
                    error="structured judge returned None",
                    final=final,
                    search_actions=search_actions,
                    search_results=search_results,
                    raw=result.raw,
                    parsing_error=result.parsing_error,
                    attempt=attempt,
                    max_attempts=_JUDGE_RETRY_LIMIT,
                )
                return None
            if attempt > 1:
                self._trace_recovery(
                    stage="evidence_judge",
                    record=record,
                    note="structured judge recovered after retry",
                    raw=last_failed_raw or result.raw,
                    parsing_error=last_failed_parsing_error,
                    recovery_method="retry_with_judge_error_feedback",
                    final=final,
                    search_actions=search_actions,
                    search_results=search_results,
                    attempt=attempt,
                    max_attempts=_JUDGE_RETRY_LIMIT,
                )
            return result.parsed
        return None

    def _search_text(self, context: _SessionEvidenceContext, pattern: str) -> list[dict]:
        if pattern not in context.text_search_cache:
            context.text_search_cache[pattern] = _search_text_turns(context.turns, pattern)
        return context.text_search_cache[pattern]

    def verify(self, record: KnowledgeRecord, session_xml: str) -> KnowledgeRecord:
        return self._verify_with_context(record, _build_session_context(record.session_id, session_xml))

    def embedded_session_ids(self) -> set[str]:
        return self._turn_store.embedded_session_ids()

    def _verify_with_context(self, record: KnowledgeRecord, context: _SessionEvidenceContext) -> KnowledgeRecord:
        try:
            plan = self._plan(record)
            results = self._execute(plan.searches, context)
            search_actions = list(plan.searches)
            search_results = results
            final = False
            judgment = self._judge(record, results, final=False, search_actions=plan.searches)
            if judgment is None:
                return _rejected_record(record)

            if judgment.verdict == "need_more" and judgment.additional_searches:
                additional = self._execute(judgment.additional_searches, context)
                all_results = results + "\n\n=== 补充搜索 ===\n" + additional
                search_actions = plan.searches + judgment.additional_searches
                search_results = all_results
                final = True
                judgment = self._judge(
                    record,
                    all_results,
                    final=True,
                    search_actions=search_actions,
                )
                if judgment is None:
                    return _rejected_record(record)

            if judgment.verdict == "admit" and judgment.evidence_turns:
                return record.model_copy(
                    update={
                        "evidence_turns": judgment.evidence_turns,
                        "evidence_count": len(judgment.evidence_turns),
                    }
                )
            self._trace_reject(
                record=record,
                judgment=judgment,
                reject_reason="empty_admit_evidence" if judgment.verdict == "admit" else "judge_reject",
                final=final,
                search_actions=search_actions,
                search_results=search_results,
            )
        except Exception as exc:  # noqa: BLE001 - failed verification should keep the original record.
            _progress(f"evidence_agent verify failed record_id={record.id} error={exc}")
            return record

        return _rejected_record(record)

    def verify_batch(
        self,
        records: list[KnowledgeRecord],
        session_xmls: dict[str, str],
    ) -> list[KnowledgeRecord]:
        records_by_session: dict[str, list[KnowledgeRecord]] = {}
        for record in records:
            if record.session_id in session_xmls:
                records_by_session.setdefault(record.session_id, []).append(record)
        ineligible = [r for r in records if r.session_id not in session_xmls]

        results: list[KnowledgeRecord] = list(ineligible)
        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            futures = {
                executor.submit(
                    self._verify_session_records,
                    session_id,
                    session_records,
                    session_xmls[session_id],
                ): session_id
                for session_id, session_records in records_by_session.items()
            }
            for future in as_completed(futures):
                session_id = futures[future]
                try:
                    results.extend(future.result())
                except Exception as exc:  # noqa: BLE001 - one failed record should not stop the batch.
                    _progress(f"evidence_agent batch failed session_id={session_id} error={exc}")
                    results.extend(records_by_session[session_id])
        return results

    def _verify_session_records(
        self,
        session_id: str,
        records: list[KnowledgeRecord],
        session_xml: str,
    ) -> list[KnowledgeRecord]:
        context = _build_session_context(session_id, session_xml)
        return [self._verify_with_context(record, context) for record in records]

    def _trace_failure(
        self,
        *,
        stage: str,
        record: KnowledgeRecord,
        error: str,
        exception_class: str | None = None,
        final: bool | None = None,
        search_actions: list[SearchAction] | None = None,
        search_results: str | None = None,
        raw: object | None = None,
        parsing_error: object | None = None,
        attempt: int | None = None,
        max_attempts: int | None = None,
    ) -> None:
        if self._failure_trace_path is None:
            return
        payload: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event_type": "failure",
            "stage": stage,
            "record_id": record.id,
            "session_id": record.session_id,
            "title": record.title,
            "insight": record.insight,
            "error": error,
        }
        if attempt is not None:
            payload["attempt"] = attempt
        if max_attempts is not None:
            payload["max_attempts"] = max_attempts
        if exception_class is not None:
            payload["exception_class"] = exception_class
        if raw is not None:
            payload.update(_raw_trace_payload(raw))
        if parsing_error is not None:
            payload["parsing_error"] = str(parsing_error)[:_RAW_PREVIEW_CHARS]
        if final is not None:
            payload["final"] = final
        if search_actions is not None:
            payload["search_actions"] = [action.model_dump(mode="json") for action in search_actions]
        if search_results is not None:
            payload["search_results_chars"] = len(search_results)
            payload["search_results_hash"] = hashlib.sha1(search_results.encode("utf-8")).hexdigest()
            payload["search_results_preview"] = search_results[:_TRACE_PREVIEW_CHARS]

        self._write_trace(payload, self._failure_trace_path)

    def _trace_recovery(
        self,
        *,
        stage: str,
        record: KnowledgeRecord,
        note: str,
        exception_class: str | None = None,
        final: bool | None = None,
        search_actions: list[SearchAction] | None = None,
        search_results: str | None = None,
        raw: object | None = None,
        parsing_error: object | None = None,
        recovered_search_count: int | None = None,
        recovery_method: str | None = None,
        attempt: int | None = None,
        max_attempts: int | None = None,
    ) -> None:
        if self._failure_trace_path is None:
            return
        payload: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event_type": "recovery",
            "stage": stage,
            "record_id": record.id,
            "session_id": record.session_id,
            "title": record.title,
            "insight": record.insight,
            "note": note,
        }
        if recovered_search_count is not None:
            payload["recovered_search_count"] = recovered_search_count
        if recovery_method is not None:
            payload["recovery_method"] = recovery_method
        if attempt is not None:
            payload["attempt"] = attempt
        if max_attempts is not None:
            payload["max_attempts"] = max_attempts
        if exception_class is not None:
            payload["exception_class"] = exception_class
        if raw is not None:
            payload.update(_raw_trace_payload(raw))
        if parsing_error is not None:
            payload["parsing_error"] = str(parsing_error)[:_RAW_PREVIEW_CHARS]
        if final is not None:
            payload["final"] = final
        if search_actions is not None:
            payload["search_actions"] = [action.model_dump(mode="json") for action in search_actions]
        if search_results is not None:
            payload["search_results_chars"] = len(search_results)
            payload["search_results_hash"] = hashlib.sha1(search_results.encode("utf-8")).hexdigest()
            payload["search_results_preview"] = search_results[:_TRACE_PREVIEW_CHARS]

        self._write_trace(payload, self._failure_trace_path)

    def _trace_reject(
        self,
        *,
        record: KnowledgeRecord,
        judgment: JudgmentOutput,
        reject_reason: str,
        final: bool,
        search_actions: list[SearchAction],
        search_results: str,
    ) -> None:
        if self._reject_trace_path is None:
            return
        payload: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "evidence_reject",
            "record_id": record.id,
            "session_id": record.session_id,
            "title": record.title,
            "insight": record.insight,
            "final": final,
            "reject_reason": reject_reason,
            "judge_verdict": judgment.verdict,
            "judge_reasoning": judgment.reasoning,
            "judge_evidence_turns": judgment.evidence_turns,
            "judge_additional_searches": [action.model_dump(mode="json") for action in judgment.additional_searches],
            "search_actions": [action.model_dump(mode="json") for action in search_actions],
            "search_results_chars": len(search_results),
            "search_results_hash": hashlib.sha1(search_results.encode("utf-8")).hexdigest(),
            "search_results_preview": search_results[:_TRACE_PREVIEW_CHARS],
        }
        self._write_trace(payload, self._reject_trace_path)

    def _write_trace(self, payload: dict, path: Path | None) -> None:
        if path is None:
            return
        with self._trace_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _progress(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)


def _build_session_context(session_id: str, session_xml: str) -> _SessionEvidenceContext:
    turns = {int(match.group(1)): match.group(2).strip() for match in TURN_PATTERN.finditer(session_xml)}
    return _SessionEvidenceContext(session_id=session_id, xml=session_xml, turns=turns)


def _rejected_record(record: KnowledgeRecord) -> KnowledgeRecord:
    return record.model_copy(
        update={
            "evidence_turns": [],
            "evidence_count": 0,
        }
    )


def _plan_retry_messages(
    base_messages: list[SystemMessage | HumanMessage],
    *,
    error: str,
    raw: object | None = None,
    parsing_error: object | None = None,
) -> list[SystemMessage | HumanMessage]:
    feedback = [
        "Previous EvidencePlan attempt failed.",
        f"Error: {_truncate_text(error, _RETRY_ERROR_PREVIEW_CHARS)}",
    ]
    if parsing_error is not None:
        feedback.append(f"Parsing error: {_truncate_text(str(parsing_error), _RETRY_ERROR_PREVIEW_CHARS)}")
    invalid_tool_calls = getattr(raw, "invalid_tool_calls", None)
    if invalid_tool_calls:
        feedback.append(
            "Invalid tool-call args excerpt:\n"
            f"{_truncate_text(json.dumps(invalid_tool_calls, ensure_ascii=False, default=str), _RETRY_ARGS_PREVIEW_CHARS)}"
        )
    feedback.append(
        'Retry with exactly one valid EvidencePlan tool call. Required shape: {"searches":[{"type":"text|semantic","query":"..."}]}. '
        'Use the key "query", not malformed keys like "query:". Return 1-4 searches.'
    )
    return [*base_messages, HumanMessage(content="\n\n".join(feedback))]


def _judge_retry_messages(
    base_messages: list[SystemMessage | HumanMessage],
    *,
    error: str,
    raw: object | None = None,
    parsing_error: object | None = None,
    final: bool = False,
) -> list[SystemMessage | HumanMessage]:
    feedback = [
        "Previous JudgmentOutput attempt failed because the structured tool-call output was invalid.",
        f"Error: {_truncate_text(error, _RETRY_ERROR_PREVIEW_CHARS)}",
    ]
    if parsing_error is not None:
        feedback.append(f"Parsing error: {_truncate_text(str(parsing_error), _RETRY_ERROR_PREVIEW_CHARS)}")
    invalid_tool_calls = getattr(raw, "invalid_tool_calls", None)
    if invalid_tool_calls:
        feedback.append(
            "Invalid tool-call args excerpt:\n"
            f"{_truncate_text(json.dumps(invalid_tool_calls, ensure_ascii=False, default=str), _RETRY_ARGS_PREVIEW_CHARS)}"
        )
    verdicts = "admit|reject" if final else "admit|reject|need_more"
    need_more_note = " Final Judge must not use need_more." if final else ""
    feedback.append(
        "Retry with exactly one valid JudgmentOutput tool call. Do not manually copy raw JSON, code, regex, shell, or "
        "backslash-heavy snippets into reasoning; summarize the evidence in one short natural-language sentence. "
        f'Required shape: {{"reasoning":"short summary only","verdict":"{verdicts}",'
        '"evidence_turns":[...],"additional_searches":[]}}.'
        f"{need_more_note}"
    )
    return [*base_messages, HumanMessage(content="\n\n".join(feedback))]


def _parse_structured_result(result: object) -> _StructuredResult | None:
    if result is None:
        return None
    if isinstance(result, dict) and ("parsed" in result or "raw" in result or "parsing_error" in result):
        return _StructuredResult(
            parsed=result.get("parsed"),
            raw=result.get("raw"),
            parsing_error=result.get("parsing_error"),
        )
    if isinstance(result, BaseModel):
        return _StructuredResult(parsed=result)
    return _StructuredResult(parsed=None, raw=result)


def _truncate_plan_from_validation_error(exc: object) -> EvidencePlan | None:
    if not isinstance(exc, ValidationError):
        return None
    for error in exc.errors():
        if error.get("loc") == ("searches",) and error.get("type") == "too_long":
            raw_searches = error.get("input")
            if isinstance(raw_searches, list):
                try:
                    return EvidencePlan(searches=raw_searches[:4])
                except ValidationError:
                    return None
    return None


def _raw_trace_payload(raw: object) -> dict[str, object]:
    if not isinstance(raw, AIMessage):
        return {"raw_repr_preview": repr(raw)[:_RAW_PREVIEW_CHARS], "raw_type": type(raw).__name__}
    content = getattr(raw, "content", None)
    payload: dict[str, object] = {}
    if content:
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        payload["raw_content_chars"] = len(text)
        payload["raw_content_hash"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
        payload["raw_content_preview"] = text[:_RAW_PREVIEW_CHARS]
    additional_kwargs = getattr(raw, "additional_kwargs", None)
    if additional_kwargs:
        text = json.dumps(additional_kwargs, ensure_ascii=False, default=str)
        payload["raw_additional_kwargs_chars"] = len(text)
        payload["raw_additional_kwargs_hash"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
        payload["raw_additional_kwargs_preview"] = text[:_RAW_PREVIEW_CHARS]
    tool_calls = getattr(raw, "tool_calls", None)
    if tool_calls:
        text = json.dumps(tool_calls, ensure_ascii=False, default=str)
        payload["raw_tool_calls_chars"] = len(text)
        payload["raw_tool_calls_hash"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
        payload["raw_tool_calls_preview"] = text[:_RAW_PREVIEW_CHARS]
    invalid_tool_calls = getattr(raw, "invalid_tool_calls", None)
    if invalid_tool_calls:
        text = json.dumps(invalid_tool_calls, ensure_ascii=False, default=str)
        payload["raw_invalid_tool_calls_chars"] = len(text)
        payload["raw_invalid_tool_calls_hash"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
        payload["raw_invalid_tool_calls_preview"] = text[:_RAW_PREVIEW_CHARS]
    if not payload:
        text = repr(raw)
        payload["raw_repr_chars"] = len(text)
        payload["raw_repr_hash"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
        payload["raw_repr_preview"] = text[:_RAW_PREVIEW_CHARS]
    return payload


def _search_text_turns(turns: dict[int, str], pattern: str, snippet_chars: int = 50) -> list[dict]:
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error:
        compiled = re.compile(re.escape(pattern), re.IGNORECASE)
    results = []
    for turn_index, content in turns.items():
        match = compiled.search(content)
        if match:
            start = max(0, match.start() - snippet_chars // 2)
            end = min(len(content), match.end() + snippet_chars // 2)
            snippet = content[start:end].replace("\n", " ").strip()
            results.append({"turn_index": turn_index, "snippet": snippet})
    return results


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"
