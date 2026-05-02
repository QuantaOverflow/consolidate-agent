from __future__ import annotations

import json
import re
from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel

from consolidate_agent.config import Settings
from consolidate_agent.observability import AgentInvocationRecorder


def _invoke_with_retry(
    *,
    stage: str,
    structured_model: object,
    prompt: ChatPromptTemplate,
    retry_prompt: ChatPromptTemplate,
    values: dict[str, str],
    recorder: AgentInvocationRecorder,
    run_id: str | None,
    validator: object,
    empty_error: str,
) -> object:
    started = perf_counter()
    result = None
    try:
        result = _invoke_structured(structured_model, prompt, values, empty_error)
        validator(result)
    except Exception as exc:
        recorder.record_agent_invocation(
            run_id=run_id,
            stage=stage,
            status="validation_failed",
            input_payload=values,
            output_payload=result.model_dump(mode="json") if isinstance(result, BaseModel) else None,
            validation_error=str(exc),
            latency_ms=_elapsed_ms(started),
        )
        retry_values = dict(values)
        retry_values["validation_error"] = str(exc)
        retry_values["invalid_result_json"] = (
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
            if isinstance(result, BaseModel)
            else "null"
        )
        retry_started = perf_counter()
        retry_result = None
        try:
            retry_result = _invoke_structured(structured_model, retry_prompt, retry_values, empty_error)
            validator(retry_result)
        except Exception as retry_exc:
            recorder.record_agent_invocation(
                run_id=run_id,
                stage=f"{stage}_retry",
                status="failure",
                input_payload=retry_values,
                output_payload=retry_result.model_dump(mode="json") if isinstance(retry_result, BaseModel) else None,
                validation_error=f"{exc}; retry: {retry_exc}",
                latency_ms=_elapsed_ms(retry_started),
            )
            raise
        recorder.record_agent_invocation(
            run_id=run_id,
            stage=f"{stage}_retry",
            status="success",
            input_payload=retry_values,
            output_payload=retry_result.model_dump(mode="json"),
            latency_ms=_elapsed_ms(retry_started),
        )
        return retry_result
    recorder.record_agent_invocation(
        run_id=run_id,
        stage=stage,
        status="success",
        input_payload=values,
        output_payload=result.model_dump(mode="json"),
        latency_ms=_elapsed_ms(started),
    )
    return result


def _invoke_structured(structured_model: object, prompt: ChatPromptTemplate, values: dict[str, str], empty_error: str) -> BaseModel:
    prompt_value = prompt.invoke(values)
    result = structured_model.invoke(prompt_value)
    if result is None:
        raise ValueError(empty_error)
    return result


def _chat_model(settings: Settings) -> ChatQwen:
    if not settings.dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY is not configured. Fill .env before running LLM taxonomy consolidation.")
    return ChatQwen(
        model=settings.qwen_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_api_base,
        temperature=0,
    )


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def _token_overlap(left: str, right: str) -> int:
    tokens_left = {token for token in re.findall(r"[a-z0-9_]+", left) if len(token) > 3}
    tokens_right = {token for token in re.findall(r"[a-z0-9_]+", right) if len(token) > 3}
    return len(tokens_left & tokens_right)
