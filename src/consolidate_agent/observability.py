from __future__ import annotations

from typing import Protocol


class AgentInvocationRecorder(Protocol):
    def record_agent_invocation(
        self,
        *,
        run_id: str | None,
        stage: str,
        status: str,
        input_payload: dict[str, object],
        output_payload: dict[str, object] | None = None,
        repaired_output_payload: dict[str, object] | None = None,
        validation_error: str | None = None,
        latency_ms: int = 0,
    ) -> None:
        ...


class NullAgentInvocationRecorder:
    def record_agent_invocation(
        self,
        *,
        run_id: str | None,
        stage: str,
        status: str,
        input_payload: dict[str, object],
        output_payload: dict[str, object] | None = None,
        repaired_output_payload: dict[str, object] | None = None,
        validation_error: str | None = None,
        latency_ms: int = 0,
    ) -> None:
        return
