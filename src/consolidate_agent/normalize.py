from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from consolidate_agent.types import MessageKind, MessageRole, Transcript, TranscriptMessage

LOW_SIGNAL_ASSISTANT_PATTERNS = (
    "我先看一下",
    "我再看一下",
    "我继续确认",
    "我再补一眼",
    "我先核对",
)

INDEX_ENTRY_ADAPTER = TypeAdapter(dict[str, Any])


class SessionIndex:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    @classmethod
    def load(cls, path: Path | None) -> "SessionIndex":
        mapping: dict[str, str] = {}
        if not path or not path.exists():
            return cls(mapping)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = INDEX_ENTRY_ADAPTER.validate_json(line)
                session_id = str(data.get("id") or "").strip()
                thread_name = str(data.get("thread_name") or "").strip()
                if session_id and thread_name:
                    mapping[session_id] = thread_name
        return cls(mapping)

    def get_thread_name(self, session_id: str) -> str | None:
        return self.mapping.get(session_id)


class SessionNormalizer:
    def __init__(self, index: SessionIndex | None = None):
        self.index = index or SessionIndex({})

    def normalize_file(self, path: Path) -> Transcript:
        raw_events = self._read_jsonl(path)
        meta = next((event for event in raw_events if event.get("type") == "session_meta"), {})
        payload = meta.get("payload", {})
        session_id = str(payload.get("id") or path.stem)
        transcript = Transcript(
            session_id=session_id,
            thread_name=self.index.get_thread_name(session_id),
            source=payload.get("source") or payload.get("originator"),
            cwd=payload.get("cwd"),
            started_at=payload.get("timestamp"),
            messages=[],
        )

        counter = 1
        for event in raw_events:
            normalized = self._normalize_event(event, counter)
            if normalized is None:
                continue
            transcript.messages.append(normalized)
            counter += 1
        return transcript

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def _normalize_event(self, event: dict[str, Any], counter: int) -> TranscriptMessage | None:
        event_type = event.get("type")
        timestamp = event.get("timestamp")
        if event_type == "response_item":
            payload = event.get("payload", {})
            payload_type = payload.get("type")
            if payload_type == "message":
                role = payload.get("role")
                text = self._flatten_content(payload.get("content"))
                if not text or self._is_excluded_message(role, text):
                    return None
                return TranscriptMessage(
                    ref=f"msg_{counter:04d}",
                    timestamp=timestamp,
                    role=MessageRole(role),
                    kind=MessageKind.MESSAGE,
                    text=self._compress_text(text),
                )
            if payload_type == "function_call":
                name = str(payload.get("name") or "")
                arguments = str(payload.get("arguments") or "").strip()
                text = f"{name}({arguments})" if arguments else name
                return TranscriptMessage(
                    ref=f"tool_{counter:04d}",
                    timestamp=timestamp,
                    role=MessageRole.TOOL,
                    kind=MessageKind.TOOL_CALL,
                    tool_name=name or None,
                    text=self._compress_text(text),
                )
            if payload_type == "function_call_output":
                text = self._extract_tool_output(payload.get("output"))
                if not text:
                    return None
                return TranscriptMessage(
                    ref=f"tool_{counter:04d}",
                    timestamp=timestamp,
                    role=MessageRole.TOOL,
                    kind=MessageKind.TOOL_OUTPUT,
                    text=text,
                )
            return None
        if event_type == "event_msg":
            payload = event.get("payload", {})
            if not self._is_high_signal_event(payload):
                return None
            text = self._extract_event_text(payload)
            if not text:
                return None
            return TranscriptMessage(
                ref=f"evt_{counter:04d}",
                timestamp=timestamp,
                role=MessageRole.TOOL,
                kind=MessageKind.EVENT,
                tool_name=payload.get("type"),
                text=text,
            )
        return None

    def _flatten_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    for key in ("text", "output_text", "input_text"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            parts.append(value)
                            break
            return "\n".join(parts)
        return ""

    def _is_excluded_message(self, role: str | None, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if role == "developer":
            return True
        if stripped.startswith("<environment_context>"):
            return True
        if role == "assistant" and any(stripped.startswith(prefix) for prefix in LOW_SIGNAL_ASSISTANT_PATTERNS):
            return True
        return False

    def _compress_text(self, text: str, limit: int = 1200) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _extract_tool_output(self, output: Any) -> str:
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        lower = output.lower()
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        high_signal: list[str] = []
        for line in lines:
            candidate = line.lower()
            if any(token in candidate for token in ("error", "exception", "traceback", "command not found", "exit code", "permission denied", "operation not permitted", "timed out", "no such file")):
                high_signal.append(line)
        if not high_signal:
            if "process exited with code" in lower or "output:" in lower:
                high_signal = lines[:6]
            else:
                return ""
        return self._compress_text(" | ".join(high_signal), limit=600)

    def _is_high_signal_event(self, payload: dict[str, Any]) -> bool:
        payload_type = payload.get("type")
        if payload_type == "exec_command_end":
            return True
        if payload_type and "error" in str(payload_type).lower():
            return True
        return bool(payload.get("status") == "failed" or payload.get("exit_code"))

    def _extract_event_text(self, payload: dict[str, Any]) -> str:
        payload_type = str(payload.get("type") or "event")
        parts: list[str] = [payload_type]
        command = payload.get("command")
        if isinstance(command, list):
            parts.append("command=" + " ".join(str(piece) for piece in command[:6]))
        exit_code = payload.get("exit_code")
        if exit_code not in (None, 0):
            parts.append(f"exit_code={exit_code}")
        aggregated = str(payload.get("aggregated_output") or "").strip()
        if aggregated:
            tool_output = self._extract_tool_output(aggregated)
            if tool_output:
                parts.append(tool_output)
        return self._compress_text(" | ".join(parts), limit=600)
