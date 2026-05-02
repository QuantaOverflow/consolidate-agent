from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from pydantic import BaseModel


READ_COMMANDS = {"cat", "sed", "head", "tail", "less", "awk", "bat"}
TEST_SUMMARY_RE = re.compile(r"\b(passed|failed|error|errors|skipped|xfailed|xpassed)\b", re.I)
TRACEBACK_RE = re.compile(r"traceback|error|exception|failed|failure", re.I)


class ContextStats(BaseModel):
    raw_chars: int
    processed_chars: int
    compression_ratio: float


class ProcessedSession(BaseModel):
    session_id: str
    cwd: str | None
    thread_name: str | None
    started_at: str | None
    is_sub_agent: bool
    xml: str
    stats: ContextStats


@dataclass(frozen=True)
class _XmlEvent:
    timestamp: str | None
    xml: str


@dataclass(frozen=True)
class _SessionData:
    path: Path
    raw_text: str
    events: list[dict[str, Any]]
    session_id: str
    cwd: str | None
    thread_name: str | None
    started_at: str | None
    is_sub_agent: bool
    source: Any


@dataclass(frozen=True)
class _SubAgentSummary:
    session_id: str
    started_at: str | None
    thread_name: str | None
    summary: str


class SessionContextEngineer:
    def __init__(self, sessions_dir: Path | None = None):
        self.sessions_dir = sessions_dir or Path.home() / ".codex" / "sessions"

    def process(self, path: Path) -> ProcessedSession | None:
        data = self._load_session(path)
        if data is None or not data.events:
            return None

        xml = self._render_session(data)
        raw_chars = len(data.raw_text)
        processed_chars = len(xml)
        ratio = 0.0 if raw_chars == 0 else max(0.0, 1 - processed_chars / raw_chars)
        return ProcessedSession(
            session_id=data.session_id,
            cwd=data.cwd,
            thread_name=data.thread_name,
            started_at=data.started_at,
            is_sub_agent=data.is_sub_agent,
            xml=xml,
            stats=ContextStats(
                raw_chars=raw_chars,
                processed_chars=processed_chars,
                compression_ratio=ratio,
            ),
        )

    def _load_session(self, path: Path) -> _SessionData | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            raw_text = path.read_text(encoding="utf-8")
            events = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None

        session_id = path.stem.removeprefix("rollout-")
        cwd = None
        thread_name = None
        started_at = None
        source = None

        for event in events:
            event_type = event.get("type")
            payload = _as_dict(event.get("payload"))
            if event_type == "session_meta":
                session_id = str(payload.get("id") or session_id)
                cwd = _optional_str(payload.get("cwd"))
                started_at = _optional_str(payload.get("timestamp") or event.get("timestamp"))
                source = payload.get("source")
            elif event_type == "event_msg" and payload.get("type") == "thread_name_updated":
                thread_name = _optional_str(payload.get("thread_name"))

        if started_at is None and events:
            started_at = _optional_str(events[0].get("timestamp"))

        return _SessionData(
            path=path,
            raw_text=raw_text,
            events=events,
            session_id=session_id,
            cwd=cwd,
            thread_name=thread_name,
            started_at=started_at,
            is_sub_agent=isinstance(source, dict),
            source=source,
        )

    def _render_session(self, data: _SessionData) -> str:
        xml_events = self._convert_events(data)
        if not data.is_sub_agent:
            xml_events.extend(self._load_sub_agent_events(data))
        xml_events.sort(key=lambda item: item.timestamp or "")

        attrs = []
        if data.cwd is not None:
            attrs.append(f"cwd={quoteattr(data.cwd)}")
        if data.thread_name is not None:
            attrs.append(f"thread={quoteattr(data.thread_name)}")
        if data.started_at is not None:
            attrs.append(f"started_at={quoteattr(data.started_at)}")
        open_tag = "<session" + (f" {' '.join(attrs)}" if attrs else "") + ">"

        body = _group_into_turns(xml_events)
        return f"{open_tag}\n\n{body}\n\n</session>" if body else f"{open_tag}\n\n</session>"

    def _convert_events(self, data: _SessionData) -> list[_XmlEvent]:
        result: list[_XmlEvent] = []
        command_by_call_id: dict[str, str] = {}
        read_call_ids: set[str] = set()

        for event in data.events:
            timestamp = _optional_str(event.get("timestamp"))
            payload = _as_dict(event.get("payload"))
            if event.get("type") == "response_item":
                xml = self._convert_response_item(payload, command_by_call_id, read_call_ids)
            elif event.get("type") == "event_msg":
                xml = self._convert_event_msg(payload, command_by_call_id, read_call_ids)
            elif event.get("type") == "compacted":
                message = _optional_str(payload.get("message"))
                xml = _text_element("context_compacted", _truncate(message, 2000)) if message else "<context_compacted/>"
            else:
                xml = None
            if xml:
                result.append(_XmlEvent(timestamp=timestamp, xml=xml))
        return result

    def _convert_response_item(
        self,
        payload: dict[str, Any],
        command_by_call_id: dict[str, str],
        read_call_ids: set[str],
    ) -> str | None:
        payload_type = payload.get("type")
        if payload_type == "message":
            role = payload.get("role")
            text = _content_text(payload.get("content"))
            if role == "user":
                text = _clean_user_message(text)
                if not text:
                    return None
                return _text_element("user", text)
            if role == "assistant":
                return _text_element("assistant", _truncate_middle(text, 1000, 500, 1500))
            return None

        if payload_type == "function_call" and payload.get("name") in {"exec_command", "shell"}:
            args = _json_object(payload.get("arguments"))
            cmd = _command_from_arguments(args)
            call_id = _optional_str(payload.get("call_id"))
            if call_id is not None:
                command_by_call_id[call_id] = cmd
            if _is_file_read_command(cmd):
                if call_id is not None:
                    read_call_ids.add(call_id)
                path = _extract_read_path(cmd)
                return f"<file_read file={quoteattr(path or cmd)}/>"
            return _text_element("bash", _truncate(cmd, 200))

        if payload_type == "custom_tool_call" and payload.get("name") == "apply_patch":
            return _file_edit_element(_extract_patch_files(str(payload.get("input") or "")))

        if payload_type == "web_search_call":
            action = _as_dict(payload.get("action"))
            queries = action.get("queries")
            query = _optional_str(
                action.get("query")
                or (queries[0] if isinstance(queries, list) and queries else None)
                or payload.get("query")
            )
            if not query:
                return None
            action_type = _optional_str(action.get("type")) or "search"
            return f"<web_search action={quoteattr(action_type)} query={quoteattr(query)}/>"

        return None

    def _convert_event_msg(
        self,
        payload: dict[str, Any],
        command_by_call_id: dict[str, str],
        read_call_ids: set[str],
    ) -> str | None:
        payload_type = payload.get("type")
        if payload_type == "exec_command_end":
            call_id = _optional_str(payload.get("call_id"))
            cmd = _optional_str(payload.get("cmd")) or command_by_call_id.get(call_id or "", "")
            if call_id in read_call_ids or _is_file_read_command(cmd):
                return None
            exit_code = _exit_code(payload)
            output = _command_output(payload)
            if exit_code != 0:
                return _text_element(
                    "bash_result",
                    _truncate(_error_output(output), 600),
                    {"status": "failed"},
                )
            if _looks_like_test_output(output):
                return _text_element("bash_result", _test_summary(output), {"status": "passed"})
            return _text_element("bash_result", _truncate("\n".join(output.splitlines()[:5]), 300))

        if payload_type == "patch_apply_end":
            status = "success" if bool(payload.get("success", True)) else "failed"
            return f"<file_edit_result status={quoteattr(status)}/>"

        if payload_type == "web_search_end":
            query = _optional_str(
                payload.get("query")
                or _as_dict(payload.get("action")).get("query")
            )
            if query:
                return f"<web_search_done query={quoteattr(query)}/>"
            return "<web_search_done/>"

        if payload_type in {"rollback", "turn_rollback", "thread_rolled_back"}:
            return "<rollback/>"

        if payload_type == "context_compacted":
            return "<context_compacted/>"

        return None

    def _load_sub_agent_events(self, parent: _SessionData) -> list[_XmlEvent]:
        return [
            _XmlEvent(timestamp=summary.started_at, xml=_sub_agent_element(summary))
            for summary in self._find_sub_agents(parent)
        ]

    def _find_sub_agents(self, parent: _SessionData) -> list[_SubAgentSummary]:
        if not self.sessions_dir.exists():
            return []
        summaries: list[_SubAgentSummary] = []
        for path in self.sessions_dir.rglob("*.jsonl"):
            if path == parent.path:
                continue
            data = self._load_session(path)
            if data is None or not data.is_sub_agent or data.cwd != parent.cwd:
                continue
            if _parent_ref(data.source) != parent.session_id:
                continue
            summaries.append(
                _SubAgentSummary(
                    session_id=data.session_id,
                    started_at=data.started_at,
                    thread_name=data.thread_name,
                    summary=_sub_agent_summary(data),
                )
            )
        return summaries


def _group_into_turns(events: list[_XmlEvent]) -> str:
    preamble: list[str] = []
    turns: list[tuple[str | None, list[str]]] = []  # (started_at, items)
    current: list[str] | None = None
    current_ts: str | None = None

    for item in events:
        if not item.xml:
            continue
        if item.xml.startswith("<user>"):
            current = [item.xml]
            current_ts = item.timestamp
            turns.append((current_ts, current))
        elif current is None:
            if not preamble or preamble[-1] != item.xml:
                preamble.append(item.xml)
        else:
            if not current or current[-1] != item.xml:
                current.append(item.xml)

    parts: list[str] = []
    if preamble:
        parts.append("\n\n".join(preamble))
    for index, (started_at, turn_items) in enumerate(turns, 1):
        attrs = f"index={quoteattr(str(index))}"
        if started_at:
            attrs += f" started_at={quoteattr(started_at)}"
        inner = "\n\n".join(turn_items)
        parts.append(f"<turn {attrs}>\n{inner}\n</turn>")
    return "\n\n".join(parts)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


_INJECTED_USER_PREFIXES = (
    "# AGENTS.md instructions for ",
    "<environment_context>",
)
_IDE_CONTEXT_MARKER = "## My request for Codex:"


_IMAGE_TAG_RE = re.compile(r"<image[^>]*>.*?</image>", re.DOTALL)


def _clean_user_message(text: str) -> str:
    stripped = text.strip()
    for prefix in _INJECTED_USER_PREFIXES:
        if stripped.startswith(prefix):
            return ""
    if _IDE_CONTEXT_MARKER in stripped:
        stripped = stripped.split(_IDE_CONTEXT_MARKER, 1)[1].strip()
    stripped = _IMAGE_TAG_RE.sub("", stripped).strip()
    return stripped


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if text is None:
                text = item.get("content")
            if text is not None:
                parts.append(str(text))
    return "\n".join(parts)


def _text_element(tag: str, text: str, attrs: dict[str, str] | None = None) -> str:
    attr_text = ""
    if attrs:
        attr_text = " " + " ".join(f"{key}={quoteattr(value)}" for key, value in attrs.items())
    return f"<{tag}{attr_text}>{escape(text)}</{tag}>"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def _truncate_middle(text: str, head: int, tail: int, full_limit: int) -> str:
    if len(text) <= full_limit:
        return text
    return text[:head] + text[-tail:]


def _command_from_arguments(args: dict[str, Any]) -> str:
    if "cmd" in args:
        return str(args.get("cmd") or "")
    command = args.get("command")
    if isinstance(command, list):
        if len(command) >= 3 and command[0] in {"bash", "sh", "zsh"} and command[1] in {"-c", "-lc"}:
            return str(command[2])
        return " ".join(str(part) for part in command)
    return str(command or "")


def _main_command(cmd: str) -> str | None:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    while parts and (parts[0].endswith("=") or "=" in parts[0] and not parts[0].startswith("-")):
        parts.pop(0)
    if not parts:
        return None
    command = Path(parts[0]).name
    if command in {"bash", "sh", "zsh"} and len(parts) >= 3 and parts[1] in {"-c", "-lc"}:
        return _main_command(parts[2])
    return command


def _is_file_read_command(cmd: str | None) -> bool:
    return _main_command(cmd or "") in READ_COMMANDS


def _extract_read_path(cmd: str) -> str | None:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    command = _main_command(cmd)
    if command is None:
        return None
    for index, part in enumerate(parts):
        if Path(part).name == command:
            args = parts[index + 1 :]
            break
    else:
        args = parts[1:]
    candidates = [arg for arg in args if not arg.startswith("-") and not re.match(r"^\d", arg)]
    return candidates[-1] if candidates else None


def _exit_code(payload: dict[str, Any]) -> int:
    for key in ("exit_code", "status"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    metadata = _as_dict(payload.get("metadata"))
    value = metadata.get("exit_code")
    return int(value) if isinstance(value, int) or str(value).isdigit() else 0


def _command_output(payload: dict[str, Any]) -> str:
    for key in ("aggregated_output", "output", "stderr", "stdout"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    return ""


def _error_output(output: str) -> str:
    lines = output.splitlines()
    if not lines:
        return output
    for index, line in enumerate(lines):
        if "traceback" in line.lower():
            return "\n".join(lines[index : index + 5])
    selected = [line for line in lines if TRACEBACK_RE.search(line)]
    if selected:
        return "\n".join(selected[:5])
    return "\n".join(lines[:5])


def _looks_like_test_output(output: str) -> bool:
    return bool(TEST_SUMMARY_RE.search(output))


def _test_summary(output: str) -> str:
    lines = [line for line in output.splitlines() if TEST_SUMMARY_RE.search(line)]
    return _truncate("\n".join(lines), 600)


def _extract_patch_files(patch: str) -> list[str]:
    files: list[str] = []
    for line in patch.splitlines():
        prefix = None
        filename = None
        if line.startswith("*** Add File: "):
            prefix = "A"
            filename = line.removeprefix("*** Add File: ").strip()
        elif line.startswith("*** Update File: "):
            prefix = "M"
            filename = line.removeprefix("*** Update File: ").strip()
        elif line.startswith("*** Delete File: "):
            prefix = "D"
            filename = line.removeprefix("*** Delete File: ").strip()
        if prefix and filename:
            files.append(f"{prefix} {filename}")
    return files


def _file_edit_element(files: list[str]) -> str:
    return _text_element("file_edit", "\n".join(files))


def _sub_agent_element(summary: _SubAgentSummary) -> str:
    text = f"thread: {summary.thread_name or ''} | {summary.summary}".strip()
    return _text_element("sub_agent", text, {"ref": summary.session_id})


def _parent_ref(source: Any) -> str | None:
    if not isinstance(source, dict):
        return None
    for key in ("parent_session_id", "parent", "parent_id"):
        value = source.get(key)
        if value:
            return str(value)
    subagent = source.get("subagent")
    if isinstance(subagent, dict):
        for key in ("parent_session_id", "parent", "parent_id"):
            value = subagent.get(key)
            if value:
                return str(value)
        thread_spawn = _as_dict(subagent.get("thread_spawn"))
        value = thread_spawn.get("parent_thread_id")
        if value:
            return str(value)
    return None


def _sub_agent_summary(data: _SessionData) -> str:
    first_user = ""
    for event in data.events:
        payload = _as_dict(event.get("payload"))
        if event.get("type") == "event_msg" and payload.get("type") == "task_complete":
            message = str(payload.get("last_agent_message") or "")
            parsed = _json_object(message)
            summary = str(parsed.get("rationale") or message)
            return _truncate(summary, 200)
        if not first_user and event.get("type") == "response_item" and payload.get("type") == "message":
            if payload.get("role") == "user":
                first_user = _content_text(payload.get("content"))
    return _truncate(first_user, 200)
