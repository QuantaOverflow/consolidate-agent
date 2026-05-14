from __future__ import annotations

import json
from pathlib import Path

from consolidate_agent.context.engineer import SessionContextEngineer


def write_jsonl(path: Path, events: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in events), encoding="utf-8")
    return path


def event(timestamp: str, event_type: str, payload: dict) -> dict:
    return {"timestamp": timestamp, "type": event_type, "payload": payload}


def meta(session_id: str = "parent", *, source: object = "cli", cwd: str = "/repo") -> dict:
    return event(
        "2026-04-17T09:08:38Z",
        "session_meta",
        {"id": session_id, "timestamp": "2026-04-17T09:08:38Z", "cwd": cwd, "source": source},
    )


def message(role: str, text: str, timestamp: str = "2026-04-17T09:08:39Z") -> dict:
    return event(
        timestamp,
        "response_item",
        {"type": "message", "role": role, "content": [{"type": "output_text", "text": text}]},
    )


def exec_call(cmd: str, call_id: str = "call_1", timestamp: str = "2026-04-17T09:08:40Z") -> dict:
    return event(
        timestamp,
        "response_item",
        {
            "type": "function_call",
            "name": "exec_command",
            "arguments": json.dumps({"cmd": cmd}),
            "call_id": call_id,
        },
    )


def exec_end(
    output: str,
    *,
    exit_code: int = 0,
    call_id: str = "call_1",
    timestamp: str = "2026-04-17T09:08:41Z",
) -> dict:
    return event(
        timestamp,
        "event_msg",
        {
            "type": "exec_command_end",
            "call_id": call_id,
            "exit_code": exit_code,
            "aggregated_output": output,
        },
    )


def test_converts_core_event_types(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "session.jsonl",
        [
            meta(),
            event(
                "2026-04-17T09:08:38Z",
                "event_msg",
                {"type": "thread_name_updated", "thread_name": "重构认证模块"},
            ),
            message("user", "能不能把认证模块重构一下"),
            message("assistant", "看了下现有结构，主要问题是..."),
            exec_call("uv run pytest tests/ -v"),
            exec_end("3 failed, 12 passed in 1.2s"),
            event(
                "2026-04-17T09:08:42Z",
                "response_item",
                {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "input": "\n".join(
                        [
                            "*** Begin Patch",
                            "*** Update File: src/auth/login.py",
                            "*** Add File: tests/unit/test_auth.py",
                            "*** Delete File: old.py",
                            "*** End Patch",
                        ]
                    ),
                },
            ),
            event("2026-04-17T09:08:43Z", "event_msg", {"type": "patch_apply_end", "success": True}),
            event(
                "2026-04-17T09:08:44Z",
                "response_item",
                {"type": "web_search_call", "query": "langchain structured output None"},
            ),
            event(
                "2026-04-17T09:08:45Z",
                "event_msg",
                {"type": "web_search_end", "result": "When using with_structured_output, returning None usually indicates..."},
            ),
            event("2026-04-17T09:08:46Z", "event_msg", {"type": "rollback"}),
            event("2026-04-17T09:08:47Z", "event_msg", {"type": "token_count", "info": {"total": 1}}),
        ],
    )

    processed = SessionContextEngineer(sessions_dir=tmp_path).process(path)

    assert processed is not None
    assert processed.session_id == "parent"
    assert processed.cwd == "/repo"
    assert processed.thread_name == "重构认证模块"
    assert '<session cwd="/repo" thread="重构认证模块" started_at="2026-04-17T09:08:38Z">' in processed.xml
    assert "<user>能不能把认证模块重构一下</user>" in processed.xml
    assert "<assistant>看了下现有结构，主要问题是...</assistant>" in processed.xml
    assert "<bash>uv run pytest tests/ -v</bash>" in processed.xml
    assert '<bash_result status="passed">3 failed, 12 passed in 1.2s</bash_result>' in processed.xml
    assert "M src/auth/login.py" in processed.xml
    assert "A tests/unit/test_auth.py" in processed.xml
    assert "D old.py" in processed.xml
    assert '<file_edit_result status="success"/>' in processed.xml
    assert 'query="langchain structured output None"' in processed.xml
    assert "<web_search_done" in processed.xml
    assert "<rollback/>" in processed.xml
    assert "token_count" not in processed.xml


def test_file_read_command_becomes_file_read_without_output(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "session.jsonl",
        [
            meta(),
            exec_call("sed -n '1,120p' src/auth/login.py", call_id="read_1"),
            exec_end("secret file contents", call_id="read_1"),
        ],
    )

    processed = SessionContextEngineer(sessions_dir=tmp_path).process(path)

    assert processed is not None
    assert '<file_read file="src/auth/login.py"/>' in processed.xml
    assert "secret file contents" not in processed.xml


def test_assistant_long_message_keeps_first_1000_and_last_500(tmp_path: Path) -> None:
    text = "A" * 1000 + "M" * 700 + "Z" * 500
    path = write_jsonl(tmp_path / "session.jsonl", [meta(), message("assistant", text)])

    processed = SessionContextEngineer(sessions_dir=tmp_path).process(path)

    assert processed is not None
    body = processed.xml
    assert "A" * 1000 in body
    assert "Z" * 500 in body
    assert "M" * 700 not in body


def test_failed_command_keeps_error_signal_and_traceback_head(tmp_path: Path) -> None:
    output = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "app.py", line 1, in <module>',
            "ValueError: bad input",
            "extra error detail",
            "another failure line",
            "ignored failure line",
        ]
    )
    path = write_jsonl(tmp_path / "session.jsonl", [meta(), exec_call("python app.py"), exec_end(output, exit_code=1)])

    processed = SessionContextEngineer(sessions_dir=tmp_path).process(path)

    assert processed is not None
    assert '<bash_result status="failed">' in processed.xml
    assert "Traceback (most recent call last):" in processed.xml
    assert "ValueError: bad input" in processed.xml
    assert "ignored failure line" not in processed.xml


def test_success_non_test_output_is_first_five_lines(tmp_path: Path) -> None:
    output = "\n".join([f"line {index}" for index in range(1, 8)])
    path = write_jsonl(tmp_path / "session.jsonl", [meta(), exec_call("ls"), exec_end(output)])

    processed = SessionContextEngineer(sessions_dir=tmp_path).process(path)

    assert processed is not None
    assert "<bash_result>line 1" in processed.xml
    assert "line 5" in processed.xml
    assert "line 6" not in processed.xml


def test_sub_agent_process_returns_marked_session_and_parent_inlines_summary(tmp_path: Path) -> None:
    parent = write_jsonl(
        tmp_path / "parent.jsonl",
        [
            meta("parent"),
            message("user", "parent start", "2026-04-17T09:00:00Z"),
            message("assistant", "parent end", "2026-04-17T09:10:00Z"),
        ],
    )
    child = write_jsonl(
        tmp_path / "child.jsonl",
        [
            meta("child", source={"subagent": {"parent_session_id": "parent"}}),
            event(
                "2026-04-17T09:05:00Z",
                "event_msg",
                {"type": "thread_name_updated", "thread_name": "fix-test-failures"},
            ),
            message("user", "fix tests", "2026-04-17T09:05:01Z"),
            event(
                "2026-04-17T09:06:00Z",
                "event_msg",
                {"type": "task_complete", "last_agent_message": json.dumps({"rationale": "修复了 3 个测试失败，更新了 retry 逻辑"})},
            ),
        ],
    )

    child_processed = SessionContextEngineer(sessions_dir=tmp_path).process(child)
    parent_processed = SessionContextEngineer(sessions_dir=tmp_path).process(parent)

    assert child_processed is not None
    assert child_processed.is_sub_agent is True
    assert parent_processed is not None
    assert '<sub_agent ref="child">thread: fix-test-failures | 修复了 3 个测试失败，更新了 retry 逻辑</sub_agent>' in parent_processed.xml
    assert parent_processed.xml.index("<user>parent start</user>") < parent_processed.xml.index("<sub_agent")
    assert parent_processed.xml.index("<sub_agent") < parent_processed.xml.index("<assistant>parent end</assistant>")


def test_thread_rolled_back_and_context_compacted(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "session.jsonl",
        [
            meta(),
            event("2026-04-17T09:08:39Z", "event_msg", {"type": "thread_rolled_back"}),
            event("2026-04-17T09:08:40Z", "event_msg", {"type": "context_compacted"}),
        ],
    )
    processed = SessionContextEngineer(sessions_dir=tmp_path).process(path)
    assert processed is not None
    assert "<rollback/>" in processed.xml
    assert "<context_compacted/>" in processed.xml


def test_sub_agent_with_unknown_parent_not_included(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "child.jsonl",
        [meta("child", source={"subagent": {}})],  # no parent_session_id
    )
    parent = write_jsonl(
        tmp_path / "parent.jsonl",
        [meta("parent"), message("user", "hi")],
    )
    processed = SessionContextEngineer(sessions_dir=tmp_path).process(parent)
    assert processed is not None
    assert "<sub_agent" not in processed.xml


def test_empty_missing_and_invalid_files_return_none(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{bad json", encoding="utf-8")

    engineer = SessionContextEngineer(sessions_dir=tmp_path)

    assert engineer.process(empty) is None
    assert engineer.process(invalid) is None
    assert engineer.process(tmp_path / "missing.jsonl") is None
