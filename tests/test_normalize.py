from __future__ import annotations

import json
from pathlib import Path

from consolidate_agent.normalize import SessionIndex, SessionNormalizer


def write_session(path: Path) -> None:
    events = [
        {
            "timestamp": "2026-04-14T09:15:26.524Z",
            "type": "session_meta",
            "payload": {
                "id": "session-1",
                "timestamp": "2026-04-14T09:15:26.524Z",
                "cwd": "/tmp/project",
                "source": "cli",
            },
        },
        {
            "timestamp": "2026-04-14T09:15:26.525Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "ignore me"}],
            },
        },
        {
            "timestamp": "2026-04-14T09:15:27.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "please fix startup"}],
            },
        },
        {
            "timestamp": "2026-04-14T09:15:28.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "问题已经明确了：asyncio.run 嵌套冲突"}],
            },
        },
        {
            "timestamp": "2026-04-14T09:15:29.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"python main.py report"}',
            },
        },
        {
            "timestamp": "2026-04-14T09:15:30.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "output": "Chunk ID: abc\nProcess exited with code 127\nOutput:\nzsh:1: command not found: python\n",
            },
        },
        {
            "timestamp": "2026-04-14T09:15:31.000Z",
            "type": "event_msg",
            "payload": {
                "type": "exec_command_end",
                "command": ["/bin/zsh", "-lc", "python main.py report"],
                "exit_code": 127,
                "aggregated_output": "zsh:1: command not found: python",
            },
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def test_normalizer_keeps_high_signal_items(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    write_session(session_path)
    index = SessionIndex({"session-1": "Fix startup"})

    transcript = SessionNormalizer(index).normalize_file(session_path)

    assert transcript.session_id == "session-1"
    assert transcript.thread_name == "Fix startup"
    assert [item.role.value for item in transcript.messages] == ["user", "assistant", "tool", "tool", "tool"]
    assert transcript.messages[1].text.startswith("问题已经明确了")
    assert "command not found: python" in transcript.messages[3].text
    assert transcript.messages[-1].kind.value == "event"
