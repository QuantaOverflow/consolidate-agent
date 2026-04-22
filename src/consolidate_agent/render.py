from __future__ import annotations

import json
from pathlib import Path

from consolidate_agent.types import CursorState, PitfallCandidate, PitfallRecord


def write_candidates(path: Path, candidates: list[PitfallCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [candidate.model_dump(mode="json") for candidate in candidates]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_pitfalls_markdown(path: Path, records: list[PitfallRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pitfall Library", ""]
    grouped: dict[str, list[PitfallRecord]] = {}
    for record in records:
        grouped.setdefault(record.category.value, []).append(record)
    for category in sorted(grouped):
        lines.append(f"## {category}")
        lines.append("")
        for record in sorted(grouped[category], key=lambda item: item.updated_at, reverse=True):
            lines.append(f"### {record.title}")
            lines.append(f"- Scope: `{record.scope.value}`")
            lines.append(f"- Confidence: `{record.confidence:.2f}`")
            lines.append(f"- Trigger: {record.trigger}")
            lines.append(f"- Failure mode: {record.failure_mode}")
            lines.append(f"- Preventive rule: {record.preventive_rule}")
            lines.append(f"- Evidence refs: {', '.join(record.evidence.message_refs)}")
            lines.append(f"- Session ids: {', '.join(record.evidence.session_ids)}")
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_cursor(path: Path, cursor: CursorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cursor.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")


def read_cursor(path: Path) -> CursorState:
    if not path.exists():
        return CursorState()
    return CursorState.model_validate(json.loads(path.read_text(encoding="utf-8")))
