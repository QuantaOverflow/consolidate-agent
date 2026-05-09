from __future__ import annotations

import json
from pathlib import Path

from consolidate_agent.types import KnowledgeRecord


def export_to_obsidian(
    records: list[KnowledgeRecord],
    tags_map: dict[str, list[str]],
    output_dir: Path,
    assignments_map: dict[str, list[str]] | None = None,
) -> int:
    """Export admitted records as Obsidian Markdown notes and return the file count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        source_map = assignments_map if assignments_map is not None else tags_map
        tags = source_map.get(record.id, [])
        (output_dir / f"{record.id}.md").write_text(_render_note(record, tags), encoding="utf-8")
    return len(records)


def _render_note(record: KnowledgeRecord, tags: list[str]) -> str:
    frontmatter_tags = ", ".join(tags)
    return (
        "---\n"
        f"title: {json.dumps(record.title, ensure_ascii=False)}\n"
        f"scope: {record.scope.value}\n"
        f"evidence_count: {record.evidence_count}\n"
        f"tags: [{frontmatter_tags}]\n"
        f"session_id: {record.session_id}\n"
        f"evidence_turns: {record.evidence_turns}\n"
        f"created_at: {record.created_at.isoformat()}\n"
        "---\n\n"
        "## Insight\n\n"
        f"{record.insight}\n\n"
        "## Applicability\n\n"
        f"{record.applicability}\n"
    )
