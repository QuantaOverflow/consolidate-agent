from __future__ import annotations

import os
from pathlib import Path

import pytest

from consolidate_agent.config import Settings
from consolidate_agent.consolidation.pipeline import run_consolidation
from consolidate_agent.extraction.pipeline import ConsolidationGraph
from consolidate_agent.types import PipelineState


@pytest.mark.skipif(not os.getenv("DASHSCOPE_API_KEY"), reason="DASHSCOPE_API_KEY is required")
def test_full_pipeline_extracts_and_consolidates_with_real_llm(tmp_path: Path) -> None:
    sessions_dir = Path("~/.codex/sessions").expanduser()
    if not sessions_dir.exists():
        pytest.skip("~/.codex/sessions does not exist")
    if not list(sessions_dir.rglob("*.jsonl"))[:3]:
        pytest.skip("no Codex session jsonl files found")

    db_path = tmp_path / "knowledge.db"
    state = PipelineState(
        input_dir=str(sessions_dir),
        output_dir=str(tmp_path / "outputs"),
        cursor_path=str(tmp_path / "cursor.json"),
        processed_index_path=str(tmp_path / "processed-index.json"),
        knowledge_db_path=str(db_path),
        sample_limit=3,
    )
    settings = Settings()

    result = ConsolidationGraph(settings).invoke(state)
    consolidation_stats = run_consolidation(db_path, settings=settings)

    assert result.stats.processed_sessions > 0
    assert result.stats.accepted_count > 0
    assert consolidation_stats.canonical_created > 0
