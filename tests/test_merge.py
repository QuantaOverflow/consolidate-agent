from __future__ import annotations

from consolidate_agent.merge import merge_session_candidates
from consolidate_agent.types import PitfallCandidate, PitfallCategory, PitfallScope


def candidate(ref: str, chunk_id: str, confidence: float = 0.5) -> PitfallCandidate:
    return PitfallCandidate(
        candidate_id=f"candidate-{ref}",
        session_id="session-1",
        title="Default interpreter assumption breaks validation",
        category=PitfallCategory.TOOLING_ENVIRONMENT,
        trigger="The workflow assumes python is available in PATH.",
        failure_mode="Validation fails before the real task is checked.",
        impact="Adds noisy debugging rounds.",
        preventive_rule="Use the project virtual environment interpreter for validation.",
        scope=PitfallScope.GLOBAL,
        evidence_refs=[ref],
        confidence=confidence,
        chunk_id=chunk_id,
    )


def test_merge_session_candidates_dedupes_and_merges_evidence() -> None:
    merged = merge_session_candidates([candidate("tool_0002", "chunk-2", 0.6), candidate("tool_0001", "chunk-1", 0.8)])

    assert len(merged) == 1
    assert merged[0].evidence_refs == ["tool_0001", "tool_0002"]
    assert merged[0].confidence == 0.8
    assert merged[0].chunk_id == "chunk-2"
