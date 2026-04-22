from __future__ import annotations

from consolidate_agent.admit import PitfallLibrary
from consolidate_agent.types import AdmissionStatus, PitfallCandidate, PitfallCategory, PitfallScope


def make_candidate(scope: PitfallScope = PitfallScope.GLOBAL) -> PitfallCandidate:
    return PitfallCandidate(
        candidate_id="candidate_1",
        session_id="session-1",
        title="Default interpreter assumption breaks validation",
        category=PitfallCategory.TOOLING_ENVIRONMENT,
        trigger="The workflow assumes python is available in PATH.",
        failure_mode="Validation fails before the real task is checked.",
        impact="Adds a noisy debugging round.",
        preventive_rule="Use the project virtual environment interpreter for validation.",
        scope=scope,
        evidence_refs=["tool_0001"],
        confidence=0.81,
    )


def test_session_specific_candidates_are_rejected() -> None:
    library = PitfallLibrary([])
    accepted, rejected = library.admit([make_candidate(PitfallScope.SESSION_SPECIFIC)])

    assert accepted == []
    assert rejected[0].admission_status == AdmissionStatus.REJECTED


def test_duplicate_candidates_merge_into_one_record() -> None:
    library = PitfallLibrary([])
    first = make_candidate()
    second = make_candidate()
    second.session_id = "session-2"
    second.evidence_refs = ["tool_0002"]

    accepted, rejected = library.admit([first, second])

    assert len(rejected) == 0
    assert len(library.records) == 1
    assert len(accepted) == 2
    assert sorted(library.records[0].evidence.session_ids) == ["session-1", "session-2"]
