from __future__ import annotations

from consolidate_agent.extraction.pipeline import _should_admit
from consolidate_agent.types import PitfallCandidate, PitfallCategory, PitfallScope


def make_candidate(**overrides) -> PitfallCandidate:
    data = {
        "candidate_id": "candidate-1",
        "session_id": "session-1",
        "title": "Validate JSON before parsing",
        "category": PitfallCategory.EXECUTION_STRATEGY,
        "trigger": "The model returns structured data.",
        "failure_mode": "Invalid structure reaches downstream code.",
        "impact": "The workflow fails late.",
        "preventive_rule": "Validate JSON payloads against a schema before parsing model output.",
        "scope": PitfallScope.GLOBAL,
        "evidence_refs": ["msg_0001"],
        "confidence": 0.9,
    }
    data.update(overrides)
    return PitfallCandidate(**data)


def test_should_admit_rejects_session_specific_scope() -> None:
    candidate = make_candidate(scope=PitfallScope.SESSION_SPECIFIC)

    assert _should_admit(candidate) is False


def test_should_admit_rejects_empty_required_fields() -> None:
    for field_name in ("trigger", "failure_mode", "preventive_rule"):
        candidate = make_candidate(**{field_name: ""})

        assert _should_admit(candidate) is False


def test_should_admit_rejects_empty_evidence_refs() -> None:
    candidate = make_candidate(evidence_refs=[])

    assert _should_admit(candidate) is False


def test_should_admit_accepts_valid_global_scope_candidate() -> None:
    candidate = make_candidate(scope=PitfallScope.GLOBAL)

    assert _should_admit(candidate) is True


def test_should_admit_accepts_valid_project_specific_scope_candidate() -> None:
    candidate = make_candidate(scope=PitfallScope.PROJECT_SPECIFIC)

    assert _should_admit(candidate) is True
