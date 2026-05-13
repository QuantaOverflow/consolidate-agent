"""Unit tests for measure_v2 — Pydantic models and vocab loader.

Does NOT test LLM calls (external dependency).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from consolidate_agent.vocab_maintenance.measure_v2 import (
    FacetedBatchOutput,
    FacetedRecordAssignment,
    TagSelection,
    load_faceted_vocab,
)

VOCAB_PATH = Path(__file__).parent.parent / "fixtures" / "vocab_v2.json"
REAL_VOCAB_PATH = Path(__file__).parent.parent.parent / "docs" / "plans" / "vocab_v2.json"


def _vocab_path() -> Path:
    if REAL_VOCAB_PATH.exists():
        return REAL_VOCAB_PATH
    if VOCAB_PATH.exists():
        return VOCAB_PATH
    pytest.skip("vocab_v2.json not found")


# ── load_faceted_vocab ────────────────────────────────────────────────────────


class TestLoadFacetedVocab:
    def test_facets_not_empty(self):
        vocab = load_faceted_vocab(_vocab_path())
        assert vocab["facets"]["matter"]["tags"], "matter tags must not be empty"
        assert vocab["facets"]["activity"]["tags"], "activity tags must not be empty"
        assert vocab["facets"]["pattern"]["tags"], "pattern tags must not be empty"

    def test_lesson_type_values_not_empty(self):
        vocab = load_faceted_vocab(_vocab_path())
        assert vocab["lesson_type"]["values"], "lesson_type.values must not be empty"

    def test_matter_tag_count_in_range(self):
        vocab = load_faceted_vocab(_vocab_path())
        count = len(vocab["facets"]["matter"]["tags"])
        assert 8 <= count <= 22, f"matter tag count {count} outside warn range 8-22"

    def test_activity_tag_count_in_range(self):
        vocab = load_faceted_vocab(_vocab_path())
        count = len(vocab["facets"]["activity"]["tags"])
        assert 4 <= count <= 12, f"activity tag count {count} outside warn range 4-12"

    def test_pattern_tag_count_in_range(self):
        vocab = load_faceted_vocab(_vocab_path())
        count = len(vocab["facets"]["pattern"]["tags"])
        assert 3 <= count <= 12, f"pattern tag count {count} outside warn range 3-12"

    def test_lesson_type_exactly_3_values(self):
        vocab = load_faceted_vocab(_vocab_path())
        count = len(vocab["lesson_type"]["values"])
        assert count == 3, f"expected 3 lesson_type values, got {count}"

    def test_tags_have_name_and_definition(self):
        vocab = load_faceted_vocab(_vocab_path())
        for facet_name in ("matter", "activity", "pattern"):
            for tag in vocab["facets"][facet_name]["tags"]:
                assert "name" in tag, f"{facet_name} tag missing 'name': {tag}"
                assert "definition" in tag, f"{facet_name} tag missing 'definition': {tag}"


# ── FacetedRecordAssignment ───────────────────────────────────────────────────


def _valid_assignment(**overrides) -> dict:
    base = {
        "record_idx": 1,
        "matter_tags": [{"name": "langgraph_state", "confidence": "high"}],
        "activity_tag": {"name": "designing", "confidence": "high"},
        "pattern_tags": [{"name": "silent_failure", "confidence": "medium"}],
        "lesson_type": "anti_pattern",
        "reason": "some reason",
    }
    base.update(overrides)
    return base


class TestFacetedRecordAssignment:
    def test_accepts_valid_4_fields(self):
        a = FacetedRecordAssignment(**_valid_assignment())
        assert a.record_idx == 1
        assert len(a.matter_tags) == 1
        assert a.activity_tag.name == "designing"
        assert a.lesson_type == "anti_pattern"

    def test_accepts_empty_pattern_tags(self):
        a = FacetedRecordAssignment(**_valid_assignment(pattern_tags=[]))
        assert a.pattern_tags == []

    def test_rejects_empty_matter_tags(self):
        with pytest.raises(ValidationError):
            FacetedRecordAssignment(**_valid_assignment(matter_tags=[]))

    def test_rejects_pattern_tags_exceeding_max(self):
        too_many = [
            {"name": "silent_failure", "confidence": "high"},
            {"name": "explicit_contract", "confidence": "medium"},
            {"name": "early_validation", "confidence": "medium"},
        ]
        with pytest.raises(ValidationError):
            FacetedRecordAssignment(**_valid_assignment(pattern_tags=too_many))

    def test_rejects_invalid_lesson_type(self):
        with pytest.raises(ValidationError):
            FacetedRecordAssignment(**_valid_assignment(lesson_type="unknown_type"))

    def test_rejects_invalid_lesson_type_empty(self):
        with pytest.raises(ValidationError):
            FacetedRecordAssignment(**_valid_assignment(lesson_type=""))

    def test_accepts_all_lesson_type_variants(self):
        for lt in ("anti_pattern", "discovery", "best_practice"):
            a = FacetedRecordAssignment(**_valid_assignment(lesson_type=lt))
            assert a.lesson_type == lt

    def test_accepts_two_matter_tags(self):
        two = [
            {"name": "langgraph_state", "confidence": "high"},
            {"name": "persistence_db", "confidence": "medium"},
        ]
        a = FacetedRecordAssignment(**_valid_assignment(matter_tags=two))
        assert len(a.matter_tags) == 2

    def test_accepts_two_pattern_tags(self):
        two = [
            {"name": "silent_failure", "confidence": "high"},
            {"name": "explicit_contract", "confidence": "medium"},
        ]
        a = FacetedRecordAssignment(**_valid_assignment(pattern_tags=two))
        assert len(a.pattern_tags) == 2

    def test_reason_defaults_to_empty_string(self):
        data = _valid_assignment()
        del data["reason"]
        a = FacetedRecordAssignment(**data)
        assert a.reason == ""


# ── FacetedBatchOutput ────────────────────────────────────────────────────────


class TestFacetedBatchOutput:
    def test_accepts_list_of_assignments(self):
        out = FacetedBatchOutput(assignments=[
            FacetedRecordAssignment(**_valid_assignment(record_idx=1)),
            FacetedRecordAssignment(**_valid_assignment(record_idx=2, lesson_type="discovery")),
        ])
        assert len(out.assignments) == 2

    def test_accepts_empty_assignments_list(self):
        out = FacetedBatchOutput(assignments=[])
        assert out.assignments == []
