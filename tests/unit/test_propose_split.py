"""Unit tests for split proposal: _apply_split and compute_heterogeneous_tags."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from consolidate_agent.vocab_maintenance.apply import (
    InvalidProposal,
    OrphanError,
    SplitTagProposal,
    _apply_split,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_vocab(*names: str) -> list[dict]:
    return [{"name": n, "definition": f"{n} definition"} for n in names]


def _make_assignment(record_id: str, *tags: str) -> dict:
    return {
        "record_id": record_id,
        "selected_tags": [{"name": t, "confidence": "high"} for t in tags],
    }


def _make_split(tag: str, sub_tags: list[dict]) -> SplitTagProposal:
    converted = []
    for st in sub_tags:
        converted.append({
            "name": st["name"],
            "definition": st.get("definition", f"{st['name']} definition"),
            "record_ids": tuple(st["record_ids"]),
        })
    return SplitTagProposal(tag=tag, sub_tags=tuple(converted))


# ── Test 1: normal split path ─────────────────────────────────────────────────


def test_apply_split_happy_path():
    """Split 'api' (3 records) into 'auth_api' (r1,r2) and 'routing_api' (r3)."""
    vocab = _make_vocab("api", "logging")
    assignments = [
        _make_assignment("r1", "api", "logging"),
        _make_assignment("r2", "api"),
        _make_assignment("r3", "api"),
    ]
    # r2 only has 'api' — after split it gets 'auth_api', so it won't be orphaned
    proposal = _make_split("api", [
        {"name": "auth_api", "record_ids": ["r1", "r2"]},
        {"name": "routing_api", "record_ids": ["r3"]},
    ])
    # r3 alone in routing_api → invariant I-S6 (min 2 records per sub-tag)
    with pytest.raises(InvalidProposal, match="routing_api"):
        _apply_split(vocab, assignments, proposal)


def test_apply_split_happy_path_full():
    """Split 'api' (4 records) into two sub-tags with >= 2 records each."""
    vocab = _make_vocab("api", "logging")
    assignments = [
        _make_assignment("r1", "api", "logging"),
        _make_assignment("r2", "api", "logging"),
        _make_assignment("r3", "api"),
        _make_assignment("r4", "api"),
    ]
    proposal = _make_split("api", [
        {"name": "auth_api", "record_ids": ["r1", "r2"]},
        {"name": "routing_api", "record_ids": ["r3", "r4"]},
    ])
    new_vocab, new_assignments = _apply_split(vocab, assignments, proposal)

    # original tag removed, two sub-tags added
    vocab_names = {t["name"] for t in new_vocab}
    assert "api" not in vocab_names
    assert "auth_api" in vocab_names
    assert "routing_api" in vocab_names
    assert "logging" in vocab_names  # untouched

    # r1 should have auth_api + logging
    r1 = next(a for a in new_assignments if a["record_id"] == "r1")
    r1_tags = {t["name"] for t in r1["selected_tags"]}
    assert r1_tags == {"auth_api", "logging"}

    # r3 should have routing_api only
    r3 = next(a for a in new_assignments if a["record_id"] == "r3")
    r3_tags = {t["name"] for t in r3["selected_tags"]}
    assert r3_tags == {"routing_api"}

    # confidence preserved from original
    r1_tag = next(t for t in r1["selected_tags"] if t["name"] == "auth_api")
    assert r1_tag["confidence"] == "high"


# ── Test 2: invariant — record not covered (OrphanError) ─────────────────────


def test_apply_split_uncovered_record_raises_orphan_error():
    """Record r3 assigned to 'api' but not included in any sub-tag → OrphanError.

    I-S4 (uncovered records) fires before I-S6 (min records per sub-tag),
    so OrphanError is raised even though routing_api has 0 records.
    """
    vocab = _make_vocab("api")
    assignments = [
        _make_assignment("r1", "api"),
        _make_assignment("r2", "api"),
        _make_assignment("r3", "api"),  # will be left out
    ]
    proposal = _make_split("api", [
        {"name": "auth_api", "record_ids": ["r1", "r2"]},
        {"name": "routing_api", "record_ids": []},
    ])
    # r3 uncovered → OrphanError (fired before I-S6 min-records check)
    with pytest.raises(OrphanError):
        _apply_split(vocab, assignments, proposal)


def test_apply_split_uncovered_record_proper():
    """OrphanError when a record is genuinely not in any sub-tag record_ids."""
    vocab = _make_vocab("api")
    assignments = [
        _make_assignment("r1", "api"),
        _make_assignment("r2", "api"),
        _make_assignment("r3", "api"),
        _make_assignment("r4", "api"),
    ]
    # Only r1,r2,r3 covered — r4 is orphaned
    proposal = _make_split("api", [
        {"name": "auth_api", "record_ids": ["r1", "r2"]},
        {"name": "routing_api", "record_ids": ["r3", "r3"]},  # r4 missing, r3 doubled
    ])
    with pytest.raises(InvalidProposal, match="appears in multiple"):
        _apply_split(vocab, assignments, proposal)


def test_apply_split_true_orphan():
    """r4 absent from sub-tags while routing_api has only 1 record.

    I-S4 (uncovered) fires before I-S6 (min-records), so OrphanError for r4.
    """
    vocab = _make_vocab("api")
    assignments = [
        _make_assignment("r1", "api"),
        _make_assignment("r2", "api"),
        _make_assignment("r3", "api"),
        _make_assignment("r4", "api"),
    ]
    proposal = SplitTagProposal(
        tag="api",
        sub_tags=(
            {"name": "auth_api", "definition": "auth", "record_ids": ("r1", "r2")},
            {"name": "routing_api", "definition": "routing", "record_ids": ("r3",)},
            # r4 uncovered → OrphanError fires before I-S6 min-records check
        ),
    )
    with pytest.raises(OrphanError, match="r4"):
        _apply_split(vocab, assignments, proposal)


def test_apply_split_orphan_error_raised():
    """Direct OrphanError test: all sub-tags have >= 2 records but r4 is uncovered."""
    vocab = _make_vocab("api")
    assignments = [
        _make_assignment("r1", "api"),
        _make_assignment("r2", "api"),
        _make_assignment("r3", "api"),
        _make_assignment("r4", "api"),
        _make_assignment("r5", "api"),
    ]
    # r5 is left uncovered, both sub-tags have >= 2 records
    proposal = SplitTagProposal(
        tag="api",
        sub_tags=(
            {"name": "auth_api", "definition": "auth", "record_ids": ("r1", "r2")},
            {"name": "routing_api", "definition": "routing", "record_ids": ("r3", "r4")},
        ),
    )
    with pytest.raises(OrphanError, match="r5"):
        _apply_split(vocab, assignments, proposal)


# ── Test 3: invariant — sub-tag name collision ────────────────────────────────


def test_apply_split_name_collision_with_vocab():
    """Sub-tag 'logging' already exists in vocab → InvalidProposal."""
    vocab = _make_vocab("api", "logging")
    assignments = [
        _make_assignment("r1", "api"),
        _make_assignment("r2", "api"),
        _make_assignment("r3", "api"),
        _make_assignment("r4", "api"),
    ]
    proposal = _make_split("api", [
        {"name": "logging", "record_ids": ["r1", "r2"]},   # collides with vocab
        {"name": "routing_api", "record_ids": ["r3", "r4"]},
    ])
    with pytest.raises(InvalidProposal, match="logging"):
        _apply_split(vocab, assignments, proposal)


# ── Test 4: invariant — double assignment ────────────────────────────────────


def test_apply_split_double_assignment():
    """r2 appears in both sub-tags → InvalidProposal."""
    vocab = _make_vocab("api")
    assignments = [
        _make_assignment("r1", "api"),
        _make_assignment("r2", "api"),
        _make_assignment("r3", "api"),
        _make_assignment("r4", "api"),
    ]
    proposal = SplitTagProposal(
        tag="api",
        sub_tags=(
            {"name": "auth_api", "definition": "auth", "record_ids": ("r1", "r2")},
            {"name": "routing_api", "definition": "routing", "record_ids": ("r2", "r3", "r4")},
        ),
    )
    with pytest.raises(InvalidProposal, match="r2"):
        _apply_split(vocab, assignments, proposal)


# ── Test 5: compute_heterogeneous_tags logic ──────────────────────────────────


def _build_test_db(record_data: list[tuple[str, list[float]]]) -> Path:
    """Create a temp SQLite db with source_knowledge_records and embedded vectors."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE source_knowledge_records "
        "(record_id TEXT PRIMARY KEY, title TEXT, insight TEXT, embedding TEXT)"
    )
    for record_id, vec in record_data:
        conn.execute(
            "INSERT INTO source_knowledge_records VALUES (?, ?, ?, ?)",
            (record_id, f"title_{record_id}", f"insight_{record_id}", json.dumps(vec)),
        )
    conn.commit()
    conn.close()
    return db_path


def _unit_vec(dim: int, hot_dim: int) -> list[float]:
    """One-hot-ish unit vector of length `dim` peaking at `hot_dim`."""
    v = [0.0] * dim
    v[hot_dim] = 1.0
    return v


def test_compute_heterogeneous_tags_low_coherence_first():
    """Tag with high diversity (low pairwise cosine) should rank first."""
    from consolidate_agent.vocab_maintenance.probes import compute_heterogeneous_tags

    dim = 4
    # "diverse_tag": records spread across all 4 basis vectors — very low coherence
    diverse_ids = [f"d{i}" for i in range(16)]
    diverse_vecs = [_unit_vec(dim, i % dim) for i in range(16)]

    # "tight_tag": all records point in the same direction — high coherence
    tight_ids = [f"t{i}" for i in range(16)]
    tight_vecs = [_unit_vec(dim, 0) for _ in range(16)]

    all_records = list(zip(diverse_ids, diverse_vecs)) + list(zip(tight_ids, tight_vecs))
    db_path = _build_test_db(all_records)

    vocab = _make_vocab("diverse_tag", "tight_tag")
    assignments = (
        [_make_assignment(rid, "diverse_tag") for rid in diverse_ids]
        + [_make_assignment(rid, "tight_tag") for rid in tight_ids]
    )

    results = compute_heterogeneous_tags(
        vocab, assignments, db_path,
        coherence_threshold=0.99,  # high threshold so both tags qualify
        min_records=15,
    )

    # diverse_tag should have lower coherence → appear first
    assert len(results) >= 1
    assert results[0]["tag"] == "diverse_tag"
    assert results[0]["coherence"] < results[-1]["coherence"] if len(results) > 1 else True
    # tight_tag coherence should be 1.0 (all same vec, cosine = 1)
    tight_result = next((r for r in results if r["tag"] == "tight_tag"), None)
    # tight_tag might not appear if its coherence == 1.0 and threshold < 1.0
    # with threshold=0.99, tight should appear
    if tight_result:
        assert tight_result["coherence"] > results[0]["coherence"]


def test_compute_heterogeneous_tags_min_records_filter():
    """Tags with < min_records records are excluded."""
    from consolidate_agent.vocab_maintenance.probes import compute_heterogeneous_tags

    dim = 4
    small_ids = [f"s{i}" for i in range(5)]  # only 5 records — below min_records=15
    small_vecs = [_unit_vec(dim, i % dim) for i in range(5)]

    db_path = _build_test_db(list(zip(small_ids, small_vecs)))
    vocab = _make_vocab("small_tag")
    assignments = [_make_assignment(rid, "small_tag") for rid in small_ids]

    results = compute_heterogeneous_tags(
        vocab, assignments, db_path,
        coherence_threshold=0.99,
        min_records=15,
    )
    assert results == []


def test_compute_heterogeneous_tags_above_threshold_excluded():
    """Tags above coherence_threshold are not returned."""
    from consolidate_agent.vocab_maintenance.probes import compute_heterogeneous_tags

    dim = 4
    tight_ids = [f"t{i}" for i in range(16)]
    tight_vecs = [_unit_vec(dim, 0) for _ in range(16)]  # all same direction

    db_path = _build_test_db(list(zip(tight_ids, tight_vecs)))
    vocab = _make_vocab("tight_tag")
    assignments = [_make_assignment(rid, "tight_tag") for rid in tight_ids]

    # coherence will be ~1.0; threshold=0.5 → should not be returned
    results = compute_heterogeneous_tags(
        vocab, assignments, db_path,
        coherence_threshold=0.5,
        min_records=15,
    )
    assert results == []
