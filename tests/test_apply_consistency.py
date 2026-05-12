"""Spec tests for apply_proposal — defines required behavior + consistency invariants.

This is the contract apply.py must satisfy. Run before implementing apply,
expect all tests to fail (TDD red). After implementation, all should pass.

Usage:
    uv run python tests/test_apply_consistency.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from consolidate_agent.vocab_maintenance.apply import (
    DeprecateProposal,
    InvalidProposal,
    MergeProposal,
    NewTagProposal,
    OrphanError,
    apply_proposal,
    check_invariants,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def mk_vocab(names: list[str]) -> list[dict]:
    return [{"name": n, "definition": f"def of {n}"} for n in names]


def mk_assignments(spec: dict[str, list[str]]) -> list[dict]:
    """spec: {record_id: [tag_a, tag_b, ...]}"""
    out = []
    for rid, tags in spec.items():
        if tags:
            out.append({
                "record_id": rid,
                "title": f"title {rid}",
                "missing": False,
                "selected_tags": [{"name": t, "confidence": "high"} for t in tags],
                "missing_concept": "",
                "reason": "",
            })
        else:
            out.append({
                "record_id": rid,
                "title": f"title {rid}",
                "missing": True,
                "selected_tags": [],
                "missing_concept": "",
                "reason": "",
            })
    return out


def tags_of(assignments: list[dict], rid: str) -> list[str]:
    a = next(a for a in assignments if a["record_id"] == rid)
    return [t["name"] for t in a["selected_tags"]]


# ── Tests ────────────────────────────────────────────────────────────────────


TESTS = []


def test(fn):
    """Decorator to register a test."""
    TESTS.append(fn)
    return fn


# Test 1: apply_new — only vocab changes, assignments untouched
@test
def test_apply_new_does_not_touch_assignments():
    vocab = mk_vocab(["a", "b", "c"])
    assignments = mk_assignments({"r1": ["a", "b"], "r2": ["c"]})
    original_assignments = [dict(a, selected_tags=list(a["selected_tags"])) for a in assignments]

    new_vocab, new_assignments = apply_proposal(
        vocab, assignments, NewTagProposal(name="d", definition="def of d")
    )

    names = [t["name"] for t in new_vocab]
    assert "d" in names, f"new tag d not added to vocab: {names}"
    assert set(names) == {"a", "b", "c", "d"}, f"unexpected vocab: {names}"
    # assignments are unchanged
    assert len(new_assignments) == len(original_assignments), "record count changed"
    for orig, new in zip(original_assignments, new_assignments):
        assert tags_of([new], new["record_id"]) == tags_of([orig], orig["record_id"]), \
            f"assignments changed for {new['record_id']}"
    check_invariants(new_vocab, new_assignments)


# Test 2: apply_new rejects duplicate name
@test
def test_apply_new_rejects_duplicate_name():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    try:
        apply_proposal(vocab, assignments, NewTagProposal(name="a", definition="..."))
        raise AssertionError("expected InvalidProposal for duplicate name 'a'")
    except InvalidProposal:
        pass


# Test 3: apply_merge renames assignments
@test
def test_apply_merge_renames_assignments():
    vocab = mk_vocab(["a", "b", "c"])
    assignments = mk_assignments({"r1": ["a"], "r2": ["a", "b"], "r3": ["c"]})

    new_vocab, new_assignments = apply_proposal(
        vocab, assignments, MergeProposal(keep_tag="b", discard_tag="a")
    )

    names = [t["name"] for t in new_vocab]
    assert "a" not in names, f"discarded tag 'a' still in vocab: {names}"
    assert "b" in names and "c" in names
    # r1: a → b
    assert tags_of(new_assignments, "r1") == ["b"]
    # r2: a, b → b (dedupe)
    assert tags_of(new_assignments, "r2") == ["b"], \
        f"merge didn't dedupe: r2 has {tags_of(new_assignments, 'r2')}"
    # r3: unchanged
    assert tags_of(new_assignments, "r3") == ["c"]
    check_invariants(new_vocab, new_assignments)


# Test 4: apply_merge rejects nonexistent tag
@test
def test_apply_merge_rejects_nonexistent_source():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    try:
        apply_proposal(vocab, assignments, MergeProposal(keep_tag="b", discard_tag="zzz"))
        raise AssertionError("expected InvalidProposal for nonexistent 'zzz'")
    except InvalidProposal:
        pass


# Test 5: apply_merge rejects nonexistent keep_tag
@test
def test_apply_merge_rejects_nonexistent_keep():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    try:
        apply_proposal(vocab, assignments, MergeProposal(keep_tag="zzz", discard_tag="a"))
        raise AssertionError("expected InvalidProposal for nonexistent keep 'zzz'")
    except InvalidProposal:
        pass


# Test 6: apply_merge rejects self-merge
@test
def test_apply_merge_rejects_self_merge():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    try:
        apply_proposal(vocab, assignments, MergeProposal(keep_tag="a", discard_tag="a"))
        raise AssertionError("expected InvalidProposal for self-merge")
    except InvalidProposal:
        pass


# Test 7: apply_deprecate without orphans succeeds
@test
def test_apply_deprecate_no_orphans():
    vocab = mk_vocab(["x", "y"])
    # x's records all have y too — no orphans
    assignments = mk_assignments({"r1": ["x", "y"], "r2": ["x", "y"], "r3": ["y"]})

    new_vocab, new_assignments = apply_proposal(
        vocab, assignments, DeprecateProposal(tag="x")
    )

    names = [t["name"] for t in new_vocab]
    assert "x" not in names
    assert tags_of(new_assignments, "r1") == ["y"]
    assert tags_of(new_assignments, "r2") == ["y"]
    assert tags_of(new_assignments, "r3") == ["y"]
    check_invariants(new_vocab, new_assignments)


# Test 8: apply_deprecate with orphans rejects (no state change)
@test
def test_apply_deprecate_rejects_when_orphans_exist():
    vocab = mk_vocab(["x", "y"])
    # r1 only has x — deprecating x would orphan it
    assignments = mk_assignments({"r1": ["x"], "r2": ["x", "y"]})

    try:
        apply_proposal(vocab, assignments, DeprecateProposal(tag="x"))
        raise AssertionError("expected OrphanError")
    except OrphanError as e:
        assert "r1" in e.orphan_record_ids, f"r1 should be reported orphan, got {e.orphan_record_ids}"

    # state must be unchanged
    assert {t["name"] for t in vocab} == {"x", "y"}, "vocab mutated despite reject"
    assert tags_of(assignments, "r1") == ["x"], "assignments mutated despite reject"


# Test 9: apply_deprecate rejects nonexistent tag
@test
def test_apply_deprecate_rejects_nonexistent():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    try:
        apply_proposal(vocab, assignments, DeprecateProposal(tag="zzz"))
        raise AssertionError("expected InvalidProposal for nonexistent tag")
    except InvalidProposal:
        pass


# Test 10: invariants pass on healthy state
@test
def test_check_invariants_passes_on_healthy_state():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"], "r2": ["a", "b"]})
    check_invariants(vocab, assignments)  # should not raise


# Test 11: check_invariants catches dangling reference
@test
def test_check_invariants_catches_dangling_reference():
    vocab = mk_vocab(["a"])
    assignments = mk_assignments({"r1": ["a", "zzz"]})  # zzz not in vocab
    try:
        check_invariants(vocab, assignments)
        raise AssertionError("invariant check should have caught dangling ref")
    except AssertionError as e:
        if "dangling" not in str(e):
            raise


# Test 12: check_invariants catches duplicate tags in record
@test
def test_check_invariants_catches_duplicate_tags():
    vocab = mk_vocab(["a", "b"])
    assignments = [{
        "record_id": "r1",
        "title": "title",
        "missing": False,
        "selected_tags": [{"name": "a", "confidence": "high"}, {"name": "a", "confidence": "low"}],
        "missing_concept": "",
        "reason": "",
    }]
    try:
        check_invariants(vocab, assignments)
        raise AssertionError("invariant check should have caught duplicate")
    except AssertionError as e:
        if "duplicate" not in str(e):
            raise


# Test 13: merge dedupe preserves higher confidence
@test
def test_apply_merge_dedupe_keeps_max_confidence():
    vocab = mk_vocab(["a", "b"])
    # r1 has a(high) and b(medium); merging a→b should yield b with confidence=high
    assignments = [{
        "record_id": "r1",
        "title": "title",
        "missing": False,
        "selected_tags": [
            {"name": "a", "confidence": "high"},
            {"name": "b", "confidence": "medium"},
        ],
        "missing_concept": "",
        "reason": "",
    }, {
        "record_id": "r2",
        "title": "title",
        "missing": False,
        "selected_tags": [
            {"name": "a", "confidence": "low"},
            {"name": "b", "confidence": "high"},
        ],
        "missing_concept": "",
        "reason": "",
    }]

    new_vocab, new_assignments = apply_proposal(
        vocab, assignments, MergeProposal(keep_tag="b", discard_tag="a")
    )

    # r1: a(high)+b(medium) → b(high) — high wins
    r1 = next(a for a in new_assignments if a["record_id"] == "r1")
    assert len(r1["selected_tags"]) == 1
    assert r1["selected_tags"][0]["name"] == "b"
    assert r1["selected_tags"][0]["confidence"] == "high", \
        f"expected high, got {r1['selected_tags'][0]['confidence']}"

    # r2: a(low)+b(high) → b(high) — high wins
    r2 = next(a for a in new_assignments if a["record_id"] == "r2")
    assert r2["selected_tags"][0]["confidence"] == "high"
    check_invariants(new_vocab, new_assignments)


# Test 14: apply doesn't touch missing records
@test
def test_apply_does_not_touch_missing_records():
    vocab = mk_vocab(["a", "b", "c"])
    assignments = mk_assignments({
        "r1": ["a"],
        "r2": [],         # missing
        "r3": ["a", "b"],
        "r4": [],         # missing
    })

    # all three proposal types should leave missing records intact
    for proposal in [
        NewTagProposal(name="d", definition="..."),
        MergeProposal(keep_tag="b", discard_tag="a"),
    ]:
        new_vocab, new_assignments = apply_proposal(vocab, assignments, proposal)
        for rid in ["r2", "r4"]:
            orig = next(a for a in assignments if a["record_id"] == rid)
            new = next(a for a in new_assignments if a["record_id"] == rid)
            assert new["missing"] == orig["missing"], f"{rid} missing field changed"
            assert new["selected_tags"] == orig["selected_tags"], f"{rid} selected_tags changed"

    # deprecate (where x has no orphan to worry about)
    vocab2 = mk_vocab(["x", "y"])
    assignments2 = mk_assignments({
        "r1": ["x", "y"],
        "r2": [],  # missing
        "r3": ["y"],
    })
    new_vocab, new_assignments = apply_proposal(vocab2, assignments2, DeprecateProposal(tag="x"))
    r2_new = next(a for a in new_assignments if a["record_id"] == "r2")
    assert r2_new["missing"] is True
    assert r2_new["selected_tags"] == []


# Test 15: record_id count never changes
@test
def test_apply_preserves_record_count():
    vocab = mk_vocab(["a", "b", "c"])
    assignments = mk_assignments({"r1": ["a"], "r2": ["b"], "r3": ["a", "b"], "r4": []})
    original_count = len(assignments)
    original_ids = sorted(a["record_id"] for a in assignments)

    for proposal in [
        NewTagProposal(name="d", definition="..."),
        MergeProposal(keep_tag="b", discard_tag="a"),
    ]:
        _, new_assignments = apply_proposal(vocab, assignments, proposal)
        assert len(new_assignments) == original_count, \
            f"{proposal.type}: record count changed from {original_count} to {len(new_assignments)}"
        new_ids = sorted(a["record_id"] for a in new_assignments)
        assert new_ids == original_ids, f"{proposal.type}: record ids changed"


# Test 16: deprecate reports ALL orphans
@test
def test_apply_deprecate_reports_all_orphans():
    vocab = mk_vocab(["x", "y"])
    # r1 and r2 are both orphans (only have x); r3 has both, not orphan
    assignments = mk_assignments({
        "r1": ["x"],
        "r2": ["x"],
        "r3": ["x", "y"],
    })

    try:
        apply_proposal(vocab, assignments, DeprecateProposal(tag="x"))
        raise AssertionError("expected OrphanError")
    except OrphanError as e:
        orphans = sorted(e.orphan_record_ids)
        assert orphans == ["r1", "r2"], \
            f"expected orphans=[r1,r2], got {orphans}"


# Test 17: new tag preserves full definition
@test
def test_apply_new_preserves_definition():
    vocab = mk_vocab(["a"])
    assignments = mk_assignments({"r1": ["a"]})
    long_def = (
        "A long, detailed definition with multiple sentences. "
        "Includes nuances about edge cases and specific examples. "
        "Should be preserved exactly."
    )
    new_vocab, _ = apply_proposal(
        vocab, assignments, NewTagProposal(name="x", definition=long_def)
    )
    new_tag = next(t for t in new_vocab if t["name"] == "x")
    assert new_tag["definition"] == long_def, \
        f"definition mangled: {new_tag['definition']!r}"


# Test 18: merge keep_tag definition is NOT overwritten by discard_tag's
@test
def test_apply_merge_keeps_keep_tag_definition():
    vocab = [
        {"name": "a", "definition": "DEFINITION_OF_A_THAT_SHOULD_BE_DISCARDED"},
        {"name": "b", "definition": "DEFINITION_OF_B_THAT_SHOULD_BE_KEPT"},
    ]
    assignments = mk_assignments({"r1": ["a", "b"]})

    new_vocab, _ = apply_proposal(vocab, assignments, MergeProposal(keep_tag="b", discard_tag="a"))
    b = next(t for t in new_vocab if t["name"] == "b")
    assert b["definition"] == "DEFINITION_OF_B_THAT_SHOULD_BE_KEPT", \
        f"b's definition got overwritten: {b['definition']!r}"
    # a should be gone
    assert not any(t["name"] == "a" for t in new_vocab)


# ── Test runner ──────────────────────────────────────────────────────────────


def main():
    passed = 0
    failed = 0
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except NotImplementedError:
            print(f"  ⏸  {name}  (NotImplementedError — apply.py stub)")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}")
            print(f"     {type(e).__name__}: {e}")
            tb_lines = traceback.format_exc().split("\n")
            for line in tb_lines[-5:-1]:
                if line.strip():
                    print(f"     {line}")
            failed += 1

    print(f"\n{'─' * 50}")
    print(f"Total: {passed + failed}, Passed: {passed}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
