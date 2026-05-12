"""TagRecordNetwork unit tests — no LLM, fast.

Tests:
  N1  construct + validate on healthy state
  N2  validate rejects duplicate vocab names
  N3  validate rejects dangling assignment refs (via check_invariants)
  N4  save → load roundtrip preserves all fields
  N5  load rejects schema_version mismatch
  N6  from_legacy_files drops dangling refs + records become missing
  N7  apply(NewTagProposal) grows vocab + updates vocab_hash
  N8  apply(MergeProposal) rewrites assignments deterministically
  N9  apply(DeprecateProposal) shrinks vocab + clears refs
  N10 diagnostics() matches build_diagnostics output
  N11 hit_rate() derives from diagnostics correctly
  N12 vocab_hash stable across vocab list reordering
  N13 save uses atomic tmp file pattern
  N14 apply updates metadata (last_modified, vocab_hash)
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

from consolidate_agent.vocab_maintenance.apply import (
    DeprecateProposal,
    MergeProposal,
    NewTagProposal,
)
from consolidate_agent.vocab_maintenance.measure import build_diagnostics
from consolidate_agent.vocab_maintenance.network import (
    SCHEMA_VERSION,
    TagRecordNetwork,
    _vocab_hash,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def mk_vocab(specs: list[tuple[str, str]]) -> list[dict]:
    return [{"name": n, "definition": d} for n, d in specs]


def mk_assignments(rows: dict[str, list[str]]) -> list[dict]:
    """rows: record_id → list of tag names (all confidence=high). Missing if empty."""
    out = []
    for rid, tags in rows.items():
        out.append({
            "record_id": rid,
            "title": f"title {rid}",
            "selected_tags": [{"name": t, "confidence": "high"} for t in tags],
            "missing": not tags,
            "missing_concept": "" if tags else "no fit",
            "reason": "",
        })
    return out


# ── Tests ────────────────────────────────────────────────────────────────────


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def test_n1_construct_and_validate():
    vocab = mk_vocab([("a", "def a"), ("b", "def b")])
    assignments = mk_assignments({"r1": ["a"], "r2": ["b"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)
    net.validate()  # should not raise
    assert net.vocab is vocab
    assert net.assignments is assignments


@test
def test_n2_duplicate_vocab_rejected():
    vocab = mk_vocab([("a", "def 1"), ("a", "def 2")])
    assignments = mk_assignments({"r1": ["a"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)
    try:
        net.validate()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "duplicate" in str(e).lower()


@test
def test_n3_dangling_ref_rejected():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a", "b"]})  # b not in vocab
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)
    try:
        net.validate()
        assert False, "expected check_invariants to raise"
    except AssertionError:
        pass


@test
def test_n4_save_load_roundtrip():
    vocab = mk_vocab([("a", "def a"), ("b", "def b")])
    assignments = mk_assignments({"r1": ["a"], "r2": ["b"]})
    original = TagRecordNetwork(vocab=vocab, assignments=assignments, metadata={"foo": "bar"})

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "net.json"
        original.save(path)
        loaded = TagRecordNetwork.load(path)

    assert loaded.vocab == original.vocab
    assert loaded.assignments == original.assignments
    assert loaded.metadata["foo"] == "bar"
    assert loaded.metadata["schema_version"] == SCHEMA_VERSION
    assert loaded.metadata["vocab_hash"] == _vocab_hash(original.vocab)


@test
def test_n5_load_schema_mismatch_rejected():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"]})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "net.json"
        path.write_text(json.dumps({
            "vocab": vocab,
            "assignments": assignments,
            "metadata": {"schema_version": SCHEMA_VERSION + 999},
        }), encoding="utf-8")
        try:
            TagRecordNetwork.load(path)
            assert False, "expected schema mismatch ValueError"
        except ValueError as e:
            assert "schema" in str(e).lower()


@test
def test_n6_from_legacy_files_drops_dangling():
    vocab = mk_vocab([("a", "def a")])
    # Legacy assignment references both 'a' and a now-removed 'b'
    legacy_assignments = mk_assignments({"r1": ["a", "b"], "r2": ["b"]})

    with tempfile.TemporaryDirectory() as d:
        vp = Path(d) / "vocab.json"
        ap = Path(d) / "assign.json"
        vp.write_text(json.dumps({"vocab": vocab}), encoding="utf-8")
        ap.write_text(json.dumps(legacy_assignments), encoding="utf-8")

        net = TagRecordNetwork.from_legacy_files(vp, ap)

    # r1 keeps 'a', drops 'b'
    r1 = next(a for a in net.assignments if a["record_id"] == "r1")
    assert [t["name"] for t in r1["selected_tags"]] == ["a"]
    assert r1["missing"] is False

    # r2 had only 'b' → all dropped → missing
    r2 = next(a for a in net.assignments if a["record_id"] == "r2")
    assert r2["selected_tags"] == []
    assert r2["missing"] is True

    assert net.metadata["import_dangling_refs_dropped"] == 2  # one from r1, one from r2


@test
def test_n7_apply_new_tag():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments, metadata={})
    old_hash = _vocab_hash(net.vocab)

    net.apply(NewTagProposal(name="b", definition="def b"))

    assert any(t["name"] == "b" for t in net.vocab), "new tag added"
    assert _vocab_hash(net.vocab) != old_hash
    assert net.metadata["vocab_hash"] == _vocab_hash(net.vocab)


@test
def test_n8_apply_merge_rewrites_assignments():
    vocab = mk_vocab([("a", "def a"), ("b", "def b")])
    assignments = mk_assignments({"r1": ["a"], "r2": ["b"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)

    # merge b → a (discard b, keep a)
    net.apply(MergeProposal(keep_tag="a", discard_tag="b"))

    assert {t["name"] for t in net.vocab} == {"a"}
    # r2 originally had 'b', now has 'a'
    r2 = next(a for a in net.assignments if a["record_id"] == "r2")
    assert [t["name"] for t in r2["selected_tags"]] == ["a"]


@test
def test_n9_apply_deprecate_shrinks_vocab():
    vocab = mk_vocab([("a", "def a"), ("b", "def b")])
    assignments = mk_assignments({"r1": ["a", "b"], "r2": ["a"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)

    net.apply(DeprecateProposal(tag="b"))

    assert {t["name"] for t in net.vocab} == {"a"}
    # r1 had [a, b] → now [a]
    r1 = next(a for a in net.assignments if a["record_id"] == "r1")
    assert [t["name"] for t in r1["selected_tags"]] == ["a"]
    assert r1["missing"] is False


@test
def test_n10_diagnostics_matches_build_diag():
    vocab = mk_vocab([("a", "def a"), ("b", "def b")])
    assignments = mk_assignments({"r1": ["a"], "r2": ["b"], "r3": []})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)

    expected = build_diagnostics(vocab, assignments)
    actual = net.diagnostics()
    assert actual == expected


@test
def test_n11_hit_rate():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"], "r2": ["a"], "r3": []})  # 2/3 assigned
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)
    assert abs(net.hit_rate() - 2 / 3) < 1e-6


@test
def test_n12_vocab_hash_stable_across_reorder():
    v1 = mk_vocab([("a", "def a"), ("b", "def b")])
    v2 = mk_vocab([("b", "def b"), ("a", "def a")])  # reordered
    assert _vocab_hash(v1) == _vocab_hash(v2)


@test
def test_n13_save_atomic_no_tmp_left():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "net.json"
        net.save(path)
        assert path.exists()
        # tmp file shouldn't linger
        tmp = path.with_suffix(path.suffix + ".tmp")
        assert not tmp.exists()


@test
def test_n14_apply_updates_metadata():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments, metadata={"created_at": 1.0, "last_modified": 1.0})

    pre_modified = net.metadata["last_modified"]
    net.apply(NewTagProposal(name="b", definition="def b"))

    assert net.metadata["last_modified"] > pre_modified
    assert net.metadata["vocab_hash"] == _vocab_hash(net.vocab)


# ── Runner ───────────────────────────────────────────────────────────────────


def main() -> int:
    passed = 0
    failed = 0
    print(f"\nRunning {len(TESTS)} network tests…\n")
    print("─" * 50)
    for fn in TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}")
            print(f"     {type(e).__name__}: {e}")
            tb_lines = traceback.format_exc().split("\n")
            for line in tb_lines[-6:-1]:
                if line.strip():
                    print(f"     {line}")
            failed += 1

    print(f"\n{'─' * 50}")
    print(f"Total: {passed + failed}, Passed: {passed}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
