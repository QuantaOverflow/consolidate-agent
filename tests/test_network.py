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
def test_n_build_diag_low_confidence_record():
    """Top-pick confidence == 'low' → record listed in low_confidence_records."""
    vocab = mk_vocab([("a", "def a"), ("b", "def b")])
    assignments = [
        {"record_id": "r1", "title": "t1",
         "selected_tags": [{"name": "a", "confidence": "low"}, {"name": "b", "confidence": "low"}],
         "missing": False, "missing_concept": "", "reason": "weak match"},
        {"record_id": "r2", "title": "t2",
         "selected_tags": [{"name": "a", "confidence": "high"}],
         "missing": False, "missing_concept": "", "reason": ""},
    ]
    diag = build_diagnostics(vocab, assignments)
    low = diag["low_confidence_records"]
    assert len(low) == 1
    assert low[0]["record_id"] == "r1"


@test
def test_n_build_diag_boundary_blur_excludes_all_low():
    """All-low tags signal 'no fit', not 'ambiguity' — exclude from boundary_blur."""
    vocab = mk_vocab([("a", "def a"), ("b", "def b"), ("c", "def c")])
    assignments = [
        # All low → NOT boundary blur (vocab doesn't fit)
        {"record_id": "r1", "title": "t1",
         "selected_tags": [{"name": "a", "confidence": "low"}, {"name": "b", "confidence": "low"}],
         "missing": False, "missing_concept": "", "reason": ""},
        # All high → boundary blur (ambiguous fit)
        {"record_id": "r2", "title": "t2",
         "selected_tags": [{"name": "a", "confidence": "high"}, {"name": "b", "confidence": "high"}],
         "missing": False, "missing_concept": "", "reason": ""},
        # Mixed high+med → NOT boundary blur (different levels)
        {"record_id": "r3", "title": "t3",
         "selected_tags": [{"name": "a", "confidence": "high"}, {"name": "b", "confidence": "medium"}],
         "missing": False, "missing_concept": "", "reason": ""},
    ]
    diag = build_diagnostics(vocab, assignments)
    blur_ids = [r["record_id"] for r in diag["boundary_blur_records"]]
    assert blur_ids == ["r2"]


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
def test_n15_themes_save_load_roundtrip():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"], "r2": ["a"]})
    themes = {"r1": "theme for r1", "r2": "theme for r2"}
    net = TagRecordNetwork(vocab=vocab, assignments=assignments, themes=themes)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "net.json"
        net.save(path)
        loaded = TagRecordNetwork.load(path)

    assert loaded.themes == themes


@test
def test_n16_load_legacy_snapshot_defaults_themes_empty():
    """Old snapshots written before themes-as-first-class load with themes={}."""
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"]})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "net.json"
        path.write_text(json.dumps({
            "vocab": vocab,
            "assignments": assignments,
            "metadata": {"schema_version": SCHEMA_VERSION, "vocab_hash": _vocab_hash(vocab)},
            # NOTE: no "themes" key — old snapshot format
        }), encoding="utf-8")
        net = TagRecordNetwork.load(path)
    assert net.themes == {}


@test
def test_n17_themes_default_empty_on_construct():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)
    assert net.themes == {}
    # Empty themes shouldn't break validate
    net.validate()


@test
def test_n14_apply_updates_metadata():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments, metadata={"created_at": 1.0, "last_modified": 1.0})

    pre_modified = net.metadata["last_modified"]
    net.apply(NewTagProposal(name="b", definition="def b"))

    assert net.metadata["last_modified"] > pre_modified
    assert net.metadata["vocab_hash"] == _vocab_hash(net.vocab)


# ── Ingest helpers ──────────────────────────────────────────────────────────


def mk_raw_records(spec: list[tuple[str, str, str]]) -> list[dict]:
    """spec: [(record_id, title, insight), ...]"""
    return [{"record_id": rid, "title": t, "insight": i} for rid, t, i in spec]


def make_reverse_check_fn(assignment_for: dict[str, dict]):
    """Returns fake reverse_check that yields pre-built assignments by record_id.

    Tracks call_count for assertions.
    """
    call_count = [0]
    seen_record_ids = []

    def fn(records, vocab, concurrency):
        call_count[0] += 1
        results = []
        for r in records:
            seen_record_ids.append(r["record_id"])
            base = assignment_for.get(r["record_id"], {
                "record_id": r["record_id"],
                "title": r["title"],
                "selected_tags": [],
                "missing": True,
                "missing_concept": "default missing",
                "reason": "",
            })
            results.append(dict(base))
        return results

    fn.call_count = call_count
    fn.seen_record_ids = seen_record_ids
    return fn


def make_propose_new_fn(candidates: list[dict]):
    """Returns fake propose_new that yields canned candidates."""
    call_count = [0]
    seen_pool_sizes = []

    def fn(vocab, assignments, focus):
        call_count[0] += 1
        seen_pool_sizes.append(sum(1 for a in assignments if a.get("missing")))
        return list(candidates)

    fn.call_count = call_count
    fn.seen_pool_sizes = seen_pool_sizes
    return fn


def make_records_fetcher(records_by_id: dict[str, dict]):
    def fn(record_ids):
        return [records_by_id[rid] for rid in record_ids if rid in records_by_id]
    return fn


# ── Ingest tests ────────────────────────────────────────────────────────────


@test
def test_i1_empty_batch_zero_llm():
    vocab = mk_vocab([("a", "def a")])
    assignments = mk_assignments({"r1": ["a"]})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)

    rc = make_reverse_check_fn({})
    pn = make_propose_new_fn([])
    result = net.ingest_batch([], reverse_check_fn=rc, propose_new_fn=pn)

    assert result.ingested == 0
    assert result.skipped_duplicates == 0
    assert rc.call_count[0] == 0
    assert pn.call_count[0] == 0


@test
def test_i2_all_assigned_no_trigger():
    vocab = mk_vocab([("a", "def a"), ("b", "def b")])
    net = TagRecordNetwork(vocab=vocab, assignments=[])

    fresh = mk_raw_records([("r1", "t1", "i1"), ("r2", "t2", "i2"), ("r3", "t3", "i3")])
    rc = make_reverse_check_fn({
        "r1": {"record_id": "r1", "title": "t1", "selected_tags": [{"name": "a", "confidence": "high"}], "missing": False, "missing_concept": "", "reason": ""},
        "r2": {"record_id": "r2", "title": "t2", "selected_tags": [{"name": "b", "confidence": "high"}], "missing": False, "missing_concept": "", "reason": ""},
        "r3": {"record_id": "r3", "title": "t3", "selected_tags": [{"name": "a", "confidence": "high"}], "missing": False, "missing_concept": "", "reason": ""},
    })
    pn = make_propose_new_fn([])
    result = net.ingest_batch(fresh, reverse_check_fn=rc, propose_new_fn=pn, propose_threshold=10)

    assert result.ingested == 3
    assert result.newly_assigned == 3
    assert result.newly_missing == 0
    assert result.new_tags_added == []
    assert pn.call_count[0] == 0, "propose_new shouldn't fire with empty pool"


@test
def test_i3_threshold_triggers_propose_new():
    vocab = mk_vocab([("a", "def a")])
    net = TagRecordNetwork(vocab=vocab, assignments=[])

    # 5 fresh records, all missing
    fresh = mk_raw_records([(f"r{i}", f"t{i}", f"i{i}") for i in range(1, 6)])
    # rc returns all missing on first call (initial classification);
    # on re-check call (after propose), 4 of them get the new tag.
    rc_call = [0]

    def rc(records, vocab, concurrency):
        rc_call[0] += 1
        if rc_call[0] == 1:
            # initial classification — all missing
            return [{
                "record_id": r["record_id"], "title": r["title"],
                "selected_tags": [], "missing": True, "missing_concept": "x", "reason": "",
            } for r in records]
        # re-check after new tag — 4 absorbed, 1 still missing
        out = []
        for i, r in enumerate(records):
            if i < 4:
                out.append({
                    "record_id": r["record_id"], "title": r["title"],
                    "selected_tags": [{"name": "newtag", "confidence": "high"}],
                    "missing": False, "missing_concept": "", "reason": "",
                })
            else:
                out.append({
                    "record_id": r["record_id"], "title": r["title"],
                    "selected_tags": [], "missing": True, "missing_concept": "x", "reason": "",
                })
        return out

    pn = make_propose_new_fn([{"name": "newtag", "definition": "def newtag"}])
    fetcher = make_records_fetcher({r["record_id"]: r for r in fresh})

    result = net.ingest_batch(
        fresh,
        reverse_check_fn=rc,
        propose_new_fn=pn,
        records_fetcher=fetcher,
        propose_threshold=5,  # trigger right at pool=5
    )

    assert result.ingested == 5
    assert result.newly_missing == 5  # all initially missing
    assert pn.call_count[0] == 1, "propose_new should fire"
    assert result.new_tags_added == ["newtag"]
    assert "newtag" in [t["name"] for t in net.vocab]
    assert result.absorbed_from_pool == 4
    assert result.total_pending_after == 1


@test
def test_i4_dedup_skips_existing():
    vocab = mk_vocab([("a", "def a")])
    # r1 already in network
    existing = mk_assignments({"r1": ["a"]})
    net = TagRecordNetwork(vocab=vocab, assignments=existing)

    fresh = mk_raw_records([("r1", "t1", "i1"), ("r2", "t2", "i2")])
    rc = make_reverse_check_fn({
        "r2": {"record_id": "r2", "title": "t2", "selected_tags": [{"name": "a", "confidence": "high"}], "missing": False, "missing_concept": "", "reason": ""},
    })
    pn = make_propose_new_fn([])
    result = net.ingest_batch(fresh, reverse_check_fn=rc, propose_new_fn=pn, propose_threshold=10)

    assert result.ingested == 1
    assert result.skipped_duplicates == 1
    assert rc.seen_record_ids == ["r2"], "rc only called on fresh r2"


@test
def test_i5_empty_vocab_raises():
    net = TagRecordNetwork(vocab=[], assignments=[])
    fresh = mk_raw_records([("r1", "t1", "i1")])
    try:
        net.ingest_batch(fresh, reverse_check_fn=make_reverse_check_fn({}))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "vocab is empty" in str(e)


@test
def test_i6_propose_new_returns_empty_no_crash():
    vocab = mk_vocab([("a", "def a")])
    net = TagRecordNetwork(vocab=vocab, assignments=[])

    fresh = mk_raw_records([("r1", "t1", "i1"), ("r2", "t2", "i2")])
    rc = make_reverse_check_fn({})  # both missing by default
    pn = make_propose_new_fn([])  # returns nothing
    result = net.ingest_batch(fresh, reverse_check_fn=rc, propose_new_fn=pn, propose_threshold=2)

    assert pn.call_count[0] == 1, "propose_new called"
    assert result.new_tags_added == []
    assert result.absorbed_from_pool == 0
    assert result.total_pending_after == 2  # all still pending


@test
def test_i7_idempotent_double_ingest():
    vocab = mk_vocab([("a", "def a")])
    net = TagRecordNetwork(vocab=vocab, assignments=[])

    fresh = mk_raw_records([("r1", "t1", "i1"), ("r2", "t2", "i2")])
    rc1 = make_reverse_check_fn({
        "r1": {"record_id": "r1", "title": "t1", "selected_tags": [{"name": "a", "confidence": "high"}], "missing": False, "missing_concept": "", "reason": ""},
        "r2": {"record_id": "r2", "title": "t2", "selected_tags": [{"name": "a", "confidence": "high"}], "missing": False, "missing_concept": "", "reason": ""},
    })
    pn = make_propose_new_fn([])

    r1 = net.ingest_batch(fresh, reverse_check_fn=rc1, propose_new_fn=pn, propose_threshold=10)
    assert r1.ingested == 2

    # Second ingest with same records — should all be dedup'd
    rc2 = make_reverse_check_fn({})
    r2 = net.ingest_batch(fresh, reverse_check_fn=rc2, propose_new_fn=pn, propose_threshold=10)
    assert r2.ingested == 0
    assert r2.skipped_duplicates == 2
    assert rc2.call_count[0] == 0, "no LLM on second ingest"


@test
def test_i8_validate_after_ingest():
    vocab = mk_vocab([("a", "def a")])
    net = TagRecordNetwork(vocab=vocab, assignments=[])

    fresh = mk_raw_records([("r1", "t1", "i1")])
    rc = make_reverse_check_fn({
        "r1": {"record_id": "r1", "title": "t1", "selected_tags": [{"name": "a", "confidence": "high"}], "missing": False, "missing_concept": "", "reason": ""},
    })
    net.ingest_batch(fresh, reverse_check_fn=rc, propose_new_fn=make_propose_new_fn([]), propose_threshold=10)
    net.validate()  # must not raise


@test
def test_i9_save_load_after_ingest():
    vocab = mk_vocab([("a", "def a")])
    net = TagRecordNetwork(vocab=vocab, assignments=[])

    fresh = mk_raw_records([("r1", "t1", "i1")])
    rc = make_reverse_check_fn({
        "r1": {"record_id": "r1", "title": "t1", "selected_tags": [], "missing": True, "missing_concept": "x", "reason": ""},
    })
    net.ingest_batch(fresh, reverse_check_fn=rc, propose_new_fn=make_propose_new_fn([]), propose_threshold=100)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "n.json"
        net.save(path)
        net2 = TagRecordNetwork.load(path)

    assert net2.vocab == net.vocab
    assert net2.assignments == net.assignments
    # Pending pool persists
    assert sum(1 for a in net2.assignments if a.get("missing")) == 1


@test
def test_i10_flush_pending_triggers_on_existing_pool():
    """flush_pending forces propose_new on already-accumulated pool without new records."""
    vocab = mk_vocab([("a", "def a")])
    # Pre-load pool with 5 missing records
    assignments = mk_assignments({f"r{i}": [] for i in range(1, 6)})
    net = TagRecordNetwork(vocab=vocab, assignments=assignments)

    raw_records = {f"r{i}": {"record_id": f"r{i}", "title": f"t{i}", "insight": f"i{i}"} for i in range(1, 6)}

    rc_call = [0]
    def rc(records, vocab, concurrency):
        rc_call[0] += 1
        # re-check absorbs all 5 with new tag
        return [{
            "record_id": r["record_id"], "title": r["title"],
            "selected_tags": [{"name": "newtag", "confidence": "high"}],
            "missing": False, "missing_concept": "", "reason": "",
        } for r in records]

    pn = make_propose_new_fn([{"name": "newtag", "definition": "def newtag"}])
    fetcher = make_records_fetcher(raw_records)

    result = net.flush_pending(
        reverse_check_fn=rc,
        propose_new_fn=pn,
        records_fetcher=fetcher,
        propose_threshold=5,
    )

    assert result.ingested == 0, "flush doesn't ingest new records"
    assert result.new_tags_added == ["newtag"]
    assert result.absorbed_from_pool == 5
    assert result.total_pending_after == 0


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
