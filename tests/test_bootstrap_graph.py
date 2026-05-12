"""Bootstrap StateGraph unit tests — no LLM, no SQLite I/O.

Tests the graph mechanics: nodes, conditional routing, HITL interrupt cycle.
LLM calls injected as fakes via build_bootstrap_graph kwargs.
"""
from __future__ import annotations

import sys
import traceback

from langgraph.types import Command

from consolidate_agent.vocab_maintenance.graphs.bootstrap import build_bootstrap_graph
from consolidate_agent.vocab_maintenance.graphs.checkpointer import memory_checkpointer


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ── Fake LLM impls ───────────────────────────────────────────────────────────


def fake_distill(db_path: str, batch_size: int) -> list[dict]:
    return [
        {"record_id": "r1", "title": "t1", "theme": "pattern about X"},
        {"record_id": "r2", "title": "t2", "theme": "pattern about Y"},
        {"record_id": "r3", "title": "t3", "theme": "pattern about Z"},
    ]


def fake_synthesize_seq(vocabs: list[list[dict]], notes: str = "fake"):
    """Returns synthesize_fn that yields each vocab in turn."""
    state = {"i": 0}
    def fn(themes):
        v = vocabs[state["i"]] if state["i"] < len(vocabs) else vocabs[-1]
        state["i"] += 1
        return {"vocab": v, "notes": notes}
    fn._call_count = state
    return fn


def fake_reverse_check(db_path, vocab, concurrency):
    tag = vocab[0]["name"] if vocab else "none"
    return [
        {"record_id": f"r{i}", "title": f"t{i}",
         "selected_tags": [{"name": tag, "confidence": "high"}],
         "missing": False, "missing_concept": "", "reason": ""}
        for i in range(1, 4)
    ]


def drive(graph, initial, config, *, on_review=None):
    """Drive graph + HITL loop until terminal."""
    r = graph.invoke(initial, config)
    while "__interrupt__" in r:
        payload = r["__interrupt__"][0].value
        d = on_review(payload) if on_review else "accept"
        r = graph.invoke(Command(resume=d), config)
    return r


# ── Tests ────────────────────────────────────────────────────────────────────


@test
def test_b1_unattended_e2e_auto_accept():
    vocab = [{"name": "tag_a", "definition": "def a"}]
    g = build_bootstrap_graph(memory_checkpointer(),
                              distill_fn=fake_distill,
                              synthesize_fn=fake_synthesize_seq([vocab]),
                              reverse_check_fn=fake_reverse_check)
    r = g.invoke({"db_path": "fake.db", "auto_accept": True},
                 {"configurable": {"thread_id": "t1"}})
    assert "__interrupt__" not in r
    assert r["vocab"] == vocab
    assert len(r["assignments"]) == 3
    assert r["synthesize_attempts"] == 1


@test
def test_b2_hitl_accept_first_attempt():
    vocab = [{"name": "tag_a", "definition": "def a"}]
    syn = fake_synthesize_seq([vocab])
    g = build_bootstrap_graph(memory_checkpointer(),
                              distill_fn=fake_distill,
                              synthesize_fn=syn,
                              reverse_check_fn=fake_reverse_check)
    seen = []
    def on_review(p):
        seen.append(p)
        return "accept"
    r = drive(g, {"db_path": "fake.db"}, {"configurable": {"thread_id": "t2"}}, on_review=on_review)
    assert len(seen) == 1
    assert seen[0]["stage"] == "vocab_review"
    assert seen[0]["vocab"] == vocab
    assert seen[0]["attempt"] == 1
    assert syn._call_count["i"] == 1


@test
def test_b3_hitl_regenerate_then_accept():
    v1 = [{"name": "v1", "definition": "1"}]
    v2 = [{"name": "v2", "definition": "2"}]
    syn = fake_synthesize_seq([v1, v2])
    g = build_bootstrap_graph(memory_checkpointer(),
                              distill_fn=fake_distill,
                              synthesize_fn=syn,
                              reverse_check_fn=fake_reverse_check)
    decisions = iter(["regenerate", "accept"])
    r = drive(g, {"db_path": "fake.db"}, {"configurable": {"thread_id": "t3"}},
              on_review=lambda p: next(decisions))
    assert syn._call_count["i"] == 2
    assert r["vocab"] == v2
    assert r["synthesize_attempts"] == 2


@test
def test_b4_hitl_abort():
    vocab = [{"name": "tag_a", "definition": "a"}]
    g = build_bootstrap_graph(memory_checkpointer(),
                              distill_fn=fake_distill,
                              synthesize_fn=fake_synthesize_seq([vocab]),
                              reverse_check_fn=fake_reverse_check)
    r = drive(g, {"db_path": "fake.db"}, {"configurable": {"thread_id": "t4"}},
              on_review=lambda p: "abort")
    assert r.get("abort_reason")
    assert not r.get("assignments")


@test
def test_b5_filter_dangling_refs():
    vocab = [{"name": "good_tag", "definition": "kept"}]
    def rc_with_bad(db_path, vocab, concurrency):
        return [
            {"record_id": "r1", "title": "t1",
             "selected_tags": [{"name": "good_tag", "confidence": "high"},
                                {"name": "bad_tag", "confidence": "high"}],
             "missing": False, "missing_concept": "", "reason": ""},
            {"record_id": "r2", "title": "t2",
             "selected_tags": [{"name": "bad_tag", "confidence": "high"}],
             "missing": False, "missing_concept": "", "reason": ""},
        ]
    g = build_bootstrap_graph(memory_checkpointer(),
                              distill_fn=fake_distill,
                              synthesize_fn=fake_synthesize_seq([vocab]),
                              reverse_check_fn=rc_with_bad)
    r = g.invoke({"db_path": "fake.db", "auto_accept": True},
                 {"configurable": {"thread_id": "t5"}})
    r1 = next(a for a in r["assignments"] if a["record_id"] == "r1")
    assert [t["name"] for t in r1["selected_tags"]] == ["good_tag"]
    assert r1["missing"] is False
    r2 = next(a for a in r["assignments"] if a["record_id"] == "r2")
    assert r2["selected_tags"] == []
    assert r2["missing"] is True
    assert r["fake_tag_drops"] == 2


@test
def test_b6_invalid_resume_defaults_to_accept():
    vocab = [{"name": "tag_a", "definition": "a"}]
    g = build_bootstrap_graph(memory_checkpointer(),
                              distill_fn=fake_distill,
                              synthesize_fn=fake_synthesize_seq([vocab]),
                              reverse_check_fn=fake_reverse_check)
    r = drive(g, {"db_path": "fake.db"}, {"configurable": {"thread_id": "t6"}},
              on_review=lambda p: "garbage_value")
    assert len(r["assignments"]) == 3
    assert r["review_decision"] == "accept"


@test
def test_b7_checkpointer_persists_state_across_invokes():
    vocab = [{"name": "tag_a", "definition": "a"}]
    cp = memory_checkpointer()
    g = build_bootstrap_graph(cp,
                              distill_fn=fake_distill,
                              synthesize_fn=fake_synthesize_seq([vocab]),
                              reverse_check_fn=fake_reverse_check)
    config = {"configurable": {"thread_id": "t7"}}
    r1 = g.invoke({"db_path": "fake.db"}, config)
    assert "__interrupt__" in r1
    r2 = g.invoke(Command(resume="accept"), config)
    assert "__interrupt__" not in r2
    assert len(r2["assignments"]) == 3
    assert r2["vocab"] == vocab


# ── Runner ───────────────────────────────────────────────────────────────────


def main() -> int:
    passed = 0
    failed = 0
    print(f"\nRunning {len(TESTS)} bootstrap_graph tests…\n")
    print("─" * 50)
    for fn in TESTS:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {fn.__name__}")
            print(f"     {type(e).__name__}: {e}")
            tb_lines = traceback.format_exc().split("\n")
            for line in tb_lines[-8:-1]:
                if line.strip():
                    print(f"     {line}")
            failed += 1
    print(f"\n{'─' * 50}")
    print(f"Total: {passed + failed}, Passed: {passed}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
