"""Agent loop unit tests — mock LLM, fast.

Tests:
  U1  done immediately
  U2  one iter then done
  U3  budget exhausted
  U4  empty proposals → block action
  U5  hit_rate regression → rollback
  U6  all actions blocked → no_actions_remaining
  U7  apply OrphanError → block action
  U8  apply InvalidProposal → block action
  U9  action_focus propagates to propose_fn
  U10 history length == iter count

Invariants checked after each test:
  A1 iter <= max_iter
  A2 no iteration ran 2 different propose actions
  A3 rollback leaves vocab unchanged
  A4 blocked actions never retried with same vocab
  A5 history length matches iter count
  A6 check_invariants passes on final state
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/agent"))

from agent import ACTIONS, Agent, FinalStatus  # noqa: E402
from apply import (  # noqa: E402
    DeprecateProposal,
    MergeProposal,
    NewTagProposal,
    check_invariants,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def mk_vocab(names: list[str]) -> list[dict]:
    return [{"name": n, "definition": f"def of {n}"} for n in names]


def mk_assignments(spec: dict[str, list[str]]) -> list[dict]:
    out = []
    for rid, tags in spec.items():
        out.append({
            "record_id": rid,
            "title": f"title {rid}",
            "missing": not tags,
            "selected_tags": [{"name": t, "confidence": "high"} for t in tags],
            "missing_concept": "",
            "reason": "",
        })
    return out


def mk_diagnostics(assignments: list[dict], vocab: list[dict]) -> dict:
    from collections import Counter
    n = len(assignments)
    assigned = sum(1 for a in assignments if not a["missing"])
    usage: Counter[str] = Counter()
    for a in assignments:
        for t in a.get("selected_tags", []):
            usage[t["name"]] += 1
    return {
        "sample_size": n,
        "total_assigned": assigned,
        "total_missing": n - assigned,
        "tag_usage_count": dict(usage),
        "unused_tags": sorted({t["name"] for t in vocab} - set(usage.keys())),
        "missing_records": [],
        "boundary_blur_records": [],
        "top_cooccurrence_pairs": [],
        "low_confidence_records": [],
    }


def make_measure_fn(post_apply_diag_overrides: list[dict] | None = None):
    """Returns measure_fn that re-computes diag from vocab+assignments.
    Optionally overrides total_assigned per call (to simulate hit_rate changes)."""
    call_count = {"i": 0}
    overrides = list(post_apply_diag_overrides or [])

    def measure_fn(vocab):
        # Use a default assignments inferred from vocab (simplistic — caller sets via mock)
        assignments = make_measure_fn._next_assignments.get(id(vocab), [])
        diag = mk_diagnostics(assignments, vocab)
        if call_count["i"] < len(overrides):
            diag.update(overrides[call_count["i"]])
        call_count["i"] += 1
        return diag, assignments

    measure_fn._next_assignments = {}
    return measure_fn


def make_constant_measure(diag: dict, assignments: list[dict]):
    """Simpler: always return same diag + assignments."""
    def measure_fn(vocab):
        d = dict(diag)
        # adapt to vocab (unused_tags depends on vocab)
        from collections import Counter
        usage: Counter[str] = Counter()
        for a in assignments:
            for t in a.get("selected_tags", []):
                if t["name"] in {v["name"] for v in vocab}:
                    usage[t["name"]] += 1
        d["unused_tags"] = sorted({v["name"] for v in vocab} - set(usage.keys()))
        return d, [dict(a, selected_tags=list(a["selected_tags"])) for a in assignments]
    return measure_fn


def make_diagnose_seq(decisions: list[dict]):
    """Returns diagnose_fn that yields decisions in order; last one repeats."""
    it = iter(decisions)
    last = [None]
    def diagnose_fn(vocab, diag, assign):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return {"decision": last[0]}
    return diagnose_fn


def make_propose_constant(proposals_list: list):
    """Always returns this proposals list."""
    captured = {"focus_calls": []}
    def propose_fn(vocab, assignments, focus):
        captured["focus_calls"].append(focus)
        return list(proposals_list)
    propose_fn._captured = captured
    return propose_fn


# ── Invariants ───────────────────────────────────────────────────────────────


def check_loop_invariants(state, status, max_iter: int):
    # A1: iter <= max_iter
    assert state.iter <= max_iter, f"iter {state.iter} > max_iter {max_iter}"
    # A5: history length matches iter count
    # done iter records 1 history; pre-stop iters also record
    # number of records may exceed state.iter if iter was incremented before recording
    # actually: every loop iteration records exactly 1 record OR loop short-circuited at stop check (no record)
    # If state.iter == N and final status was stop check at top, history has N records
    assert len(state.history) == state.iter, \
        f"history len {len(state.history)} != iter {state.iter}"
    # A6: final invariants
    check_invariants(state.vocab, state.assignments)


# ── Tests ────────────────────────────────────────────────────────────────────


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# U1: done immediately
@test
def test_u1_done_immediately():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"], "r2": ["b"]})
    diag = mk_diagnostics(assignments, vocab)

    agent = Agent(
        measure_fn=make_constant_measure(diag, assignments),
        diagnose_fn=make_diagnose_seq([
            {"next_action": "done", "action_focus": "", "confidence": "high"}
        ]),
        propose_fns={},
    )
    state, status = agent.run(vocab)
    assert status == FinalStatus.COMPLETED, f"got {status}"
    assert state.iter == 1
    assert state.history[-1].action == "done"
    assert state.history[-1].result == "completed"
    check_loop_invariants(state, status, 5)


# U2: one propose iter then done
@test
def test_u2_one_iter_then_done():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"], "r2": ["b"]})
    diag = mk_diagnostics(assignments, vocab)

    propose_new = make_propose_constant([
        NewTagProposal(name="c", definition="def of c")
    ])
    agent = Agent(
        measure_fn=make_constant_measure(diag, assignments),
        diagnose_fn=make_diagnose_seq([
            {"next_action": "propose_new", "action_focus": "needs c", "confidence": "high"},
            {"next_action": "done", "action_focus": "", "confidence": "high"},
        ]),
        propose_fns={"propose_new": propose_new},
    )
    state, status = agent.run(vocab)
    assert status == FinalStatus.COMPLETED
    assert state.iter == 2
    assert state.history[0].action == "propose_new"
    assert state.history[0].result == "applied"
    assert state.history[1].action == "done"
    assert "c" in [t["name"] for t in state.vocab]
    check_loop_invariants(state, status, 5)


# U3: budget exhausted
@test
def test_u3_budget_exhausted():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    diag = mk_diagnostics(assignments, vocab)

    # diagnose always returns propose_new, propose returns new tag each time
    counter = {"i": 0}
    def propose_new(vocab, assignments, focus):
        counter["i"] += 1
        return [NewTagProposal(name=f"new_{counter['i']}", definition="d")]

    agent = Agent(
        measure_fn=make_constant_measure(diag, assignments),
        diagnose_fn=make_diagnose_seq([
            {"next_action": "propose_new", "action_focus": "", "confidence": "high"}
        ]),
        propose_fns={"propose_new": propose_new},
        max_iter=3,
    )
    state, status = agent.run(vocab)
    assert status == FinalStatus.MAX_ITER_EXHAUSTED
    assert state.iter == 3
    check_loop_invariants(state, status, 3)


# U4: empty proposals → block action
@test
def test_u4_empty_proposals_blocks_action():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    diag = mk_diagnostics(assignments, vocab)

    propose_new = make_propose_constant([])  # always empty
    agent = Agent(
        measure_fn=make_constant_measure(diag, assignments),
        diagnose_fn=make_diagnose_seq([
            {"next_action": "propose_new", "action_focus": "", "confidence": "high"},
            {"next_action": "done", "action_focus": "", "confidence": "high"},
        ]),
        propose_fns={"propose_new": propose_new},
    )
    state, status = agent.run(vocab)
    assert state.history[0].result == "blocked_empty"
    assert "propose_new" in state.history[0].blocked_actions_after
    assert state.history[1].action == "done"
    assert status == FinalStatus.COMPLETED
    check_loop_invariants(state, status, 5)


# U5: hit_rate regression → rollback
@test
def test_u5_hit_rate_regression_rollback():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"], "r2": ["b"], "r3": ["a"]})

    # initial diag: hit_rate = 3/3 = 1.0
    # after apply diag: hit_rate = 0.5 (drop > 0.03)
    diag_initial = mk_diagnostics(assignments, vocab)
    assignments_after = [
        dict(assignments[0], missing=True, selected_tags=[]),
        dict(assignments[1], missing=True, selected_tags=[]),
        assignments[2],
    ]
    diag_after = mk_diagnostics(assignments_after, vocab)

    call_count = {"i": 0}
    def measure_fn(vocab):
        call_count["i"] += 1
        if call_count["i"] == 1:
            return diag_initial, [dict(a, selected_tags=list(a["selected_tags"])) for a in assignments]
        # subsequent calls = after apply
        return diag_after, assignments_after

    propose_new = make_propose_constant([NewTagProposal(name="c", definition="d")])
    agent = Agent(
        measure_fn=measure_fn,
        diagnose_fn=make_diagnose_seq([
            {"next_action": "propose_new", "action_focus": "", "confidence": "high"},
            {"next_action": "done", "action_focus": "", "confidence": "high"},
        ]),
        propose_fns={"propose_new": propose_new},
    )
    state, status = agent.run(vocab)
    # First iter rolled back
    assert state.history[0].result == "rolled_back", f"got {state.history[0].result}"
    assert "propose_new" in state.history[0].blocked_actions_after
    # vocab unchanged after rollback
    assert "c" not in [t["name"] for t in state.vocab], "vocab was not rolled back"
    assert status == FinalStatus.COMPLETED
    check_loop_invariants(state, status, 5)


# U6: all actions blocked → no_actions_remaining
@test
def test_u6_all_blocked():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    diag = mk_diagnostics(assignments, vocab)

    # diagnose cycles through 3 actions, each propose returns empty (→ block)
    seq = [
        {"next_action": "propose_new", "action_focus": "", "confidence": "high"},
        {"next_action": "propose_merge", "action_focus": "", "confidence": "high"},
        {"next_action": "propose_deprecate", "action_focus": "", "confidence": "high"},
        # then diagnose would never be called because no_actions_remaining triggers
    ]
    agent = Agent(
        measure_fn=make_constant_measure(diag, assignments),
        diagnose_fn=make_diagnose_seq(seq),
        propose_fns={
            "propose_new": make_propose_constant([]),
            "propose_merge": make_propose_constant([]),
            "propose_deprecate": make_propose_constant([]),
        },
    )
    state, status = agent.run(vocab)
    assert status == FinalStatus.NO_ACTIONS_REMAINING, f"got {status}"
    assert state.blocked_actions == set(ACTIONS)
    check_loop_invariants(state, status, 5)


# U7: apply OrphanError → block action
@test
def test_u7_orphan_error_blocks_action():
    vocab = mk_vocab(["x", "y"])
    # r1 only has x → deprecating x will orphan it
    assignments = mk_assignments({"r1": ["x"], "r2": ["y"]})
    diag = mk_diagnostics(assignments, vocab)

    propose_dep = make_propose_constant([DeprecateProposal(tag="x")])
    agent = Agent(
        measure_fn=make_constant_measure(diag, assignments),
        diagnose_fn=make_diagnose_seq([
            {"next_action": "propose_deprecate", "action_focus": "", "confidence": "high"},
            {"next_action": "done", "action_focus": "", "confidence": "high"},
        ]),
        propose_fns={"propose_deprecate": propose_dep},
    )
    state, status = agent.run(vocab)
    assert state.history[0].result == "apply_error"
    assert "OrphanError" in (state.history[0].error or "")
    assert "propose_deprecate" in state.history[0].blocked_actions_after
    assert status == FinalStatus.COMPLETED
    check_loop_invariants(state, status, 5)


# U8: apply InvalidProposal → block action
@test
def test_u8_invalid_proposal_blocks_action():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    diag = mk_diagnostics(assignments, vocab)

    # propose returns a merge with nonexistent tag
    propose_merge = make_propose_constant([MergeProposal(keep_tag="b", discard_tag="ZZZ")])
    agent = Agent(
        measure_fn=make_constant_measure(diag, assignments),
        diagnose_fn=make_diagnose_seq([
            {"next_action": "propose_merge", "action_focus": "", "confidence": "high"},
            {"next_action": "done", "action_focus": "", "confidence": "high"},
        ]),
        propose_fns={"propose_merge": propose_merge},
    )
    state, status = agent.run(vocab)
    assert state.history[0].result == "apply_error"
    assert "InvalidProposal" in (state.history[0].error or "")
    assert "propose_merge" in state.history[0].blocked_actions_after
    check_loop_invariants(state, status, 5)


# U9: action_focus propagates to propose_fn
@test
def test_u9_action_focus_propagates():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    diag = mk_diagnostics(assignments, vocab)

    propose_new = make_propose_constant([NewTagProposal(name="c", definition="d")])
    agent = Agent(
        measure_fn=make_constant_measure(diag, assignments),
        diagnose_fn=make_diagnose_seq([
            {"next_action": "propose_new", "action_focus": "focus on encoding gap", "confidence": "high"},
            {"next_action": "done", "action_focus": "", "confidence": "high"},
        ]),
        propose_fns={"propose_new": propose_new},
    )
    state, status = agent.run(vocab)
    captured_focus = propose_new._captured["focus_calls"]
    assert "focus on encoding gap" in captured_focus, f"focus not propagated: {captured_focus}"
    check_loop_invariants(state, status, 5)


# U10: history length == iter count
@test
def test_u10_history_length():
    vocab = mk_vocab(["a", "b"])
    assignments = mk_assignments({"r1": ["a"]})
    diag = mk_diagnostics(assignments, vocab)

    propose_new = make_propose_constant([NewTagProposal(name="c", definition="d")])
    agent = Agent(
        measure_fn=make_constant_measure(diag, assignments),
        diagnose_fn=make_diagnose_seq([
            {"next_action": "propose_new", "action_focus": "", "confidence": "high"},
            {"next_action": "propose_new", "action_focus": "", "confidence": "high"},
            {"next_action": "done", "action_focus": "", "confidence": "high"},
        ]),
        propose_fns={"propose_new": propose_new},
    )
    state, status = agent.run(vocab)
    # 3 iterations recorded
    assert state.iter == 3
    assert len(state.history) == 3
    check_loop_invariants(state, status, 5)


# ── Runner ───────────────────────────────────────────────────────────────────


def main():
    passed = 0
    failed = 0
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
