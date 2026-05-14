"""Tests for vocab_maintenance/agent_brain: AgentMemory, AgentDecision, gate, OutcomeLog."""
from __future__ import annotations

import tempfile
from pathlib import Path
from dataclasses import field

import pytest

from consolidate_agent.vocab_maintenance.agent_brain import (
    AgentDecision,
    AgentMemory,
    DecisionOutcome,
    GateResult,
    NetworkState,
    OutcomeLog,
    RoundSummary,
    ToolResult,
    apply_reversibility_default,
    gate_decision,
    make_decision_id,
)
from consolidate_agent.vocab_maintenance.agent_brain.memory import _detect_critical_patterns
from consolidate_agent.vocab_maintenance.agent_brain.tools import BrainContext, call_tool_with_cache, TOOLS_BY_NAME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_network_state(**kwargs) -> NetworkState:
    defaults = dict(
        tag_sizes={"foo": 10, "bar": 5},
        recent_operations={},
        total_records=15,
        total_edges=15,
    )
    defaults.update(kwargs)
    return NetworkState(**defaults)


def _make_memory(**kwargs) -> AgentMemory:
    return AgentMemory(_make_network_state(**kwargs))


def _make_round_summary(
    idx: int,
    target: str = "tag_a",
    action: str | None = "refine",
    committed: bool = True,
) -> RoundSummary:
    return RoundSummary(
        round_idx=idx,
        tools_called=["inspect_tag"],
        target=target,
        decision_action=action,
        decision_certainty="medium",
        outcome="coherence 0.40→0.47",
        committed=committed,
        rolled_back=False,
    )


def _make_clean_decision(action: str = "refine", target: str = "my_tag") -> AgentDecision:
    """A decision designed to pass all gates."""
    return AgentDecision(
        action=action,
        target=target,
        reasoning="x" * 80,
        certainty="medium",
        supporting_observations=["obs1", "obs2"],
        opposing_observations=[],
        preview_reviewed=True,
        affected_records_estimate=5,
        reversibility="clean_rollback",
    )


# ---------------------------------------------------------------------------
# AgentMemory — FIFO caps
# ---------------------------------------------------------------------------

def test_working_fifo_at_8_items():
    mem = _make_memory()
    for i in range(9):
        mem.record_tool_call("tool", {"i": i}, f"result_{i}")
    assert len(mem.working) == 8
    # oldest (i=0) should be evicted
    assert all(tr.args["i"] != 0 for tr in mem.working)
    assert mem.working[0].args["i"] == 1
    assert mem.working[-1].args["i"] == 8


def test_history_fifo_at_15_rounds():
    mem = _make_memory()
    for i in range(16):
        mem.archive_round(_make_round_summary(i))
    assert len(mem.history) == 15
    # round 0 should be evicted
    assert mem.history[0].round_idx == 1
    assert mem.history[-1].round_idx == 15


def test_facts_fifo_at_30_items():
    mem = _make_memory()
    for i in range(31):
        mem.add_fact(f"fact_{i}")
    assert len(mem.facts) == 30
    assert mem.facts[0] == "fact_1"
    assert mem.facts[-1] == "fact_30"


# ---------------------------------------------------------------------------
# AgentMemory — render_for_llm
# ---------------------------------------------------------------------------

def test_render_for_llm_in_correct_order():
    mem = _make_memory()
    mem.add_fact("some fact")
    mem.archive_round(_make_round_summary(0))
    mem.record_tool_call("inspect_tag", {}, "ok")

    output = mem.render_for_llm()
    idx_state = output.index("=== Current network state ===")
    idx_facts = output.index("=== Run facts ===")
    idx_history = output.index("=== Recent rounds (last 8) ===")
    idx_working = output.index("=== This round so far ===")

    assert idx_state < idx_facts < idx_history < idx_working


def test_render_for_llm_truncates_history_when_oversized():
    mem = _make_memory()
    # Pack history with fat summaries by using long outcome strings
    for i in range(15):
        rs = RoundSummary(
            round_idx=i,
            tools_called=["t"] * 3,
            target="a" * 200,
            decision_action="refine",
            decision_certainty="medium",
            outcome="x" * 500,
            committed=True,
            rolled_back=False,
        )
        mem.history.append(rs)
    # Pack working with fat tool results
    for i in range(8):
        mem.working.append(
            ToolResult(
                tool="inspect_tag",
                args={"key": "v"},
                result="r" * 200,
                round_idx=99,
                timestamp="2026-01-01T00:00:00+00:00",
            )
        )

    output = mem.render_for_llm()
    assert len(output) <= 10_000
    # working must still be present
    assert "=== This round so far ===" in output
    assert "inspect_tag" in output


def test_archive_round_updates_network_state():
    mem = _make_memory(tag_sizes={"existing": 3}, total_records=3)
    rs = _make_round_summary(0)
    mem.archive_round(rs, applied_changes={"added_tags": ["foo"], "removed_tags": [], "modified_tags": [], "record_reassignments": 0})
    assert "foo" in mem.state.tag_sizes
    assert "foo" in mem.state.recent_operations
    assert "added" in mem.state.recent_operations["foo"]


def test_archive_round_removes_tag_from_sizes():
    mem = _make_memory(tag_sizes={"to_remove": 5}, total_records=5)
    rs = _make_round_summary(0)
    mem.archive_round(rs, applied_changes={"added_tags": [], "removed_tags": ["to_remove"], "modified_tags": [], "record_reassignments": 0})
    assert "to_remove" not in mem.state.tag_sizes


def test_archive_round_clears_working():
    mem = _make_memory()
    mem.record_tool_call("t", {}, "r")
    assert len(mem.working) == 1
    mem.archive_round(_make_round_summary(0))
    assert len(mem.working) == 0


def test_archive_round_increments_round_idx():
    mem = _make_memory()
    assert mem.current_round_idx == 0
    mem.archive_round(_make_round_summary(0))
    assert mem.current_round_idx == 1


# ---------------------------------------------------------------------------
# gate_decision
# ---------------------------------------------------------------------------

def test_stop_always_auto():
    mem = _make_memory()
    d = AgentDecision(
        action="stop",
        target="",
        reasoning="x" * 80,
        certainty="high",
        supporting_observations=["obs1"],
        stop_reason="network looks good",
    )
    gd = gate_decision(d, mem)
    assert gd.result == GateResult.AUTO
    assert "stop_always_auto" in gd.triggered_gates


def test_high_impact_no_preview_review():
    mem = _make_memory()
    d = _make_clean_decision()
    d = d.model_copy(update={"affected_records_estimate": 50, "preview_reviewed": False})
    gd = gate_decision(d, mem)
    assert gd.result == GateResult.REVIEW
    assert "high_impact_no_preview" in gd.triggered_gates


def test_high_cert_no_preview_review():
    mem = _make_memory()
    d = _make_clean_decision(action="split")
    d = d.model_copy(update={"certainty": "high", "preview_reviewed": False})
    gd = gate_decision(d, mem)
    assert gd.result == GateResult.REVIEW
    assert "high_cert_no_preview" in gd.triggered_gates


def test_flipflop_detection():
    mem = _make_memory()
    # Add same target twice in recent history with actionable decisions
    for i in range(3):
        mem.history.append(
            RoundSummary(
                round_idx=i,
                tools_called=["t"],
                target="conflicted_tag",
                decision_action="refine",
                decision_certainty="medium",
                outcome=None,
                committed=True,
                rolled_back=False,
            )
        )
    d = _make_clean_decision(action="refine", target="conflicted_tag")
    gd = gate_decision(d, mem)
    assert gd.result == GateResult.REVIEW
    assert "recent_flipflop" in gd.triggered_gates


def test_clean_decision_passes_all_gates():
    mem = _make_memory()
    d = _make_clean_decision(action="refine", target="new_tag")
    gd = gate_decision(d, mem)
    assert gd.result == GateResult.AUTO
    assert "all_gates_pass" in gd.triggered_gates


def test_apply_reversibility_default_deprecate():
    d = AgentDecision(
        action="deprecate",
        target="old_tag",
        reasoning="x" * 80,
        certainty="medium",
        supporting_observations=["obs1", "obs2"],
    )
    # pydantic default is "clean_rollback"; action's semantic default is "messy_rollback"
    assert d.reversibility == "clean_rollback"
    d2 = apply_reversibility_default(d)
    assert d2.reversibility == "messy_rollback"


def test_apply_reversibility_default_respects_explicit_override():
    d = AgentDecision(
        action="split",
        target="big_tag",
        reasoning="x" * 80,
        certainty="medium",
        supporting_observations=["obs1", "obs2"],
        reversibility="messy_rollback",  # LLM override
    )
    d2 = apply_reversibility_default(d)
    # LLM said messy; we should keep that even though split default is clean
    assert d2.reversibility == "messy_rollback"


# ---------------------------------------------------------------------------
# OutcomeLog
# ---------------------------------------------------------------------------

def _make_outcome(decision_id: str, action: str = "refine", certainty: str = "medium") -> DecisionOutcome:
    return DecisionOutcome(
        decision_id=decision_id,
        decision={"action": action, "certainty": certainty},
        gate_result={"result": "auto", "triggered_gates": ["all_gates_pass"], "reason": "ok"},
        timestamp="2026-01-01T00:00:00+00:00",
    )


def test_log_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        log = OutcomeLog(Path(tmpdir) / "sub" / "decisions.jsonl")
        for i in range(3):
            log.log(_make_outcome(f"run1.r{i}.refine.tag"))
        loaded = log.load_all()
        assert len(loaded) == 3
        assert loaded[0].decision_id == "run1.r0.refine.tag"
        assert loaded[2].decision_id == "run1.r2.refine.tag"


def test_query_filter_by_action():
    with tempfile.TemporaryDirectory() as tmpdir:
        log = OutcomeLog(Path(tmpdir) / "d.jsonl")
        log.log(_make_outcome("id1", action="refine"))
        log.log(_make_outcome("id2", action="split"))
        log.log(_make_outcome("id3", action="refine"))

        splits = log.query(action="split")
        assert len(splits) == 1
        assert splits[0].decision_id == "id2"

        refines = log.query(action="refine")
        assert len(refines) == 2


def test_make_decision_id_format():
    did = make_decision_id("run42", 3, "split", "my_tag")
    parts = did.split(".")
    assert parts[0] == "run42"
    assert parts[1] == "r3"
    assert parts[2] == "split"
    assert parts[3] == "my_tag"
    assert len(parts[4]) == 8


# ---------------------------------------------------------------------------
# Pattern Detector
# ---------------------------------------------------------------------------

def _make_review_summary(idx: int, target: str, action: str) -> RoundSummary:
    return RoundSummary(
        round_idx=idx,
        tools_called=["inspect_tag"],
        target=target,
        decision_action=action,
        decision_certainty="high",
        outcome="gate=REVIEW (gates triggered: high_cert_no_preview), not applied",
        committed=False,
        rolled_back=False,
    )


def test_pattern_detect_repeated_target_action():
    """History with same target+action REVIEW-gated >= 2 times → Pattern A triggers."""
    mem = _make_memory()
    for i in range(3):
        mem.history.append(_make_review_summary(i, "my_tag", "refine"))

    result = _detect_critical_patterns(mem)
    assert result is not None
    assert "Pattern A" in result
    assert "refine my_tag" in result
    assert "⚠⚠⚠ CRITICAL PATTERNS DETECTED" in result


def test_pattern_detect_repeated_tool_call():
    """Working list with same tool+args called >= 3 times → Pattern B triggers."""
    mem = _make_memory()
    for _ in range(3):
        mem.working.append(ToolResult(
            tool="inspect_tag",
            args={"tag_name": "my_tag"},
            result={"ok": True},
            round_idx=0,
            timestamp="2026-01-01T00:00:00+00:00",
        ))

    result = _detect_critical_patterns(mem)
    assert result is not None
    assert "Pattern B" in result
    assert "inspect_tag" in result


def test_pattern_no_false_trigger():
    """Normal history (different targets, different tools) → no pattern triggered."""
    mem = _make_memory()
    # Different targets, different actions, all committed
    mem.history.append(RoundSummary(
        round_idx=0, tools_called=["assess_global_health"], target="tag_a",
        decision_action="refine", decision_certainty="medium",
        outcome="applied: vocab 10→10 tags", committed=True, rolled_back=False,
    ))
    mem.history.append(RoundSummary(
        round_idx=1, tools_called=["inspect_tag"], target="tag_b",
        decision_action="split", decision_certainty="high",
        outcome="applied: vocab 10→11 tags", committed=True, rolled_back=False,
    ))
    # Different tool calls in working
    mem.working.append(ToolResult(
        tool="assess_global_health", args={}, result={"ok": True},
        round_idx=2, timestamp="2026-01-01T00:00:00+00:00",
    ))
    mem.working.append(ToolResult(
        tool="inspect_tag", args={"tag_name": "tag_c"}, result={"ok": True},
        round_idx=2, timestamp="2026-01-01T00:00:00+00:00",
    ))

    result = _detect_critical_patterns(mem)
    assert result is None


def test_pattern_a_when_only_one_round_history():
    """Only 1 REVIEW round → Pattern A NOT triggered (needs >= 2)."""
    mem = _make_memory()
    mem.history.append(_make_review_summary(0, "my_tag", "refine"))

    result = _detect_critical_patterns(mem)
    # Should have no Pattern A (only 1 review)
    if result is not None:
        assert "Pattern A" not in result


# ---------------------------------------------------------------------------
# Tool Repeat Cache
# ---------------------------------------------------------------------------

def _make_brain_context() -> BrainContext:
    return BrainContext(
        db_path=Path("/tmp/fake.db"),
        vocab=[],
        assignments=[],
        golden=[],
    )


def test_tool_cache_blocks_duplicate():
    """Same tool + same args the second time returns error with duplicate_call=True."""
    ctx = _make_brain_context()
    tool = TOOLS_BY_NAME["assess_global_health"]

    # Override impl to avoid real DB calls
    original_impl = tool.impl
    tool.impl = lambda args, c: {"ok": True, "result": {"tag_count": 5}, "error": None}

    try:
        r1 = call_tool_with_cache(tool, {}, ctx)
        assert r1["ok"] is True

        r2 = call_tool_with_cache(tool, {}, ctx)
        assert r2["ok"] is False
        assert r2.get("duplicate_call") is True
        assert "already called" in r2["error"]
    finally:
        tool.impl = original_impl


def test_tool_cache_per_round():
    """After clearing _tool_cache_this_round (simulating archive), tool can be called again."""
    ctx = _make_brain_context()
    tool = TOOLS_BY_NAME["assess_global_health"]

    original_impl = tool.impl
    tool.impl = lambda args, c: {"ok": True, "result": {"tag_count": 5}, "error": None}

    try:
        r1 = call_tool_with_cache(tool, {}, ctx)
        assert r1["ok"] is True

        # Simulate archive_round clearing the cache
        ctx._tool_cache_this_round.clear()

        # Should succeed again in new round
        r2 = call_tool_with_cache(tool, {}, ctx)
        assert r2["ok"] is True
        assert r2.get("duplicate_call") is None
    finally:
        tool.impl = original_impl
