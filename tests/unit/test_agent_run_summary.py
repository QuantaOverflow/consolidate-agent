"""Tests for compute_run_summary and decide_final_status pure functions."""
from __future__ import annotations

import pytest

from consolidate_agent.vocab_maintenance.graphs.agent import (
    _DEGRADATION_THRESHOLD,
    _FAILURE_RESULTS,
    _VALID_RESULTS,
    compute_run_summary,
    decide_final_status,
)


# ── compute_run_summary ───────────────────────────────────────────────────────

def test_compute_run_summary_returns_zeros_when_empty_history():
    summary = compute_run_summary([])
    assert summary["processed"] == 0
    assert summary["failure_rate"] == 0.0
    for key in ("applied", "blocked_empty", "rolled_back", "apply_error", "propose_error", "unknown_action"):
        assert summary[key] == 0


def test_compute_run_summary_counts_each_result_correctly_when_mixed_history():
    history = [
        {"result": "applied"},
        {"result": "applied"},
        {"result": "propose_error"},
        {"result": "blocked_empty"},
    ]
    summary = compute_run_summary(history)
    assert summary["applied"] == 2
    assert summary["propose_error"] == 1
    assert summary["blocked_empty"] == 1
    assert summary["apply_error"] == 0
    assert summary["rolled_back"] == 0
    assert summary["unknown_action"] == 0
    assert summary["processed"] == 4


def test_compute_run_summary_computes_failure_rate_correctly_when_3_of_10_fail():
    history = (
        [{"result": "applied"}] * 5
        + [{"result": "propose_error"}] * 2
        + [{"result": "apply_error"}] * 1
        + [{"result": "blocked_empty"}] * 1
        + [{"result": "rolled_back"}] * 1
    )
    assert len(history) == 10
    summary = compute_run_summary(history)
    assert summary["processed"] == 10
    assert summary["failure_rate"] == pytest.approx(3 / 10)


def test_compute_run_summary_ignores_unknown_result_strings_when_present():
    history = [
        {"result": "applied"},
        {"result": "totally_unknown_result"},
        {"result": "propose_error"},
    ]
    summary = compute_run_summary(history)
    assert summary["processed"] == 2
    assert summary["applied"] == 1
    assert summary["propose_error"] == 1


# ── decide_final_status ───────────────────────────────────────────────────────

def test_decide_final_status_returns_completed_when_processed_is_zero():
    summary = compute_run_summary([])
    assert decide_final_status("max_iter_exhausted", summary) == "completed"


def test_decide_final_status_returns_degraded_when_failure_rate_above_threshold_and_cause_completed():
    summary = {"processed": 2, "failure_rate": 0.5}
    assert decide_final_status("completed", summary) == "degraded"


def test_decide_final_status_returns_degraded_when_failure_rate_above_threshold_and_cause_max_iter():
    summary = {"processed": 2, "failure_rate": 0.5}
    assert decide_final_status("max_iter_exhausted", summary) == "degraded"


def test_decide_final_status_returns_completed_when_failure_rate_below_threshold():
    summary = {"processed": 10, "failure_rate": 0.2}
    assert decide_final_status("completed", summary) == "completed"


def test_decide_final_status_returns_max_iter_when_failure_rate_below_threshold_and_cause_max_iter():
    summary = {"processed": 10, "failure_rate": 0.2}
    assert decide_final_status("max_iter_exhausted", summary) == "max_iter_exhausted"


def test_decide_final_status_returns_degraded_when_failure_rate_equals_threshold():
    summary = {"processed": 10, "failure_rate": _DEGRADATION_THRESHOLD}
    assert decide_final_status("completed", summary) == "degraded"


# ── acceptance fixture: 5 propose_error iters → degraded ─────────────────────

def test_decide_final_status_returns_degraded_and_failure_rate_1_when_all_propose_error():
    history = [{"result": "propose_error"}] * 5
    summary = compute_run_summary(history)
    assert summary["failure_rate"] == pytest.approx(1.0)
    status = decide_final_status("completed", summary)
    assert status == "degraded"
