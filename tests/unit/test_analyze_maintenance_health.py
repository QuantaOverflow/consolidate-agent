"""Tests for compute_tag_streaks pure function."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from analyze_maintenance_health import compute_tag_streaks


def test_empty_input_returns_empty_list():
    assert compute_tag_streaks([], 3) == []


def test_single_tag_all_applied_streak_is_zero():
    runs = [
        {"run_id": "r1", "tag_iters": [("tag_x", "applied"), ("tag_x", "applied")]},
    ]
    result = compute_tag_streaks(runs, 3)
    assert len(result) == 1
    assert result[0]["tag"] == "tag_x"
    assert result[0]["streak"] == 0


def test_single_tag_streak_stops_at_applied():
    # sequence (oldest → newest): applied, blocked_empty, rolled_back, blocked_empty
    # streak = 3 (last 3 are all non-applied; the applied before them breaks the streak)
    runs = [
        {"run_id": "r1", "tag_iters": [("tag_a", "applied")]},
        {"run_id": "r2", "tag_iters": [("tag_a", "blocked_empty")]},
        {"run_id": "r3", "tag_iters": [("tag_a", "rolled_back")]},
        {"run_id": "r4", "tag_iters": [("tag_a", "blocked_empty")]},
    ]
    result = compute_tag_streaks(runs, 3)
    assert result[0]["tag"] == "tag_a"
    assert result[0]["streak"] == 3


def test_merge_focus_split_tuple_list_streak():
    # tag_x: run1 blocked_empty, run2 rolled_back → streak=2
    runs = [
        {"run_id": "r1", "tag_iters": [("tag_x", "blocked_empty")]},
        {"run_id": "r2", "tag_iters": [("tag_x", "rolled_back")]},
    ]
    result = compute_tag_streaks(runs, 3)
    tag_x = next(r for r in result if r["tag"] == "tag_x")
    assert tag_x["streak"] == 2


def test_last_results_returns_newest_five_regardless_of_streak():
    # 7 results; last 5 newest are what we want, newest first
    results_seq = ["applied", "blocked_empty", "applied", "rolled_back", "applied", "apply_error", "propose_error"]
    runs = [{"run_id": f"r{i}", "tag_iters": [("t", r)]} for i, r in enumerate(results_seq)]
    result = compute_tag_streaks(runs, 3)
    t_entry = next(r for r in result if r["tag"] == "t")
    # The 5 most recent (newest first): propose_error, apply_error, applied, rolled_back, applied
    assert t_entry["last_results"] == ["propose_error", "apply_error", "applied", "rolled_back", "applied"]
    # streak: propose_error, apply_error → then applied → stop; streak=2
    assert t_entry["streak"] == 2
