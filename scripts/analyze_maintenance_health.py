#!/usr/bin/env python3
"""Operator tool: cross-run maintenance health analysis from JSONL run logs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Pure functions ────────────────────────────────────────────────────────────


def compute_tag_streaks(
    runs: list[dict],
    threshold: int = 3,
) -> list[dict]:
    """Return per-tag streak summary sorted by streak descending.

    Each run dict: {"run_id": str, "tag_iters": [(tag_name, result), ...]}
    streak = consecutive non-"applied" results counting backwards from latest.
    last_results = up to 5 most-recent results (newest first), regardless of value.
    """
    # Build per-tag full result sequence (chronological order, oldest first)
    tag_results: dict[str, list[str]] = {}
    for run in runs:
        for tag, result in run.get("tag_iters", []):
            if tag not in tag_results:
                tag_results[tag] = []
            tag_results[tag].append(result)

    out: list[dict] = []
    for tag, results in tag_results.items():
        # streak: count backwards from end while result != "applied"
        streak = 0
        for r in reversed(results):
            if r == "applied":
                break
            streak += 1
        # last_results: newest 5 (reversed slice)
        last_results = list(reversed(results[-5:])) if results else []
        out.append({"tag": tag, "streak": streak, "last_results": last_results})

    out.sort(key=lambda x: x["streak"], reverse=True)
    return out


# ── JSONL parsing ─────────────────────────────────────────────────────────────


def _load_run(path: Path) -> dict | None:
    """Parse one jsonl file; return run summary dict or None if no agent.run.done found."""
    run_done: dict | None = None
    tag_iters: list[tuple[str, str]] = []
    pending_starts: dict[int, dict] = {}  # iter -> start event

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            op = ev.get("op", "")
            if op == "agent.run.done":
                run_done = ev
            elif op == "agent.iter.start":
                pending_starts[ev.get("iter", -1)] = ev
            elif op == "agent.iter.done":
                it = ev.get("iter", -1)
                result = ev.get("result", "")
                start = pending_starts.get(it, {})
                focus = start.get("focus", "")
                # Split merge focus "tag_a + tag_b"
                if " + " in focus:
                    tags = [t.strip() for t in focus.split(" + ")]
                else:
                    tags = [focus] if focus else []
                for tag in tags:
                    if tag:
                        tag_iters.append((tag, result))

    if run_done is None:
        return None

    return {
        "run_id": path.name,
        "status": run_done.get("status", ""),
        "processed": run_done.get("processed", 0),
        "applied": run_done.get("applied", 0),
        "blocked_empty": run_done.get("blocked_empty", 0),
        "rolled_back": run_done.get("rolled_back", 0),
        "apply_error": run_done.get("apply_error", 0),
        "propose_error": run_done.get("propose_error", 0),
        "unknown_action": run_done.get("unknown_action", 0),
        "failure_rate": run_done.get("failure_rate", 0.0),
        "iters": run_done.get("iters", 0),
        "queue_remaining": run_done.get("queue_remaining", 0),
        "tag_iters": tag_iters,
    }


def load_runs(runs_dir: Path) -> list[dict]:
    """Load all runs from dir, sorted lexicographically by filename."""
    files = sorted(runs_dir.glob("*.jsonl"))
    runs = []
    for f in files:
        r = _load_run(f)
        if r is not None:
            runs.append(r)
    return runs


# ── Display ───────────────────────────────────────────────────────────────────

_STATUS_WIDTH = 22
_RUN_WIDTH = 52


def _no_op_rate(run: dict) -> float:
    processed = run["processed"]
    if processed == 0:
        return 0.0
    non_applied = processed - run["applied"]
    return non_applied / processed


def _print_run_table(runs: list[dict], last_n: int) -> None:
    recent = runs[-last_n:] if len(runs) > last_n else runs
    header = (
        f"{'Run':<{_RUN_WIDTH}}"
        f"{'Status':<{_STATUS_WIDTH}}"
        f"{'Processed':>10}"
        f"{'Applied':>8}"
        f"{'Fail%':>8}"
        f"{'No-op%':>8}"
    )
    print(f"=== Last {last_n} runs ===")
    print(header)
    for r in recent:
        noop = _no_op_rate(r)
        fail_pct = r["failure_rate"] * 100
        flags = ""
        if r["status"] == "degraded":
            flags += " ⚠ degraded"
        elif noop >= 1.0 and r["processed"] > 0:
            flags += " ⚠ no-op"
        print(
            f"{r['run_id']:<{_RUN_WIDTH}}"
            f"{r['status']:<{_STATUS_WIDTH}}"
            f"{r['processed']:>10}"
            f"{r['applied']:>8}"
            f"{fail_pct:>7.1f}%"
            f"{noop * 100:>7.1f}%"
            f"{flags}"
        )
    print()


def _print_streak_table(runs: list[dict], threshold: int) -> None:
    print("=== Per-tag failure streaks (consecutive non-applied across all runs) ===")
    streaks = compute_tag_streaks(runs, threshold)
    above = [s for s in streaks if s["streak"] >= threshold]
    below = [s for s in streaks if s["streak"] < threshold and s["streak"] > 0]
    display = above + below

    if not display:
        print("No persistent failures detected ✓")
    else:
        header = f"{'Tag':<26}{'Streak':>8}   {'Last 5 results'}"
        print(header)
        for s in display:
            last = ", ".join(s["last_results"])
            flag = "  ⚠" if s["streak"] >= threshold else ""
            print(f"{s['tag']:<26}{s['streak']:>8}   {last}{flag}")
    print()


def _print_summary(runs: list[dict], threshold: int) -> None:
    print("=== Summary ===")
    total = len(runs)
    degraded = sum(1 for r in runs if r["status"] == "degraded")
    fatal = sum(1 for r in runs if r["status"] == "fatal_error")
    noop_rates = [_no_op_rate(r) for r in runs]
    mean_noop = sum(noop_rates) / total if total else 0.0
    streaks = compute_tag_streaks(runs, threshold)
    n_streaks = sum(1 for s in streaks if s["streak"] >= threshold)
    print(f"Total runs analyzed: {total}")
    print(f"Runs with status=degraded: {degraded}")
    print(f"Runs with status=fatal_error: {fatal}")
    print(f"Mean no-op rate: {mean_noop * 100:.1f}%")
    print(f"Tags with streak >= {threshold}: {n_streaks}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze vocab maintenance run health.")
    parser.add_argument("--runs-dir", default="outputs/runs", type=Path)
    parser.add_argument("--last", default=10, type=int)
    parser.add_argument("--streak-threshold", default=3, type=int)
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    if not runs_dir.exists():
        print(f"No runs found in {runs_dir}")
        sys.exit(0)

    runs = load_runs(runs_dir)
    if not runs:
        print(f"No runs found in {runs_dir}")
        sys.exit(0)

    _print_run_table(runs, args.last)
    _print_streak_table(runs, args.streak_threshold)
    _print_summary(runs, args.streak_threshold)


if __name__ == "__main__":
    main()
