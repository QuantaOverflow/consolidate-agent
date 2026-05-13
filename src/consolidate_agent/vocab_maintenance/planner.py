"""Plan-and-execute planner — scan signals once, emit a prioritized work queue.

Phase E replaces the per-iter LLM diagnose+decide step with a single
rule-based planning pass. The planner walks the standard signal probes
(`compute_fit_signals`, `find_similar_pairs`, diagnostics `unused_tags`)
and turns each surface signal into a WorkItem. The agent loop then pops
items in priority order and executes them — no LLM judgment between
iters, no convergence threshold.

Termination conditions (no soft "we agree" gate):
  - work queue exhausted -> FinalStatus.COMPLETED
  - iter >= max_iter      -> FinalStatus.MAX_ITER_EXHAUSTED
  - apply / propose raises a fatal error -> FinalStatus.FATAL_ERROR
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .probes import compute_fit_signals
from .similarity import find_similar_pairs


# Priority bands. Lower = higher priority. Within a band: severity ascending.
# Bands are spaced by 1000 so per-item severity (0-999) cannot bleed across.
_PRI_REFINE_BASE = 1000
_PRI_MERGE_BASE = 2000
_PRI_DEPRECATE_BASE = 3000

DEFAULT_FORCED_FIT_THRESHOLD = 0.5
DEFAULT_SIMILAR_PAIRS_THRESHOLD = 0.80


@dataclass(frozen=True)
class WorkItem:
    """One surfaced issue + the action that addresses it."""
    action: str             # "propose_refine" | "propose_merge" | "propose_deprecate"
    focus: str              # tag name (single tag for refine/deprecate, "a + b" for merge)
    reason: str             # 1-line justification recorded in audit log
    priority: int           # lower = higher priority

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "focus": self.focus,
            "reason": self.reason,
            "priority": self.priority,
        }


def build_work_queue(
    vocab: list[dict],
    assignments: list[dict],
    diagnostics: dict,
    db_path: Path,
    *,
    disabled_actions: set[str] | None = None,
    forced_fit_threshold: float = DEFAULT_FORCED_FIT_THRESHOLD,
    similar_pairs_threshold: float = DEFAULT_SIMILAR_PAIRS_THRESHOLD,
) -> list[WorkItem]:
    """Rules-based scan -> prioritized work queue. No LLM.

    Priority bands (lower = higher priority):
      10-19 propose_refine  — precision fix, low blast radius
      20-29 propose_merge   — consolidation, medium risk
      30-39 propose_deprecate — vocab shrink, highest per-item risk

    Within a band: sort by signal severity (worst signal first).
    """
    disabled = disabled_actions or set()
    queue: list[WorkItem] = []

    if "propose_refine" not in disabled:
        fit_signals = compute_fit_signals(
            vocab, assignments, db_path,
            low_fit_threshold=forced_fit_threshold,
        )
        for c in fit_signals.get("forced_fit_candidates", []):
            mean_fit = c["mean_fit"]
            queue.append(WorkItem(
                action="propose_refine",
                focus=c["tag"],
                reason=f"forced_fit mean_fit={mean_fit} sample={c['sample']}",
                # severity = how far below threshold; worst (lowest mean_fit) first
                priority=_PRI_REFINE_BASE + int(mean_fit * 1000),
            ))

    if "propose_merge" not in disabled:
        pairs = find_similar_pairs(
            vocab, threshold=similar_pairs_threshold, top_n=10,
        )
        for p in pairs:
            sim = p["similarity"]
            queue.append(WorkItem(
                action="propose_merge",
                focus=f"{p['tag_a']} + {p['tag_b']}",
                reason=f"def similarity {sim}",
                # higher similarity = higher priority (lower number)
                priority=_PRI_MERGE_BASE + int((1.0 - sim) * 1000),
            ))

    if "propose_deprecate" not in disabled:
        for tag_name in diagnostics.get("unused_tags", []) or []:
            queue.append(WorkItem(
                action="propose_deprecate",
                focus=tag_name,
                reason="zero usage in sample",
                priority=_PRI_DEPRECATE_BASE,
            ))

    queue.sort(key=lambda w: (w.priority, w.action, w.focus))
    return queue


def sanity_check_metrics(
    before: dict[str, float],
    after: dict[str, float],
    *,
    catastrophic_threshold: float = 0.10,
) -> tuple[str, str]:
    """Pure-function replacement for LLM-as-judge.

    Rollback only when a non-coverage dim drops by >= catastrophic_threshold.
    Coverage drops are expected for merge/deprecate/refine actions, so the
    coverage dim is intentionally excluded from the catastrophe check.

    Returns (verdict, reason). verdict in {"commit", "rollback"}.
    """
    if not before or not after:
        return ("commit", "no prior metrics — accepting first commit")
    monitored = ("coherence", "distinctness", "granularity", "multi_axis")
    worst_dim = None
    worst_delta = 0.0
    for dim in monitored:
        delta = after.get(dim, 0.0) - before.get(dim, 0.0)
        if delta < worst_delta:
            worst_delta = delta
            worst_dim = dim
    if worst_dim is not None and worst_delta <= -catastrophic_threshold:
        return (
            "rollback",
            f"catastrophic regression on {worst_dim}: {worst_delta:+.3f} "
            f"(threshold -{catastrophic_threshold})",
        )
    return (
        "commit",
        f"max drop {worst_delta:+.3f} on {worst_dim or 'none'} "
        f"(< -{catastrophic_threshold} catastrophe threshold)",
    )
