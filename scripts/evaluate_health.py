"""Compute a 0-1 vocab health score from diagnostics + assignments.

Pure arithmetic — no LLM. Used by agent to decide whether maintain
is needed and when to stop.

Score = weighted sum of 6 sub-metrics:
  hit_rate              0.30  records that got at least one tag
  usage_balance         0.20  tag size distribution (lower CV → better)
  multi_label_rate      0.15  records with >=2 tags (info preservation)
  boundary_clarity      0.15  1 - boundary_blur_rate
  no_unused             0.10  1 - unused_rate
  low_missing           0.10  1 - missing_rate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


WEIGHTS = {
    "hit_rate":         0.30,
    "usage_balance":    0.20,
    "multi_label_rate": 0.15,
    "boundary_clarity": 0.15,
    "no_unused":        0.10,
    "low_missing":      0.10,
}


def compute_usage_balance(tag_usage: dict[str, int]) -> float:
    """1 - coefficient_of_variation, clamped to [0, 1].

    cv=0 (perfectly uniform) → 1.0
    cv=1 (std == mean)        → 0.5
    cv>=2                     → 0.0
    """
    usages = list(tag_usage.values())
    if not usages:
        return 0.0
    mean = sum(usages) / len(usages)
    if mean <= 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in usages) / len(usages)
    std = math.sqrt(variance)
    cv = std / mean
    return max(0.0, min(1.0, 1.0 - cv / 2))


def compute_multi_label_rate(assignments: list[dict]) -> float:
    """Fraction of assigned records that have >=2 selected tags."""
    assigned = [a for a in assignments if not a.get("missing") and a.get("selected_tags")]
    if not assigned:
        return 0.0
    multi = sum(1 for a in assigned if len(a["selected_tags"]) >= 2)
    return multi / len(assigned)


def evaluate(diagnostics: dict, vocab_size: int, assignments: list[dict]) -> dict:
    sample_size = diagnostics["sample_size"]
    total_assigned = diagnostics["total_assigned"]
    total_missing = diagnostics["total_missing"]
    unused_tags = diagnostics["unused_tags"]
    boundary_blur = diagnostics["boundary_blur_records"]
    tag_usage = diagnostics["tag_usage_count"]

    hit_rate = total_assigned / sample_size if sample_size else 0.0
    missing_rate = total_missing / sample_size if sample_size else 1.0
    unused_rate = len(unused_tags) / vocab_size if vocab_size else 1.0
    boundary_blur_rate = (
        len(boundary_blur) / sample_size if sample_size else 0.0
    )

    metrics = {
        "hit_rate": hit_rate,
        "usage_balance": compute_usage_balance(tag_usage),
        "multi_label_rate": compute_multi_label_rate(assignments),
        "boundary_clarity": 1.0 - boundary_blur_rate,
        "no_unused": 1.0 - unused_rate,
        "low_missing": 1.0 - missing_rate,
    }

    score = sum(WEIGHTS[k] * metrics[k] for k in WEIGHTS)

    return {
        "score": round(score, 4),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "weights": WEIGHTS,
        "raw": {
            "vocab_size": vocab_size,
            "sample_size": sample_size,
            "assigned": total_assigned,
            "missing": total_missing,
            "unused_count": len(unused_tags),
            "blur_count": len(boundary_blur),
        },
    }


def run(diagnostics_path: Path, assignments_path: Path, vocab_path: Path) -> None:
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))["vocab"]
    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))

    result = evaluate(diagnostics, len(vocab), assignments)

    print(f"\n=== Vocab Health Report ===")
    print(f"  vocab:       {vocab_path.name}  ({result['raw']['vocab_size']} tags)")
    print(f"  diagnostics: {diagnostics_path.name}  ({result['raw']['sample_size']} records)")
    print(f"\n  📊 SCORE: {result['score']}")
    print(f"\n  Breakdown:")
    for metric, value in result["metrics"].items():
        bar_len = int(value * 30)
        bar = "█" * bar_len + "·" * (30 - bar_len)
        print(f"    {metric:20s} {bar} {value:.4f}  (weight {WEIGHTS[metric]})")
    print(f"\n  Raw counts:")
    for k, v in result["raw"].items():
        print(f"    {k:20s} {v}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    args = parser.parse_args()

    run(diagnostics_path=args.diagnostics, assignments_path=args.assignments, vocab_path=args.vocab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
