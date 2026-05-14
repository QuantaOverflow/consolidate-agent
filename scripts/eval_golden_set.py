#!/usr/bin/env python3
"""Gate 4: evaluate LLM tagging against the 80-record golden set.

For v3 (Matter + lesson_type only), compares:
  - matter_tags exact / partial match vs golden
  - lesson_type exact match vs golden

Activity / pattern fields in golden are ignored (v3 dropped those facets per ADR-0004).
`acceptable_*_alternatives` in golden allows LLM to pick any listed alt without penalty.

Thresholds from docs/plans/2026-05-stage1-acceptance.md dimension ④.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


GOLDEN = Path("tests/fixtures/golden_80.jsonl")


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def load_assignments(network_path: Path) -> dict[str, dict]:
    data = json.loads(network_path.read_text())
    return {a["record_id"]: a for a in data["assignments"]}


def matter_match(expected_set: set[str], actual_set: set[str], alts: list[list[str]] | None) -> tuple[bool, bool]:
    """Returns (exact_match, partial_overlap).

    If alts (acceptable_matter_alternatives) given, exact_match passes if actual
    equals expected OR any alt.
    """
    if actual_set == expected_set:
        return True, True
    if alts:
        for alt in alts:
            if actual_set == set(alt):
                return True, True
    partial = bool(expected_set & actual_set)
    return False, partial


def lesson_match(expected: str, actual: str | None, alts: list[str] | None) -> bool:
    if actual == expected:
        return True
    if alts and actual in alts:
        return True
    return False


def evaluate(network_path: Path) -> int:
    golden = load_golden()
    assigns = load_assignments(network_path)

    # Per-difficulty stats
    stats = defaultdict(lambda: {
        "n": 0,
        "matter_exact": 0,
        "matter_partial": 0,
        "lesson_exact": 0,
        "matter_missing": [],  # golden tags LLM should have but didn't
        "lesson_disagreements": [],  # (rid, expected, actual)
    })
    missing_records = []

    for g in golden:
        rid = g["record_id"]
        diff = g["difficulty"]
        exp = g["expected"]
        a = assigns.get(rid)
        if a is None:
            missing_records.append(rid)
            continue

        stats[diff]["n"] += 1

        # Matter
        exp_matter = set(exp["matter_tags"])
        act_matter = {t["name"] for t in (a.get("matter_tags") or [])}
        alts = g.get("acceptable_matter_alternatives")
        exact, partial = matter_match(exp_matter, act_matter, alts)
        if exact:
            stats[diff]["matter_exact"] += 1
        if partial:
            stats[diff]["matter_partial"] += 1
        else:
            stats[diff]["matter_missing"].append({
                "rid": rid,
                "expected": sorted(exp_matter),
                "actual": sorted(act_matter),
            })

        # lesson_type
        exp_lesson = exp["lesson_type"]
        act_lesson = a.get("lesson_type")
        lesson_alts = g.get("acceptable_lesson_type_alternatives")
        if lesson_match(exp_lesson, act_lesson, lesson_alts):
            stats[diff]["lesson_exact"] += 1
        else:
            stats[diff]["lesson_disagreements"].append({
                "rid": rid,
                "expected": exp_lesson,
                "actual": act_lesson,
            })

    # Print report
    print("=" * 70)
    print(f"GOLDEN SET EVALUATION ({network_path})")
    print("=" * 70)
    if missing_records:
        print(f"\n⚠ {len(missing_records)} golden record_ids not found in network output")

    # Thresholds per acceptance plan dimension ④
    thresholds = {
        "easy":   {"matter_exact": 0.65, "matter_partial": 0.88, "lesson": 0.82},
        "medium": {"matter_exact": 0.45, "matter_partial": 0.72, "lesson": 0.82},
        "hard":   {"matter_exact": 0.25, "matter_partial": 0.45, "lesson": 0.82},
    }
    warn_drop = 0.10  # WARN if hits PASS - 10%, FAIL if PASS - 20%

    def verdict(actual: float, pass_thresh: float) -> str:
        if actual >= pass_thresh:
            return "✓ PASS"
        if actual >= pass_thresh - warn_drop:
            return "⚠ WARN"
        return "✗ FAIL"

    overall_pass = True
    total_n = 0
    total_matter_exact = 0
    total_matter_partial = 0
    total_lesson_exact = 0

    for diff in ["easy", "medium", "hard"]:
        s = stats[diff]
        n = s["n"]
        if n == 0:
            continue
        total_n += n
        total_matter_exact += s["matter_exact"]
        total_matter_partial += s["matter_partial"]
        total_lesson_exact += s["lesson_exact"]
        mx = s["matter_exact"] / n
        mp = s["matter_partial"] / n
        lx = s["lesson_exact"] / n
        thresh = thresholds[diff]
        print(f"\n── {diff.upper()} ({n} records) ──")
        v1 = verdict(mx, thresh["matter_exact"])
        v2 = verdict(mp, thresh["matter_partial"])
        v3 = verdict(lx, thresh["lesson"])
        print(f"  Matter exact match    : {mx*100:5.1f}%  {v1}  (PASS ≥{thresh['matter_exact']*100:.0f}%)")
        print(f"  Matter partial overlap: {mp*100:5.1f}%  {v2}  (PASS ≥{thresh['matter_partial']*100:.0f}%)")
        print(f"  lesson_type exact     : {lx*100:5.1f}%  {v3}  (PASS ≥{thresh['lesson']*100:.0f}%)")
        if "✗" in v1 + v2 + v3:
            overall_pass = False

    # Overall lesson_type (cross-difficulty, single threshold 82%)
    if total_n:
        lx_overall = total_lesson_exact / total_n
        print(f"\n── OVERALL ({total_n} records) ──")
        print(f"  Matter exact (all)    : {total_matter_exact/total_n*100:5.1f}%")
        print(f"  Matter partial (all)  : {total_matter_partial/total_n*100:5.1f}%")
        print(f"  lesson_type exact (all): {lx_overall*100:5.1f}%  (PASS ≥82%)")

    # Top 10 disagreement examples per difficulty
    print("\n── Top mismatches ──")
    for diff in ["easy", "medium", "hard"]:
        misses = stats[diff]["matter_missing"][:5]
        if misses:
            print(f"\n  Matter disagreements [{diff}]:")
            for m in misses:
                print(f"    {m['rid']}: expected={m['expected']} actual={m['actual']}")
    print()
    for diff in ["easy", "medium", "hard"]:
        ld = stats[diff]["lesson_disagreements"][:5]
        if ld:
            print(f"  Lesson disagreements [{diff}]:")
            for d in ld:
                print(f"    {d['rid']}: expected={d['expected']} actual={d['actual']}")

    print()
    print("=" * 70)
    print(f"Overall: {'PASS' if overall_pass else 'WARN or FAIL'}")
    return 0 if overall_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=Path("outputs/network_v3.json"), type=Path)
    args = parser.parse_args()
    if not args.in_path.exists():
        sys.exit(f"{args.in_path} not found")
    sys.exit(evaluate(args.in_path))


if __name__ == "__main__":
    main()
