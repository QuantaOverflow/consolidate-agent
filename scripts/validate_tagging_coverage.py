#!/usr/bin/env python3
"""Validate faceted assignments file against Stage 1 Dimension ② Tagging Coverage.

Usage:
    python scripts/validate_tagging_coverage.py [assignments_file]

Default assignments file: outputs/network_v2.json

Exit codes:
    0 = PASS or file-not-found (Gate 2 expected state)
    1 = WARN only
    2 = any FAIL
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_ASSIGNMENTS_PATH = Path("outputs/network_v2.json")

# ── Thresholds ───────────────────────────────────────────────────────────────

MATTER_PASS_RATE = 0.98
MATTER_WARN_RATE = 0.90

ACTIVITY_PASS_RATE = 0.95
ACTIVITY_WARN_RATE = 0.85

PATTERN_PASS_MIN = 0.60
PATTERN_PASS_MAX = 0.85
PATTERN_WARN_MIN = 0.40
PATTERN_WARN_MAX = 0.90  # >90% is also WARN

LESSON_TYPE_PASS_RATE = 1.00

MEAN_TAGS_PASS_MIN = 3.0
MEAN_TAGS_PASS_MAX = 4.5
MEAN_TAGS_WARN_MIN = 2.5
MEAN_TAGS_WARN_MAX = 5.5

ORPHAN_PASS_RATE = 0.01   # <1%
ORPHAN_WARN_RATE = 0.03   # 1-3%

LONGTAIL_PASS = 2
LONGTAIL_WARN = 4

LONGTAIL_THRESHOLD = 2    # tags used by ≤ this many records → long-tail


def _rate_label(rate: float) -> str:
    return f"{rate*100:.1f}%"


def _check_rate(value: float, pass_thresh: float, warn_thresh: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= pass_thresh:
            return "PASS"
        if value >= warn_thresh:
            return "WARN"
        return "FAIL"
    else:
        # lower is better (e.g. orphan rate)
        if value < pass_thresh:
            return "PASS"
        if value < warn_thresh:
            return "WARN"
        return "FAIL"


def validate(assignments_path: Path) -> int:
    raw = json.loads(assignments_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "assignments" in raw:
        assignments: list[dict] = raw["assignments"]
    else:
        assignments = raw  # raw list form
    n = len(assignments)
    if n == 0:
        print("No assignments found. Cannot validate.", file=sys.stderr)
        return 2

    results: list[str] = []

    print("=== Tagging Coverage ===\n")
    print(f"Records: {n}\n")

    # ── matter_tag coverage ──────────────────────────────────────────────────
    matter_covered = sum(1 for a in assignments if len(a.get("matter_tags") or []) >= 1)
    matter_rate = matter_covered / n
    matter_status = _check_rate(matter_rate, MATTER_PASS_RATE, MATTER_WARN_RATE)
    icon = "✓" if matter_status == "PASS" else ("⚠" if matter_status == "WARN" else "✗")
    print(f"  Records with ≥1 matter_tag    : {_rate_label(matter_rate)}  {icon} {matter_status} (≥98% PASS)")
    results.append(matter_status)

    # ── activity_tag coverage ────────────────────────────────────────────────
    activity_covered = sum(1 for a in assignments if a.get("activity_tag"))
    activity_rate = activity_covered / n
    activity_status = _check_rate(activity_rate, ACTIVITY_PASS_RATE, ACTIVITY_WARN_RATE)
    icon = "✓" if activity_status == "PASS" else ("⚠" if activity_status == "WARN" else "✗")
    print(f"  Records with activity_tag      : {_rate_label(activity_rate)}  {icon} {activity_status} (≥95% PASS)")
    results.append(activity_status)

    # ── pattern_tag coverage (range-based) ───────────────────────────────────
    pattern_covered = sum(1 for a in assignments if len(a.get("pattern_tags") or []) >= 1)
    pattern_rate = pattern_covered / n
    if PATTERN_PASS_MIN <= pattern_rate <= PATTERN_PASS_MAX:
        pattern_status = "PASS"
    elif pattern_rate < PATTERN_WARN_MIN or pattern_rate > PATTERN_WARN_MAX:
        pattern_status = "FAIL"
    else:
        pattern_status = "WARN"
    icon = "✓" if pattern_status == "PASS" else ("⚠" if pattern_status == "WARN" else "✗")
    print(f"  Records with ≥1 pattern_tag    : {_rate_label(pattern_rate)}  {icon} {pattern_status} (60-85% PASS)")
    results.append(pattern_status)

    # ── lesson_type coverage ─────────────────────────────────────────────────
    lt_covered = sum(1 for a in assignments if a.get("lesson_type"))
    lt_rate = lt_covered / n
    lt_status = "PASS" if lt_rate >= LESSON_TYPE_PASS_RATE else "FAIL"
    icon = "✓" if lt_status == "PASS" else "✗"
    print(f"  Records with lesson_type       : {_rate_label(lt_rate)}  {icon} {lt_status} (100% required)")
    results.append(lt_status)

    # ── Mean tags/record (matter + activity + pattern) ───────────────────────
    tag_counts = []
    for a in assignments:
        m = len(a.get("matter_tags") or [])
        act = 1 if a.get("activity_tag") else 0
        p = len(a.get("pattern_tags") or [])
        tag_counts.append(m + act + p)
    mean_tags = sum(tag_counts) / n
    if MEAN_TAGS_PASS_MIN <= mean_tags <= MEAN_TAGS_PASS_MAX:
        mean_status = "PASS"
    elif mean_tags < MEAN_TAGS_WARN_MIN or mean_tags > MEAN_TAGS_WARN_MAX:
        mean_status = "FAIL"
    else:
        mean_status = "WARN"
    icon = "✓" if mean_status == "PASS" else ("⚠" if mean_status == "WARN" else "✗")
    print(f"  Mean tags/record               : {mean_tags:.2f}  {icon} {mean_status} (3.0-4.5 PASS)")
    results.append(mean_status)

    # ── Orphan rate ──────────────────────────────────────────────────────────
    orphan_count = sum(
        1 for a in assignments
        if not (a.get("matter_tags") or a.get("activity_tag") or a.get("pattern_tags"))
    )
    orphan_rate = orphan_count / n
    orphan_status = _check_rate(orphan_rate, ORPHAN_PASS_RATE, ORPHAN_WARN_RATE, higher_is_better=False)
    icon = "✓" if orphan_status == "PASS" else ("⚠" if orphan_status == "WARN" else "✗")
    print(f"  Orphan rate (no tags at all)   : {_rate_label(orphan_rate)}  {icon} {orphan_status} (<1% PASS)")
    results.append(orphan_status)

    # ── Long-tail tags per facet ─────────────────────────────────────────────
    print("\nLong-tail tags (≤2 records assigned) per facet:")

    def _longtail(facet_key: str, single_tag: bool = False) -> str:
        usage: Counter[str] = Counter()
        for a in assignments:
            if single_tag:
                tag = a.get(facet_key)
                if tag:
                    name = tag.get("name") if isinstance(tag, dict) else str(tag)
                    if name:
                        usage[name] += 1
            else:
                for t in (a.get(facet_key) or []):
                    name = t.get("name") if isinstance(t, dict) else str(t)
                    if name:
                        usage[name] += 1
        longtail_tags = [name for name, cnt in usage.items() if cnt <= LONGTAIL_THRESHOLD]
        count = len(longtail_tags)
        if count <= LONGTAIL_PASS:
            status = "PASS"
            icon = "✓"
        elif count <= LONGTAIL_WARN:
            status = "WARN"
            icon = "⚠"
        else:
            status = "FAIL"
            icon = "✗"
        label = facet_key.replace("_tags", "").replace("_tag", "")
        print(f"  {label:<10}: {count} long-tail tags  {icon} {status}  (≤2 PASS)")
        if longtail_tags:
            print(f"             {longtail_tags}")
        return status

    results.append(_longtail("matter_tags"))
    results.append(_longtail("activity_tag", single_tag=True))
    results.append(_longtail("pattern_tags"))

    # ── Overall ──────────────────────────────────────────────────────────────
    if "FAIL" in results:
        overall = "FAIL"
        exit_code = 2
    elif "WARN" in results:
        overall = "WARN"
        exit_code = 1
    else:
        overall = "PASS"
        exit_code = 0

    print(f"\n=== Overall: {overall} ===")
    return exit_code


def main() -> int:
    if len(sys.argv) >= 2:
        assignments_path = Path(sys.argv[1])
    else:
        assignments_path = DEFAULT_ASSIGNMENTS_PATH

    if not assignments_path.exists():
        print(f"expected at {assignments_path} (run Gate 3 first)")
        return 0

    return validate(assignments_path)


if __name__ == "__main__":
    sys.exit(main())
