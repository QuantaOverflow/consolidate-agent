#!/usr/bin/env python3
"""Validate vocab_v2.json against Stage 1 Dimension ① Vocab Structure Compliance.

Usage:
    python scripts/validate_vocab_structure.py docs/plans/vocab_v2.json

Exit codes:
    0 = PASS
    1 = WARN only
    2 = any FAIL
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Facet size thresholds ────────────────────────────────────────────────────

MATTER_PASS = (12, 18)
MATTER_WARN = (8, 22)   # 8-11 or 19-22 → outside PASS but inside WARN

ACTIVITY_PASS = (6, 10)
ACTIVITY_WARN = (4, 12)

PATTERN_PASS = (5, 9)
PATTERN_WARN = (3, 12)

LESSON_TYPE_EXPECTED = 3

# Definition name-keyword anchoring thresholds
ANCHOR_PASS_RATE = 0.90
ANCHOR_WARN_RATE = 0.70

# Definition word count mean thresholds
DEFLEN_PASS_MIN = 15
DEFLEN_PASS_MAX = 40
DEFLEN_FAIL_MIN = 10  # < 10 → FAIL
DEFLEN_FAIL_MAX = 60  # > 60 → FAIL


def _check_facet_size(name: str, count: int, pass_range: tuple, warn_range: tuple) -> str:
    lo_pass, hi_pass = pass_range
    lo_warn, hi_warn = warn_range
    if lo_pass <= count <= hi_pass:
        return "PASS"
    if lo_warn <= count <= hi_warn:
        return "WARN"
    return "FAIL"


def _contains_name_keyword(definition: str, tag_name: str) -> bool:
    """Return True if definition contains at least one word from the tag name (split on _)."""
    words = tag_name.lower().split("_")
    def_lower = definition.lower()
    return any(w in def_lower for w in words if len(w) > 1)


def _facet_size_label(pass_range: tuple, warn_range: tuple) -> str:
    lp, hp = pass_range
    lw, hw = warn_range
    return f"{lp}-{hp} PASS / {lw}-{lp-1} or {hp+1}-{hw} WARN"


def validate(vocab_path: Path) -> int:
    data = json.loads(vocab_path.read_text(encoding="utf-8"))

    facets = data.get("facets", {})
    matter_tags: list[dict] = facets.get("matter", {}).get("tags", [])
    activity_tags: list[dict] = facets.get("activity", {}).get("tags", [])
    pattern_tags: list[dict] = facets.get("pattern", {}).get("tags", [])
    lesson_type_values: list[dict] = data.get("lesson_type", {}).get("values", [])

    all_tags: list[dict] = matter_tags + activity_tags + pattern_tags

    results: list[str] = []  # "PASS", "WARN", "FAIL"

    print("=== Vocab Structure Compliance ===\n")

    # ── Facet sizes ──────────────────────────────────────────────────────────
    print("Facet sizes:")

    def _print_facet(label: str, count: int, pass_range: tuple, warn_range: tuple) -> str:
        status = _check_facet_size(label, count, pass_range, warn_range)
        icon = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
        lp, hp = pass_range
        lw, hw = warn_range
        hint = f"{lp}-{hp}"
        print(f"  {label:<10}: {count} tags  {icon} {status} ({hint})")
        return status

    results.append(_print_facet("Matter", len(matter_tags), MATTER_PASS, MATTER_WARN))
    results.append(_print_facet("Activity", len(activity_tags), ACTIVITY_PASS, ACTIVITY_WARN))
    results.append(_print_facet("Pattern", len(pattern_tags), PATTERN_PASS, PATTERN_WARN))

    # ── lesson_type enum ─────────────────────────────────────────────────────
    lt_count = len(lesson_type_values)
    lt_status = "PASS" if lt_count == LESSON_TYPE_EXPECTED else "FAIL"
    lt_icon = "✓" if lt_status == "PASS" else "✗"
    print(f"\nlesson_type enum: {lt_count} values  {lt_icon} {lt_status}")
    results.append(lt_status)

    # ── Cross-facet name uniqueness ──────────────────────────────────────────
    names = [t["name"] for t in all_tags]
    name_set = set(names)
    dup_status = "PASS" if len(names) == len(name_set) else "FAIL"
    dup_icon = "✓" if dup_status == "PASS" else "✗"
    if dup_status == "PASS":
        print(f"\nCross-facet uniqueness: {dup_icon} PASS (no duplicate names)")
    else:
        dupes = [n for n in name_set if names.count(n) > 1]
        print(f"\nCross-facet uniqueness: {dup_icon} FAIL (duplicates: {dupes})")
    results.append(dup_status)

    # ── Name-keyword anchoring ───────────────────────────────────────────────
    print("\nName-keyword anchoring:")

    def _check_anchoring(label: str, tags: list[dict]) -> str:
        if not tags:
            print(f"  ✓ 0/0 {label} tags (empty facet)")
            return "PASS"
        missing = [t["name"] for t in tags if not _contains_name_keyword(t.get("definition", ""), t["name"])]
        hit_count = len(tags) - len(missing)
        rate = hit_count / len(tags)
        if rate >= ANCHOR_PASS_RATE:
            status = "PASS"
            icon = "✓"
        elif rate >= ANCHOR_WARN_RATE:
            status = "WARN"
            icon = "⚠"
        else:
            status = "FAIL"
            icon = "✗"
        if missing:
            print(f"  {icon} {hit_count}/{len(tags)} {label} tags include name keyword in definition")
            for m in missing:
                print(f"      missing keyword: {m}")
        else:
            print(f"  {icon} {hit_count}/{len(tags)} {label} tags include name keyword in definition")
        return status

    results.append(_check_anchoring("Matter", matter_tags))
    results.append(_check_anchoring("Activity", activity_tags))
    results.append(_check_anchoring("Pattern", pattern_tags))

    # ── Definition word count mean ───────────────────────────────────────────
    if all_tags:
        word_counts = [len(t.get("definition", "").split()) for t in all_tags]
        mean_words = sum(word_counts) / len(word_counts)
    else:
        mean_words = 0.0

    if DEFLEN_PASS_MIN <= mean_words <= DEFLEN_PASS_MAX:
        deflen_status = "PASS"
        deflen_icon = "✓"
    elif mean_words < DEFLEN_FAIL_MIN or mean_words > DEFLEN_FAIL_MAX:
        deflen_status = "FAIL"
        deflen_icon = "✗"
    else:
        deflen_status = "WARN"
        deflen_icon = "⚠"
    print(f"\nDefinition length: mean={mean_words:.0f} ({DEFLEN_PASS_MIN}-{DEFLEN_PASS_MAX}) {deflen_icon} {deflen_status}")
    results.append(deflen_status)

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
    if len(sys.argv) < 2:
        print("Usage: validate_vocab_structure.py <vocab_v2.json>", file=sys.stderr)
        return 2
    vocab_path = Path(sys.argv[1])
    if not vocab_path.exists():
        print(f"File not found: {vocab_path}", file=sys.stderr)
        return 2
    return validate(vocab_path)


if __name__ == "__main__":
    sys.exit(main())
