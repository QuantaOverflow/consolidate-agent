#!/usr/bin/env python3
"""Post-process: filter out vocab-external (hallucinated) tags from network_v2.json.

LLM occasionally emits tag names that don't exist in the v2 vocab. This script:
  1. Removes each tag entry whose name isn't in its facet's vocab
  2. Logs all removals (for vocab gap analysis)
  3. Writes the cleaned file (default in-place, with --backup creating .bak)
  4. Reports stats per-facet

Hallucinated tags are categorized:
  - cross_facet  : the name exists in a DIFFERENT facet (e.g., 'documenting' in matter_tags)
  - novel        : the name doesn't exist in any facet (real vocab gap signal)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=Path("outputs/network_v2.json"), type=Path)
    parser.add_argument("--out", default=None, type=Path, help="default: overwrite --in")
    parser.add_argument("--backup", action="store_true", help="create .bak before overwrite")
    parser.add_argument("--log", default=Path("outputs/network_v2_hallucinations.json"), type=Path)
    args = parser.parse_args()

    if not args.in_path.exists():
        sys.exit(f"{args.in_path} not found")

    data = json.loads(args.in_path.read_text(encoding="utf-8"))
    vocab = data["vocab"]
    facets = vocab.get("facets", {})
    matter_set = {t["name"] for t in facets.get("matter", {}).get("tags", [])}
    activity_set = {t["name"] for t in facets.get("activity", {}).get("tags", [])}
    pattern_set = {t["name"] for t in facets.get("pattern", {}).get("tags", [])}
    all_vocab = matter_set | activity_set | pattern_set

    removed: dict[str, list[dict]] = defaultdict(list)
    cross_facet = Counter()
    novel = Counter()

    for a in data["assignments"]:
        # Matter
        new_matter = []
        for t in a.get("matter_tags") or []:
            name = t.get("name")
            if name in matter_set:
                new_matter.append(t)
            else:
                cat = "cross_facet" if name in all_vocab else "novel"
                removed["matter"].append({"record_id": a["record_id"], "name": name, "category": cat})
                (cross_facet if cat == "cross_facet" else novel)[("matter", name)] += 1
        a["matter_tags"] = new_matter

        # Activity (single tag)
        at = a.get("activity_tag")
        if isinstance(at, dict):
            name = at.get("name")
            if name not in activity_set:
                cat = "cross_facet" if name in all_vocab else "novel"
                removed["activity"].append({"record_id": a["record_id"], "name": name, "category": cat})
                (cross_facet if cat == "cross_facet" else novel)[("activity", name)] += 1
                a["activity_tag"] = None  # mark missing for downstream

        # Pattern
        new_pattern = []
        for t in a.get("pattern_tags") or []:
            name = t.get("name")
            if name in pattern_set:
                new_pattern.append(t)
            else:
                cat = "cross_facet" if name in all_vocab else "novel"
                removed["pattern"].append({"record_id": a["record_id"], "name": name, "category": cat})
                (cross_facet if cat == "cross_facet" else novel)[("pattern", name)] += 1
        a["pattern_tags"] = new_pattern

    total_removed = sum(len(v) for v in removed.values())

    print("=== Hallucination filter ===")
    print(f"Total tag removals: {total_removed}")
    print(f"  matter:   {len(removed['matter'])}")
    print(f"  activity: {len(removed['activity'])}")
    print(f"  pattern:  {len(removed['pattern'])}")
    print(f"  cross_facet: {sum(cross_facet.values())} (vocab tag used in wrong facet)")
    print(f"  novel:       {sum(novel.values())} (concept not in any facet — gap signal)")

    # Recount facet coverage after cleanup
    n = len(data["assignments"])
    matter_covered = sum(1 for a in data["assignments"] if a.get("matter_tags"))
    activity_covered = sum(1 for a in data["assignments"] if a.get("activity_tag"))
    pattern_covered = sum(1 for a in data["assignments"] if a.get("pattern_tags"))
    print(f"\nPost-clean coverage:")
    print(f"  matter   ≥1 tag : {matter_covered}/{n} ({matter_covered/n*100:.1f}%)")
    print(f"  activity tag    : {activity_covered}/{n} ({activity_covered/n*100:.1f}%)")
    print(f"  pattern  ≥1 tag : {pattern_covered}/{n} ({pattern_covered/n*100:.1f}%)")

    out_path = args.out or args.in_path
    if args.backup and out_path == args.in_path:
        bak = args.in_path.with_suffix(args.in_path.suffix + ".bak")
        shutil.copy2(args.in_path, bak)
        print(f"\nbackup: {bak}")
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote cleaned: {out_path}")

    # Hallucination log
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        "total_removed": total_removed,
        "by_facet": {k: len(v) for k, v in removed.items()},
        "cross_facet_top": [{"facet": f, "name": n, "count": c} for (f, n), c in cross_facet.most_common()],
        "novel_top": [{"facet": f, "name": n, "count": c} for (f, n), c in novel.most_common()],
        "details": dict(removed),
    }
    args.log.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote hallucination log: {args.log}")


if __name__ == "__main__":
    main()
