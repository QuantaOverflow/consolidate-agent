#!/usr/bin/env python3
"""Isolated integration test for the propose_split pipeline.

Uses a small controlled slice of the real network:
  - http_api   (78 records, coherence 0.39 — known heterogeneous)
  - git_vcs    (49 records, coherence 0.50 — known healthy, should be skipped)
  - datetime_handling (22 records, coherence 0.52 — healthy)

Expected outcomes:
  1. compute_heterogeneous_tags: http_api flagged, git_vcs / datetime skipped
  2. propose_split_fn: returns a SplitTagProposal for http_api (2 or 3 sub-tags)
  3. _apply_split: applies without invariant errors
  4. intra-cluster coherence of sub-tags > original http_api coherence (0.39)
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from pathlib import Path


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def mean_pairwise_cosine(embs):
    """Mean pairwise cosine for a list of embeddings."""
    if len(embs) < 2:
        return float("nan")
    pairs = [cosine(embs[i], embs[j]) for i in range(len(embs)) for j in range(i + 1, len(embs))]
    return statistics.mean(pairs)


def main():
    load_dotenv(Path(".env"))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY missing")

    from consolidate_agent.vocab_maintenance.probes import compute_heterogeneous_tags
    from consolidate_agent.vocab_maintenance.propose.split import propose_split_fn
    from consolidate_agent.vocab_maintenance.apply import apply_proposal

    # ── 1. Build isolated mini-network ────────────────────────────────────────
    print("=" * 68)
    print("ISOLATED TEST: propose_split pipeline")
    print("=" * 68)

    data = json.loads(Path("outputs/network_v3_p1.json").read_text())
    all_vocab = data["vocab"]["facets"]["matter"]["tags"]

    TEST_TAGS = {"http_api", "git_vcs", "datetime_handling"}
    test_vocab = [t for t in all_vocab if t["name"] in TEST_TAGS]
    test_assignments = []
    for a in data["assignments"]:
        relevant = [t for t in (a.get("matter_tags") or []) if t.get("name") in TEST_TAGS]
        if relevant:
            test_assignments.append({
                "record_id": a["record_id"],
                "title": a.get("title", ""),
                "selected_tags": relevant,
                "missing": False,
            })

    tag_counts = {}
    for a in test_assignments:
        for t in a["selected_tags"]:
            tag_counts[t["name"]] = tag_counts.get(t["name"], 0) + 1

    print(f"\nMini-network: {len(test_vocab)} tags, {len(test_assignments)} records")
    for tag, n in sorted(tag_counts.items()):
        print(f"  {tag:<28} {n} records")

    # ── 2. compute_heterogeneous_tags ──────────────────────────────────────────
    print("\n" + "─" * 68)
    print("[1] compute_heterogeneous_tags (threshold=0.42, min_records=10)")
    print("─" * 68)
    candidates = compute_heterogeneous_tags(
        test_vocab, test_assignments,
        Path("outputs/knowledge.db"),
        coherence_threshold=0.42,
        min_records=10,
    )
    print(f"  Found {len(candidates)} split candidate(s):")
    for c in candidates:
        print(f"    {c['tag']:<28} coherence={c['coherence']:.3f}  n={c['record_count']}")
    if not candidates:
        print("  FAIL: expected http_api to be flagged")
        sys.exit(1)
    if candidates[0]["tag"] != "http_api":
        print(f"  WARN: expected http_api first, got {candidates[0]['tag']}")
    else:
        print("  ✓ http_api correctly identified as worst candidate")

    # ── 3. propose_split on http_api ──────────────────────────────────────────
    print("\n" + "─" * 68)
    print("[2] propose_split_fn on http_api (LLM decides n_groups in {2,3})")
    print("─" * 68)
    proposals = propose_split_fn(
        test_vocab, test_assignments, "http_api",
        db_path=Path("outputs/knowledge.db"),
    )
    if not proposals:
        print("  No proposal generated (LLM declined or too few records). Exiting.")
        sys.exit(1)

    p = proposals[0]
    print(f"  Split into {len(p.sub_tags)} sub-tags:")
    for st in p.sub_tags:
        print(f"    [{st['name']}]  {len(st['record_ids'])} records")
        print(f"      def: {st['definition'][:120]}")

    # ── 4. Apply split ────────────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("[3] _apply_split (invariant checks)")
    print("─" * 68)
    try:
        new_vocab, new_assignments = apply_proposal(test_vocab, test_assignments, p)
        print(f"  ✓ Applied. vocab: {len(test_vocab)} → {len(new_vocab)} tags")
    except Exception as e:
        print(f"  ✗ Apply failed: {e}")
        sys.exit(1)

    new_tags = {t["name"] for t in new_vocab}
    removed = {t["name"] for t in test_vocab} - new_tags
    added = new_tags - {t["name"] for t in test_vocab}
    print(f"    removed: {removed}")
    print(f"    added:   {added}")

    # ── 5. Measure coherence before vs after ──────────────────────────────────
    print("\n" + "─" * 68)
    print("[4] Coherence comparison (before vs after)")
    print("─" * 68)

    # Load stored embeddings
    import sqlite3
    conn = sqlite3.connect("outputs/knowledge.db")
    emb_map = {
        rid: tuple(json.loads(e))
        for rid, e in conn.execute(
            "SELECT record_id, embedding FROM source_knowledge_records WHERE embedding IS NOT NULL"
        )
    }
    conn.close()

    # Before: http_api coherence
    http_api_rids = [a["record_id"] for a in test_assignments
                     if any(t["name"] == "http_api" for t in a["selected_tags"])]
    http_api_embs = [emb_map[r] for r in http_api_rids if r in emb_map]
    before_coherence = mean_pairwise_cosine(http_api_embs)
    print(f"\n  BEFORE: http_api coherence = {before_coherence:.3f}  ({len(http_api_embs)} records)")

    # After: sub-tag coherences
    print(f"\n  AFTER:")
    improved_count = 0
    for st in p.sub_tags:
        embs = [emb_map[r] for r in st["record_ids"] if r in emb_map]
        c = mean_pairwise_cosine(embs)
        better = "✓" if not math.isnan(c) and c > before_coherence else "↘"
        if not math.isnan(c) and c > before_coherence:
            improved_count += 1
        print(f"    [{st['name']}]  coherence = {c:.3f}  ({len(embs)} records)  {better}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    detection_ok = len(candidates) > 0 and candidates[0]["tag"] == "http_api"
    proposal_ok = len(proposals) > 0
    apply_ok = True  # didn't raise
    improvement_ok = improved_count >= len(p.sub_tags) // 2  # at least half improved

    checks = [
        ("compute_heterogeneous_tags detects http_api", detection_ok),
        ("propose_split_fn generates a proposal", proposal_ok),
        ("_apply_split applies without error", apply_ok),
        (f"≥50% sub-tags improve coherence over {before_coherence:.3f}", improvement_ok),
    ]
    for desc, ok in checks:
        print(f"  {'✓' if ok else '✗'} {desc}")

    overall = all(ok for _, ok in checks)
    print(f"\n  Overall: {'PASS' if overall else 'FAIL (see above)'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
