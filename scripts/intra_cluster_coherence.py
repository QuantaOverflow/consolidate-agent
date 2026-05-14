#!/usr/bin/env python3
"""Intra-cluster coherence metric for Matter facet.

For each tag, computes mean pairwise cosine across its assigned records'
stored embeddings (text-embedding-v4 from knowledge.db).

- High intra-cluster cosine → tag's members are semantically tight (concept-unified)
- Low intra-cluster cosine → tag's members are scattered (concept-overbroad)

Complements silhouette (which measures "is THIS tag better than others") with
"are my members really alike". A tag can have silhouette > 0 globally but low
intra coherence (= "barely better than chaos" rather than "tightly clustered").
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def pairwise_mean(embs: list[tuple]) -> float:
    """Mean of all C(n, 2) pairwise cosines. n <= 1 returns NaN-equivalent (None)."""
    n = len(embs)
    if n < 2:
        return float("nan")
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine(embs[i], embs[j])
            count += 1
    return total / count if count else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=Path("outputs/network_v3.json"), type=Path)
    parser.add_argument("--db", default=Path("outputs/knowledge.db"), type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    data = json.loads(args.in_path.read_text())
    vocab = data["vocab"]
    assignments = data["assignments"]

    # Load record embeddings
    print("loading record embeddings...", file=sys.stderr)
    conn = sqlite3.connect(args.db)
    rec_embs = {
        rid: tuple(json.loads(emb_json))
        for rid, emb_json in conn.execute(
            "SELECT record_id, embedding FROM source_knowledge_records WHERE embedding IS NOT NULL"
        )
    }
    conn.close()
    print(f"  loaded {len(rec_embs)} embeddings", file=sys.stderr)

    facets = vocab.get("facets", {})
    matter_tags = facets.get("matter", {}).get("tags", [])
    matter_names = {t["name"] for t in matter_tags}

    # Group records by matter tag (works on cleaned v3 file format)
    tag_to_records: dict[str, list[str]] = {t["name"]: [] for t in matter_tags}
    for a in assignments:
        for t in (a.get("matter_tags") or []):
            name = t.get("name") if isinstance(t, dict) else t
            if name in matter_names:
                tag_to_records.setdefault(name, []).append(a["record_id"])

    # Compute intra-cluster coherence
    out = []
    out.append("=" * 70)
    out.append(f"INTRA-CLUSTER COHERENCE (Matter facet, {args.in_path})")
    out.append("=" * 70)
    out.append("")
    out.append(f"For each tag: mean pairwise cosine across record embeddings.")
    out.append(f"Higher = members semantically tight (concept-unified).")
    out.append(f"Lower  = members scattered (concept-overbroad).")
    out.append("")
    out.append(f"  {'tag':<28} {'n':>5} {'mean':>8} {'min':>8} {'max':>8}  verdict")

    rows = []
    for tag, rids in tag_to_records.items():
        embs = [rec_embs[r] for r in rids if r in rec_embs]
        if len(embs) < 2:
            rows.append((tag, len(rids), float("nan"), float("nan"), float("nan")))
            continue
        # Pairwise (can be expensive for big tags but n is small here)
        pairs = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                pairs.append(cosine(embs[i], embs[j]))
        mean = statistics.mean(pairs)
        rows.append((tag, len(rids), mean, min(pairs), max(pairs)))

    # Sort by mean coherence desc (best at top)
    rows.sort(key=lambda r: -(r[2] if not math.isnan(r[2]) else -1))

    for tag, n, mean, mn, mx in rows:
        if math.isnan(mean):
            out.append(f"  {tag:<28} {n:>5} {'n/a':>8} {'n/a':>8} {'n/a':>8}  (size < 2)")
            continue
        # Verdict — relative thresholds based on observed v4 range (~0.3-0.7 typical for cohesive SE clusters)
        if mean >= 0.55:
            verdict = "✓ tight"
        elif mean >= 0.45:
            verdict = "✓ ok"
        elif mean >= 0.35:
            verdict = "⚠ loose"
        else:
            verdict = "✗ scattered"
        out.append(f"  {tag:<28} {n:>5} {mean:>8.3f} {mn:>8.3f} {mx:>8.3f}  {verdict}")

    # Overall stats
    valid = [r[2] for r in rows if not math.isnan(r[2])]
    if valid:
        out.append("")
        out.append(f"Overall mean intra-cluster: {statistics.mean(valid):.3f}")
        out.append(f"Tight tags  (≥0.55): {sum(1 for m in valid if m >= 0.55)}/{len(valid)}")
        out.append(f"OK tags     (≥0.45): {sum(1 for m in valid if 0.45 <= m < 0.55)}")
        out.append(f"Loose tags  (≥0.35): {sum(1 for m in valid if 0.35 <= m < 0.45)}")
        out.append(f"Scattered   (<0.35): {sum(1 for m in valid if m < 0.35)}")

    report = "\n".join(out)
    print(report)

    if args.out:
        args.out.write_text(report)
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
