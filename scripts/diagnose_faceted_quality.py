#!/usr/bin/env python3
"""Facet-aware silhouette diagnostic for v2 assignments.

For each (record, assigned_tag) edge, computes silhouette within the assigned
tag's facet (compared only to other tags in the SAME facet, not across).

Outputs:
  outputs/network_v2_diagnostics.txt   (full report)

Stage 1 Acceptance Plan Dimension ③ thresholds:
  Matter   mean silhouette  ≥ +0.05 PASS
  Activity mean silhouette  ≥ +0.05 PASS
  Pattern  mean silhouette  ≥  0.00 PASS
  Healthy tag ratio (per facet): Matter/Activity ≥70%, Pattern ≥50%
  LLM top-1 cosine hit rate (within facet): ≥50%
"""
from __future__ import annotations

import inspect
import json
import math
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
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


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def compute_facet_metrics(
    facet_name: str,
    tag_defs: dict[str, str],
    rec_embs: dict[str, tuple],
    assignments: list[dict],
    assignment_field: str,  # "matter_tags" / "pattern_tags" (lists) or "activity_tag" (single)
    is_single: bool,
    embed_fn,
) -> dict:
    """Compute per-edge silhouette for one facet."""
    # Embed tag defs in this facet
    tag_embs = {name: embed_fn(definition) for name, definition in tag_defs.items()}
    tag_names = list(tag_embs.keys())

    edges = []
    tag_silhouettes = defaultdict(list)
    skipped = 0

    for a in assignments:
        rid = a["record_id"]
        rec_emb = rec_embs.get(rid)
        if rec_emb is None:
            skipped += 1
            continue

        if is_single:
            tag_dict = a.get(assignment_field)
            assigned_tags = [tag_dict] if isinstance(tag_dict, dict) and tag_dict.get("name") in tag_embs else []
        else:
            assigned_tags = [t for t in (a.get(assignment_field) or []) if t.get("name") in tag_embs]

        if not assigned_tags:
            continue

        # Score record against ALL tag defs in this facet
        scores = {name: cosine(rec_emb, tag_embs[name]) for name in tag_names}
        sorted_scores = sorted(scores.items(), key=lambda kv: -kv[1])

        for t in assigned_tags:
            name = t["name"]
            sim_a = scores[name]
            # b = best alternative in this facet (excluding the assigned)
            alts = [s for n, s in scores.items() if n != name]
            sim_b = max(alts) if alts else 0.0
            margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) >= 2 else 0.0
            denom = max(sim_a, sim_b)
            sil = (sim_a - sim_b) / denom if denom > 0 else 0.0
            edges.append({
                "record_id": rid,
                "title": a.get("title", "")[:80],
                "assigned": name,
                "cosine": sim_a,
                "margin": margin,
                "silhouette": sil,
                "top1_match": sorted_scores[0][0] == name,
                "top3_match": name in {n for n, _ in sorted_scores[:3]},
                "better_alt": sorted_scores[0][0] if sorted_scores[0][0] != name else "",
                "better_alt_score": sorted_scores[0][1],
            })
            tag_silhouettes[name].append(sil)

    return {
        "facet": facet_name,
        "edges": edges,
        "tag_silhouettes": dict(tag_silhouettes),
        "skipped": skipped,
        "n_tags": len(tag_names),
    }


def format_facet_section(result: dict, threshold_mean: float, threshold_healthy: float) -> list[str]:
    out = []
    facet = result["facet"]
    edges = result["edges"]
    out.append(f"── {facet.upper()} facet ({result['n_tags']} tags, {len(edges)} edges) ──")
    if not edges:
        out.append("  (no edges)")
        return out

    sils = [e["silhouette"] for e in edges]
    cosines = [e["cosine"] for e in edges]
    mean_sil = statistics.mean(sils)
    out.append(f"  silhouette mean={mean_sil:+.3f}  median={statistics.median(sils):+.3f}  "
               f"p25={percentile(sils, 0.25):+.3f}  p75={percentile(sils, 0.75):+.3f}")
    out.append(f"  cosine     mean={statistics.mean(cosines):.3f}  median={statistics.median(cosines):.3f}  "
               f"min={min(cosines):.3f}  max={max(cosines):.3f}")
    neg = sum(1 for s in sils if s < 0)
    pos = sum(1 for s in sils if s > 0.01)
    out.append(f"  silhouette buckets: pos={pos} ({100*pos/len(sils):.1f}%)  neg={neg} ({100*neg/len(sils):.1f}%)")

    # LLM vs cosine
    top1 = sum(1 for e in edges if e["top1_match"])
    top3 = sum(1 for e in edges if e["top3_match"])
    out.append(f"  LLM top-1 within facet: {top1}/{len(edges)} ({100*top1/len(edges):.1f}%)")
    out.append(f"  LLM top-3 within facet: {top3}/{len(edges)} ({100*top3/len(edges):.1f}%)")

    # PASS / FAIL gates per acceptance plan
    pass_mean = "PASS" if mean_sil >= threshold_mean else ("WARN" if mean_sil >= threshold_mean - 0.10 else "FAIL")
    out.append(f"  ⇒ Mean silhouette {pass_mean} (≥{threshold_mean:+.2f} PASS)")

    # Per-tag table
    out.append(f"\n  per-tag silhouette (sorted asc):")
    out.append(f"  {'tag':<28} {'edges':>6} {'mean':>8} {'neg':>5}  verdict")
    tag_stats = []
    for tag, sl in result["tag_silhouettes"].items():
        if not sl:
            continue
        m = statistics.mean(sl)
        ng = sum(1 for s in sl if s < 0)
        tag_stats.append((tag, len(sl), m, ng))
    tag_stats.sort(key=lambda x: x[2])
    healthy = sum(1 for _, _, m, _ in tag_stats if m > 0)
    healthy_ratio = healthy / max(1, len(tag_stats))
    for tag, n, m, ng in tag_stats:
        if m < 0:
            verdict = "✗ negative"
        elif m < 0.05:
            verdict = "⚠ weak"
        elif ng / max(1, n) > 0.3:
            verdict = "⚠ many neg"
        else:
            verdict = "✓"
        out.append(f"  {tag:<28} {n:>6} {m:>+8.3f} {ng:>5}  {verdict}")

    pass_h = "PASS" if healthy_ratio >= threshold_healthy else ("WARN" if healthy_ratio >= threshold_healthy - 0.20 else "FAIL")
    out.append(f"  ⇒ Healthy tag ratio: {healthy}/{len(tag_stats)} = {100*healthy_ratio:.1f}% {pass_h} (≥{100*threshold_healthy:.0f}% PASS)")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=Path("outputs/network_v2.json"), type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()
    in_path: Path = args.in_path
    out_path: Path = args.out or in_path.parent / f"{in_path.stem}_diagnostics.txt"

    load_dotenv(Path(".env"))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY missing")

    from consolidate_agent.vocab_maintenance.similarity import _embed

    embed_model = inspect.signature(_embed).parameters["model"].default

    data = json.loads(in_path.read_text())
    vocab = data["vocab"]
    assignments = data["assignments"]

    # Load record embeddings from db
    print("loading record embeddings from db...", file=sys.stderr)
    conn = sqlite3.connect("outputs/knowledge.db")
    rec_embs = {
        rid: tuple(json.loads(emb_json))
        for rid, emb_json in conn.execute(
            "SELECT record_id, embedding FROM source_knowledge_records WHERE embedding IS NOT NULL"
        )
    }
    conn.close()
    print(f"  loaded {len(rec_embs)} record embeddings", file=sys.stderr)

    facets = vocab.get("facets", {})
    matter_defs = {t["name"]: t["definition"] for t in facets.get("matter", {}).get("tags", [])}
    activity_defs = {t["name"]: t["definition"] for t in facets.get("activity", {}).get("tags", [])}
    pattern_defs = {t["name"]: t["definition"] for t in facets.get("pattern", {}).get("tags", [])}

    total_tags = len(matter_defs) + len(activity_defs) + len(pattern_defs)
    print(f"embedding {total_tags} tag defs...", file=sys.stderr)

    out: list[str] = []
    out.append("=" * 80)
    out.append("FACETED ASSIGNMENT DIAGNOSTICS")
    out.append("=" * 80)
    out.append(f"Source: {in_path}")
    out.append(f"Records: {len(assignments)}, embedding model: {embed_model}")
    out.append("")

    facet_results = []

    if matter_defs:
        matter_result = compute_facet_metrics(
            "matter", matter_defs, rec_embs, assignments, "matter_tags", is_single=False, embed_fn=_embed
        )
        out.extend(format_facet_section(matter_result, threshold_mean=0.05, threshold_healthy=0.70))
        out.append("")
        facet_results.append(("matter", matter_result))

    if activity_defs:
        activity_result = compute_facet_metrics(
            "activity", activity_defs, rec_embs, assignments, "activity_tag", is_single=True, embed_fn=_embed
        )
        out.extend(format_facet_section(activity_result, threshold_mean=0.05, threshold_healthy=0.70))
        out.append("")
        facet_results.append(("activity", activity_result))

    if pattern_defs:
        pattern_result = compute_facet_metrics(
            "pattern", pattern_defs, rec_embs, assignments, "pattern_tags", is_single=False, embed_fn=_embed
        )
        out.extend(format_facet_section(pattern_result, threshold_mean=0.0, threshold_healthy=0.50))
        out.append("")
        facet_results.append(("pattern", pattern_result))

    # Worst N edges per facet (only present facets)
    for label, result in facet_results:
        worst = sorted(result["edges"], key=lambda e: e["silhouette"])[:5]
        if worst and worst[0]["silhouette"] < -0.3:
            out.append(f"── 5 worst {label} edges (sil < -0.3) ──")
            for e in worst:
                if e["silhouette"] >= -0.3:
                    break
                out.append(f"  sil={e['silhouette']:+.3f} cos={e['cosine']:.3f}  {e['title']}")
                out.append(f"    assigned: {e['assigned']:<24} better_alt: {e['better_alt']}")
            out.append("")

    report = "\n".join(out)
    out_path.write_text(report)
    print(report)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
