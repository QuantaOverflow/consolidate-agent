#!/usr/bin/env python3
"""Objective record-tag assignment quality via embedding cosine.

Computes per-edge cosine, margin (top1 - top2), and silhouette score.
Silhouette < 0 means a better alternative exists in current vocab.

Inputs:
  outputs/network.json                       (vocab + assignments)
  outputs/knowledge.db                       (record embeddings, pre-cached)

Output:
  outputs/network_diagnostics.txt            (full report)
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


def main():
    load_dotenv(Path(".env"))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY missing (looked in .env)")

    import inspect
    from consolidate_agent.vocab_maintenance.similarity import _embed

    _embed_model = inspect.signature(_embed).parameters["model"].default

    net = json.loads(Path("outputs/network.json").read_text())
    vocab = net["vocab"]
    assignments = net["assignments"]

    # 1. Load record embeddings from DB (free, pre-cached)
    print("loading record embeddings from db...")
    conn = sqlite3.connect("outputs/knowledge.db")
    rec_embs = {}
    for rid, emb_json in conn.execute(
        "SELECT record_id, embedding FROM source_knowledge_records WHERE embedding IS NOT NULL"
    ):
        rec_embs[rid] = tuple(json.loads(emb_json))
    conn.close()
    print(f"  loaded {len(rec_embs)} record embeddings")

    # 2. Embed tag definitions
    print(f"embedding {len(vocab)} tag definitions...")
    tag_embs = {}
    for i, t in enumerate(vocab):
        tag_embs[t["name"]] = _embed(t["definition"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(vocab)}")
    tag_names = list(tag_embs.keys())

    # 3. For each tagged record, compute cosine vs ALL 49 tag defs, then derive metrics
    print("computing per-edge metrics...")
    edges = []  # (record_id, assigned_tag, cosine, margin, silhouette, better_alt)
    tag_silhouettes = defaultdict(list)
    skipped_no_emb = 0

    for a in assignments:
        if a.get("missing") or not a.get("selected_tags"):
            continue
        rid = a["record_id"]
        rec_emb = rec_embs.get(rid)
        if rec_emb is None:
            skipped_no_emb += 1
            continue
        # Score against all tag defs
        scores = {name: cosine(rec_emb, tag_embs[name]) for name in tag_names}
        sorted_scores = sorted(scores.items(), key=lambda kv: -kv[1])
        top1, top2 = sorted_scores[0], sorted_scores[1]
        # For each tag actually assigned to this record, compute edge metric
        assigned_set = {t["name"] for t in a["selected_tags"]}
        for t in a["selected_tags"]:
            name = t["name"]
            sim_a = scores.get(name, 0.0)
            # b = best alternative NOT this tag
            alt_best = max(((n, s) for n, s in scores.items() if n != name),
                          key=lambda kv: kv[1])
            sim_b = alt_best[1]
            margin = top1[1] - top2[1]
            denom = max(sim_a, sim_b)
            sil = (sim_a - sim_b) / denom if denom > 0 else 0.0
            edges.append({
                "record_id": rid,
                "title": a["title"][:80],
                "assigned": name,
                "confidence": t.get("confidence", ""),
                "cosine": sim_a,
                "margin": margin,
                "silhouette": sil,
                "better_alt": alt_best[0] if sim_b > sim_a else "",
                "better_alt_score": sim_b,
                "top1_match": top1[0] == name,
                "top3_match": name in {n for n, _ in sorted_scores[:3]},
            })
            tag_silhouettes[name].append(sil)

    if skipped_no_emb:
        print(f"  WARN: skipped {skipped_no_emb} records (no embedding in db)")
    print(f"  produced {len(edges)} edges")

    # 4. Aggregate + format report
    out = []
    out.append("=" * 80)
    out.append("RECORD-TAG ASSIGNMENT DIAGNOSTICS")
    out.append("=" * 80)
    out.append(f"Total edges analyzed: {len(edges)}")
    out.append(f"Tags: {len(vocab)}, embeddings model: {_embed_model}")
    out.append("")

    cosines = [e["cosine"] for e in edges]
    margins = [e["margin"] for e in edges]
    sils = [e["silhouette"] for e in edges]

    out.append("── Overall distribution ──")
    for label, xs in [("cosine(rec, assigned)", cosines), ("margin (top1-top2)", margins), ("silhouette", sils)]:
        out.append(f"  {label:<28} mean={statistics.mean(xs):+.3f}  median={statistics.median(xs):+.3f}  "
                   f"p25={percentile(xs, 0.25):+.3f}  p75={percentile(xs, 0.75):+.3f}  "
                   f"min={min(xs):+.3f}  max={max(xs):+.3f}")
    out.append("")

    neg_sil = [e for e in edges if e["silhouette"] < 0]
    zero_sil = [e for e in edges if -0.01 <= e["silhouette"] <= 0.01]
    pos_sil = [e for e in edges if e["silhouette"] > 0.01]
    out.append("── Silhouette buckets (negative = better alternative exists) ──")
    out.append(f"  silhouette < 0       : {len(neg_sil):>4}  ({100*len(neg_sil)/len(edges):.1f}%)  ← objectively misassigned")
    out.append(f"  silhouette ≈ 0       : {len(zero_sil):>4}  ({100*len(zero_sil)/len(edges):.1f}%)  ← tie with alternative")
    out.append(f"  silhouette > 0       : {len(pos_sil):>4}  ({100*len(pos_sil)/len(edges):.1f}%)  ← clear winner")
    out.append("")

    # Top-1 / Top-3 agreement (LLM choice vs cosine ranking)
    n_top1 = sum(1 for e in edges if e["top1_match"])
    n_top3 = sum(1 for e in edges if e["top3_match"])
    out.append("── LLM choice vs cosine ranking ──")
    out.append(f"  LLM picked top-1 cosine match: {n_top1}/{len(edges)} ({100*n_top1/len(edges):.1f}%)")
    out.append(f"  LLM picked top-3 cosine match: {n_top3}/{len(edges)} ({100*n_top3/len(edges):.1f}%)")
    out.append("")

    # Per-tag silhouette
    out.append("── Per-tag mean silhouette (sorted asc — worst first) ──")
    out.append(f"  {'tag':<30} {'records':>7}  {'mean_sil':>8}  {'neg_count':>9}  {'verdict'}")
    tag_stats = []
    for tag, sils_list in tag_silhouettes.items():
        mean_sil = statistics.mean(sils_list)
        neg = sum(1 for s in sils_list if s < 0)
        tag_stats.append((tag, len(sils_list), mean_sil, neg))
    tag_stats.sort(key=lambda x: x[2])  # asc by mean silhouette
    for tag, n, mean_sil, neg in tag_stats:
        if mean_sil < 0:
            verdict = "✗ deprecate candidate"
        elif mean_sil < 0.05:
            verdict = "⚠ weak cohesion"
        elif neg / n > 0.3:
            verdict = "⚠ many misassignments"
        else:
            verdict = "✓"
        out.append(f"  {tag:<30} {n:>7}  {mean_sil:+8.3f}  {neg:>9}  {verdict}")
    out.append("")

    # Worst 15 edges
    out.append("── 15 worst edges (most negative silhouette) ──")
    worst = sorted(edges, key=lambda e: e["silhouette"])[:15]
    for e in worst:
        out.append(f"  sil={e['silhouette']:+.3f}  cos={e['cosine']:.3f}  ")
        out.append(f"    record: {e['record_id']} — {e['title']}")
        out.append(f"    assigned: {e['assigned']:<26} (conf={e['confidence']})")
        out.append(f"    better alt: {e['better_alt']:<26} (score={e['better_alt_score']:.3f})")
        out.append("")

    # Strongest 10 (top1, high cosine)
    out.append("── 10 strongest edges (highest silhouette) ──")
    best = sorted(edges, key=lambda e: -e["silhouette"])[:10]
    for e in best:
        out.append(f"  sil={e['silhouette']:+.3f}  cos={e['cosine']:.3f}  {e['assigned']:<26}  ← {e['record_id']}")
    out.append("")

    report = "\n".join(out)
    Path("outputs/network_diagnostics.txt").write_text(report)
    print()
    print(report)
    print()
    print("Wrote outputs/network_diagnostics.txt")


if __name__ == "__main__":
    main()
