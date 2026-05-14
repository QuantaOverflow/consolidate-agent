#!/usr/bin/env python3
"""Quality scorecard: multi-tag-friendly metrics (replaces silhouette PASS/FAIL).

Per ADR-0005, silhouette is no longer the primary quality metric. This script
runs the 3-tier metric stack:

Primary (PASS/FAIL):
  - Golden set: Matter exact, partial, lesson_type vs human-labeled 80
  - Intra-cluster coherence: mean pairwise cosine per tag

Secondary (diagnostic):
  - Tag plausibility per record: assigned tags in top-K cosine neighbors

Sanity:
  - Hallucination rate
  - Coverage
  - Mean tags/record
  - Tag size distribution
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


GOLDEN = Path("tests/fixtures/golden_80.jsonl")


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


def verdict(actual: float, pass_th: float, warn_drop: float = 0.10) -> tuple[str, str]:
    if actual >= pass_th:
        return "PASS", "✓"
    if actual >= pass_th - warn_drop:
        return "WARN", "⚠"
    return "FAIL", "✗"


# ── Primary metric 1: Golden set ─────────────────────────────────────────


def matter_set_match(expected: set[str], actual: set[str], alts: list[list[str]] | None) -> tuple[bool, bool]:
    if actual == expected:
        return True, True
    if alts:
        for alt in alts:
            if actual == set(alt):
                return True, True
    return False, bool(expected & actual)


def evaluate_golden(assignments_by_rid: dict, golden: list[dict]) -> dict:
    matter_exact = 0
    matter_partial = 0
    lesson_exact = 0
    n_present = 0
    for g in golden:
        rid = g["record_id"]
        a = assignments_by_rid.get(rid)
        if a is None:
            continue
        n_present += 1
        exp_m = set(g["expected"]["matter_tags"])
        act_m = {t["name"] for t in (a.get("matter_tags") or [])}
        e, p = matter_set_match(exp_m, act_m, g.get("acceptable_matter_alternatives"))
        if e: matter_exact += 1
        if p: matter_partial += 1
        exp_l = g["expected"]["lesson_type"]
        act_l = a.get("lesson_type")
        l_alts = g.get("acceptable_lesson_type_alternatives") or []
        if act_l == exp_l or act_l in l_alts:
            lesson_exact += 1
    return {
        "n": n_present,
        "matter_exact_pct": matter_exact / n_present * 100 if n_present else 0,
        "matter_partial_pct": matter_partial / n_present * 100 if n_present else 0,
        "lesson_exact_pct": lesson_exact / n_present * 100 if n_present else 0,
    }


# ── Primary metric 2: Intra-cluster coherence ────────────────────────────


def intra_cluster_coherence(tag_to_recs: dict[str, list[str]], rec_embs: dict[str, tuple]) -> dict[str, float]:
    """Mean pairwise cosine per tag."""
    results = {}
    for tag, rids in tag_to_recs.items():
        embs = [rec_embs[r] for r in rids if r in rec_embs]
        if len(embs) < 2:
            results[tag] = float("nan")
            continue
        pairs = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                pairs.append(cosine(embs[i], embs[j]))
        results[tag] = sum(pairs) / len(pairs)
    return results


# ── Secondary metric: Tag plausibility per record ────────────────────────


def per_record_plausibility(
    assignments: list[dict],
    rec_embs: dict[str, tuple],
    tag_embs: dict[str, tuple],
    k: int = 5,
) -> dict:
    """For each record, fraction of assigned matter_tags in record's top-K cosine neighbors among all tags."""
    plaus = []
    for a in assignments:
        rid = a["record_id"]
        rec_emb = rec_embs.get(rid)
        if rec_emb is None:
            continue
        assigned = [t["name"] for t in (a.get("matter_tags") or []) if t.get("name") in tag_embs]
        if not assigned:
            continue
        # Get top-K cosine neighbors among all tags
        scores = sorted(
            ((name, cosine(rec_emb, te)) for name, te in tag_embs.items()),
            key=lambda kv: -kv[1],
        )
        top_k = {n for n, _ in scores[:k]}
        hits = sum(1 for t in assigned if t in top_k)
        plaus.append(hits / len(assigned))
    return {
        "n": len(plaus),
        "mean": statistics.mean(plaus) if plaus else 0,
        "median": statistics.median(plaus) if plaus else 0,
        "perfect_pct": sum(1 for p in plaus if p == 1.0) / max(1, len(plaus)) * 100,
        "any_pct": sum(1 for p in plaus if p > 0) / max(1, len(plaus)) * 100,
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=Path("outputs/network_v3.json"), type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--k", default=5, type=int, help="top-K for plausibility")
    args = parser.parse_args()

    load_dotenv(Path(".env"))

    data = json.loads(args.in_path.read_text())
    vocab = data["vocab"]
    assignments = data["assignments"]
    assignments_by_rid = {a["record_id"]: a for a in assignments}

    facets = vocab.get("facets", {})
    matter_defs = {t["name"]: t["definition"] for t in facets.get("matter", {}).get("tags", [])}

    # Load record embeddings
    print("loading record embeddings...", file=sys.stderr)
    conn = sqlite3.connect("outputs/knowledge.db")
    rec_embs = {
        rid: tuple(json.loads(emb_json))
        for rid, emb_json in conn.execute(
            "SELECT record_id, embedding FROM source_knowledge_records WHERE embedding IS NOT NULL"
        )
    }
    conn.close()

    # Embed tag defs (using same v4 default as records)
    from consolidate_agent.vocab_maintenance.similarity import _embed
    embed_model = inspect.signature(_embed).parameters["model"].default
    print(f"embedding {len(matter_defs)} matter tags...", file=sys.stderr)
    tag_embs = {name: _embed(d) for name, d in matter_defs.items()}

    # Build tag→records map
    tag_to_recs: dict[str, list[str]] = {name: [] for name in matter_defs}
    for a in assignments:
        for t in (a.get("matter_tags") or []):
            name = t.get("name") if isinstance(t, dict) else t
            if name in tag_to_recs:
                tag_to_recs[name].append(a["record_id"])

    # ── Compute metrics ──
    golden = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    golden_results = evaluate_golden(assignments_by_rid, golden)
    intra = intra_cluster_coherence(tag_to_recs, rec_embs)
    plaus = per_record_plausibility(assignments, rec_embs, tag_embs, k=args.k)

    n_records = len(assignments)
    covered = sum(1 for a in assignments if a.get("matter_tags"))
    total_matter_tags = sum(len(a.get("matter_tags") or []) for a in assignments)
    mean_tags = total_matter_tags / max(1, n_records)
    matter_names = set(matter_defs.keys())
    hallucinations = sum(
        1 for a in assignments
        for t in (a.get("matter_tags") or [])
        if t.get("name") not in matter_names
    )
    halluc_rate = hallucinations / max(1, total_matter_tags) * 100

    # Tag size distribution
    tag_sizes = Counter()
    for a in assignments:
        for t in (a.get("matter_tags") or []):
            n = t.get("name") if isinstance(t, dict) else t
            tag_sizes[n] += 1
    max_share = max(tag_sizes.values()) / max(1, total_matter_tags) * 100 if tag_sizes else 0
    long_tail = sum(1 for c in tag_sizes.values() if c <= 2)

    # ── Format report ──
    out = []
    out.append("=" * 72)
    out.append(f"QUALITY SCORECARD — multi-tag aware (per ADR-0005)")
    out.append("=" * 72)
    out.append(f"Source        : {args.in_path}")
    out.append(f"Records       : {n_records}")
    out.append(f"Matter tags   : {len(matter_defs)}")
    out.append(f"Embedding     : {embed_model}")
    out.append("")

    # ── Primary tier ──
    out.append("─" * 72)
    out.append("PRIMARY METRICS (drive PASS/FAIL)")
    out.append("─" * 72)
    out.append("")
    out.append("[1] Golden set (n=80 human labels) ")
    # Thresholds calibrated from inter-rater experiment (2026-05-14):
    # intra-annotator = 86.7% exact / 100% partial / 86.7% lesson
    # true inter-rater estimated at intra - 12pp → 70% / 90% / 78%
    v1, s1 = verdict(golden_results["matter_exact_pct"], 70)
    v2, s2 = verdict(golden_results["matter_partial_pct"], 90)
    v3, s3 = verdict(golden_results["lesson_exact_pct"], 78)
    out.append(f"  Matter exact match     : {golden_results['matter_exact_pct']:5.1f}%  {s1} {v1}  (PASS ≥70%, inter-rater anchored)")
    out.append(f"  Matter partial overlap : {golden_results['matter_partial_pct']:5.1f}%  {s2} {v2}  (PASS ≥90%, inter-rater anchored)")
    out.append(f"  lesson_type exact      : {golden_results['lesson_exact_pct']:5.1f}%  {s3} {v3}  (PASS ≥78%, inter-rater anchored)")
    out.append("")

    out.append("[2] Intra-cluster coherence (mean pairwise cosine per tag)")
    intra_vals = sorted(((t, v) for t, v in intra.items() if not math.isnan(v)), key=lambda x: -x[1])
    n_tight = sum(1 for _, v in intra_vals if v >= 0.55)
    n_ok = sum(1 for _, v in intra_vals if 0.45 <= v < 0.55)
    n_loose = sum(1 for _, v in intra_vals if 0.35 <= v < 0.45)
    n_scat = sum(1 for _, v in intra_vals if v < 0.35)
    total = len(intra_vals)
    ok_or_better_pct = (n_tight + n_ok) / max(1, total) * 100
    v_intra, s_intra = verdict(ok_or_better_pct, 50)
    out.append(f"  Tags OK+ (≥0.45)       : {n_tight + n_ok}/{total} = {ok_or_better_pct:.1f}%  {s_intra} {v_intra}  (PASS ≥50%)")
    out.append(f"     - tight (≥0.55)     : {n_tight}")
    out.append(f"     - OK   (0.45-0.54)  : {n_ok}")
    out.append(f"     - loose(0.35-0.44)  : {n_loose}")
    out.append(f"     - scattered (<0.35) : {n_scat}")
    overall_intra = statistics.mean([v for _, v in intra_vals]) if intra_vals else 0
    out.append(f"  Overall mean           : {overall_intra:.3f}")
    out.append("")

    # ── Secondary tier ──
    out.append("─" * 72)
    out.append("SECONDARY METRICS (diagnostic)")
    out.append("─" * 72)
    out.append("")
    out.append(f"[3] Tag plausibility per record (top-{args.k} cosine neighbor check)")
    out.append(f"  Records evaluated      : {plaus['n']}")
    out.append(f"  Mean plausibility      : {plaus['mean']:.3f}  (1.0 = all tags in top-{args.k})")
    out.append(f"  Perfect plausibility   : {plaus['perfect_pct']:.1f}%")
    out.append(f"  Any-tag in top-{args.k}     : {plaus['any_pct']:.1f}%")
    out.append("")

    # ── Sanity tier ──
    out.append("─" * 72)
    out.append("SANITY METRICS (hygiene)")
    out.append("─" * 72)
    out.append("")
    v_cov, s_cov = verdict(covered / n_records * 100, 98)
    v_mean = "PASS ✓" if 1.3 <= mean_tags <= 2.0 else ("WARN ⚠" if 1.0 <= mean_tags <= 2.3 else "FAIL ✗")
    v_halluc = "PASS ✓" if halluc_rate <= 1.0 else ("WARN ⚠" if halluc_rate <= 3.0 else "FAIL ✗")
    v_dom = "PASS ✓" if max_share <= 30 else ("WARN ⚠" if max_share <= 40 else "FAIL ✗")
    out.append(f"[4] Coverage              : {covered}/{n_records} = {covered/n_records*100:.1f}%  {s_cov} {v_cov} (PASS ≥98%)")
    out.append(f"[5] Mean tags/record      : {mean_tags:.2f}                {v_mean} (PASS 1.3-2.0)")
    out.append(f"[6] Hallucination rate    : {halluc_rate:.2f}% ({hallucinations}/{total_matter_tags} edges)  {v_halluc} (PASS ≤1%)")
    out.append(f"[7] Max tag share         : {max_share:.1f}%                {v_dom} (PASS ≤30%)")
    out.append(f"[8] Long-tail tags (≤2)   : {long_tail}                       (no PASS gate)")
    out.append("")

    # ── Top + worst tags ──
    out.append("─" * 72)
    out.append("Top 5 best tags by intra-cluster coherence")
    out.append("─" * 72)
    out.append(f"  {'tag':<28} {'n':>5} {'coherence':>10}")
    for tag, v in intra_vals[:5]:
        out.append(f"  {tag:<28} {len(tag_to_recs[tag]):>5} {v:>10.3f}")
    out.append("")
    out.append("Bottom 5 tags by intra-cluster coherence")
    out.append("─" * 72)
    out.append(f"  {'tag':<28} {'n':>5} {'coherence':>10}")
    for tag, v in intra_vals[-5:]:
        out.append(f"  {tag:<28} {len(tag_to_recs[tag]):>5} {v:>10.3f}")
    out.append("")

    # Overall verdict
    primary_passes = [v1 == "PASS", v2 == "PASS", v3 == "PASS", v_intra == "PASS"]
    sanity_passes = [
        v_cov == "PASS",
        "PASS" in v_mean,
        "PASS" in v_halluc,
        "PASS" in v_dom,
    ]
    overall = "PASS" if all(primary_passes + sanity_passes) else ("WARN" if all(sanity_passes) else "FAIL")
    out.append("=" * 72)
    out.append(f"OVERALL: {overall}  (primary={sum(primary_passes)}/4 PASS, sanity={sum(sanity_passes)}/4 PASS)")
    out.append("=" * 72)

    report = "\n".join(out)
    print(report)
    if args.out:
        args.out.write_text(report)
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
