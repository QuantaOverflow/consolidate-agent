#!/usr/bin/env python3
"""P2: maintenance on v3 vocab using existing agent functions directly.

Runs: compute_fit_signals → propose_refine (top N forced_fit tags) → apply_proposal
Saves updated vocab as docs/plans/vocab_v3.1.json.
Does NOT run the full LangGraph agent loop — calls the functions directly
to avoid LangGraph infra complexity while still using all production logic.

Usage:
  python scripts/run_v3_maintenance.py [--in outputs/network_v3_p1.json]
                                        [--vocab-out docs/plans/vocab_v3.1.json]
                                        [--max-tags N]   # forced_fit tags to refine, default 5
"""
from __future__ import annotations

import argparse
import json
import os
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


def v3_to_v1_view(data: dict) -> tuple[list[dict], list[dict]]:
    """Extract matter tags as flat vocab + rename matter_tags → selected_tags."""
    vocab = data["vocab"]["facets"]["matter"]["tags"]
    assignments = []
    for a in data["assignments"]:
        assignments.append({
            "record_id": a["record_id"],
            "title": a.get("title", ""),
            "selected_tags": list(a.get("matter_tags") or []),
            "missing": not bool(a.get("matter_tags")),
        })
    return vocab, assignments


def writeback_vocab(data: dict, new_vocab: list[dict]) -> dict:
    """Replace matter tags in v3 data with updated vocab."""
    import copy
    result = copy.deepcopy(data)
    result["vocab"]["facets"]["matter"]["tags"] = new_vocab
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=Path("outputs/network_v3_p1.json"), type=Path)
    parser.add_argument("--vocab-out", default=Path("docs/plans/vocab_v3.1.json"), type=Path)
    parser.add_argument("--max-tags", default=5, type=int, help="max forced_fit tags to attempt refine on")
    parser.add_argument("--db", default=Path("outputs/knowledge.db"), type=Path)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY missing")

    from consolidate_agent.vocab_maintenance.probes import compute_fit_signals
    from consolidate_agent.vocab_maintenance.propose.refine import propose_refine_fn
    from consolidate_agent.vocab_maintenance.apply import apply_proposal, InvalidProposal, OrphanError

    # ── Load ──────────────────────────────────────────────────────────────────
    data = json.loads(args.in_path.read_text())
    vocab, assignments = v3_to_v1_view(data)
    print(f"Loaded: {len(vocab)} tags, {len(assignments)} records (v1-view)")
    multi_tag = sum(1 for a in assignments if len(a["selected_tags"]) >= 2)
    print(f"Prunable records (≥2 matter tags): {multi_tag}")

    # ── Step 1: Probe forced_fit_candidates ───────────────────────────────────
    print("\n[Step 1] compute_fit_signals...")
    signals = compute_fit_signals(
        vocab, assignments, args.db,
        max_records=len(assignments),
        low_fit_threshold=0.5,
        min_tag_sample=5,
    )
    candidates = signals["forced_fit_candidates"]
    print(f"  forced_fit_candidates ({len(candidates)} top):")
    for c in candidates:
        print(f"    {c['tag']:<28} mean_fit={c['mean_fit']}  sample={c['sample']}")

    if not candidates:
        print("  No forced_fit candidates — vocab already well-fitted. Exiting.")
        return

    # ── Step 2: propose_refine on each candidate ─────────────────────────────
    print(f"\n[Step 2] propose_refine on top {min(args.max_tags, len(candidates))} candidates...")
    commits = 0
    for c in candidates[:args.max_tags]:
        tag_name = c["tag"]
        print(f"\n  → {tag_name} (mean_fit={c['mean_fit']})")
        proposals = propose_refine_fn(
            vocab, assignments, focus=tag_name,
            db_path=args.db, max_targets=1,
        )
        if not proposals:
            print(f"    no proposal generated (no prunable outliers or LLM declined)")
            continue
        p = proposals[0]
        print(f"    new def: {p.new_definition[:120]}")
        print(f"    prune {len(p.prune_record_ids)} records")
        try:
            vocab, assignments = apply_proposal(vocab, assignments, p)
            commits += 1
            print(f"    ✓ applied")
        except (InvalidProposal, OrphanError) as e:
            print(f"    ✗ invariant violation: {e}")

    print(f"\n[Step 2] {commits} proposals applied")

    # ── Step 3: Save updated vocab ────────────────────────────────────────────
    print(f"\n[Step 3] Saving updated vocab → {args.vocab_out}")
    updated_data = writeback_vocab(data, vocab)
    updated_data["vocab"]["version"] = "v3.1-post-maintenance"
    updated_data["vocab"]["notes"] = (
        f"v3.0 vocab after {commits} propose_refine maintenance iterations "
        f"on top forced_fit candidates. "
        "Run scripts/run_minimal_tagging.py --vocab docs/plans/vocab_v3.1.json "
        "--out outputs/network_v3_p2.json to re-tag with updated definitions."
    )
    args.vocab_out.write_text(json.dumps(updated_data["vocab"], ensure_ascii=False, indent=2))
    print(f"  wrote {args.vocab_out}")

    # Summary of changed definitions
    orig_defs = {t["name"]: t["definition"] for t in json.loads(args.in_path.read_text())["vocab"]["facets"]["matter"]["tags"]}
    new_defs = {t["name"]: t["definition"] for t in vocab}
    changed = {n for n in orig_defs if orig_defs[n] != new_defs.get(n, "")}
    print(f"\n  Changed tag definitions ({len(changed)}):")
    for n in sorted(changed):
        print(f"    [{n}]")
        print(f"      old: {orig_defs[n][:100]}")
        print(f"      new: {new_defs[n][:100]}")


if __name__ == "__main__":
    main()
