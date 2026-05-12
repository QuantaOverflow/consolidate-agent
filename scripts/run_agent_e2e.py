"""End-to-end agent runner with real LLM workflows.

Wires together vocab_maintenance package components:
  - measure (reverse_check) — parallel 10
  - diagnose (probe-driven reviewer)
  - propose.new / propose.merge / propose.deprecate
  - apply_proposal (invariant-checked)

Usage:
    uv run python scripts/run_agent_e2e.py [--max-iter N] [--sample-size N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from consolidate_agent.vocab_maintenance.agent import Agent, FinalStatus
from consolidate_agent.vocab_maintenance.diagnose import diagnose as diagnose_call
from consolidate_agent.vocab_maintenance.measure import (
    build_diagnostics,
    load_records,
    reverse_check_subset,
)
from consolidate_agent.vocab_maintenance.propose.new import propose_new_fn
from consolidate_agent.vocab_maintenance.propose.merge import propose_merge_fn
from consolidate_agent.vocab_maintenance.propose.deprecate import propose_deprecate_fn


BASE = Path(__file__).resolve().parents[1]


# ── Wiring ────────────────────────────────────────────────────────────────────


def build_measure_fn(
    db_path: Path,
    cached_assignments_path: Path,
    sample_size: int,
    concurrency: int = 10,
):
    """Cache-driven measure_fn.

    Initial call (action=None): load cached assignments (no LLM).
    After merge/deprecate apply: zero LLM — apply.py has already updated
      assignments deterministically; we just re-aggregate diagnostics.
    After propose_new apply: LLM patch only the records that were `missing`
      in the previous round (the new tag may now cover them).
    """
    cached = json.loads(cached_assignments_path.read_text(encoding="utf-8"))
    records_by_id = {r["record_id"]: r for r in load_records(db_path)}

    # Sample cap: deterministic slice (sorted by record_id) so reruns are stable
    if sample_size > 0 and sample_size < len(cached):
        cached = sorted(cached, key=lambda a: a["record_id"])[:sample_size]
    print(f"  [measure] loaded {len(cached)} cached assignments", flush=True)

    def _patch_missing_with_new_vocab(vocab, previous_assignments, current_assignments):
        """For propose_new: re-LLM the records that were `missing` last round."""
        prev_missing_ids = {a["record_id"] for a in previous_assignments if a.get("missing")}
        if not prev_missing_ids:
            return current_assignments
        to_check = [records_by_id[rid] for rid in prev_missing_ids if rid in records_by_id]
        print(f"  [measure] propose_new patch: re-checking {len(to_check)} previously-missing records", flush=True)
        delta = reverse_check_subset(to_check, vocab, batch_size=10, concurrency=concurrency)
        # Filter vocab-external fake tags (LLM hallucination guard)
        vocab_names = {t["name"] for t in vocab}
        for a in delta:
            original = a.get("selected_tags", []) or []
            filtered = [t for t in original if t["name"] in vocab_names]
            a["selected_tags"] = filtered
            if not filtered and not a.get("missing"):
                a["missing"] = True
                a["missing_concept"] = a.get("missing_concept") or "all selected_tags were vocab-external"
        delta_by_id = {a["record_id"]: a for a in delta}
        return [delta_by_id.get(a["record_id"], a) for a in current_assignments]

    def measure_fn(vocab: list[dict], **ctx):
        action = ctx.get("action")
        current_assignments = ctx.get("current_assignments")
        previous_assignments = ctx.get("previous_assignments")

        if action is None:
            # Initial call: use cache as the assignment snapshot.
            assignments = [dict(a) for a in cached]
        elif action == "propose_new":
            assignments = _patch_missing_with_new_vocab(
                vocab, previous_assignments or [], list(current_assignments or [])
            )
        else:
            # merge / deprecate / others: apply.py already updated assignments
            assignments = list(current_assignments or [])

        return build_diagnostics(vocab, assignments), assignments

    return measure_fn


def diagnose_fn(vocab, diagnostics, assignments, **kwargs):
    db = BASE / "outputs/knowledge.db"
    return diagnose_call(vocab, diagnostics, assignments, db, **kwargs)


def build_propose_fns(db_path: Path):
    return {
        "propose_new": lambda v, a, f: propose_new_fn(v, a, f, db_path=db_path),
        "propose_merge": lambda v, a, f: propose_merge_fn(v, a, f, db_path=db_path),
        "propose_deprecate": lambda v, a, f: propose_deprecate_fn(v, a, f, db_path=db_path),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def serialize_state_history(history) -> list[dict]:
    out = []
    for rec in history:
        d = asdict(rec)
        # serialize Proposal dataclasses
        d["proposals"] = [
            ({"type": p.type, **{k: v for k, v in vars(p).items() if k != "type"}} if is_dataclass(p) else p)
            for p in d.get("proposals", [])
        ]
        out.append(d)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="outputs/tag_extraction_v2_vocab_v1.1.json", type=Path)
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument(
        "--cached-assignments",
        default="outputs/full_assignment/reverse_check_assignments.json",
        type=Path,
        help="prior full reverse_check output — used as initial assignment snapshot",
    )
    parser.add_argument("--output-dir", default="outputs/agent_e2e", type=Path)
    parser.add_argument("--max-iter", default=3, type=int)
    parser.add_argument("--sample-size", default=0, type=int, help="cap on cached records (0 = use all)")
    parser.add_argument("--concurrency", default=10, type=int)
    args = parser.parse_args()

    initial_vocab = json.loads(args.vocab.read_text(encoding="utf-8"))["vocab"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Agent E2E ===")
    print(f"  vocab: {args.vocab.name} ({len(initial_vocab)} tags)")
    print(f"  cache: {args.cached_assignments.name}  sample_cap: {args.sample_size or 'full'}  concurrency: {args.concurrency}")
    print(f"  max_iter: {args.max_iter}")
    print()

    measure_fn = build_measure_fn(args.db, args.cached_assignments, args.sample_size, args.concurrency)
    propose_fns = build_propose_fns(args.db)

    agent = Agent(
        measure_fn=measure_fn,
        diagnose_fn=diagnose_fn,
        propose_fns=propose_fns,
        max_iter=args.max_iter,
    )

    t0 = time.perf_counter()
    state, status = agent.run(initial_vocab)
    elapsed = time.perf_counter() - t0

    # Print summary
    print(f"\n{'='*60}\n=== Final ===\n{'='*60}")
    print(f"status: {status}")
    print(f"iters:  {state.iter}")
    print(f"elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"vocab:  {len(initial_vocab)} → {len(state.vocab)} tags")
    print(f"blocked actions: {sorted(state.blocked_actions)}")
    print()
    print("History:")
    for rec in state.history:
        print(f"  iter {rec.iter}: action={rec.action}  result={rec.result}  "
              f"hit_rate {rec.hit_rate_before:.2f}→{rec.hit_rate_after:.2f}  "
              f"proposals={len(rec.proposals)}  blocked_after={rec.blocked_actions_after}")
        if rec.error:
            print(f"    error: {rec.error}")
        if rec.decision and rec.decision.get("action_focus"):
            print(f"    focus: {rec.decision['action_focus'][:120]}")

    # Save
    report = {
        "status": status.value,
        "iter": state.iter,
        "elapsed_seconds": round(elapsed, 1),
        "initial_vocab_size": len(initial_vocab),
        "final_vocab_size": len(state.vocab),
        "blocked_actions": sorted(state.blocked_actions),
        "history": serialize_state_history(state.history),
    }
    (args.output_dir / "agent_run_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "final_vocab.json").write_text(
        json.dumps({"vocab": state.vocab}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nReport → {args.output_dir / 'agent_run_report.json'}")


if __name__ == "__main__":
    main()
