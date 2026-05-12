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
from consolidate_agent.vocab_maintenance.health import measure_health
from consolidate_agent.vocab_maintenance.judge import llm_judge
from consolidate_agent.vocab_maintenance.measure import (
    build_diagnostics,
    load_records,
    reverse_check_subset,
)
from consolidate_agent.vocab_maintenance.network import TagRecordNetwork
from consolidate_agent.vocab_maintenance.observability import RunLogger, set_default_logger
from consolidate_agent.vocab_maintenance.propose.merge import propose_merge_fn
from consolidate_agent.vocab_maintenance.propose.deprecate import propose_deprecate_fn
from consolidate_agent.vocab_maintenance.propose.refine import propose_refine_fn


BASE = Path(__file__).resolve().parents[1]


# ── Wiring ────────────────────────────────────────────────────────────────────


def build_measure_fn(
    network: TagRecordNetwork,
    db_path: Path,
    concurrency: int = 10,
):
    """Network-driven measure_fn (no LLM for initial / merge / deprecate).

    Initial call (action=None): return network's current diagnostics + assignments.
    After merge/deprecate apply: zero LLM — apply.py has already updated
      assignments deterministically; we just re-aggregate diagnostics.
    After propose_new apply: LLM patch only the records that were `missing`
      in the previous round (the new tag may now cover them).
    """
    records_by_id = {r["record_id"]: r for r in load_records(db_path)}
    print(f"  [measure] network has {len(network.assignments)} assignments, {len(network.vocab)} tags", flush=True)

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
            # Initial call: use network's snapshot directly (no LLM).
            assignments = [dict(a) for a in network.assignments]
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


def build_health_fn(db_path: Path):
    """Closure capturing db_path so the graph layer doesn't need to know it."""
    def health_fn(vocab: list[dict], assignments: list[dict]):
        return measure_health(vocab, assignments, db_path)
    return health_fn


def build_propose_fns(db_path: Path):
    """Maintenance agent's tool kit: merge + deprecate + refine.

    propose_new is intentionally absent — vocab growth belongs to ingest_batch,
    not to the maintenance loop. Maintenance is consolidation-only.
    propose_refine sharpens a tag's definition + prunes <=10 misfit records.
    """
    return {
        "propose_merge": lambda v, a, f: propose_merge_fn(v, a, f, db_path=db_path),
        "propose_deprecate": lambda v, a, f: propose_deprecate_fn(v, a, f, db_path=db_path),
        "propose_refine": lambda v, a, f: propose_refine_fn(v, a, f, db_path=db_path),
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
    parser.add_argument("--network", default="outputs/network.json", type=Path,
                        help="path to canonical TagRecordNetwork snapshot")
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument(
        "--legacy-vocab",
        default="outputs/tag_extraction_v2_vocab_v1.1.json",
        type=Path,
        help="(migration only) legacy vocab file — used if --network doesn't exist",
    )
    parser.add_argument(
        "--legacy-assignments",
        default="outputs/full_assignment/reverse_check_assignments.json",
        type=Path,
        help="(migration only) legacy assignments file — used if --network doesn't exist",
    )
    parser.add_argument("--output-dir", default="outputs/agent_e2e", type=Path)
    parser.add_argument("--max-iter", default=3, type=int)
    parser.add_argument("--concurrency", default=10, type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up run log (one jsonl per invocation)
    log_path = BASE / "outputs" / "runs" / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}_agent_e2e.jsonl"
    run_logger = RunLogger(log_path)
    set_default_logger(run_logger)
    run_logger.event("run.start", script="run_agent_e2e", network=str(args.network))
    print(f"  run log → {log_path}")

    t_run = time.perf_counter()
    try:
        # Load network (or migrate from legacy split files on first run)
        if args.network.exists():
            network = TagRecordNetwork.load(args.network)
            print(f"  loaded network from {args.network}")
        else:
            print(f"  network not found at {args.network}, migrating from legacy files…")
            network = TagRecordNetwork.from_legacy_files(args.legacy_vocab, args.legacy_assignments)
            network.save(args.network)
            print(f"  migrated → saved to {args.network}")

        initial_vocab_size = len(network.vocab)
        initial_assignment_count = len(network.assignments)

        print(f"=== Agent E2E ===")
        print(f"  network: vocab={initial_vocab_size} tags, assignments={initial_assignment_count} records")
        print(f"  concurrency: {args.concurrency}  max_iter: {args.max_iter}")
        print()

        measure_fn = build_measure_fn(network, args.db, args.concurrency)
        propose_fns = build_propose_fns(args.db)

        agent = Agent(
            measure_fn=measure_fn,
            diagnose_fn=diagnose_fn,
            propose_fns=propose_fns,
            max_iter=args.max_iter,
            disabled_actions={"propose_new"},  # growth handled by ingest_batch
            health_fn=build_health_fn(args.db),
            judge_fn=llm_judge,
        )

        t0 = time.perf_counter()
        state, status = agent.run(network.vocab)
        elapsed = time.perf_counter() - t0

        # Mirror final agent state back into network and persist atomically
        network.vocab = state.vocab
        network.assignments = state.assignments
        network.metadata["last_agent_run"] = {
            "status": status.value,
            "iter": state.iter,
            "elapsed_seconds": round(elapsed, 1),
        }
        network.save(args.network)

        # Print summary
        print(f"\n{'='*60}\n=== Final ===\n{'='*60}")
        print(f"status: {status}")
        print(f"iters:  {state.iter}")
        print(f"elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"vocab:  {initial_vocab_size} → {len(state.vocab)} tags")
        print(f"blocked actions: {sorted(state.blocked_actions)}")
        print(f"network saved → {args.network}")
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

        # Run report (audit trail; the canonical state lives in --network)
        report = {
            "status": status.value,
            "iter": state.iter,
            "elapsed_seconds": round(elapsed, 1),
            "initial_vocab_size": initial_vocab_size,
            "final_vocab_size": len(state.vocab),
            "blocked_actions": sorted(state.blocked_actions),
            "history": serialize_state_history(state.history),
        }
        (args.output_dir / "agent_run_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nReport → {args.output_dir / 'agent_run_report.json'}")
        run_logger.event(
            "run.done",
            elapsed_s=round(elapsed, 1),
            status=status.value,
            iter=state.iter,
            vocab_before=initial_vocab_size,
            vocab_after=len(state.vocab),
        )
    except Exception as e:  # noqa: BLE001 — top-level audit; re-raised below
        import traceback
        run_logger.event(
            "run.error",
            elapsed_s=round(time.perf_counter() - t_run, 1),
            error_type=type(e).__name__,
            error=str(e)[:500],
            traceback=traceback.format_exc()[:3000],
        )
        raise
    finally:
        run_logger.close()


if __name__ == "__main__":
    main()
