"""Bootstrap a TagRecordNetwork from raw records, with optional HITL review.

Default flow: distill themes → synthesize vocab → [pause for review] → reverse_check.
Use --auto-accept to skip the review gate (unattended / smoke runs).

Resume support: if a previous run crashed, pass --thread-id <same id> and the
graph picks up from the last checkpoint in --checkpoint-db.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from consolidate_agent.vocab_maintenance.network import TagRecordNetwork
from consolidate_agent.vocab_maintenance.observability import RunLogger, set_default_logger


BASE = Path(__file__).resolve().parents[1]


def review_vocab_interactive(vocab: list[dict]) -> str:
    print(f"\n{'='*60}\n=== Vocab review ({len(vocab)} tags) ===\n{'='*60}")
    for i, t in enumerate(vocab, 1):
        print(f"  {i:>3}. {t['name']:30s} {t['definition'][:90]}")
    print()
    while True:
        choice = input("[a]ccept / [r]egenerate / [q]uit-abort? ").strip().lower()
        if choice in ("a", "accept"):
            return "accept"
        if choice in ("r", "regenerate"):
            return "regenerate"
        if choice in ("q", "quit", "abort"):
            return "abort"
        print("invalid — enter a / r / q")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--network-out", default="outputs/network.json", type=Path)
    parser.add_argument("--checkpoint-db", default="outputs/checkpoints.db", type=Path)
    parser.add_argument("--thread-id", default=None, type=str,
                        help="reuse to resume a crashed bootstrap")
    parser.add_argument("--batch-size", default=30, type=int, help="distill batch size")
    parser.add_argument("--concurrency", default=10, type=int)
    parser.add_argument("--auto-accept", action="store_true",
                        help="skip vocab review (unattended)")
    args = parser.parse_args()

    log_path = BASE / "outputs/runs" / f"{time.strftime('%Y-%m-%dT%H-%M-%S')}_bootstrap.jsonl"
    run_logger = RunLogger(log_path)
    set_default_logger(run_logger)
    run_logger.event("run.start", script="run_bootstrap",
                     db=str(args.db), thread_id=args.thread_id, auto_accept=args.auto_accept)

    print(f"=== Bootstrap ===")
    print(f"  db:            {args.db}")
    print(f"  network out:   {args.network_out}")
    print(f"  checkpoint db: {args.checkpoint_db}")
    print(f"  thread_id:     {args.thread_id or '(auto-generated)'}")
    print(f"  auto_accept:   {args.auto_accept}")
    print(f"  run log:       {log_path}")
    print()

    on_review = None if args.auto_accept else review_vocab_interactive

    t0 = time.perf_counter()
    try:
        network = TagRecordNetwork.bootstrap(
            args.db,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            on_vocab_review=on_review,
            thread_id=args.thread_id,
            checkpoint_db=args.checkpoint_db,
        )
    except ValueError as e:
        if "aborted" in str(e):
            print(f"\n  bootstrap aborted by user.")
            run_logger.event("run.aborted", reason=str(e))
            run_logger.close()
            return 1
        raise

    elapsed = time.perf_counter() - t0
    network.save(args.network_out)

    print(f"\n{'='*60}\n=== Done ({elapsed:.1f}s = {elapsed/60:.1f} min) ===\n{'='*60}")
    print(f"  vocab:       {len(network.vocab)} tags")
    print(f"  assignments: {len(network.assignments)} records")
    print(f"  hit_rate:    {network.hit_rate():.3f}")
    print(f"  thread_id:   {network.metadata.get('bootstrap_thread_id')}")
    print(f"  attempts:    {network.metadata.get('synthesize_attempts', 1)} (synthesize)")
    print(f"  network →    {args.network_out}")

    run_logger.event("run.done", elapsed_s=round(elapsed, 1),
                     vocab_size=len(network.vocab),
                     assignments=len(network.assignments),
                     hit_rate=round(network.hit_rate(), 4))
    run_logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
