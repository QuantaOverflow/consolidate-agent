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

from langchain_core.prompts import ChatPromptTemplate

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model
from consolidate_agent.vocab_maintenance.agent import Agent, FinalStatus
from consolidate_agent.vocab_maintenance.diagnose import diagnose as diagnose_call
from consolidate_agent.vocab_maintenance.measure import (
    SYSTEM_PROMPT as RC_SYSTEM,
    USER_PROMPT as RC_USER,
    BatchAssignmentOutput,
    _run_single_batch,
    format_vocab,
    format_records,
    load_records,
)
from consolidate_agent.vocab_maintenance.propose.new import propose_new_fn
from consolidate_agent.vocab_maintenance.propose.merge import propose_merge_fn
from consolidate_agent.vocab_maintenance.propose.deprecate import propose_deprecate_fn


BASE = Path(__file__).resolve().parents[1]


# ── Wiring ────────────────────────────────────────────────────────────────────


def build_measure_fn(db_path: Path, sample_size: int, concurrency: int = 10):
    """Wrap reverse_check as an in-memory measure_fn for the agent."""
    import random as _r
    settings = Settings()
    records = load_records(db_path)
    rng = _r.Random(42)
    rng.shuffle(records)
    sample = records[:sample_size] if sample_size > 0 else records

    def measure_fn(vocab: list[dict]):
        model = _chat_model(settings).with_structured_output(BatchAssignmentOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", RC_SYSTEM),
            ("user", RC_USER),
        ])
        batch_size = 10
        total_batches = (len(sample) + batch_size - 1) // batch_size
        batches = []
        for i in range(total_batches):
            batches.append((i, sample[i * batch_size : (i + 1) * batch_size]))

        from concurrent.futures import ThreadPoolExecutor, as_completed
        results_by_idx: dict[int, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_run_single_batch, idx, recs, model, prompt, vocab, total_batches): idx
                       for idx, recs in batches}
            for fut in as_completed(futures):
                idx, batch_out, _ = fut.result()
                results_by_idx[idx] = batch_out

        assignments = [r for i in sorted(results_by_idx) for r in results_by_idx[i]]
        # Filter LLM-invented fake tags (not in vocab)
        # This is necessary because reverse_check's LLM occasionally outputs
        # tag names outside the provided vocabulary (~3% empirically).
        # Without this, check_invariants fails on dangling references.
        vocab_names = {t["name"] for t in vocab}
        fake_filtered = 0
        for a in assignments:
            original = a.get("selected_tags", []) or []
            filtered = [t for t in original if t["name"] in vocab_names]
            if len(filtered) < len(original):
                fake_filtered += len(original) - len(filtered)
            a["selected_tags"] = filtered
            # if no tags left → mark as missing
            if not filtered and not a.get("missing"):
                a["missing"] = True
                a["missing_concept"] = a.get("missing_concept") or "all selected_tags were vocab-external"
            elif a.get("missing") and not filtered:
                # keep missing=True, just ensured selected_tags=[]
                pass
        if fake_filtered:
            print(f"  [measure] filtered {fake_filtered} vocab-external fake tags", flush=True)
        diag = _build_diag(vocab, assignments)
        return diag, assignments

    return measure_fn


def _build_diag(vocab: list[dict], assignments: list[dict]) -> dict:
    from collections import Counter, defaultdict
    tag_usage: Counter[str] = Counter()
    cooccur: Counter[tuple[str, str]] = Counter()
    missing_records: list[dict] = []
    boundary_blur: list[dict] = []

    for a in assignments:
        if a.get("missing"):
            missing_records.append({"record_id": a["record_id"], "title": a.get("title", ""), "missing_concept": a.get("missing_concept", "")})
            continue
        tags = a.get("selected_tags", [])
        for t in tags:
            tag_usage[t["name"]] += 1
        names = [t["name"] for t in tags]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cooccur[tuple(sorted([names[i], names[j]]))] += 1
        confs = [t.get("confidence", "") for t in tags]
        if len(confs) >= 2 and len(set(confs)) == 1:
            boundary_blur.append({"record_id": a["record_id"], "title": a.get("title", "")})

    return {
        "sample_size": len(assignments),
        "total_assigned": sum(1 for a in assignments if not a.get("missing")),
        "total_missing": len(missing_records),
        "tag_usage_count": dict(tag_usage.most_common()),
        "unused_tags": sorted({t["name"] for t in vocab} - set(tag_usage)),
        "missing_records": missing_records,
        "boundary_blur_records": boundary_blur,
        "low_confidence_records": [],
        "top_cooccurrence_pairs": [{"pair": list(p), "count": c} for p, c in cooccur.most_common(15)],
    }


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
    parser.add_argument("--output-dir", default="outputs/agent_e2e", type=Path)
    parser.add_argument("--max-iter", default=3, type=int)
    parser.add_argument("--sample-size", default=200, type=int, help="reverse_check sample (200 = ~30s/measure)")
    parser.add_argument("--concurrency", default=10, type=int)
    args = parser.parse_args()

    initial_vocab = json.loads(args.vocab.read_text(encoding="utf-8"))["vocab"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Agent E2E ===")
    print(f"  vocab: {args.vocab.name} ({len(initial_vocab)} tags)")
    print(f"  sample_size: {args.sample_size}  concurrency: {args.concurrency}")
    print(f"  max_iter: {args.max_iter}")
    print()

    measure_fn = build_measure_fn(args.db, args.sample_size, args.concurrency)
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
