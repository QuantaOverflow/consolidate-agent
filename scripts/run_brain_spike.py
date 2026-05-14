#!/usr/bin/env python3
"""Phase F spike: run the brain loop on network_v3_p1.json.

Usage:
    python scripts/run_brain_spike.py [--network outputs/network_v3_p1.json] [--max-rounds 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project src is on path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from consolidate_agent.config import Settings
from consolidate_agent.vocab_maintenance.agent_brain.graph import run_brain_graph
from consolidate_agent.vocab_maintenance.observability import RunLogger, set_default_logger

GOLDEN_PATH = Path(__file__).parent.parent / "tests/fixtures/golden_80.jsonl"
DEFAULT_NETWORK = Path("outputs/network_v3_p1.json")


def load_network(path: Path) -> tuple[list[dict], list[dict], Path]:
    """Load network JSON, extract vocab (matter facet) + assignments in v1-view."""
    with open(path) as f:
        data = json.load(f)

    # Extract matter facet vocab
    facets = data["vocab"]["facets"]
    matter_tags: list[dict] = facets["matter"]["tags"]

    # Convert assignments: matter_tags → selected_tags (v1-view)
    raw_assignments = data["assignments"]
    assignments: list[dict] = []
    for a in raw_assignments:
        selected = a.get("matter_tags", [])
        assignments.append({
            "record_id": a["record_id"],
            "selected_tags": selected,
            "lesson_type": a.get("lesson_type", ""),
            "missing": a.get("missing", False),
        })

    # db path from settings
    settings = Settings()
    db_path = settings.knowledge_db_path

    return matter_tags, assignments, db_path


def load_golden(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Warning: golden file not found at {path}, using empty golden set")
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_timeline(result: dict) -> str:
    lines = ["=== Phase F Spike ==="]
    decisions = result["decisions"]

    # Group decisions by round
    rounds_seen: set[int] = set()
    decisions_by_round: dict[int, dict] = {}
    for d in decisions:
        r = d["round"]
        decisions_by_round[r] = d
        rounds_seen.add(r)

    total_rounds = result["rounds"]
    for r in range(total_rounds):
        d = decisions_by_round.get(r)
        if d is None:
            lines.append(f"Round {r + 1}: (no decision)")
            continue

        dec = d["decision"]
        gate = d["gate"]
        gate_str = gate["result"].upper()
        action = dec["action"]
        target = dec.get("target", "") or ""
        certainty = dec.get("certainty", "?")
        committed = d.get("committed", False)
        outcome = d.get("outcome", "")

        if action == "stop":
            lines.append(f"Round {r + 1}: STOP — cert={certainty}, gate={gate_str}")
        elif committed:
            lines.append(
                f"Round {r + 1}: {action} {target} (cert={certainty}) — gate={gate_str} → APPLIED"
                + (f"\n  outcome: {outcome}" if outcome else "")
            )
        else:
            lines.append(
                f"Round {r + 1}: {action} {target} (cert={certainty}) — gate={gate_str}"
                + (f" | {outcome}" if outcome else "")
            )

    total_calls = result["total_llm_calls"]
    applied = result["applied_count"]
    # rough cost estimate: qwen-flash ~$0.0005/1k tokens, ~1k tokens/call
    cost_est = round(total_calls * 0.001 * 0.5, 3)
    lines.append(
        f"\nTotal: {total_rounds} rounds, {total_calls} LLM calls (~${cost_est} estimated), "
        f"{applied} commits, 0 rollbacks"
    )
    lines.append(f"Stop reason: {result['stop_reason']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase F brain loop spike")
    parser.add_argument("--network", default=str(DEFAULT_NETWORK), help="Path to network JSON")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max rounds (default 5 for spike)")
    args = parser.parse_args()

    network_path = Path(args.network)
    if not network_path.exists():
        print(f"Error: network file not found: {network_path}")
        sys.exit(1)

    print(f"Loading network from {network_path}...")
    vocab, assignments, db_path = load_network(network_path)
    print(f"  vocab: {len(vocab)} matter tags")
    print(f"  assignments: {len(assignments)} records")
    print(f"  db_path: {db_path}")

    print(f"Loading golden set from {GOLDEN_PATH}...")
    golden = load_golden(GOLDEN_PATH)
    print(f"  golden: {len(golden)} records")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    trace_path = outputs_dir / f"spike_trace_{ts}.jsonl"
    summary_path = outputs_dir / f"spike_summary_{ts}.json"

    logger = RunLogger(trace_path)
    set_default_logger(logger)

    print(f"\nStarting brain graph (max_rounds={args.max_rounds})...")
    t_start = time.perf_counter()

    result = run_brain_graph(
        db_path=db_path,
        vocab=vocab,
        assignments=assignments,
        max_rounds=args.max_rounds,
        max_tools_per_round=3,
        cost_cap_calls=60,
        logger=logger,
    )

    elapsed = round(time.perf_counter() - t_start, 1)
    logger.close()

    # Persist final vocab + assignments snapshot (post-apply state) so judges
    # and follow-up spikes can diff before/after or load the new network.
    final_vocab = result.pop("_final_vocab", [])
    final_assignments = result.pop("_final_assignments", [])
    vocab_path = outputs_dir / f"spike_vocab_{ts}.json"
    with open(vocab_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "source_network": str(network_path),
            "vocab": final_vocab,
            "assignments": final_assignments,
        }, f, indent=2, default=str)

    # Write summary JSON (without the heavy vocab/assignments payload)
    summary = {
        "timestamp": ts,
        "network": str(network_path),
        "max_rounds": args.max_rounds,
        "elapsed_s": elapsed,
        "final_vocab_snapshot": str(vocab_path),
        **result,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nTrace: {trace_path}")
    print(f"Summary: {summary_path}")
    print(f"Final vocab: {vocab_path}")
    print()
    print(format_timeline(result))


if __name__ == "__main__":
    main()
