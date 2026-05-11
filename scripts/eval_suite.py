"""Evaluate propose_merge and propose_deprecate against ground truth.

Loads testset.json (human-labeled expected decisions), runs each workflow
on the test cases, computes accuracy + confusion matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model

sys.path.insert(0, str(Path(__file__).parent))
from propose_merge import (  # noqa: E402
    MergeJudgement,
    SYSTEM_PROMPT as MERGE_SYSTEM,
    USER_PROMPT as MERGE_USER,
    compute_cooccurrence,
    load_record_details,
    format_evidence,
    judge_pair,
)
from propose_deprecate import (  # noqa: E402
    DeprecateJudgement,
    SYSTEM_PROMPT as DEPRECATE_SYSTEM,
    USER_PROMPT as DEPRECATE_USER,
    compute_tag_records,
    judge_tag,
)


def load_vocab(path: Path) -> tuple[list[dict], dict[str, dict]]:
    vocab = json.loads(path.read_text(encoding="utf-8"))["vocab"]
    return vocab, {t["name"]: t for t in vocab}


def eval_propose_merge(
    cases: list[dict],
    vocab: list[dict],
    vocab_by_name: dict[str, dict],
    assignments_path: Path,
    db_path: Path,
) -> dict:
    settings = Settings()
    model = _chat_model(settings).with_structured_output(MergeJudgement)
    prompt = ChatPromptTemplate.from_messages([("system", MERGE_SYSTEM), ("user", MERGE_USER)])

    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    pair_counts, tag_record_ids, tag_usage = compute_cooccurrence(assignments)

    results = []
    print(f"\n=== Evaluating propose_merge ({len(cases)} cases) ===\n", flush=True)

    for idx, case in enumerate(cases, 1):
        a, b = case["tag_a"], case["tag_b"]
        if a not in vocab_by_name or b not in vocab_by_name:
            print(f"  [{idx}] {a} + {b}: ❌ vocab missing", flush=True)
            results.append({"case": case, "predicted": "error", "correct": False, "elapsed": 0})
            continue

        pair = tuple(sorted([a, b]))
        cooccur = pair_counts.get(pair, 0)
        shared = list(tag_record_ids.get(a, set()) & tag_record_ids.get(b, set()))[:5]
        details = load_record_details(db_path, shared)
        evidence = [details[rid] for rid in shared if rid in details]

        t0 = time.perf_counter()
        result = judge_pair(model, prompt, vocab_by_name[a], vocab_by_name[b], tag_usage, cooccur, evidence)
        elapsed = time.perf_counter() - t0

        if result is None:
            print(f"  [{idx}] {a} + {b}: ❌ LLM None ({elapsed:.1f}s)", flush=True)
            results.append({"case": case, "predicted": "error", "correct": False, "elapsed": elapsed})
            continue

        predicted = result.decision
        expected = case["expected"]
        correct = (predicted == expected)
        symbol = "✅" if correct else "❌"
        cooc_info = f"(co-occur {cooccur})"

        print(f"  [{idx}] {a} + {b} {cooc_info}: {symbol} pred={predicted} | exp={expected} ({elapsed:.1f}s)", flush=True)
        if not correct:
            print(f"      LLM reasoning: {result.reasoning[:150]}", flush=True)
            print(f"      Reviewer notes: {case.get('notes', '')}", flush=True)

        results.append({
            "case": case,
            "predicted": predicted,
            "correct": correct,
            "llm_reasoning": result.reasoning,
            "llm_keep_tag": result.keep_tag if predicted == "merge" else None,
            "elapsed": elapsed,
        })

    return summarize_results("propose_merge", results)


def eval_propose_deprecate(
    cases: list[dict],
    vocab: list[dict],
    vocab_by_name: dict[str, dict],
    assignments_path: Path,
    db_path: Path,
) -> dict:
    settings = Settings()
    model = _chat_model(settings).with_structured_output(DeprecateJudgement)
    prompt = ChatPromptTemplate.from_messages([("system", DEPRECATE_SYSTEM), ("user", DEPRECATE_USER)])

    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    tag_records = compute_tag_records(assignments)

    results = []
    print(f"\n=== Evaluating propose_deprecate ({len(cases)} cases) ===\n", flush=True)

    for idx, case in enumerate(cases, 1):
        tag_name = case["tag"]
        if tag_name not in vocab_by_name:
            print(f"  [{idx}] {tag_name}: ❌ vocab missing", flush=True)
            results.append({"case": case, "predicted": "error", "correct": False, "elapsed": 0})
            continue

        record_ids = tag_records.get(tag_name, [])[:8]
        details = load_record_details(db_path, record_ids)
        evidence = [details[rid] for rid in record_ids if rid in details]
        usage = len(tag_records.get(tag_name, []))

        t0 = time.perf_counter()
        result = judge_tag(model, prompt, vocab_by_name[tag_name], vocab, evidence, usage)
        elapsed = time.perf_counter() - t0

        if result is None:
            print(f"  [{idx}] {tag_name}: ❌ LLM None ({elapsed:.1f}s)", flush=True)
            results.append({"case": case, "predicted": "error", "correct": False, "elapsed": elapsed})
            continue

        predicted = result.decision
        expected = case["expected"]
        correct = (predicted == expected)
        # for merge_to, also check target matches if expected
        if correct and predicted == "merge_to" and case.get("merge_target"):
            if result.merge_target != case["merge_target"]:
                correct = False  # right decision but wrong target

        symbol = "✅" if correct else "❌"
        target_info = f" → {result.merge_target}" if predicted == "merge_to" else ""

        print(f"  [{idx}] {tag_name} (size {usage}): {symbol} pred={predicted}{target_info} | exp={expected}", flush=True)
        if not correct:
            print(f"      LLM reasoning: {result.reasoning[:150]}", flush=True)
            print(f"      Reviewer notes: {case.get('notes', '')}", flush=True)

        results.append({
            "case": case,
            "predicted": predicted,
            "correct": correct,
            "llm_reasoning": result.reasoning,
            "llm_merge_target": result.merge_target if predicted == "merge_to" else None,
            "elapsed": elapsed,
        })

    return summarize_results("propose_deprecate", results)


def summarize_results(workflow_name: str, results: list[dict]) -> dict:
    n_total = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_error = sum(1 for r in results if r["predicted"] == "error")
    accuracy = n_correct / n_total if n_total else 0.0

    # confusion matrix
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for r in results:
        confusion[(r["case"]["expected"], r["predicted"])] += 1

    return {
        "workflow": workflow_name,
        "total": n_total,
        "correct": n_correct,
        "incorrect": n_total - n_correct - n_error,
        "errors": n_error,
        "accuracy": accuracy,
        "confusion": [{"expected": k[0], "predicted": k[1], "count": v} for k, v in confusion.items()],
        "results": results,
    }


def print_summary(summary: dict) -> None:
    print(f"\n--- {summary['workflow']} summary ---")
    print(f"  Total:     {summary['total']}")
    print(f"  Correct:   {summary['correct']}")
    print(f"  Incorrect: {summary['incorrect']}")
    print(f"  Errors:    {summary['errors']}")
    print(f"  Accuracy:  {summary['accuracy']:.2%}")
    print(f"  Confusion matrix:")
    for c in summary["confusion"]:
        marker = "✓" if c["expected"] == c["predicted"] else "✗"
        print(f"    {marker}  expected={c['expected']:15s}  predicted={c['predicted']:15s}  count={c['count']}")


def run(
    testset_path: Path,
    vocab_path: Path,
    assignments_path: Path,
    db_path: Path,
    output_path: Path,
) -> None:
    testset = json.loads(testset_path.read_text(encoding="utf-8"))
    vocab, vocab_by_name = load_vocab(vocab_path)

    merge_summary = eval_propose_merge(
        testset["propose_merge_cases"], vocab, vocab_by_name, assignments_path, db_path,
    )
    print_summary(merge_summary)

    deprecate_summary = eval_propose_deprecate(
        testset["propose_deprecate_cases"], vocab, vocab_by_name, assignments_path, db_path,
    )
    print_summary(deprecate_summary)

    output_path.write_text(
        json.dumps({"propose_merge": merge_summary, "propose_deprecate": deprecate_summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✅ Full report → {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="scripts/testset.json", type=Path)
    parser.add_argument("--vocab", default="outputs/tag_extraction_v2_vocab_v1.1.json", type=Path)
    parser.add_argument("--assignments", default="outputs/full_assignment/reverse_check_assignments.json", type=Path)
    parser.add_argument("--db", default="outputs/knowledge.db", type=Path)
    parser.add_argument("--output", default="outputs/full_assignment/eval_report.json", type=Path)
    args = parser.parse_args()

    run(args.testset, args.vocab, args.assignments, args.db, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
