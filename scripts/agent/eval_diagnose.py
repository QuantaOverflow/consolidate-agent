"""Run diagnose() on each scenario fixture and score against expected.json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diagnose import diagnose  # noqa: E402


BASE = Path(__file__).resolve().parents[2]
FIXTURES = BASE / "tests/fixtures"
DB = BASE / "outputs/knowledge.db"


def load_scenario(scenario_dir: Path):
    return {
        "vocab": json.loads((scenario_dir / "vocab.json").read_text(encoding="utf-8"))["vocab"],
        "diagnostics": json.loads((scenario_dir / "diagnostics.json").read_text(encoding="utf-8")),
        "assignments": json.loads((scenario_dir / "assignments.json").read_text(encoding="utf-8")),
        "expected": json.loads((scenario_dir / "expected.json").read_text(encoding="utf-8")),
        "name": scenario_dir.name,
    }


def score_decision(decision: dict, probe_calls: list[str], expected: dict) -> dict:
    """Return per-criterion pass/fail."""
    action = decision["next_action"]
    focus = decision.get("action_focus", "")
    reasoning = decision.get("reasoning", "") + " " + decision.get("bias_assessment", "")
    probe_str = " ".join(probe_calls).lower()

    # 1. action acceptable
    acceptable = expected.get("acceptable_actions", [])
    rejected = expected.get("rejected_actions", [])
    action_ok = action in acceptable and action not in rejected

    # 2. required probes
    must_probe = expected.get("must_probe", [])
    probe_ok = all(p in probe_str for p in must_probe)

    # 3. forbidden merge pair
    forbidden_pair = expected.get("must_NOT_propose_merge_pair")
    if forbidden_pair and action == "propose_merge":
        # check focus mentions this pair
        forbidden_violation = (
            forbidden_pair[0].lower() in focus.lower()
            and forbidden_pair[1].lower() in focus.lower()
        )
        merge_safety_ok = not forbidden_violation
    else:
        merge_safety_ok = True

    # 4. probe target mention
    target_keyword = expected.get("probe_target_must_include")
    if target_keyword:
        target_ok = target_keyword.lower() in probe_str or target_keyword.lower() in focus.lower()
    else:
        target_ok = True

    # 5. reasoning mentions key concepts
    must_mention = expected.get("reasoning_must_mention", [])
    full_text = (reasoning + " " + focus).lower()
    mention_count = sum(1 for kw in must_mention if kw.lower() in full_text)
    mention_ratio = mention_count / len(must_mention) if must_mention else 1.0

    # 6. confidence calibration
    expected_confidence = expected.get("expected_confidence", ["high", "medium", "low"])
    predicted_confidence = decision.get("confidence", "high")
    confidence_ok = predicted_confidence in expected_confidence

    passed_count = sum([action_ok, probe_ok, merge_safety_ok, target_ok, mention_ratio >= 0.5, confidence_ok])
    total_criteria = 6

    return {
        "action_ok": action_ok,
        "probe_ok": probe_ok,
        "merge_safety_ok": merge_safety_ok,
        "target_ok": target_ok,
        "mention_ratio": round(mention_ratio, 2),
        "confidence_ok": confidence_ok,
        "predicted_confidence": predicted_confidence,
        "score": passed_count / total_criteria,
        "predicted_action": action,
    }


def run(scenarios: list[str] | None = None):
    scenario_dirs = sorted([d for d in FIXTURES.iterdir() if d.is_dir() and d.name.startswith("scenario_")])
    if scenarios:
        scenario_dirs = [d for d in scenario_dirs if d.name in scenarios]

    all_results = []
    for sd in scenario_dirs:
        scenario = load_scenario(sd)
        print(f"\n{'='*70}")
        print(f"=== {scenario['name']} ===")
        print(f"{'='*70}")
        print(f"Description: {scenario['expected'].get('description', '')}")
        print(f"Acceptable actions: {scenario['expected'].get('acceptable_actions', [])}")
        print(f"Rejected actions:   {scenario['expected'].get('rejected_actions', [])}")
        print()

        t0 = time.perf_counter()
        try:
            result = diagnose(
                scenario["vocab"],
                scenario["diagnostics"],
                scenario["assignments"],
                DB,
            )
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_results.append({"scenario": scenario["name"], "error": str(e)})
            continue
        elapsed = time.perf_counter() - t0

        decision = result["decision"]
        probe_calls = result["probe_calls"]

        print(f"⏱  elapsed: {elapsed:.1f}s")
        print(f"\n📋 Plan:")
        print(f"   raw_signals: {result['plan']['raw_signals_observed'][:200]}")
        print(f"   biases:      {result['plan']['suspected_biases'][:200]}")
        print(f"\n🔍 Probes:")
        for p in probe_calls:
            print(f"   - {p}")
        print(f"\n🎯 Decision:")
        print(f"   next_action:  {decision['next_action']}")
        print(f"   focus:        {decision['action_focus'][:200]}")
        print(f"   probe_findings: {decision['probe_findings_summary'][:200]}")
        print(f"   bias_assessment: {decision['bias_assessment'][:200]}")
        print(f"   reasoning:    {decision['reasoning'][:200]}")

        score = score_decision(decision, probe_calls, scenario["expected"])
        symbol = "✅" if score["score"] >= 0.8 else ("⚠️" if score["score"] >= 0.5 else "❌")
        print(f"\n{symbol} Score: {score['score']:.0%}")
        print(f"   action_ok:        {'✓' if score['action_ok'] else '✗'} (predicted={score['predicted_action']})")
        print(f"   confidence_ok:    {'✓' if score['confidence_ok'] else '✗'} (predicted={score['predicted_confidence']}, expected={scenario['expected'].get('expected_confidence')})")
        print(f"   probe_ok:         {'✓' if score['probe_ok'] else '✗'}")
        print(f"   merge_safety_ok:  {'✓' if score['merge_safety_ok'] else '✗'}")
        print(f"   target_ok:        {'✓' if score['target_ok'] else '✗'}")
        print(f"   mention_ratio:    {score['mention_ratio']}")
        if decision.get('uncertainty_reasons'):
            print(f"   uncertainty_reasons:")
            for r in decision['uncertainty_reasons'][:3]:
                print(f"     - {r[:150]}")

        all_results.append({
            "scenario": scenario["name"],
            "score": score,
            "decision": decision,
            "probe_calls": probe_calls,
            "elapsed": elapsed,
        })

    # Summary
    print(f"\n\n{'='*70}")
    print("=== Summary ===")
    print(f"{'='*70}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['scenario']:35s} ❌ ERROR: {r['error'][:60]}")
        else:
            s = r["score"]
            symbol = "✅" if s["score"] >= 0.8 else ("⚠️" if s["score"] >= 0.5 else "❌")
            print(f"  {r['scenario']:35s} {symbol} {s['score']:.0%}  action={s['predicted_action']:18s}")

    n_passed = sum(1 for r in all_results if "score" in r and r["score"]["score"] >= 0.8)
    n_total = len(all_results)
    print(f"\nOverall: {n_passed}/{n_total} passed")

    out = FIXTURES / "eval_results.json"
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Details → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="*", help="specific scenario names to run")
    args = parser.parse_args()
    run(args.scenarios)


if __name__ == "__main__":
    main()
