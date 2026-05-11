"""Construct 5 diagnostic test scenarios from real v1.1 data.

Each scenario directory contains:
  - vocab.json
  - diagnostics.json
  - assignments.json
  - expected.json    # ground truth for eval
"""
from __future__ import annotations

import copy
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parents[2]
SRC_VOCAB = BASE / "outputs/tag_extraction_v2_vocab_v1.1.json"
SRC_DIAG = BASE / "outputs/full_assignment/reverse_check_diagnostics.json"
SRC_ASSIGN = BASE / "outputs/full_assignment/reverse_check_assignments.json"
DST = BASE / "tests/fixtures"


def load_base() -> tuple[dict, dict, list]:
    return (
        json.loads(SRC_VOCAB.read_text(encoding="utf-8")),
        json.loads(SRC_DIAG.read_text(encoding="utf-8")),
        json.loads(SRC_ASSIGN.read_text(encoding="utf-8")),
    )


def save_scenario(name: str, vocab: dict, diag: dict, assign: list, expected: dict) -> None:
    d = DST / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False))
    (d / "diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False))
    (d / "assignments.json").write_text(json.dumps(assign, indent=2, ensure_ascii=False))
    (d / "expected.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False))
    print(f"  ✅ {name}/")


# ── Scenario 1: real v1.1 (no modification) ──────────────────────────────────

def scenario_1_real():
    """Baseline: real v1.1. 60 missing but most are false positives.
    Agent should NOT blindly propose_new — should probe missing first."""
    vocab, diag, assign = load_base()
    expected = {
        "name": "scenario_1_real_v1.1",
        "description": "Real v1.1 data — 60 missing but ~77% are LLM false negatives (verified earlier)",
        "acceptable_actions": ["propose_new", "done"],
        "rejected_actions": ["propose_merge", "propose_deprecate"],
        "expected_confidence": ["medium", "high"],
        "must_probe": ["inspect_missing_records"],
        "must_NOT_propose_merge_pair": ["llm_integration", "prompt_resilience"],
        "reasoning_must_mention": ["missing", "false positive"],
        "notes": "Critical: agent must recognize that high missing_rate (8.6%) ≠ real gap. Must probe missing records to verify before proposing new tags."
    }
    save_scenario("scenario_1_real", vocab, diag, assign, expected)


# ── Scenario 2: unused fake tag ──────────────────────────────────────────────

def scenario_2_unused_tag():
    """Inject a clearly unused niche tag. Agent should propose_deprecate it."""
    vocab, diag, assign = load_base()
    fake_tag = {
        "name": "obscure_xml_quirk",
        "definition": "Niche XML namespace ordering pitfall affecting only legacy SOAP gateways.",
    }
    vocab["vocab"].append(fake_tag)
    diag["unused_tags"] = ["obscure_xml_quirk"]  # 全量看就是 unused
    # assignments 不变，自然没有这个 tag
    expected = {
        "name": "scenario_2_unused_tag",
        "description": "Added fake niche tag with 0 records in assignments",
        "acceptable_actions": ["propose_deprecate", "propose_new"],
        "rejected_actions": ["propose_merge", "done"],
        "expected_confidence": ["high"],
        "must_probe": ["inspect_tag"],
        "probe_target_must_include": "obscure_xml_quirk",
        "reasoning_must_mention": ["obscure_xml_quirk", "unused"],
        "notes": "unused tag obvious; either deprecate it or address missing first (both defensible). Should be high confidence."
    }
    save_scenario("scenario_2_unused", vocab, diag, assign, expected)


# ── Scenario 3: split (artificial near-synonym) ──────────────────────────────

def scenario_3_split_synonym():
    """Add a near-synonym of state_isolation, then move half of state_isolation's
    records to it. Agent should detect the redundancy and propose_merge."""
    vocab, diag, assign = load_base()
    new_tag = {
        "name": "state_namespacing",
        "definition": "Techniques to scope shared state via namespaces, preventing coupling between independent components.",
    }
    vocab["vocab"].append(new_tag)

    # Move half of state_isolation assignments to state_namespacing
    state_iso_records = [a for a in assign if not a["missing"]
                         and any(t["name"] == "state_isolation" for t in a["selected_tags"])]
    rng = random.Random(42)
    rng.shuffle(state_iso_records)
    to_rename = state_iso_records[:len(state_iso_records) // 2]
    for a in to_rename:
        for t in a["selected_tags"]:
            if t["name"] == "state_isolation":
                t["name"] = "state_namespacing"

    # Recompute diagnostics
    diag = _recompute_diag(vocab, diag, assign)

    expected = {
        "name": "scenario_3_split_synonym",
        "description": "Injected state_namespacing as near-synonym of state_isolation, 50% records moved",
        "acceptable_actions": ["propose_merge", "done", "propose_new"],
        "rejected_actions": ["propose_deprecate"],
        "expected_confidence": ["low", "medium"],
        "must_probe": ["state_namespacing", "state_isolation"],
        "probe_pair_must_include": ["state_isolation", "state_namespacing"],
        "reasoning_must_mention": ["state_isolation", "state_namespacing"],
        "notes": "Borderline case: defs look similar (0.82) but LLM can find legitimate distinctions. Agent should mark low/medium confidence to flag for human review."
    }
    save_scenario("scenario_3_split", vocab, diag, assign, expected)


# ── Scenario 4: misleading cooccur ────────────────────────────────────────────

def scenario_4_fake_cooccur():
    """Force cli_design and git_workflow to high cooccur via fake co-tagging.
    These tags are legitimately distinct domains — agent should probe and
    realize the records actually span multiple legitimate axes, not synonymy."""
    vocab, diag, assign = load_base()

    # Find records tagged cli_design, force-add git_workflow tag to many
    rng = random.Random(42)
    candidates = [a for a in assign if not a["missing"]
                  and any(t["name"] == "cli_design" for t in a["selected_tags"])]
    rng.shuffle(candidates)
    # Force-add git_workflow to first 15
    for a in candidates[:15]:
        existing = {t["name"] for t in a["selected_tags"]}
        if "git_workflow" not in existing:
            a["selected_tags"].append({"name": "git_workflow", "confidence": "medium"})

    diag = _recompute_diag(vocab, diag, assign)

    expected = {
        "name": "scenario_4_fake_cooccur",
        "description": "Forced cli_design + git_workflow cooccurrence by adding git_workflow tag to 15 cli records (legitimately distinct domains)",
        "acceptable_actions": ["propose_new", "done"],  # NOT propose_merge!
        "rejected_actions": ["propose_merge"],
        "expected_confidence": ["high", "medium"],
        "must_probe": ["inspect_cooccur_pair"],
        "probe_pair_must_include": ["cli_design", "git_workflow"],
        "reasoning_must_mention": ["different domain", "distinct"],
        "must_NOT_propose_merge_pair": ["cli_design", "git_workflow"],
        "notes": "CRITICAL bias test: agent must probe high cooccur and recognize false signal. Should be high/medium confidence on the rejection."
    }
    save_scenario("scenario_4_fake_cooccur", vocab, diag, assign, expected)


# ── Scenario 5: healthy vocab ────────────────────────────────────────────────

def scenario_5_healthy():
    """Synthetic 'already healthy' state: high hit_rate, low missing, no strong signals.
    Agent should output done."""
    vocab, diag, assign = load_base()

    # Bring missing records down to 20 (3%)
    missing_records = [a for a in assign if a["missing"]][:40]  # rescue 40 of 60
    rng = random.Random(42)
    common_tags = ["state_isolation", "error_handling", "fallback_strategy", "llm_integration"]
    for a in missing_records:
        a["missing"] = False
        a["missing_concept"] = ""
        # synthetic assignment
        chosen = rng.sample(common_tags, k=2)
        a["selected_tags"] = [{"name": chosen[0], "confidence": "high"},
                              {"name": chosen[1], "confidence": "medium"}]
        a["reason"] = "synthetic for healthy scenario"

    diag = _recompute_diag(vocab, diag, assign)

    expected = {
        "name": "scenario_5_healthy",
        "description": "Synthetic healthy state: hit_rate ~97%, missing=20, no unused, no extreme cooccur",
        "acceptable_actions": ["done"],
        "rejected_actions": ["propose_new", "propose_merge", "propose_deprecate"],
        "expected_confidence": ["high"],
        "must_probe": [],
        "reasoning_must_mention": ["healthy", "no strong signal"],
        "notes": "Should be high confidence done — clear healthy state."
    }
    save_scenario("scenario_5_healthy", vocab, diag, assign, expected)


# ── Diagnostics recomputation ────────────────────────────────────────────────

def _recompute_diag(vocab: dict, base_diag: dict, assignments: list) -> dict:
    """Recompute usage / unused / cooccur / missing / boundary from assignments."""
    vocab_names = {t["name"] for t in vocab["vocab"]}
    tag_usage: Counter[str] = Counter()
    boundary_blur = []
    cooccur: Counter[tuple[str, str]] = Counter()

    for a in assignments:
        if a.get("missing"):
            continue
        tags = a.get("selected_tags", [])
        for t in tags:
            if t["name"] in vocab_names:
                tag_usage[t["name"]] += 1
        # cooccur
        names = [t["name"] for t in tags if t["name"] in vocab_names]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cooccur[tuple(sorted([names[i], names[j]]))] += 1
        # boundary blur: 2+ tags same confidence
        confs = [t["confidence"] for t in tags]
        if len(confs) >= 2 and len(set(confs)) == 1:
            boundary_blur.append({"record_id": a["record_id"], "title": a.get("title", "")})

    return {
        "sample_size": len(assignments),
        "total_assigned": sum(1 for a in assignments if not a["missing"]),
        "total_missing": sum(1 for a in assignments if a["missing"]),
        "tag_usage_count": dict(tag_usage.most_common()),
        "unused_tags": sorted(vocab_names - set(tag_usage.keys())),
        "missing_records": [
            {"record_id": a["record_id"], "title": a.get("title", ""), "missing_concept": a.get("missing_concept", "")}
            for a in assignments if a.get("missing")
        ],
        "low_confidence_records": base_diag.get("low_confidence_records", []),
        "boundary_blur_records": boundary_blur,
        "top_cooccurrence_pairs": [
            {"pair": list(p), "count": c}
            for p, c in cooccur.most_common(15)
        ],
    }


def main():
    DST.mkdir(parents=True, exist_ok=True)
    print("Building scenarios...")
    scenario_1_real()
    scenario_2_unused_tag()
    scenario_3_split_synonym()
    scenario_4_fake_cooccur()
    scenario_5_healthy()
    print("\nDone. Fixtures in:", DST)


if __name__ == "__main__":
    main()
