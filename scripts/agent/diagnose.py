"""Diagnostic step: LLM plans probes, executes them, then decides next action.

3-step pipeline:
  1. plan: LLM sees vocab + diagnostics → outputs probe plan
  2. execute: run requested probes (deterministic)
  3. decide: LLM sees plan + findings → outputs final action
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model

import sys
sys.path.insert(0, str(Path(__file__).parent))
from probes import run_probe  # noqa: E402


# ── Schemas ──────────────────────────────────────────────────────────────────


class PlanOutput(BaseModel):
    """Step 1 output: what the LLM thinks looks important + which probes to run."""
    raw_signals_observed: str = Field(description="1-2 sentences: what stands out in the diagnostics. Cite specific numbers.")
    suspected_biases: str = Field(description="What aspects of the raw signals might be misleading? E.g. 'missing_rate may be inflated by LLM conservatism' or 'high cooccur may not mean synonymy'.")
    probe_calls_needed: list[str] = Field(
        description="1-3 probe calls in format: inspect_missing_records(n=10) | inspect_cooccur_pair(tag_a=X, tag_b=Y) | inspect_tag(name=X) | compare_tag_records(tag_a=X, tag_b=Y). Pick the most diagnostic probes.",
        max_length=3,
    )


class DiagnosticDecision(BaseModel):
    """Step 3 output: final action after probe findings."""
    probe_findings_summary: str = Field(description="STEP 1: Summarize what the probes revealed in 1-3 sentences. Cite specific record titles or counts.")
    bias_assessment: str = Field(description="STEP 2: Were the raw signals reliable? Did probes confirm or refute them? Be specific about what was misleading.")
    independence_check: str = Field(description="STEP 3 (only if considering merge/deprecate): is one tag actually subsumed by another, or are they distinct axes?")

    next_action: Literal["propose_new", "propose_merge", "propose_deprecate", "done"] = Field(description="FINAL: which workflow to run next, or done if vocab is healthy.")
    action_focus: str = Field(default="", description="Specific guidance for the chosen workflow: e.g. 'focus on the 14 truly missing records about permission/encoding' or 'target tag X'. Empty if action=done.")
    reasoning: str = Field(description="1-sentence final synthesis.")


# ── Prompts ──────────────────────────────────────────────────────────────────


PLAN_SYSTEM = """You are diagnosing the health of a knowledge tag vocabulary. You receive:
- Current vocab (tag names + definitions)
- Raw diagnostics (hit rate, missing records, unused tags, top co-occurrence pairs)

Your job in this step: identify suspicious signals and plan probes to verify them.

Known biases of these signals (use these to plan):
- missing_rate is often INFLATED — the upstream tagger is conservative and tends to mark records as "missing" even when they could fit existing tags. Roughly 70-80% of missing records turn out to be false positives.
- High cooccurrence (e.g. cooccur >= 10) does NOT prove synonymy. Many records legitimately span multiple axes.
- unused_tags from full-sample diagnostics is reliable; from small samples it's an artifact.
- low_size_tag (size < 5) might be niche-but-valuable, not necessarily redundant.

Plan 1-3 probes to verify the most suspicious signals. Don't over-probe — pick the most diagnostic.

Available probes:
- inspect_missing_records(n=10) — see actual missing records
- inspect_cooccur_pair(tag_a=X, tag_b=Y) — see records co-occurring on a pair
- inspect_tag(name=X) — see a tag's records + co-tags
- compare_tag_records(tag_a=X, tag_b=Y) — compare two tags' record sets

Be conservative: if signals look weak/balanced, don't probe everything — just say "vocab looks healthy, minimal probes needed"."""


PLAN_USER = """## Current vocab ({vocab_count} tags)
{vocab_brief}

## Diagnostics
total_records: {total}
assigned: {assigned}
missing: {missing} ({missing_rate:.1%})
unused tags: {unused}
top co-occurrence pairs:
{top_cooccur}
tag size distribution (top 10):
{top_usage}
tag size distribution (bottom 5, excluding 0):
{bottom_usage}

Plan probes to verify suspicious signals."""


DECIDE_SYSTEM = """You are now deciding the next action based on probe findings.

Decision rules:
- propose_new: only if probes confirm genuine missing concepts (not LLM false positives)
- propose_merge: only if probes confirm two tags describe the same concept (subsumption, not just relatedness)
- propose_deprecate: if a tag is truly unused or its records all fit a single other tag better
- done: if probes show the vocab is in good shape

Work through analysis steps IN ORDER. Pydantic fields are listed in thinking order — fill them sequentially.

Critical: probe findings OVERRIDE raw signals. If raw missing_rate was high but probe shows records fit existing tags → suppress propose_new. If raw cooccur was high but probe shows distinct axes → suppress propose_merge.

action_focus should be specific. E.g.:
- "focus on records about encoding/byte semantics (3 records found)" not just "missing records"
- "target tag obscure_xml_quirk (0 usage, no semantic overlap)" not just "deprecate unused"
- "" if action=done"""


DECIDE_USER = """## Original raw signals
{raw_signals}

## Suspected biases
{biases}

## Probe findings
{findings}

Decide the next action with probe-backed reasoning."""


# ── Pipeline ─────────────────────────────────────────────────────────────────


def _vocab_brief(vocab: list[dict]) -> str:
    return "\n".join(f"- {t['name']}" for t in vocab)


def _top_cooccur_str(pairs: list[dict], n: int = 5) -> str:
    if not pairs:
        return "(none)"
    return "\n".join(f"  {p['count']:>3} × {p['pair'][0]} + {p['pair'][1]}" for p in pairs[:n])


def _top_usage_str(tag_usage: dict[str, int], n: int = 10) -> str:
    sorted_t = sorted(tag_usage.items(), key=lambda x: -x[1])[:n]
    return "\n".join(f"  {c:>3}  {t}" for t, c in sorted_t)


def _bottom_usage_str(tag_usage: dict[str, int], n: int = 5) -> str:
    nonzero = [(t, c) for t, c in tag_usage.items() if c > 0]
    nonzero.sort(key=lambda x: x[1])
    return "\n".join(f"  {c:>3}  {t}" for t, c in nonzero[:n])


def diagnose(vocab: list[dict], diagnostics: dict, assignments: list[dict], db_path: Path) -> dict:
    settings = Settings()
    plan_model = _chat_model(settings).with_structured_output(PlanOutput)
    decide_model = _chat_model(settings).with_structured_output(DiagnosticDecision)

    plan_prompt = ChatPromptTemplate.from_messages([("system", PLAN_SYSTEM), ("user", PLAN_USER)])
    decide_prompt = ChatPromptTemplate.from_messages([("system", DECIDE_SYSTEM), ("user", DECIDE_USER)])

    total = diagnostics["sample_size"]
    missing = diagnostics["total_missing"]
    missing_rate = missing / total if total else 0.0

    # Step 1: plan
    plan_msg = plan_prompt.invoke({
        "vocab_count": len(vocab),
        "vocab_brief": _vocab_brief(vocab),
        "total": total,
        "assigned": diagnostics["total_assigned"],
        "missing": missing,
        "missing_rate": missing_rate,
        "unused": diagnostics["unused_tags"] or "(none)",
        "top_cooccur": _top_cooccur_str(diagnostics["top_cooccurrence_pairs"]),
        "top_usage": _top_usage_str(diagnostics["tag_usage_count"]),
        "bottom_usage": _bottom_usage_str(diagnostics["tag_usage_count"]),
    })
    plan: PlanOutput | None = plan_model.invoke(plan_msg)
    if plan is None:
        plan = plan_model.invoke(plan_msg)  # retry once
    if plan is None:
        raise ValueError("plan step returned None twice")

    # Step 2: execute probes
    findings: dict[str, dict] = {}
    for probe_call in plan.probe_calls_needed[:3]:
        try:
            findings[probe_call] = run_probe(probe_call, vocab, assignments, db_path)
        except Exception as e:
            findings[probe_call] = {"error": str(e)}

    # Step 3: decide
    findings_str = json.dumps(findings, indent=2, ensure_ascii=False)[:6000]  # cap
    decide_msg = decide_prompt.invoke({
        "raw_signals": plan.raw_signals_observed,
        "biases": plan.suspected_biases,
        "findings": findings_str,
    })
    decision: DiagnosticDecision | None = decide_model.invoke(decide_msg)
    if decision is None:
        decision = decide_model.invoke(decide_msg)
    if decision is None:
        raise ValueError("decide step returned None twice")

    return {
        "plan": plan.model_dump(),
        "probe_calls": list(plan.probe_calls_needed[:3]),
        "findings": findings,
        "decision": decision.model_dump(),
    }
