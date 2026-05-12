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
from pydantic import BaseModel, Field, create_model

ALL_ACTIONS = ("propose_new", "propose_merge", "propose_deprecate")

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model

from .observability import get_default_logger, invoke_with_retry
from .probes import run_probe
from .similarity import find_similar_pairs


# ── Schemas ──────────────────────────────────────────────────────────────────


class PlanOutput(BaseModel):
    """Step 1 output: what the LLM thinks looks important + which probes to run."""
    raw_signals_observed: str = Field(description="1-2 sentences: what stands out in the diagnostics. Cite specific numbers.")
    suspected_biases: str = Field(description="What aspects of the raw signals might be misleading? E.g. 'missing_rate may be inflated by LLM conservatism' or 'high cooccur may not mean synonymy'.")
    probe_calls_needed: list[str] = Field(
        description="1-3 probe calls. Available: inspect_missing_records(n=10) | inspect_cooccur_pair(tag_a=X, tag_b=Y) | inspect_tag(name=X) | inspect_outliers(name=X, n=5) | find_orphan_themes(n=10) | compare_tag_records(tag_a=X, tag_b=Y). Pick the most diagnostic probes.",
        max_length=3,
    )


class DiagnosticDecision(BaseModel):
    """Step 3 output: final action after probe findings."""
    probe_findings_summary: str = Field(description="STEP 1: Summarize what the probes revealed in 1-3 sentences. Cite specific record titles or counts.")
    bias_assessment: str = Field(description="STEP 2: Were the raw signals reliable? Did probes confirm or refute them? Be specific about what was misleading.")
    independence_check: str = Field(description="STEP 3 (only if considering merge/deprecate): is one tag actually subsumed by another, or are they distinct axes?")

    confidence: Literal["high", "medium", "low"] = Field(description="STEP 4 — SELF-ASSESSMENT: how confident in this decision? high=probes unanimously support; medium=probes mostly support with some tension; low=genuine ambiguity, multiple defensible decisions, would benefit from human review.")
    uncertainty_reasons: list[str] = Field(default_factory=list, description="If confidence < high, list 1-3 specific reasons (e.g. 'high def similarity but distinct usage patterns', 'probe sample too small', 'records could be re-tagged either way'). Empty if confidence=high.")

    next_action: Literal["propose_new", "propose_merge", "propose_deprecate", "done"] = Field(description="FINAL: which workflow to run next, or done if vocab is healthy.")  # overridden dynamically per-call to exclude blocked actions
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
- inspect_outliers(name=X, n=5) — within tag X, list records with LOWEST cosine fit to the tag definition. Use this when you suspect expand_coverage attached records mechanically (surface keywords) that don't actually match. The "bottom_n_by_fit" entries are the audit candidates.
- find_orphan_themes(n=10) — across a sample of records, list those whose MAX similarity to ANY vocab tag is lowest. Two failure modes appear here: (a) records with low max_sim AND missing=False → reverse_check likely forced a bad fit; (b) records with low max_sim AND missing=True → confirms vocab has a real gap; (c) records with high max_sim to a tag NOT in current_tags → potential missing co-tag or wrong assignment.
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

## Suspected redundancy pairs (high definition similarity — may indicate near-synonyms)
{similar_pairs}

## Already-tried actions (BLOCKED — structurally removed from next_action choices)
{blocked_actions}

Plan probes to verify suspicious signals. Pay attention to suspected redundancy pairs — if any pair has high def similarity, probe it before deciding action.

Blocked actions are not in your decision schema this round; you can only choose among remaining actions or 'done'. Plan probes accordingly — don't probe signals that only support a blocked action."""


DECIDE_SYSTEM = """You are now deciding the next action based on probe findings.

Decision rules (ONLY these actions are available this round):
{decision_rules}

Work through analysis steps IN ORDER. Pydantic fields are listed in thinking order — fill them sequentially.

Critical: probe findings OVERRIDE raw signals. If raw missing_rate was high but probe shows records fit existing tags → suppress propose_new. If raw cooccur was high but probe shows distinct axes → suppress propose_merge.

action_focus should be specific. E.g.:
- "focus on records about encoding/byte semantics (3 records found)" not just "missing records"
- "target tag obscure_xml_quirk (0 usage, no semantic overlap)" not just "deprecate unused"
- "" if action=done

CRITICAL — self-assess confidence HONESTLY. Default starting point: medium. Only escalate to high if the evidence is overwhelming. Only stay at medium or drop to low if uncertainty is genuine.

Calibration rules (apply these BEFORE deciding confidence):

1. **Suspected redundancy pairs were flagged in plan input** (high def similarity). For EACH such pair:
   - If you decide MERGE based on probes → confidence can be high (probes confirmed the suspicion)
   - If you decide KEEP_DISTINCT despite high similarity → confidence MUST be medium or low (you're overruling a strong static signal; reasonable people might disagree). NEVER say high in this case.

2. **Raw signals contradict probe findings**:
   - If raw missing_rate was high but probes show most are false positives → medium (probes refuted raw, but you're trusting probes over aggregate)
   - If raw cooccur was high but probes show legit multi-axis → medium

3. **Probes were insufficient**:
   - Did you probe everything the plan suggested? If you skipped suspected pairs, your decision is undertested → at most medium
   - If probe sample size was too small (e.g. 3 records to judge a 30-record tag) → at most medium

4. **Decision is action=done**:
   - If you said "vocab is healthy" but there ARE suspected pairs, low missing rate, AND some signals were ambiguous → at most medium

When to say HIGH (rare, must meet all):
- Probes gave unanimous, unambiguous signal
- No suspected redundancy pairs above 0.80 left unresolved
- Decision is consistent with both raw signals AND probe findings (no override)
- Reviewer looking at same data would clearly agree

Otherwise: medium or low.

uncertainty_reasons must be SPECIFIC and concrete — e.g. "state_isolation and state_namespacing have 0.82 def similarity but records show distinct usage axes (containment vs key-scoping)" — not generic phrases like "borderline case". Include the specific pair names or numeric thresholds that drive uncertainty."""


DECIDE_USER = """## Original raw signals
{raw_signals}

## Suspected biases
{biases}

## Probe findings
{findings}

## ⚠️ BLOCKED actions for this round (DO NOT pick — your output will be rejected)
{blocked_actions}

You MUST choose `next_action` from: {allowed_actions}

Decide the next action with probe-backed reasoning."""


# Per-action rule descriptions — assembled dynamically based on allowed_actions.
_DECIDE_RULES = {
    "propose_new": "- propose_new: only if probes confirm genuine missing concepts (not LLM false positives)",
    "propose_merge": "- propose_merge: only if probes confirm two tags describe the same concept (subsumption, not just relatedness)",
    "propose_deprecate": "- propose_deprecate: if a tag is truly unused or its records all fit a single other tag better",
    "done": "- done: if probes show the vocab is in good shape",
}


def _build_decision_rules(allowed_actions: tuple[str, ...]) -> str:
    """Render only the rules for actions that are currently allowed.

    Blocked actions are entirely absent from the prompt — LLM never sees
    them as candidates.
    """
    keys = list(allowed_actions) + ["done"]
    return "\n".join(_DECIDE_RULES[k] for k in keys if k in _DECIDE_RULES)


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


def _similar_pairs_str(pairs: list[dict]) -> str:
    if not pairs:
        return "  (none above threshold)"
    return "\n".join(f"  {p['similarity']:.3f}  {p['tag_a']} ↔ {p['tag_b']}" for p in pairs)


def _decision_model_for(allowed_actions: tuple[str, ...]) -> type[BaseModel]:
    """Build a DiagnosticDecision variant whose next_action Literal is restricted.

    Hard structural constraint: LLM cannot return a blocked action.
    'done' is always allowed.
    """
    choices = tuple(allowed_actions) + ("done",)
    action_t = Literal[choices]  # type: ignore[valid-type]
    return create_model(
        "DiagnosticDecisionRestricted",
        __base__=DiagnosticDecision,
        next_action=(action_t, Field(description=f"FINAL: which workflow to run next, or done if vocab is healthy. Allowed: {', '.join(choices)}.")),
    )


def diagnose(
    vocab: list[dict],
    diagnostics: dict,
    assignments: list[dict],
    db_path: Path,
    blocked_actions: list[str] | None = None,
) -> dict:
    settings = Settings()
    plan_model = _chat_model(settings).with_structured_output(PlanOutput)

    blocked_set = set(blocked_actions or [])
    allowed = tuple(a for a in ALL_ACTIONS if a not in blocked_set)
    DecisionCls = _decision_model_for(allowed)
    decide_model = _chat_model(settings).with_structured_output(DecisionCls)

    plan_prompt = ChatPromptTemplate.from_messages([("system", PLAN_SYSTEM), ("user", PLAN_USER)])
    decide_prompt = ChatPromptTemplate.from_messages([("system", DECIDE_SYSTEM), ("user", DECIDE_USER)])

    total = diagnostics["sample_size"]
    missing = diagnostics["total_missing"]
    missing_rate = missing / total if total else 0.0

    # Pre-compute suspected redundancy pairs via def cosine similarity
    similar_pairs = find_similar_pairs(vocab, threshold=0.80, top_n=5)
    blocked_str = ", ".join(blocked_actions) if blocked_actions else "(none)"

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
        "similar_pairs": _similar_pairs_str(similar_pairs),
        "blocked_actions": blocked_str,
    })
    logger = get_default_logger()

    plan: PlanOutput | None = invoke_with_retry(
        plan_model, plan_msg, retries=3, caller="diagnose.plan", logger=logger,
    )
    if plan is None:
        raise ValueError("plan step failed after 3 retries")

    # Step 2: execute probes
    findings: dict[str, dict] = {}
    for probe_call in plan.probe_calls_needed[:3]:
        try:
            findings[probe_call] = run_probe(probe_call, vocab, assignments, db_path)
            logger.event("probe.ok", call=probe_call)
        except Exception as e:
            findings[probe_call] = {"error": str(e)}
            logger.event("probe.error", call=probe_call, error=str(e)[:200])

    # Step 3: decide — inject blocked/allowed info so LLM doesn't keep
    # picking blocked actions and getting ValidationError.
    findings_str = json.dumps(findings, indent=2, ensure_ascii=False)[:6000]  # cap
    allowed_for_decide = tuple(allowed) + ("done",)
    blocked_for_decide = ", ".join(blocked_set) if blocked_set else "(none)"
    decide_msg = decide_prompt.invoke({
        "raw_signals": plan.raw_signals_observed,
        "biases": plan.suspected_biases,
        "findings": findings_str,
        "decision_rules": _build_decision_rules(allowed),
        "blocked_actions": blocked_for_decide,
        "allowed_actions": ", ".join(allowed_for_decide),
    })
    decision = invoke_with_retry(
        decide_model, decide_msg, retries=3, caller="diagnose.decide", logger=logger,
    )
    if decision is None:
        # Fallback: terminate the loop cleanly rather than crash.
        # Happens when LLM ignores the dynamic Literal schema (e.g., returns
        # a blocked action despite the schema excluding it).
        logger.event("diagnose.fallback_to_done", allowed_actions=list(allowed))
        print(f"  [diagnose] LLM returned invalid action 3× — fallback to done", flush=True)
        return {
            "plan": plan.model_dump(),
            "similar_pairs": similar_pairs,
            "probe_calls": list(plan.probe_calls_needed[:3]),
            "findings": findings,
            "decision": {
                "probe_findings_summary": "(LLM repeatedly returned invalid action despite schema)",
                "bias_assessment": "",
                "independence_check": "",
                "confidence": "low",
                "uncertainty_reasons": ["LLM ignored next_action schema constraint on all retries"],
                "next_action": "done",
                "action_focus": "",
                "reasoning": "diagnose retries exhausted — terminating to avoid loop",
            },
        }

    return {
        "plan": plan.model_dump(),
        "similar_pairs": similar_pairs,  # exposed to reviewer; no auto-downgrade
        "probe_calls": list(plan.probe_calls_needed[:3]),
        "findings": findings,
        "decision": decision.model_dump(),
    }
