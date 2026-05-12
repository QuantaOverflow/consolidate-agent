"""LLM-as-judge for vocab maintenance actions.

Replaces the hit_rate threshold gate. After each apply, this judge reads the
action, before/after 5-dim metrics, and produces a structured verdict:
commit / rollback / unsure. Conservative bias — only rollback when at least
one dim regresses materially outside the action's expected effects.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consolidate_agent.config import Settings
from consolidate_agent.consolidation._utils import _chat_model

from .apply import DeprecateProposal, MergeProposal, NewTagProposal
from .observability import get_default_logger, invoke_with_retry


class JudgeVerdict(BaseModel):
    """Structured output of the judge step."""

    primary_concern: str = Field(
        default="",
        description="If rollback or unsure: the dim name or aspect driving the decision (e.g. 'distinctness dropped 0.08'). Empty when committing.",
    )
    reasoning: str = Field(
        description="1-3 sentences citing specific dim names and numeric deltas. No generic phrases."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Self-assess: high = unanimous signal across dims; medium = clear trend with some noise; low = genuine ambiguity."
    )
    verdict: Literal["commit", "rollback", "unsure"] = Field(
        description="FINAL: commit if action goal achieved without material regression; rollback if at least one dim regresses materially outside expected effects; unsure only when genuinely ambiguous."
    )


JUDGE_SYSTEM = """You are evaluating whether a vocab-maintenance action just applied should be committed or rolled back.

You see:
- The action taken + the reviewer's reasoning
- Each proposal applied
- The 5 health dimensions before vs after, with per-dim delta
- The new vocab size

Decision rules (conservative defaults):
- commit: action achieves its intended effect AND no dimension regresses materially.
- rollback: at least one dim regresses materially AND the regression isn't expected for this action's goal.
- unsure: signals are genuinely mixed; do not use to dodge a hard call.

Calibration — what is "material regression"?

Per-action expectations (apply IN ORDER — first matching rule wins):

propose_merge:
  - Coverage drop <= 0.02 is expected (records may lose duplicate tags when merged).
  - Coherence is expected to RISE or stay flat (collapsing redundant tags tightens semantic fit). Coherence drop > 0.02 is suspicious.
  - Distinctness drop > 0.03 is suspicious (a merge should make vocab LESS redundant, not more — the only way distinctness drops is if the merged tag definition now overlaps more with surviving tags).
  - Multi_axis drop > 0.05 is suspicious (records losing real secondary axes).
  - Granularity: any direction is plausible — ignore.

propose_deprecate:
  - Coverage drop <= 0.05 is expected (records lose their assignments to the dropped tag).
  - Coherence usually unchanged or slightly up; drop > 0.03 means surviving tags fit worse.
  - Distinctness expected to rise slightly; drop > 0.03 is suspicious.
  - Multi_axis drop > 0.10 means too many records lost their only secondary tag.
  - Granularity: ignore.

propose_new / propose_refine (future):
  - Coverage expected to rise or be flat.
  - Coherence may shift either direction; drop > 0.05 is suspicious.
  - Distinctness drop > 0.05 suggests the new/refined tag overlaps with an existing one.

Other notes:
- A "small" delta (< 0.01 in absolute value) on any dim is noise; do not cite it.
- Cite SPECIFIC dim names and numeric deltas in reasoning. No generic phrases like "looks worse".
- 'commit' is the safe default. Only rollback when at least one regression is clearly outside the action's expected effects.
- 'unsure' is for genuine ambiguity (e.g., one dim regressed but action's goal is unclear). Do not use to avoid deciding."""


JUDGE_USER = """## Action taken
- Action: {action}
- Reviewer reasoning: {reasoning}

## Proposals applied
{proposals_summary}

## Health metrics: before -> after (delta)
- coverage:     {b_cov:.3f} -> {a_cov:.3f}  ({d_cov:+.3f})
- coherence:    {b_coh:.3f} -> {a_coh:.3f}  ({d_coh:+.3f})
- distinctness: {b_dis:.3f} -> {a_dis:.3f}  ({d_dis:+.3f})
- granularity:  {b_gra:.3f} -> {a_gra:.3f}  ({d_gra:+.3f})
- multi_axis:   {b_ma:.3f} -> {a_ma:.3f}  ({d_ma:+.3f})

## Vocab change
- vocab size: {b_vocab} -> {a_vocab} tags

Decide: commit, rollback, or unsure?"""


def summarize_proposals(proposals: list) -> str:
    """One-line summary per proposal for the judge prompt."""
    if not proposals:
        return "(none)"
    lines = []
    for p in proposals:
        if isinstance(p, MergeProposal):
            lines.append(f"- merge: discard '{p.discard_tag}' -> keep '{p.keep_tag}'")
        elif isinstance(p, DeprecateProposal):
            lines.append(f"- deprecate: '{p.tag}'")
        elif isinstance(p, NewTagProposal):
            lines.append(f"- new: '{p.name}' — {p.definition[:120]}")
        else:
            lines.append(f"- {type(p).__name__}: {p}")
    return "\n".join(lines)


def llm_judge(
    action: str,
    proposals: list,
    reasoning: str,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    before_vocab_size: int,
    after_vocab_size: int,
) -> JudgeVerdict:
    """Single LLM call returning structured JudgeVerdict.

    On LLM failure after retries: falls back to commit (conservative — don't
    reject changes just because the judge couldn't be reached).
    """
    settings = Settings()
    model = _chat_model(settings).with_structured_output(JudgeVerdict)
    prompt = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("user", JUDGE_USER)])

    def _b(k: str) -> float:
        return before_metrics.get(k, 0.0)

    def _a(k: str) -> float:
        return after_metrics.get(k, 0.0)

    msg = prompt.invoke({
        "action": action,
        "reasoning": (reasoning or "")[:500],
        "proposals_summary": summarize_proposals(proposals),
        "b_cov": _b("coverage"), "a_cov": _a("coverage"), "d_cov": _a("coverage") - _b("coverage"),
        "b_coh": _b("coherence"), "a_coh": _a("coherence"), "d_coh": _a("coherence") - _b("coherence"),
        "b_dis": _b("distinctness"), "a_dis": _a("distinctness"), "d_dis": _a("distinctness") - _b("distinctness"),
        "b_gra": _b("granularity"), "a_gra": _a("granularity"), "d_gra": _a("granularity") - _b("granularity"),
        "b_ma": _b("multi_axis"), "a_ma": _a("multi_axis"), "d_ma": _a("multi_axis") - _b("multi_axis"),
        "b_vocab": before_vocab_size, "a_vocab": after_vocab_size,
    })

    logger = get_default_logger()
    verdict = invoke_with_retry(model, msg, retries=3, caller="judge.decide", logger=logger)
    if verdict is None:
        # Conservative fallback: commit. Don't reject changes on judge failure.
        return JudgeVerdict(
            verdict="commit",
            primary_concern="judge LLM failed after retries",
            reasoning="judge unavailable — defaulting to commit per conservative bias",
            confidence="low",
        )
    return verdict
