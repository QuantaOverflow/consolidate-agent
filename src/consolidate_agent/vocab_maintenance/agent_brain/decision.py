from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .memory import AgentMemory

_HIGH_IMPACT_THRESHOLD = 30

REVERSIBILITY_DEFAULTS: dict[str, str] = {
    "split": "clean_rollback",
    "refine": "clean_rollback",
    "merge": "messy_rollback",
    "deprecate": "messy_rollback",
    "inspect_more": "clean_rollback",
    "stop": "clean_rollback",
}


class AgentDecision(BaseModel):
    action: Literal["split", "merge", "refine", "deprecate", "inspect_more", "stop"]
    target: str = ""

    reasoning: str = Field(min_length=80, max_length=2000)

    certainty: Literal["high", "medium", "low"]

    supporting_observations: list[str] = Field(min_length=1, max_length=8)
    opposing_observations: list[str] = Field(default_factory=list, max_length=8)

    preview_reviewed: bool = False

    affected_records_estimate: int = 0
    reversibility: Literal["clean_rollback", "messy_rollback", "irreversible"] = "clean_rollback"

    expected_outcome: str = Field(default="", description="qualitative description of what you expect to change semantically, e.g. 'http_api should split into 3 cleaner sub-concepts: auth, streaming, routing'")

    stop_reason: str = ""


def apply_reversibility_default(d: AgentDecision) -> AgentDecision:
    default = REVERSIBILITY_DEFAULTS.get(d.action, "clean_rollback")
    if d.reversibility == default:
        return d
    # If the model field equals its own default ("clean_rollback") and the action's
    # semantic default differs, fill in the semantic default.  If LLM explicitly
    # overrode to something *other* than pydantic's field default, keep that.
    pydantic_field_default = "clean_rollback"
    if d.reversibility == pydantic_field_default and default != pydantic_field_default:
        return d.model_copy(update={"reversibility": default})
    return d


class GateResult(str, Enum):
    AUTO = "auto"
    REVIEW = "review"
    BLOCK = "block"


@dataclass
class GateDecision:
    result: GateResult
    triggered_gates: list[str]
    reason: str


def gate_decision(d: AgentDecision, memory: "AgentMemory") -> GateDecision:
    triggered: list[str] = []

    if d.action == "stop":
        return GateDecision(GateResult.AUTO, ["stop_always_auto"], "stop allowed")

    if d.reversibility == "irreversible":
        triggered.append("irreversible_always_review")
        return GateDecision(GateResult.REVIEW, triggered, "action marked irreversible")

    if d.certainty == "low":
        triggered.append("low_certainty")

    if len(d.supporting_observations) < 2:
        triggered.append("insufficient_evidence")

    # 5. High impact + no preview (replaces former no_golden check per ADR-0007)
    if d.affected_records_estimate > _HIGH_IMPACT_THRESHOLD and not d.preview_reviewed:
        triggered.append("high_impact_no_preview")

    if d.certainty == "high" and not d.preview_reviewed and d.action in {
        "split", "merge", "refine", "deprecate"
    }:
        triggered.append("high_cert_no_preview")

    # Silent no-op guard: any modifying action without a preview means no
    # proposal is in cache to apply. Brain.py's verify step would have set
    # preview_reviewed=True if a proposal existed — so reaching here with
    # preview_reviewed=False means we'd produce a "gate=AUTO but no proposal
    # in cache" silent no-op. Block it.
    if d.action in {"split", "merge", "refine", "deprecate"} and not d.preview_reviewed:
        if "decide_without_proposal" not in triggered:
            triggered.append("decide_without_proposal")

    if d.action not in {"inspect_more", "stop"} and d.target:
        recent_targets = [
            r.target
            for r in list(memory.history)[-5:]
            if r.target and r.decision_action not in (None, "inspect_more")
        ]
        same_target_count = sum(1 for t in recent_targets if t == d.target)
        if same_target_count >= 2:
            triggered.append("recent_flipflop")

    if triggered:
        return GateDecision(
            GateResult.REVIEW,
            triggered,
            f"gates triggered: {', '.join(triggered)}",
        )
    return GateDecision(GateResult.AUTO, ["all_gates_pass"], "passed all gates")


def make_decision_id(run_id: str, round_idx: int, action: str, target: str) -> str:
    import uuid

    return f"{run_id}.r{round_idx}.{action}.{target or 'none'}.{uuid.uuid4().hex[:8]}"
