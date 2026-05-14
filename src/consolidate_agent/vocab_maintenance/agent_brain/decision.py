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
    """Admission control on state mutations.

    Only fact-based gates remain. Removed in this version: low_certainty,
    insufficient_evidence, irreversible_always_review — all depended on
    LLM-self-reported fields that spike data showed LLM does not honestly
    calibrate (cert always "high", observations padded). high_cert_no_preview
    and high_impact_no_preview are kept as diagnostic markers (they co-trigger
    with decide_without_proposal after verify_node fact-overrides preview_reviewed).
    """
    triggered: list[str] = []

    if d.action == "stop":
        return GateDecision(GateResult.AUTO, ["stop_always_auto"], "stop allowed")

    # Diagnostic markers (subset of decide_without_proposal after verify_node)
    if d.affected_records_estimate > _HIGH_IMPACT_THRESHOLD and not d.preview_reviewed:
        triggered.append("high_impact_no_preview")
    if d.certainty == "high" and not d.preview_reviewed and d.action in {
        "split", "merge", "refine", "deprecate"
    }:
        triggered.append("high_cert_no_preview")

    # Core gate: modify action must have a cached proposal (verify_node would
    # have set preview_reviewed=True if cache had a match). Reaching here with
    # preview_reviewed=False means there's nothing to apply — block to prevent
    # silent no-op.
    if d.action in {"split", "merge", "refine", "deprecate"} and not d.preview_reviewed:
        if "decide_without_proposal" not in triggered:
            triggered.append("decide_without_proposal")

    if d.action not in {"inspect_more", "stop"} and d.target:
        # Only count COMMITTED modifications — REVIEW-gated attempts don't count
        # as "flip-flop" since they never changed the network.
        recent_targets = [
            r.target
            for r in list(memory.history)[-5:]
            if r.target and r.committed
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
