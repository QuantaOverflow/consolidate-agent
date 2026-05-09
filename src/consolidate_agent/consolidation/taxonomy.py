from __future__ import annotations

import re

from pydantic import BaseModel, Field

from consolidate_agent.types import MechanismTag, MechanismTagStatus, normalize_text, utc_now


class MechanismTagDraft(BaseModel):
    name: str
    definition: str
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)


class TagMergeDecision(BaseModel):
    keep: str
    merge: str
    reason: str


class TagDeduplicationResult(BaseModel):
    merges: list[TagMergeDecision] = Field(default_factory=list)


def validate_tag_deduplication(result: TagDeduplicationResult, tags: list[MechanismTag]) -> None:
    tag_names = {normalize_tag_name(tag.name) for tag in tags}
    merge_targets: dict[str, str] = {}
    for decision in result.merges:
        keep = normalize_tag_name(decision.keep)
        merge = normalize_tag_name(decision.merge)
        if keep not in tag_names:
            raise ValueError(f"Tag deduplication keep tag is not active: {decision.keep}")
        if merge not in tag_names:
            raise ValueError(f"Tag deduplication merge tag is not active: {decision.merge}")
        if keep == merge:
            raise ValueError(f"Tag deduplication cannot merge a tag into itself: {decision.merge}")
        existing = merge_targets.get(merge)
        if existing is not None and existing != keep:
            raise ValueError(f"Tag deduplication merge target is ambiguous: {decision.merge}")
        merge_targets[merge] = keep

    for merge in merge_targets:
        seen: set[str] = set()
        current = merge
        while current in merge_targets:
            if current in seen:
                raise ValueError("Tag deduplication contains a merge cycle.")
            seen.add(current)
            current = merge_targets[current]


def tag_from_draft(draft: MechanismTagDraft) -> MechanismTag:
    now = utc_now()
    name = normalize_tag_name(draft.name)
    definition = draft.definition.strip()
    return MechanismTag(
        tag_id=tag_id(name, definition),
        name=name,
        definition=definition,
        status=MechanismTagStatus.ACTIVE,
        positive_examples=draft.positive_examples,
        negative_examples=draft.negative_examples,
        created_at=now,
        updated_at=now,
    )


def tag_id(name: str, definition: str) -> str:
    import hashlib
    key = f"{normalize_tag_name(name)}|{normalize_text(definition)}"
    return f"tag_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def normalize_tag_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", value).strip("_") or "general_failure_mechanism"
