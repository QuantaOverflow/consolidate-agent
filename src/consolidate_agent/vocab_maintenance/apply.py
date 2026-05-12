"""Apply a proposal to a (vocab, assignments) state.

Pure data transformation — no LLM calls, no DB writes.
Returns new (vocab, assignments) tuple. Caller persists.

Currently a stub — to be implemented after spec tests are reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ── Proposal types ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NewTagProposal:
    name: str
    definition: str

    type: str = "new"


@dataclass(frozen=True)
class MergeProposal:
    keep_tag: str
    discard_tag: str

    type: str = "merge"


@dataclass(frozen=True)
class DeprecateProposal:
    tag: str

    type: str = "deprecate"


@dataclass(frozen=True)
class RefineTagProposal:
    """Update a tag's definition AND detach a small set of records that don't fit.

    The cosine-distance invariant ("prune targets sit far from the new
    definition") is enforced at proposal construction time in
    `propose/refine.py:validate_refine_proposal`, because apply.py is pure
    data and has no path to record content embeddings.
    """
    tag: str
    new_definition: str
    prune_record_ids: tuple[str, ...] = ()  # tuple so the dataclass is hashable

    type: str = "refine"


# ── Exceptions ───────────────────────────────────────────────────────────────


class InvalidProposal(ValueError):
    """Proposal references nonexistent tags, self-references, etc."""


class OrphanError(ValueError):
    """Deprecate would leave records with zero tags. Caller must resolve before retrying."""

    def __init__(self, message: str, orphan_record_ids: list[str]):
        super().__init__(message)
        self.orphan_record_ids = orphan_record_ids


# ── Apply (stub) ─────────────────────────────────────────────────────────────


_CONFIDENCE_ORDER = {"high": 2, "medium": 1, "low": 0}


def _max_confidence(c1: str, c2: str) -> str:
    return c1 if _CONFIDENCE_ORDER.get(c1, 0) >= _CONFIDENCE_ORDER.get(c2, 0) else c2


def _apply_new(
    vocab: list[dict], assignments: list[dict], p: NewTagProposal
) -> tuple[list[dict], list[dict]]:
    if p.name in {t["name"] for t in vocab}:
        raise InvalidProposal(f"tag '{p.name}' already in vocab")
    new_vocab = list(vocab) + [{"name": p.name, "definition": p.definition}]
    return new_vocab, assignments


def _apply_merge(
    vocab: list[dict], assignments: list[dict], p: MergeProposal
) -> tuple[list[dict], list[dict]]:
    if p.keep_tag == p.discard_tag:
        raise InvalidProposal(f"self-merge: keep_tag == discard_tag == '{p.keep_tag}'")
    names = {t["name"] for t in vocab}
    if p.keep_tag not in names:
        raise InvalidProposal(f"keep_tag '{p.keep_tag}' not in vocab")
    if p.discard_tag not in names:
        raise InvalidProposal(f"discard_tag '{p.discard_tag}' not in vocab")

    new_vocab = [t for t in vocab if t["name"] != p.discard_tag]

    new_assignments = []
    for a in assignments:
        if a.get("missing"):
            new_assignments.append(a)
            continue
        # rename discard → keep, then dedupe (keeping max confidence)
        by_name: dict[str, dict] = {}
        for t in a["selected_tags"]:
            new_name = p.keep_tag if t["name"] == p.discard_tag else t["name"]
            entry = {"name": new_name, "confidence": t["confidence"]}
            if new_name in by_name:
                by_name[new_name] = {
                    "name": new_name,
                    "confidence": _max_confidence(by_name[new_name]["confidence"], entry["confidence"]),
                }
            else:
                by_name[new_name] = entry
        new_assignments.append(dict(a, selected_tags=list(by_name.values())))

    return new_vocab, new_assignments


def _apply_deprecate(
    vocab: list[dict], assignments: list[dict], p: DeprecateProposal
) -> tuple[list[dict], list[dict]]:
    names = {t["name"] for t in vocab}
    if p.tag not in names:
        raise InvalidProposal(f"tag '{p.tag}' not in vocab")

    # detect ALL orphans before mutating anything
    orphans: list[str] = []
    for a in assignments:
        if a.get("missing"):
            continue
        record_tag_names = [t["name"] for t in a["selected_tags"]]
        if p.tag in record_tag_names:
            other_tags = [n for n in record_tag_names if n != p.tag]
            if not other_tags:
                orphans.append(a["record_id"])
    if orphans:
        raise OrphanError(
            f"deprecating '{p.tag}' would orphan {len(orphans)} record(s): {orphans}",
            orphan_record_ids=orphans,
        )

    new_vocab = [t for t in vocab if t["name"] != p.tag]
    new_assignments = []
    for a in assignments:
        if a.get("missing"):
            new_assignments.append(a)
            continue
        filtered = [t for t in a["selected_tags"] if t["name"] != p.tag]
        new_assignments.append(dict(a, selected_tags=filtered))
    return new_vocab, new_assignments


_REFINE_PRUNE_HARD_CAP = 10


def _apply_refine(
    vocab: list[dict], assignments: list[dict], p: RefineTagProposal
) -> tuple[list[dict], list[dict]]:
    """Update tag definition + detach prune-listed records from that tag.

    Structural invariants enforced:
      I-R1: tag exists in vocab
      I-R2: prune list size <= _REFINE_PRUNE_HARD_CAP (==10)
      I-R3: every prune target currently carries `tag`
      I-R4: no record becomes orphan (loses all tags) after prune
    """
    names = {t["name"] for t in vocab}
    if p.tag not in names:
        raise InvalidProposal(f"refine target '{p.tag}' not in vocab")
    if len(p.prune_record_ids) > _REFINE_PRUNE_HARD_CAP:
        raise InvalidProposal(
            f"refine prune list too large: {len(p.prune_record_ids)} > {_REFINE_PRUNE_HARD_CAP}"
        )

    prune_set = set(p.prune_record_ids)

    orphans: list[str] = []
    for a in assignments:
        if a.get("missing"):
            continue
        if a["record_id"] not in prune_set:
            continue
        current = [t["name"] for t in a.get("selected_tags", [])]
        if p.tag not in current:
            raise InvalidProposal(
                f"refine: record {a['record_id']} not tagged with '{p.tag}', cannot prune"
            )
        if not [n for n in current if n != p.tag]:
            orphans.append(a["record_id"])
    if orphans:
        raise OrphanError(
            f"refining '{p.tag}' would orphan {len(orphans)} record(s): {orphans}",
            orphan_record_ids=orphans,
        )

    new_vocab = [
        {**t, "definition": p.new_definition} if t["name"] == p.tag else t
        for t in vocab
    ]
    new_assignments = []
    for a in assignments:
        if a.get("missing") or a["record_id"] not in prune_set:
            new_assignments.append(a)
            continue
        filtered = [t for t in a["selected_tags"] if t["name"] != p.tag]
        new_assignments.append(dict(a, selected_tags=filtered))
    return new_vocab, new_assignments


def apply_proposal(
    vocab: list[dict],
    assignments: list[dict],
    proposal: NewTagProposal | MergeProposal | DeprecateProposal | RefineTagProposal,
) -> tuple[list[dict], list[dict]]:
    """Apply proposal, returning new (vocab, assignments).

    Pure function — does not mutate inputs. Caller responsible for persistence.

    Raises:
        InvalidProposal: nonexistent tag, self-merge, name collision, refine
            tag mismatch / size violation
        OrphanError: deprecate or refine would orphan one or more records
    """
    if isinstance(proposal, NewTagProposal):
        return _apply_new(vocab, assignments, proposal)
    if isinstance(proposal, MergeProposal):
        return _apply_merge(vocab, assignments, proposal)
    if isinstance(proposal, DeprecateProposal):
        return _apply_deprecate(vocab, assignments, proposal)
    if isinstance(proposal, RefineTagProposal):
        return _apply_refine(vocab, assignments, proposal)
    raise InvalidProposal(f"unknown proposal type: {type(proposal).__name__}")


# ── Consistency checker ──────────────────────────────────────────────────────


def check_invariants(vocab: list[dict], assignments: list[dict]) -> None:
    """Verify state-level consistency. Raises AssertionError on violation."""
    vocab_names = [t["name"] for t in vocab]
    vocab_set = set(vocab_names)

    # I1: vocab tag names are unique
    assert len(vocab_names) == len(vocab_set), f"duplicate tag names in vocab: {vocab_names}"

    # I2: no dangling references in assignments
    for a in assignments:
        for t in a.get("selected_tags", []):
            assert t["name"] in vocab_set, (
                f"dangling reference: assignment {a.get('record_id')} → tag '{t['name']}' not in vocab"
            )

    # I3: no duplicate tags in same record's selected_tags
    for a in assignments:
        names = [t["name"] for t in a.get("selected_tags", [])]
        assert len(names) == len(set(names)), (
            f"duplicate tags in record {a.get('record_id')}: {names}"
        )

    # I4: assigned records have at least one tag (missing=False ⟹ selected_tags non-empty)
    for a in assignments:
        if not a.get("missing"):
            assert a.get("selected_tags"), (
                f"non-missing record {a.get('record_id')} has empty selected_tags"
            )
