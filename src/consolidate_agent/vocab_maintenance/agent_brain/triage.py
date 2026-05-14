"""Deterministic network triage — runs once per brain run, feeds initial state.

Computes signals per tag and emits a suggested_action with confidence, so the
LLM gets a prioritized worklist instead of having to discover candidates
through tool calls every round.

Pure function: vocab + assignments + db_path → TriageReport. No LLM.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from consolidate_agent.vocab_maintenance.probes import compute_heterogeneous_tags

_LOW_COHERENCE = 0.42
_LOW_HIGH_CONF = 0.60
_SMALL_TAG = 15
_LARGE_TAG = 100
_HIGH_OVERLAP = 7


@dataclass(frozen=True)
class TagTriage:
    name: str
    size: int
    coherence: float | None
    high_conf_ratio: float
    top_neighbors: list[tuple[str, int]]
    signals: list[str]
    suggested_action: str | None
    suggestion_confidence: str
    rationale: str

    def render(self) -> str:
        coh = f"{self.coherence:.2f}" if self.coherence is not None else "n/a"
        neighbors = ", ".join(f"{n}:{c}" for n, c in self.top_neighbors[:3]) or "(none)"
        sigs = ",".join(self.signals) or "(none)"
        head = (
            f"  {self.name}  size={self.size}  coh={coh}  "
            f"high_conf={self.high_conf_ratio:.0%}  neighbors=[{neighbors}]"
        )
        if self.suggested_action:
            body = (
                f"    signals: {sigs}\n"
                f"    suggested: {self.suggested_action} (confidence={self.suggestion_confidence})\n"
                f"    rationale: {self.rationale}"
            )
        else:
            body = f"    signals: {sigs}  →  {self.rationale}"
        return head + "\n" + body


@dataclass(frozen=True)
class TriageReport:
    candidates: list[TagTriage]
    healthy: list[str]
    untagged_records: int
    total_records: int
    total_tags: int

    def render(self) -> str:
        lines = ["=== Network triage ==="]
        lines.append(
            f"vocab={self.total_tags} tags  records={self.total_records}  "
            f"untagged={self.untagged_records}"
        )
        if self.candidates:
            lines.append("")
            lines.append(f"Action candidates ({len(self.candidates)}, sorted by priority):")
            for c in self.candidates:
                lines.append(c.render())
        else:
            lines.append("Action candidates: (none — network appears healthy)")
        if self.healthy:
            lines.append("")
            lines.append(
                f"Healthy tags ({len(self.healthy)}): " + ", ".join(self.healthy)
            )
        return "\n".join(lines)


def _signal_priority(t: TagTriage) -> tuple[int, int, int]:
    conf_rank = {"high": 0, "medium": 1, "low": 2, "n/a": 3}.get(
        t.suggestion_confidence, 3
    )
    return (conf_rank, -len(t.signals), -t.size)


def _classify(
    name: str,
    size: int,
    coherence: float | None,
    high_conf_ratio: float,
    top_neighbors: list[tuple[str, int]],
    recently_modified: bool,
) -> TagTriage:
    signals: list[str] = []

    if coherence is not None and coherence < _LOW_COHERENCE:
        signals.append("low_coherence")
    if high_conf_ratio < _LOW_HIGH_CONF:
        signals.append("low_high_conf")
    if size < _SMALL_TAG:
        signals.append("small_tag")
    elif size > _LARGE_TAG:
        signals.append("large_tag")
    if top_neighbors and top_neighbors[0][1] >= _HIGH_OVERLAP:
        signals.append("high_overlap")
    if recently_modified:
        signals.append("recently_modified")

    if "recently_modified" in signals:
        return TagTriage(
            name=name, size=size, coherence=coherence,
            high_conf_ratio=high_conf_ratio, top_neighbors=top_neighbors,
            signals=signals,
            suggested_action=None,
            suggestion_confidence="low",
            rationale="modified earlier this run, cooldown — skip unless evidence is strong",
        )

    if "small_tag" in signals:
        action = "deprecate"
        conf = "high"
        rationale = (
            f"size={size} below {_SMALL_TAG} — coverage too low to justify its own tag; "
            f"consider deprecate or merge into closest neighbor"
        )
    elif "low_coherence" in signals and "large_tag" in signals:
        action = "split"
        conf = "medium"
        rationale = (
            f"large ({size} records) and incoherent (coh={coherence:.2f}) — "
            f"likely multiple distinct concepts; inspect samples to confirm split axis"
        )
    elif "low_coherence" in signals and "high_overlap" in signals:
        nb = top_neighbors[0][0]
        action = "refine"
        conf = "medium"
        rationale = (
            f"low coherence (coh={coherence:.2f}) plus high overlap with `{nb}` — "
            f"boundary drift; refine definition to disambiguate, or split if two distinct clusters"
        )
    elif "low_coherence" in signals:
        action = "refine"
        conf = "medium"
        rationale = (
            f"low coherence (coh={coherence:.2f}) at moderate size ({size}) — "
            f"definition likely admits too many subtopics; refine or split"
        )
    elif "low_high_conf" in signals:
        action = "refine"
        conf = "low"
        rationale = (
            f"only {high_conf_ratio:.0%} high-confidence assignments — "
            f"definition wording may be ambiguous to the classifier"
        )
    elif "high_overlap" in signals:
        nb = top_neighbors[0][0]
        action = "inspect"
        conf = "low"
        rationale = (
            f"high overlap with `{nb}` despite ok coherence — possible boundary issue, "
            f"not yet actionable"
        )
    else:
        action = None
        conf = "n/a"
        rationale = "healthy"

    return TagTriage(
        name=name, size=size, coherence=coherence,
        high_conf_ratio=high_conf_ratio, top_neighbors=top_neighbors,
        signals=signals,
        suggested_action=action,
        suggestion_confidence=conf,
        rationale=rationale,
    )


def compute_triage(
    vocab: list[dict],
    assignments: list[dict],
    db_path: Path,
    *,
    recently_modified: set[str] | None = None,
) -> TriageReport:
    recently_modified = recently_modified or set()

    # per-tag counts
    tag_size: dict[str, int] = defaultdict(int)
    tag_high_conf: dict[str, int] = defaultdict(int)
    cooccur: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    untagged = 0

    for a in assignments:
        if a.get("missing"):
            continue
        tags = a.get("selected_tags", [])
        if not tags:
            untagged += 1
            continue
        names = [t["name"] for t in tags]
        for t in tags:
            tag_size[t["name"]] += 1
            if t.get("confidence") == "high":
                tag_high_conf[t["name"]] += 1
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                cooccur[n1][n2] += 1
                cooccur[n2][n1] += 1

    # coherence (only for eligible tags: size in [15, 300])
    coh_by_tag: dict[str, float] = {}
    try:
        het = compute_heterogeneous_tags(
            vocab, assignments, db_path,
            coherence_threshold=1.0,  # accept all so we get coherence for every eligible tag
            min_records=_SMALL_TAG,
        )
        for h in het:
            coh_by_tag[h["tag"]] = h["coherence"]
    except Exception:
        # fall back: triage without coherence signal
        pass

    candidates: list[TagTriage] = []
    healthy: list[str] = []

    for tag in vocab:
        name = tag["name"]
        size = tag_size.get(name, 0)
        if size == 0:
            healthy.append(name + "(0)")
            continue
        ratio = tag_high_conf.get(name, 0) / size if size else 0.0
        neighbors = sorted(
            cooccur.get(name, {}).items(), key=lambda x: -x[1]
        )[:5]
        t = _classify(
            name=name,
            size=size,
            coherence=coh_by_tag.get(name),
            high_conf_ratio=ratio,
            top_neighbors=neighbors,
            recently_modified=(name in recently_modified),
        )
        if t.signals:
            candidates.append(t)
        else:
            healthy.append(name)

    candidates.sort(key=_signal_priority)

    return TriageReport(
        candidates=candidates,
        healthy=healthy,
        untagged_records=untagged,
        total_records=sum(1 for a in assignments if not a.get("missing")),
        total_tags=len(vocab),
    )
