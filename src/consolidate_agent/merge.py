from __future__ import annotations

import hashlib
import re

from consolidate_agent.types import PitfallCandidate


def merge_session_candidates(candidates: list[PitfallCandidate]) -> list[PitfallCandidate]:
    merged: dict[str, PitfallCandidate] = {}
    for candidate in candidates:
        key = _dedupe_key(candidate)
        existing = merged.get(key)
        if existing is None:
            candidate.evidence_refs = _unique_sorted(candidate.evidence_refs)
            candidate.candidate_id = _candidate_id(candidate.session_id, key)
            merged[key] = candidate
            continue
        existing.evidence_refs = _unique_sorted([*existing.evidence_refs, *candidate.evidence_refs])
        existing.confidence = max(existing.confidence, candidate.confidence)
        if existing.chunk_id is None:
            existing.chunk_id = candidate.chunk_id
    return [merged[key] for key in sorted(merged)]


def _dedupe_key(candidate: PitfallCandidate) -> str:
    title = _normalize_string(candidate.title)
    rule = _normalize_string(candidate.preventive_rule)
    return f"{candidate.category.value}|{title}|{rule}"


def _normalize_string(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _candidate_id(session_id: str, key: str) -> str:
    digest = hashlib.sha1(f"{session_id}:{key}".encode("utf-8")).hexdigest()[:12]
    return f"candidate_{digest}"


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})
