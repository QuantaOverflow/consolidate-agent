from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from consolidate_agent.types import (
    AdmissionStatus,
    PitfallCandidate,
    PitfallEvidence,
    PitfallRecord,
    PitfallScope,
    utc_now,
)


class PitfallLibrary:
    def __init__(self, records: list[PitfallRecord] | None = None):
        self.records = records or []

    @classmethod
    def load(cls, path: Path) -> "PitfallLibrary":
        if not path.exists():
            return cls([])
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([PitfallRecord.model_validate(item) for item in data])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record.model_dump(mode="json") for record in self.records]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def admit(self, candidates: list[PitfallCandidate]) -> tuple[list[PitfallRecord], list[PitfallCandidate]]:
        accepted: list[PitfallRecord] = []
        rejected: list[PitfallCandidate] = []
        for candidate in candidates:
            if not self._should_admit(candidate):
                candidate.admission_status = AdmissionStatus.REJECTED
                rejected.append(candidate)
                continue
            candidate.admission_status = AdmissionStatus.ACCEPTED
            record = self._upsert(candidate)
            accepted.append(record)
        return accepted, rejected

    def _should_admit(self, candidate: PitfallCandidate) -> bool:
        if candidate.scope == PitfallScope.SESSION_SPECIFIC:
            return False
        required = [candidate.trigger, candidate.failure_mode, candidate.preventive_rule]
        if any(not item.strip() for item in required):
            return False
        if not candidate.evidence_refs:
            return False
        return True

    def _upsert(self, candidate: PitfallCandidate) -> PitfallRecord:
        key = self._dedupe_key(candidate)
        for existing in self.records:
            if self._dedupe_key(existing) == key:
                self._merge(existing, candidate)
                return existing
        now = utc_now()
        record = PitfallRecord(
            id=self._record_id(key),
            title=candidate.title,
            category=candidate.category,
            trigger=candidate.trigger,
            failure_mode=candidate.failure_mode,
            impact=candidate.impact,
            preventive_rule=candidate.preventive_rule,
            scope=candidate.scope,
            evidence=PitfallEvidence(
                session_ids=[candidate.session_id],
                message_refs=list(candidate.evidence_refs),
            ),
            confidence=candidate.confidence,
            tags=self._tags(candidate),
            created_at=now,
            updated_at=now,
        )
        self.records.append(record)
        return record

    def _merge(self, record: PitfallRecord, candidate: PitfallCandidate) -> None:
        if candidate.session_id not in record.evidence.session_ids:
            record.evidence.session_ids.append(candidate.session_id)
        for ref in candidate.evidence_refs:
            if ref not in record.evidence.message_refs:
                record.evidence.message_refs.append(ref)
        record.confidence = max(record.confidence, candidate.confidence)
        record.updated_at = utc_now()
        for tag in self._tags(candidate):
            if tag not in record.tags:
                record.tags.append(tag)

    def _dedupe_key(self, item: PitfallCandidate | PitfallRecord) -> str:
        title = self._normalize_string(item.title)
        trigger = self._normalize_string(item.trigger)
        rule = self._normalize_string(item.preventive_rule)
        return f"{item.category.value}|{title}|{trigger}|{rule}"

    def _normalize_string(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    def _record_id(self, key: str) -> str:
        return f"pitfall_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"

    def _tags(self, candidate: PitfallCandidate) -> list[str]:
        tags = {candidate.category.value}
        for token in re.findall(r"[a-zA-Z0-9_\-.]+", f"{candidate.title} {candidate.trigger}"):
            if len(token) > 2 and len(tags) < 6:
                tags.add(token.lower())
        return sorted(tags)
