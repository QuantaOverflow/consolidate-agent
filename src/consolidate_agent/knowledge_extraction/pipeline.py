from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from consolidate_agent.config import Settings
from consolidate_agent.context import ProcessedSession, SessionContextEngineer
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.knowledge_extraction.extractor import KnowledgeExtractor, KnowledgeItemInput
from consolidate_agent.types import (
    KnowledgeExtractionStats,
    KnowledgeRecord,
    KnowledgeScope,
    ProcessedIndex,
    ProcessedSessionState,
    ProcessedStatus,
    utc_now,
)

MAX_CONCURRENT_SESSIONS = 5


@dataclass(frozen=True)
class _PendingSession:
    path: Path
    session: ProcessedSession
    normalized_hash: str
    total_turns: int


def run_knowledge_extraction(
    input_dir: Path,
    output_dir: Path,
    processed_index_path: Path,
    knowledge_db_path: Path,
    settings: Settings,
    sample_limit: int | None = None,
    max_session_chars: int = 100_000,
    extractor: KnowledgeExtractor | None = None,
) -> KnowledgeExtractionStats:
    stats = KnowledgeExtractionStats()
    input_dir = input_dir.expanduser()
    output_dir = output_dir.expanduser()
    processed_index_path = processed_index_path.expanduser()
    knowledge_db_path = knowledge_db_path.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    _progress(f"knowledge discover_sessions start input_dir={input_dir} sample_limit={sample_limit or 'none'}")
    paths = sorted(input_dir.rglob("*.jsonl"))
    if sample_limit is not None:
        paths = paths[:sample_limit]
    stats.discovered_sessions = len(paths)
    _progress(f"knowledge discover_sessions done discovered={len(paths)}")

    processed_index = read_processed_index(processed_index_path)
    engineer = SessionContextEngineer(sessions_dir=input_dir)
    pending: list[_PendingSession] = []

    _progress(f"knowledge preprocess_sessions start sessions={len(paths)}")
    for index, path in enumerate(paths, start=1):
        _progress(f"knowledge preprocess_sessions session={index}/{len(paths)} path={path}")
        session = engineer.process(path)
        if session is None:
            stats.skipped_sessions += 1
            continue
        normalized_hash = str(session.stats.processed_chars)
        existing = processed_index.sessions.get(session.session_id)
        if session.is_sub_agent or session.stats.processed_chars < 200 or session.stats.processed_chars > max_session_chars:
            stats.skipped_sessions += 1
            continue
        if existing and existing.status == ProcessedStatus.PROCESSED and existing.normalized_hash == normalized_hash:
            stats.skipped_sessions += 1
            continue

        processed_index.sessions[session.session_id] = ProcessedSessionState(
            session_id=session.session_id,
            path=str(path),
            normalized_hash=normalized_hash,
            status=ProcessedStatus.PROCESSING,
            chunk_count=1,
            processed_at=None,
            error=None,
        )
        pending.append(
            _PendingSession(
                path=path,
                session=session,
                normalized_hash=normalized_hash,
                total_turns=_count_turns(session.xml),
            )
        )
    _progress(f"knowledge preprocess_sessions done pending={len(pending)} skipped={stats.skipped_sessions}")

    store = KnowledgeStore(knowledge_db_path)
    try:
        extractor = extractor or KnowledgeExtractor(settings)
        _progress(f"knowledge extract_sessions start sessions={len(pending)}")

        def extract_session(pending_session: _PendingSession) -> tuple[_PendingSession, list[KnowledgeItemInput]]:
            _progress(
                f"knowledge extract_sessions session start session_id={pending_session.session.session_id} "
                f"chars={pending_session.session.stats.processed_chars}"
            )
            items = extractor.extract(pending_session.session)
            _progress(
                f"knowledge extract_sessions session done session_id={pending_session.session.session_id} "
                f"items={len(items)}"
            )
            return pending_session, items

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SESSIONS) as executor:
            future_to_session = {executor.submit(extract_session, item): item for item in pending}
            for future in as_completed(future_to_session):
                pending_session = future_to_session[future]
                session_id = pending_session.session.session_id
                try:
                    _, items = future.result()
                except Exception as exc:  # noqa: BLE001 - one failed session must not stop the run.
                    stats.failed_sessions += 1
                    _mark_failed(processed_index, session_id, exc)
                    _progress(f"knowledge extract_sessions session failed session_id={session_id} error={_concise_error(exc)}")
                    continue

                admitted, rejected = _admit_items(items)
                stats.extracted_count += len(items)
                stats.admitted_count += len(admitted)
                stats.rejected_count += len(rejected)
                for item in admitted:
                    record = _record_from_item(
                        pending_session.session,
                        item,
                        total_turns=pending_session.total_turns,
                    )
                    store.upsert_knowledge_record(record)

                session_state = processed_index.sessions[session_id]
                session_state.status = ProcessedStatus.PROCESSED
                session_state.processed_at = utc_now()
                session_state.error = None
                stats.processed_sessions += 1
        _progress(
            f"knowledge extract_sessions done processed={stats.processed_sessions} failed={stats.failed_sessions} "
            f"extracted={stats.extracted_count} admitted={stats.admitted_count} rejected={stats.rejected_count}"
        )
    finally:
        store.close()

    write_processed_index(processed_index_path, processed_index)
    _progress(f"knowledge write_index done path={processed_index_path}")
    return stats


def _admit_items(items: list[KnowledgeItemInput]) -> tuple[list[KnowledgeItemInput], list[KnowledgeItemInput]]:
    admitted: list[KnowledgeItemInput] = []
    rejected: list[KnowledgeItemInput] = []
    for item in items:
        if item.scope == KnowledgeScope.SESSION_SPECIFIC or not item.evidence_turns:
            rejected.append(item)
        else:
            admitted.append(item)
    return admitted, rejected


def _record_from_item(session: ProcessedSession, item: KnowledgeItemInput, total_turns: int) -> KnowledgeRecord:
    now = utc_now()
    evidence_turns = sorted(set(item.evidence_turns))
    return KnowledgeRecord(
        id=_record_id(session.session_id, item.title),
        session_id=session.session_id,
        title=item.title,
        insight=item.insight,
        applicability=item.applicability,
        scope=item.scope,
        evidence_turns=evidence_turns,
        evidence_count=len(evidence_turns),
        evidence_spread=_evidence_spread(evidence_turns, total_turns),
        processed_chars=session.stats.processed_chars,
        created_at=now,
        updated_at=now,
    )


def _record_id(session_id: str, title: str) -> str:
    digest = hashlib.sha1(f"{session_id}:{title}".encode("utf-8")).hexdigest()[:12]
    return f"knowledge_{digest}"


def _evidence_spread(evidence_turns: list[int], total_turns: int) -> float:
    if total_turns <= 1 or len(evidence_turns) < 2:
        return 0.0
    return (max(evidence_turns) - min(evidence_turns)) / (total_turns - 1)


def _count_turns(xml: str) -> int:
    turn_count = len(re.findall(r"<turn(?:\s|>)", xml))
    if turn_count:
        return turn_count
    return len(re.findall(r"<user(?:\s|>)", xml))


def _mark_failed(processed_index: ProcessedIndex, session_id: str, exc: Exception) -> None:
    session_state = processed_index.sessions[session_id]
    session_state.status = ProcessedStatus.FAILED
    session_state.processed_at = utc_now()
    session_state.error = _concise_error(exc)


def _concise_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:500]


def _progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[{timestamp}] {message}", flush=True)


def read_processed_index(path: Path) -> ProcessedIndex:
    if not path.exists():
        return ProcessedIndex()
    return ProcessedIndex.model_validate(json.loads(path.read_text(encoding="utf-8")))


def write_processed_index(path: Path, processed_index: ProcessedIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    processed_index.updated_at = utc_now()
    path.write_text(
        json.dumps(processed_index.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
