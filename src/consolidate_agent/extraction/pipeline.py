from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
from langgraph.graph import END, START, StateGraph

from consolidate_agent.extraction.chunk import normalized_transcript_hash, split_transcript
from consolidate_agent.config import Settings
from consolidate_agent.extraction.extract import PitfallExtractor
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.extraction.merge import merge_session_candidates
from consolidate_agent.extraction.normalize import SessionIndex, SessionNormalizer
from consolidate_agent.types import (
    AdmissionStatus,
    CursorState,
    PitfallCandidate,
    PitfallEvidence,
    PitfallRecord,
    PitfallScope,
    PipelineState,
    ProcessedIndex,
    ProcessedSessionState,
    ProcessedStatus,
    TranscriptChunk,
    normalize_text,
    utc_now,
)

MAX_CONCURRENT_CHUNKS = 5


class ConsolidationGraph:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.graph = self._build_graph().compile()

    def invoke(self, state: PipelineState) -> PipelineState:
        return PipelineState.model_validate(self.graph.invoke(state))

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("discover_sessions", self.discover_sessions)
        graph.add_node("normalize_sessions", self.normalize_sessions)
        graph.add_node("extract_candidates", self.extract_candidates)
        graph.add_node("admit_candidates", self.admit_candidates)
        graph.add_node("render_outputs", self.render_outputs)
        graph.add_edge(START, "discover_sessions")
        graph.add_edge("discover_sessions", "normalize_sessions")
        graph.add_edge("normalize_sessions", "extract_candidates")
        graph.add_edge("extract_candidates", "admit_candidates")
        graph.add_edge("admit_candidates", "render_outputs")
        graph.add_edge("render_outputs", END)
        return graph

    def discover_sessions(self, state: PipelineState) -> PipelineState:
        input_dir = Path(state.input_dir).expanduser()
        _progress(f"discover_sessions start input_dir={input_dir} sample_limit={state.sample_limit or 'none'}")
        sessions = sorted(str(path) for path in input_dir.rglob("*.jsonl"))
        if state.sample_limit:
            sessions = sessions[: state.sample_limit]
        state.sessions = sessions
        state.stats.discovered_sessions = len(sessions)
        _progress(f"discover_sessions done discovered={len(sessions)}")
        return state

    def normalize_sessions(self, state: PipelineState) -> PipelineState:
        _progress(f"normalize_sessions start sessions={len(state.sessions)}")
        state.processed_index = read_processed_index(Path(state.processed_index_path).expanduser())
        index = SessionIndex.load(Path(state.session_index_path).expanduser() if state.session_index_path else None)
        normalizer = SessionNormalizer(index)

        transcripts = []
        for index, session_path in enumerate(state.sessions, start=1):
            _progress(f"normalize_sessions session={index}/{len(state.sessions)} path={session_path}")
            transcript = normalizer.normalize_file(Path(session_path))
            if transcript is None:
                state.stats.skipped_sessions += 1
                _progress(f"normalize_sessions skipped subagent session path={session_path}")
                continue
            normalized_hash = normalized_transcript_hash(transcript)
            existing = state.processed_index.sessions.get(transcript.session_id)
            if existing and existing.status == ProcessedStatus.PROCESSED and existing.normalized_hash == normalized_hash:
                state.stats.skipped_sessions += 1
                _progress(f"normalize_sessions skipped session_id={transcript.session_id}")
                continue

            state.transcript_paths[transcript.session_id] = session_path
            state.transcript_hashes[transcript.session_id] = normalized_hash
            state.processed_index.sessions[transcript.session_id] = ProcessedSessionState(
                session_id=transcript.session_id,
                path=session_path,
                normalized_hash=normalized_hash,
                status=ProcessedStatus.PROCESSING,
                chunk_count=0,
                processed_at=None,
                error=None,
            )
            transcripts.append(transcript)

        state.transcripts = transcripts
        _progress(f"normalize_sessions done pending={len(transcripts)} skipped={state.stats.skipped_sessions}")
        return state

    def extract_candidates(self, state: PipelineState) -> PipelineState:
        _progress(f"extract_candidates start transcripts={len(state.transcripts)}")
        extractor = PitfallExtractor(self.settings)
        all_candidates = []
        all_chunks: list[TranscriptChunk] = []
        successful_session_ids = []
        failed_session_ids = []
        chunk_results_by_session: dict[str, list] = {}
        chunk_errors_by_session: dict[str, Exception] = {}

        for transcript_index, transcript in enumerate(state.transcripts, start=1):
            chunks = split_transcript(transcript, state.max_chunk_chars, state.overlap_messages)
            _progress(
                f"extract_candidates session={transcript_index}/{len(state.transcripts)} "
                f"session_id={transcript.session_id} chunks={len(chunks)}"
            )
            all_chunks.extend(chunks)
            chunk_results_by_session[transcript.session_id] = []
            state.processed_index.sessions[transcript.session_id].chunk_count = len(chunks)

        def extract_chunk(chunk: TranscriptChunk):
            _progress(
                f"extract_candidates chunk start session_id={chunk.session_id} "
                f"chunk={chunk.chunk_index + 1}/{chunk.total_chunks} chars={chunk.char_count}"
            )
            chunk_candidates = extractor.extract_chunk(chunk)
            _progress(
                f"extract_candidates chunk done session_id={chunk.session_id} "
                f"chunk={chunk.chunk_index + 1}/{chunk.total_chunks} candidates={len(chunk_candidates)}"
            )
            return chunk, chunk_candidates

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHUNKS) as executor:
            future_to_chunk = {executor.submit(extract_chunk, chunk): chunk for chunk in all_chunks}
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    _, chunk_candidates = future.result()
                    chunk_results_by_session[chunk.session_id].extend(chunk_candidates)
                except Exception as exc:  # noqa: BLE001 - session-level failure must not stop the batch.
                    chunk_errors_by_session.setdefault(chunk.session_id, exc)

        for transcript in state.transcripts:
            session_id = transcript.session_id
            if session_id in chunk_errors_by_session:
                failed_session_ids.append(session_id)
                exc = chunk_errors_by_session[session_id]
                self._mark_failed(state, session_id, exc)
                _progress(f"extract_candidates session failed session_id={session_id} error={_concise_error(exc)}")
                continue
            session_candidates = chunk_results_by_session.get(session_id, [])
            merged_candidates = merge_session_candidates(session_candidates)
            all_candidates.extend(merged_candidates)
            successful_session_ids.append(session_id)
            _progress(
                f"extract_candidates session done session_id={session_id} "
                f"raw_candidates={len(session_candidates)} merged_candidates={len(merged_candidates)}"
            )

        state.chunks = all_chunks
        state.candidates = all_candidates
        state.successful_session_ids = successful_session_ids
        state.failed_session_ids = failed_session_ids
        state.stats.chunk_count = len(all_chunks)
        state.stats.candidate_count = len(all_candidates)
        state.stats.failed_sessions = len(failed_session_ids)
        _progress(
            f"extract_candidates done chunks={len(all_chunks)} candidates={len(all_candidates)} "
            f"successful_sessions={len(successful_session_ids)} failed_sessions={len(failed_session_ids)}"
        )
        return state

    def admit_candidates(self, state: PipelineState) -> PipelineState:
        _progress(f"admit_candidates start candidates={len(state.candidates)}")
        accepted, rejected = _admit_candidates(state.candidates)
        if state.knowledge_db_path:
            store = KnowledgeStore(Path(state.knowledge_db_path).expanduser())
            try:
                for record in accepted:
                    store.upsert_source_record(record)
            finally:
                store.close()
        state.accepted_records = accepted
        state.rejected_candidates = rejected
        state.stats.accepted_count = len(accepted)
        state.stats.rejected_count = len(rejected)
        _progress(
            f"admit_candidates done accepted={len(accepted)} rejected={len(rejected)}"
        )
        return state

    def render_outputs(self, state: PipelineState) -> PipelineState:
        _progress("render_outputs start")
        output_dir = Path(state.output_dir).expanduser()
        write_candidates(output_dir / "candidates.json", state.candidates)

        now = utc_now()
        for session_id in state.successful_session_ids:
            session_state = state.processed_index.sessions[session_id]
            session_state.status = ProcessedStatus.PROCESSED
            session_state.processed_at = now
            session_state.error = None
        state.stats.processed_sessions = len(state.successful_session_ids)

        last_session = state.sessions[-1] if state.sessions else None
        write_cursor(
            Path(state.cursor_path).expanduser(),
            CursorState(last_session_path=last_session, updated_at=now),
        )
        write_processed_index(Path(state.processed_index_path).expanduser(), state.processed_index)
        _progress(f"render_outputs done output_dir={output_dir} processed_sessions={state.stats.processed_sessions}")
        return state

    def _mark_failed(self, state: PipelineState, session_id: str, exc: Exception) -> None:
        session_state = state.processed_index.sessions[session_id]
        session_state.status = ProcessedStatus.FAILED
        session_state.processed_at = utc_now()
        session_state.error = _concise_error(exc)


def _concise_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:500]


def _admit_candidates(candidates: list[PitfallCandidate]) -> tuple[list[PitfallRecord], list[PitfallCandidate]]:
    accepted: list[PitfallRecord] = []
    rejected: list[PitfallCandidate] = []
    for candidate in candidates:
        if not _should_admit(candidate):
            candidate.admission_status = AdmissionStatus.REJECTED
            rejected.append(candidate)
            continue
        candidate.admission_status = AdmissionStatus.ACCEPTED
        accepted.append(_record_from_candidate(candidate))
    return accepted, rejected


def _should_admit(candidate: PitfallCandidate) -> bool:
    if candidate.scope == PitfallScope.SESSION_SPECIFIC:
        return False
    required = [candidate.trigger, candidate.failure_mode, candidate.preventive_rule]
    if any(not item.strip() for item in required):
        return False
    return bool(candidate.evidence_refs)


def _record_from_candidate(candidate: PitfallCandidate) -> PitfallRecord:
    now = utc_now()
    key = _dedupe_key(candidate)
    return PitfallRecord(
        id=_record_id(key),
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
        created_at=now,
        updated_at=now,
    )


def _dedupe_key(candidate: PitfallCandidate) -> str:
    title = normalize_text(candidate.title)
    rule = normalize_text(candidate.preventive_rule)
    return f"{candidate.category.value}|{title}|{rule}"


def _record_id(key: str) -> str:
    return f"pitfall_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


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
    payload = processed_index.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_candidates(path: Path, candidates: list[PitfallCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [candidate.model_dump(mode="json") for candidate in candidates]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_cursor(path: Path, cursor: CursorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cursor.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")


def read_cursor(path: Path) -> CursorState:
    if not path.exists():
        return CursorState()
    return CursorState.model_validate(json.loads(path.read_text(encoding="utf-8")))
