from __future__ import annotations

from pathlib import Path
from langgraph.graph import END, START, StateGraph

from consolidate_agent.admit import PitfallLibrary
from consolidate_agent.chunk import normalized_transcript_hash, split_transcript
from consolidate_agent.config import Settings
from consolidate_agent.extract import PitfallExtractor
from consolidate_agent.merge import merge_session_candidates
from consolidate_agent.normalize import SessionIndex, SessionNormalizer
from consolidate_agent.render import write_candidates, write_cursor, write_pitfalls_markdown
from consolidate_agent.state import read_processed_index, write_processed_index
from consolidate_agent.types import (
    CursorState,
    GraphState,
    PipelineState,
    ProcessedSessionState,
    ProcessedStatus,
    TranscriptChunk,
    utc_now,
)


class ConsolidationGraph:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.graph = self._build_graph().compile()

    def invoke(self, state: PipelineState) -> PipelineState:
        result = self.graph.invoke(state.model_dump(mode="python"))
        return PipelineState.model_validate(result)

    def _build_graph(self):
        graph = StateGraph(GraphState)
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

    def discover_sessions(self, raw_state: GraphState) -> GraphState:
        state = PipelineState.model_validate(raw_state)
        input_dir = Path(state.input_dir).expanduser()
        sessions = sorted(str(path) for path in input_dir.rglob("*.jsonl"))
        if state.sample_limit:
            sessions = sessions[: state.sample_limit]
        state.sessions = sessions
        state.stats.discovered_sessions = len(sessions)
        return state.model_dump(mode="python")

    def normalize_sessions(self, raw_state: GraphState) -> GraphState:
        state = PipelineState.model_validate(raw_state)
        state.processed_index = read_processed_index(Path(state.processed_index_path).expanduser())
        index = SessionIndex.load(Path(state.session_index_path).expanduser() if state.session_index_path else None)
        normalizer = SessionNormalizer(index)

        transcripts = []
        for session_path in state.sessions:
            transcript = normalizer.normalize_file(Path(session_path))
            normalized_hash = normalized_transcript_hash(transcript)
            existing = state.processed_index.sessions.get(transcript.session_id)
            if existing and existing.status == ProcessedStatus.PROCESSED and existing.normalized_hash == normalized_hash:
                state.stats.skipped_sessions += 1
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
        return state.model_dump(mode="python")

    def extract_candidates(self, raw_state: GraphState) -> GraphState:
        state = PipelineState.model_validate(raw_state)
        extractor: PitfallExtractor | None = None
        all_candidates = []
        all_chunks: list[TranscriptChunk] = []
        successful_session_ids = []
        failed_session_ids = []

        for transcript in state.transcripts:
            chunks = split_transcript(transcript, state.max_chunk_chars, state.overlap_messages)
            all_chunks.extend(chunks)
            state.processed_index.sessions[transcript.session_id].chunk_count = len(chunks)
            try:
                session_candidates = []
                if extractor is None:
                    extractor = PitfallExtractor(self.settings)
                for chunk in chunks:
                    session_candidates.extend(extractor.extract_chunk(chunk))
                merged_candidates = merge_session_candidates(session_candidates)
                all_candidates.extend(merged_candidates)
                successful_session_ids.append(transcript.session_id)
            except Exception as exc:  # noqa: BLE001 - session-level failure must not stop the batch.
                failed_session_ids.append(transcript.session_id)
                self._mark_failed(state, transcript.session_id, exc)

        state.chunks = all_chunks
        state.candidates = all_candidates
        state.successful_session_ids = successful_session_ids
        state.failed_session_ids = failed_session_ids
        state.stats.chunk_count = len(all_chunks)
        state.stats.candidate_count = len(all_candidates)
        state.stats.failed_sessions = len(failed_session_ids)
        return state.model_dump(mode="python")

    def admit_candidates(self, raw_state: GraphState) -> GraphState:
        state = PipelineState.model_validate(raw_state)
        output_dir = Path(state.output_dir).expanduser()
        library_path = output_dir / "pitfalls.json"
        library = PitfallLibrary.load(library_path)
        accepted, rejected = library.admit(state.candidates)
        library.save(library_path)
        state.accepted_records = accepted
        state.rejected_candidates = rejected
        state.stats.accepted_count = len(accepted)
        state.stats.rejected_count = len(rejected)
        state.stats.library_size = len(library.records)
        return state.model_dump(mode="python")

    def render_outputs(self, raw_state: GraphState) -> GraphState:
        state = PipelineState.model_validate(raw_state)
        output_dir = Path(state.output_dir).expanduser()
        library = PitfallLibrary.load(output_dir / "pitfalls.json")
        write_candidates(output_dir / "candidates.json", state.candidates)
        write_pitfalls_markdown(output_dir / "pitfalls.md", library.records)

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
        return state.model_dump(mode="python")

    def _mark_failed(self, state: PipelineState, session_id: str, exc: Exception) -> None:
        session_state = state.processed_index.sessions[session_id]
        session_state.status = ProcessedStatus.FAILED
        session_state.processed_at = utc_now()
        session_state.error = _concise_error(exc)


def _concise_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:500]
