from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from consolidate_agent.admit import PitfallLibrary
from consolidate_agent.config import Settings
from consolidate_agent.extract import PitfallExtractor
from consolidate_agent.normalize import SessionIndex, SessionNormalizer
from consolidate_agent.render import read_cursor, write_candidates, write_cursor, write_pitfalls_markdown
from consolidate_agent.types import CursorState, GraphState, PipelineState, utc_now


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
        cursor = read_cursor(Path(state.cursor_path).expanduser())
        sessions = sorted(str(path) for path in input_dir.rglob("*.jsonl"))
        if cursor.last_session_path:
            sessions = [path for path in sessions if path > cursor.last_session_path]
        if state.sample_limit:
            sessions = sessions[: state.sample_limit]
        state.sessions = sessions
        state.stats.discovered_sessions = len(sessions)
        return state.model_dump(mode="python")

    def normalize_sessions(self, raw_state: GraphState) -> GraphState:
        state = PipelineState.model_validate(raw_state)
        index = SessionIndex.load(Path(state.session_index_path).expanduser() if state.session_index_path else None)
        normalizer = SessionNormalizer(index)
        state.transcripts = [normalizer.normalize_file(Path(path)) for path in state.sessions]
        state.stats.processed_sessions = len(state.transcripts)
        return state.model_dump(mode="python")

    def extract_candidates(self, raw_state: GraphState) -> GraphState:
        state = PipelineState.model_validate(raw_state)
        extractor = PitfallExtractor(self.settings)
        candidates = []
        for transcript in state.transcripts:
            candidates.extend(extractor.extract(transcript))
        state.candidates = candidates
        state.stats.candidate_count = len(candidates)
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
        last_session = state.sessions[-1] if state.sessions else None
        write_cursor(
            Path(state.cursor_path).expanduser(),
            CursorState(last_session_path=last_session, updated_at=utc_now()),
        )
        return state.model_dump(mode="python")
