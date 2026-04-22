from __future__ import annotations

import argparse
from pathlib import Path

from consolidate_agent.config import Settings
from consolidate_agent.graph import ConsolidationGraph
from consolidate_agent.types import PipelineState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Codex pitfall extraction pipeline")
    parser.add_argument("--input-dir", help="Session input directory")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--cursor-path", help="Cursor file path")
    parser.add_argument("--session-index", help="Session index path")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit the number of sessions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings()
    input_dir = Path(args.input_dir or settings.sessions_dir_path).expanduser()
    output_dir = Path(args.output_dir or settings.output_dir_path).expanduser()
    cursor_path = Path(args.cursor_path or settings.cursor_path).expanduser()
    session_index_path = Path(args.session_index or settings.session_index_path).expanduser()

    state = PipelineState(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        cursor_path=str(cursor_path),
        session_index_path=str(session_index_path),
        sample_limit=args.sample_limit,
    )
    graph = ConsolidationGraph(settings)
    result = graph.invoke(state)
    print(f"Discovered sessions: {result.stats.discovered_sessions}")
    print(f"Processed sessions: {result.stats.processed_sessions}")
    print(f"Candidates: {result.stats.candidate_count}")
    print(f"Accepted: {result.stats.accepted_count}")
    print(f"Rejected: {result.stats.rejected_count}")
    print(f"Library size: {result.stats.library_size}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
