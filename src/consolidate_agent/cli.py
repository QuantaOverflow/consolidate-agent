from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from consolidate_agent.config import Settings
from consolidate_agent.knowledge.evidence_agent import EvidenceAgent
from consolidate_agent.knowledge_extraction.pipeline import run_knowledge_extraction
from consolidate_agent.knowledge.session_turn_store import SessionTurnStore
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.knowledge.vector import KnowledgeVectorStore, create_dashscope_embeddings, search_knowledge

_STAGES = ("extract", "verify", "embed")
_SESSION_TURNS_CHROMA_PATH = Path("outputs/chroma/session_turns")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge extraction and consolidation pipeline")
    parser.add_argument("--stage", choices=_STAGES, help="Pipeline stage to run")
    parser.add_argument("--input-dir", help="Session input directory")
    parser.add_argument("--knowledge-db-path", help="Knowledge database path")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit the number of sessions")
    parser.add_argument("--knowledge-processed-index-path", help="Processed index path for knowledge extraction")
    parser.add_argument(
        "--max-session-chars",
        type=int,
        default=100_000,
        help="Max processed chars per session for knowledge extraction",
    )
    parser.add_argument(
        "--evidence-workers",
        type=int,
        default=10,
        help="Number of session-level Evidence Agent workers",
    )
    parser.add_argument(
        "--evidence-failure-trace-path",
        help="Evidence Agent structured-output failure/recovery trace JSONL path",
    )
    parser.add_argument(
        "--evidence-reject-trace-path",
        help="Evidence Agent judge reject trace JSONL path",
    )
    parser.add_argument("--embed-turns", action="store_true", help="Build embedding index for processed session turns")
    parser.add_argument("--query", help="Search embedded knowledge records by text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of knowledge search results to return")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)

    settings = Settings()
    paths = _resolve_paths(args, settings)

    if args.embed_turns:
        _run_embed_turns(settings, paths.knowledge_db_path)
        return

    if args.query is not None:
        _run_query(settings, paths.knowledge_db_path, args)
        return

    if args.stage == "extract":
        _run_extract(settings, paths, args)
        return
    if args.stage == "verify":
        _with_store(paths.knowledge_db_path, lambda store: _run_verify(store, settings, args))
        return
    if args.stage == "embed":
        _with_store(paths.knowledge_db_path, lambda store: _run_embed(store, settings))
        return

    parser.print_help()


class _ResolvedPaths:
    def __init__(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        knowledge_processed_index_path: Path,
        knowledge_db_path: Path,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.knowledge_processed_index_path = knowledge_processed_index_path
        self.knowledge_db_path = knowledge_db_path


def _validate_args(args: argparse.Namespace) -> None:
    exclusive_modes = [
        args.stage is not None,
        args.query is not None,
        args.embed_turns,
    ]
    if sum(1 for enabled in exclusive_modes if enabled) > 1:
        print("Error: --stage, --query, and --embed-turns are mutually exclusive")
        sys.exit(1)


def _resolve_paths(args: argparse.Namespace, settings: Settings) -> _ResolvedPaths:
    return _ResolvedPaths(
        input_dir=Path(args.input_dir or settings.sessions_dir_path).expanduser(),
        output_dir=Path(settings.output_dir_path).expanduser(),
        knowledge_processed_index_path=Path(
            args.knowledge_processed_index_path or settings.knowledge_processed_index_path
        ).expanduser(),
        knowledge_db_path=Path(args.knowledge_db_path or settings.knowledge_db_path).expanduser(),
    )


def _with_store(knowledge_db_path: Path, fn) -> None:
    store = KnowledgeStore(knowledge_db_path)
    try:
        fn(store)
    finally:
        store.close()


def _run_extract(settings: Settings, paths: _ResolvedPaths, args: argparse.Namespace) -> None:
    stats = run_knowledge_extraction(
        input_dir=paths.input_dir,
        output_dir=paths.output_dir,
        processed_index_path=paths.knowledge_processed_index_path,
        knowledge_db_path=paths.knowledge_db_path,
        settings=settings,
        sample_limit=args.sample_limit,
        max_session_chars=args.max_session_chars,
        evidence_agent=None,
    )
    _print_extraction_stats(stats)


def _run_verify(store: KnowledgeStore, settings: Settings, args: argparse.Namespace) -> None:
    records = store.list_pending_unverified_records()
    if not records:
        print("No unverified records found.")
        return

    session_xmls: dict[str, str] = {}
    for record in records:
        if record.session_id in session_xmls:
            continue
        xml = store.get_processed_session(record.session_id)
        if xml:
            session_xmls[record.session_id] = xml

    session_turn_store = SessionTurnStore(
        chroma_path=_SESSION_TURNS_CHROMA_PATH,
        embeddings=create_dashscope_embeddings(settings),
    )
    evidence_agent = EvidenceAgent(
        settings,
        session_turn_store,
        workers=args.evidence_workers,
        **_evidence_agent_kwargs(args),
    )

    verified = evidence_agent.verify_batch(records, session_xmls)
    updated = 0
    for record in verified:
        if record.evidence_count > 0:
            store.upsert_knowledge_record(record)
            updated += 1
    print(f"Verified {len(records)} records, {updated} admitted with evidence.")


def _run_embed(store: KnowledgeStore, settings: Settings) -> None:
    vector_store = KnowledgeVectorStore(store, create_dashscope_embeddings(settings))
    records = store.list_knowledge_records_without_embedding()
    vector_store.embed_knowledge_records(records)
    print(f"Embedded knowledge records: {len(records)}")


def _run_embed_turns(settings: Settings, knowledge_db_path: Path) -> None:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_qwq import ChatQwen

    from consolidate_agent.prompt_loader import load_prompt

    store = KnowledgeStore(knowledge_db_path)
    try:
        llm = ChatQwen(
            model=settings.qwen_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_api_base,
            temperature=0,
        )
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("turn_summarization_system.md")),
                ("user", load_prompt("turn_summarization_user.md")),
            ],
            template_format="jinja2",
        )
        translation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("query_translation_system.md")),
                ("user", load_prompt("query_translation_user.md")),
            ],
            template_format="jinja2",
        )

        def summarizer(turn_xml: str) -> str:
            try:
                return (summarization_prompt | llm).invoke({"turn_xml": turn_xml}).content
            except Exception:
                return turn_xml[:500]

        def translator(query: str) -> str:
            return (translation_prompt | llm).invoke({"query": query}).content

        session_turn_store = SessionTurnStore(
            chroma_path=_SESSION_TURNS_CHROMA_PATH,
            embeddings=create_dashscope_embeddings(settings),
            summarizer=summarizer,
            translator=translator,
        )
        embedded_ids = session_turn_store.embedded_session_ids()
        sessions = store.list_unembedded_sessions(embedded_ids)
        embedded_turn_count = 0
        _lock = threading.Lock()

        def _embed_one(args: tuple[int, str, str]) -> int:
            index, session_id, xml = args
            added = session_turn_store.embed_session(session_id, xml)
            print(f"Embedding session {index}/{len(sessions)}: {session_id} turns={added}")
            return added

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_embed_one, (index, session_id, xml)): session_id
                for index, (session_id, xml) in enumerate(sessions, start=1)
            }
            for future in as_completed(futures):
                with _lock:
                    embedded_turn_count += future.result()
        print(f"Embedded {embedded_turn_count} turns from {len(sessions)} sessions")
    finally:
        store.close()


def _run_query(settings: Settings, knowledge_db_path: Path, args: argparse.Namespace) -> None:
    store = KnowledgeStore(knowledge_db_path)
    try:
        results = search_knowledge(
            store=store,
            embeddings=create_dashscope_embeddings(settings),
            query_text=args.query,
            top_k=args.top_k,
        )
        for index, result in enumerate(results, start=1):
            record = result["record"]
            score = result["similarity_score"]
            print(f"[{index}] score={score:.2f} | {record.scope.value}")
            print(f"    Title: {record.title}")
            print(f"    Insight: {_truncate(record.insight, 150)}")
            print(f"    Applicability: {_truncate(record.applicability, 80)}")
    finally:
        store.close()


def _evidence_agent_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    kwargs = {}
    if args.evidence_failure_trace_path:
        kwargs["failure_trace_path"] = Path(args.evidence_failure_trace_path).expanduser()
    if args.evidence_reject_trace_path:
        kwargs["reject_trace_path"] = Path(args.evidence_reject_trace_path).expanduser()
    return kwargs


def _print_extraction_stats(stats) -> None:
    print(f"Knowledge extracted: {stats.extracted_count}")
    print(f"Knowledge admitted: {stats.admitted_count}")
    print(f"Knowledge rejected: {stats.rejected_count}")
    print(f"Evidence admitted: {stats.evidence_admitted_count}")
    print(f"Evidence rejected: {stats.evidence_rejected_count}")
    print(f"Sessions processed: {stats.processed_sessions}")
    print(f"Sessions skipped: {stats.skipped_sessions}")
    if stats.skip_reasons:
        print(f"Session skip reasons: {dict(sorted(stats.skip_reasons.items()))}")
    print(f"Sessions failed: {stats.failed_sessions}")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3]}..."


if __name__ == "__main__":
    main()
