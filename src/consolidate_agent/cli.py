from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen

from consolidate_agent.consolidation.pipeline import run_consolidation
from consolidate_agent.config import Settings
from consolidate_agent.extraction.pipeline import ConsolidationGraph
from consolidate_agent.knowledge_extraction.pipeline import run_knowledge_extraction
from consolidate_agent.knowledge.session_turn_store import SessionTurnStore
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.knowledge.vector import KnowledgeVectorStore, create_dashscope_embeddings, search_knowledge
from consolidate_agent.prompt_loader import load_prompt
from consolidate_agent.types import PipelineState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Codex pitfall extraction pipeline")
    parser.add_argument("--input-dir", help="Session input directory")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--cursor-path", help="Legacy cursor file path")
    parser.add_argument("--processed-index-path", help="Processed index file path")
    parser.add_argument("--knowledge-db-path", help="Knowledge database path")
    parser.add_argument("--session-index", help="Session index path")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit the number of sessions")
    parser.add_argument("--max-chunk-chars", type=int, default=None, help="Maximum approximate chars per extraction chunk")
    parser.add_argument("--overlap-messages", type=int, default=None, help="Number of messages to overlap between chunks")
    parser.add_argument("--extract-knowledge", action="store_true", help="Run knowledge extraction pipeline on sessions")
    parser.add_argument("--knowledge-processed-index-path", help="Processed index path for knowledge extraction")
    parser.add_argument(
        "--max-session-chars",
        type=int,
        default=100_000,
        help="Max processed chars per session for knowledge extraction",
    )
    parser.add_argument("--run-evidence-agent", action="store_true", help="Run Evidence Agent to verify evidence_turns after knowledge extraction")
    parser.add_argument("--run-consolidation", action="store_true", help="Run post-processing knowledge consolidation")
    parser.add_argument("--report", action="store_true", help="Print an observability report for the latest consolidation run")
    parser.add_argument("--embed", action="store_true", help="Build embedding indexes for pitfall and knowledge records")
    parser.add_argument("--embed-turns", action="store_true", help="Build embedding index for processed session turns")
    parser.add_argument("--query", help="Search embedded knowledge records by text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of knowledge search results to return")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.embed and args.extract_knowledge:
        print("Error: --embed and --extract-knowledge are mutually exclusive")
        sys.exit(1)
    if args.query is not None and (args.embed or args.extract_knowledge or args.run_consolidation):
        print("Error: --query is mutually exclusive with --embed, --extract-knowledge, and --run-consolidation")
        sys.exit(1)
    if args.embed_turns and (args.embed or args.extract_knowledge or args.run_consolidation or args.report or args.query is not None):
        print("Error: --embed-turns is mutually exclusive with --embed, --extract-knowledge, --run-consolidation, --report, and --query")
        sys.exit(1)

    settings = Settings()
    input_dir = Path(args.input_dir or settings.sessions_dir_path).expanduser()
    output_dir = Path(args.output_dir or settings.output_dir_path).expanduser()
    cursor_path = Path(args.cursor_path or settings.cursor_path).expanduser()
    processed_index_path = Path(args.processed_index_path or settings.processed_index_path).expanduser()
    knowledge_processed_index_path = Path(
        args.knowledge_processed_index_path or settings.knowledge_processed_index_path
    ).expanduser()
    knowledge_db_path = Path(args.knowledge_db_path or settings.knowledge_db_path).expanduser()
    session_index_path = Path(args.session_index or settings.session_index_path).expanduser()

    if args.embed_turns:
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
                chroma_path=Path("outputs/chroma/session_turns"),
                embeddings=create_dashscope_embeddings(settings),
                summarizer=summarizer,
                translator=translator,
            )
            embedded_ids = session_turn_store.embedded_session_ids()
            sessions = store.list_unembedded_sessions(embedded_ids)
            embedded_turn_count = 0
            for index, (session_id, xml) in enumerate(sessions, start=1):
                added = session_turn_store.embed_session(session_id, xml)
                embedded_turn_count += added
                print(f"Embedding session {index}/{len(sessions)}: {session_id} turns={added}")
            print(f"Embedded {embedded_turn_count} turns from {len(sessions)} sessions")
        finally:
            store.close()
        return

    if args.query is not None:
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
        return

    if args.embed:
        store = KnowledgeStore(knowledge_db_path)
        try:
            vector_store = KnowledgeVectorStore(store, create_dashscope_embeddings(settings))
            canonicals = store.list_active_canonicals()
            knowledge_records = store.list_all_knowledge_records()
            missing_canonicals = sum(
                1 for canonical in canonicals if store.get_canonical_embedding(canonical.canonical_id) is None
            )
            missing_knowledge_records = len(store.list_knowledge_records_without_embedding())
            vector_store.embed_canonicals(canonicals)
            vector_store.embed_knowledge_records(knowledge_records)
            print(f"Embedded canonicals: {missing_canonicals}")
            print(f"Embedded knowledge records: {missing_knowledge_records}")
        finally:
            store.close()
        return

    if args.extract_knowledge:
        evidence_agent = None
        if args.run_evidence_agent:
            from consolidate_agent.knowledge.evidence_agent import EvidenceAgent
            session_turn_store = SessionTurnStore(
                chroma_path=Path("outputs/chroma/session_turns"),
                embeddings=create_dashscope_embeddings(settings),
            )
            evidence_agent = EvidenceAgent(settings, session_turn_store)

        stats = run_knowledge_extraction(
            input_dir=input_dir,
            output_dir=output_dir,
            processed_index_path=knowledge_processed_index_path,
            knowledge_db_path=knowledge_db_path,
            settings=settings,
            sample_limit=args.sample_limit,
            max_session_chars=args.max_session_chars,
            evidence_agent=evidence_agent,
        )
        print(f"Knowledge extracted: {stats.extracted_count}")
        print(f"Knowledge admitted: {stats.admitted_count}")
        print(f"Knowledge rejected: {stats.rejected_count}")
        print(f"Evidence admitted: {stats.evidence_admitted_count}")
        print(f"Evidence rejected: {stats.evidence_rejected_count}")
        print(f"Sessions processed: {stats.processed_sessions}")
        print(f"Sessions failed: {stats.failed_sessions}")
        return

    state = PipelineState(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        cursor_path=str(cursor_path),
        processed_index_path=str(processed_index_path),
        knowledge_db_path=str(knowledge_db_path),
        session_index_path=str(session_index_path),
        sample_limit=args.sample_limit,
        max_chunk_chars=args.max_chunk_chars or settings.consolidate_max_chunk_chars,
        overlap_messages=args.overlap_messages if args.overlap_messages is not None else settings.consolidate_overlap_messages,
    )
    graph = ConsolidationGraph(settings)
    result = graph.invoke(state)
    print(f"Discovered sessions: {result.stats.discovered_sessions}")
    print(f"Skipped sessions: {result.stats.skipped_sessions}")
    print(f"Processed sessions: {result.stats.processed_sessions}")
    print(f"Failed sessions: {result.stats.failed_sessions}")
    print(f"Chunks: {result.stats.chunk_count}")
    print(f"Candidates: {result.stats.candidate_count}")
    print(f"Accepted: {result.stats.accepted_count}")
    print(f"Rejected: {result.stats.rejected_count}")
    print(f"Outputs: {output_dir}")
    if args.run_consolidation:
        consolidation_stats = run_consolidation(
            knowledge_db_path,
            settings=settings,
        )
        print(f"Knowledge DB: {knowledge_db_path}")
        print(f"Source records seen: {consolidation_stats.source_records_seen}")
        print(f"Source records linked: {consolidation_stats.source_records_linked}")
        print(f"Canonical created: {consolidation_stats.canonical_created}")
        print(f"Canonical reused: {consolidation_stats.canonical_reused}")
        print(f"Classification failures: {consolidation_stats.classification_failures}")
        print(f"Tags created: {consolidation_stats.tags_created}")
        print(f"Tag proposals created: {consolidation_stats.tag_proposals_created}")
        print(f"Rules tagged: {consolidation_stats.rules_tagged}")
        print(f"Rules classified: {consolidation_stats.rules_classified}")
        print(f"Taxonomy governance failures: {consolidation_stats.taxonomy_governance_failures}")
        print(f"Rule classification failures: {consolidation_stats.rule_classification_failures}")
        print(f"Agent invocations: {consolidation_stats.agent_invocations}")
        print(f"Agent failures: {consolidation_stats.agent_failures}")
    if args.report:
        print_observability_report(knowledge_db_path)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3]}..."


def print_observability_report(db_path: Path) -> None:
    store = KnowledgeStore(db_path)
    try:
        run = store.runs.latest_consolidation_run()
        run_id = str(run["run_id"]) if run else None
        print("Observability report")
        if run:
            stats = json.loads(str(run["stats_json"]))
            print(f"Run ID: {run_id}")
            print(f"Run status: {run['status']}")
            print(f"Model: {run['model'] or 'n/a'}")
            print(f"Started: {run['started_at']}")
            print(f"Finished: {run['finished_at'] or 'n/a'}")
            print(
                "Stats: "
                f"rules_tagged={stats.get('rules_tagged', 0)}, "
                f"rules_classified={stats.get('rules_classified', 0)}, "
                f"taxonomy_governance_failures={stats.get('taxonomy_governance_failures', 0)}, "
                f"rule_classification_failures={stats.get('rule_classification_failures', 0)}, "
                f"agent_invocations={stats.get('agent_invocations', 0)}, "
                f"agent_failures={stats.get('agent_failures', 0)}"
            )

        print("Agent invocations:")
        for row in store.runs.agent_invocation_summary(run_id):
            print(
                f"  {row['stage']} {row['status']}: "
                f"count={row['count']}, repaired={row['repaired_count']}, "
                f"validation_errors={row['validation_error_count']}, latency_ms={row['latency_ms'] or 0}"
            )

        print("Tag proposals:")
        for row in store.proposal_decision_counts():
            print(f"  {row['decision']}: {row['count']}")

        tags = store.tag_rule_counts()
        print("Tags:")
        for row in tags:
            print(f"  {row['name']}: {row['rule_count']} rules")

        failures = store.runs.recent_consolidation_failures(run_id)
        print("Recent failures:")
        if failures:
            for row in failures:
                print(f"  {row['failure_id']} {row['stage']}: {row['error']}")
        else:
            print("  none")
    finally:
        store.close()


if __name__ == "__main__":
    main()
