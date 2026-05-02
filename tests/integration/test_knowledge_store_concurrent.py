from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.types import CanonicalKnowledge, PitfallCategory, PitfallScope


def _canonical(index: int) -> CanonicalKnowledge:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return CanonicalKnowledge(
        canonical_id=f"canonical-{index}",
        title=f"Concurrent rule {index}",
        category=PitfallCategory.EXECUTION_STRATEGY,
        summary="Concurrent SQLite write coverage.",
        preventive_rule=f"Prevent concurrent write issue {index}.",
        scope=PitfallScope.GLOBAL,
        support_count=1,
        created_at=now,
        updated_at=now,
    )


def test_save_canonical_embedding_handles_concurrent_writes(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        canonicals = [_canonical(index) for index in range(64)]
        for canonical in canonicals:
            store.create_canonical(canonical)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    store.save_canonical_embedding,
                    canonical.canonical_id,
                    [float(index), float(index + 1), float(index + 2)],
                )
                for index, canonical in enumerate(canonicals)
                for _ in range(4)
            ]

            for future in futures:
                future.result()
    finally:
        store.close()


def test_record_agent_invocation_handles_concurrent_writes(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.db")
    try:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    store.runs.record_agent_invocation,
                    run_id="integration-run",
                    stage="concurrent-stage",
                    status="success",
                    input_payload={"index": index},
                    output_payload={"ok": True},
                    latency_ms=index,
                )
                for index in range(256)
            ]

            for future in futures:
                future.result()
    finally:
        store.close()
