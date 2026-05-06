from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

TURN_PATTERN = re.compile(r'<turn index="(\d+)"[^>]*>(.*?)</turn>', re.DOTALL)
_CHUNK_SIZE = 8000
_SUMMARIZE_WORKERS = 8


class SessionTurnStore:
    def __init__(
        self,
        chroma_path: Path,
        embeddings: Embeddings,
        summarizer: Callable[[str], str] | None = None,
        translator: Callable[[str], str] | None = None,
    ):
        self._summarizer = summarizer
        self._translator = translator
        self._chroma = Chroma(
            collection_name="session_turns",
            embedding_function=embeddings,
            persist_directory=str(chroma_path),
        )

    def embed_session(self, session_id: str, xml: str) -> int:
        chunks: list[tuple[str, int, int, str]] = []  # (id, turn_index, chunk_index, content)
        turns = [(int(match.group(1)), match.group(2).strip()) for match in TURN_PATTERN.finditer(xml)]
        if not turns:
            return 0

        if self._summarizer is not None:
            with ThreadPoolExecutor(max_workers=_SUMMARIZE_WORKERS) as executor:
                summaries = list(executor.map(self._summarizer, [content for _, content in turns]))
            chunks = [
                (_chunk_id(session_id, turn_index, 0), turn_index, 0, summary)
                for (turn_index, _), summary in zip(turns, summaries)
            ]
        else:
            for turn_index, content in turns:
                for chunk_index, chunk in enumerate(_split_chunks(content, _CHUNK_SIZE)):
                    chunks.append((_chunk_id(session_id, turn_index, chunk_index), turn_index, chunk_index, chunk))

        if not chunks:
            return 0

        ids = [chunk_id for chunk_id, _, _, _ in chunks]
        existing_ids = set(self._chroma.get(ids=ids)["ids"])
        missing = [(cid, ti, ci, content) for cid, ti, ci, content in chunks if cid not in existing_ids]
        if not missing:
            return 0

        self._chroma.add_texts(
            texts=[content for _, _, _, content in missing],
            metadatas=[{"session_id": session_id, "turn_index": ti, "chunk_index": ci} for _, ti, ci, _ in missing],
            ids=[cid for cid, _, _, _ in missing],
        )
        return len(missing)

    def search_turns(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        *,
        precomputed_query: str | None = None,
    ) -> list[dict]:
        effective_query = precomputed_query if precomputed_query is not None else query
        if precomputed_query is None and self._translator is not None:
            effective_query = self._translator(query)
        results = self._chroma.similarity_search_with_relevance_scores(
            effective_query,
            k=top_k * 3,
            filter={"session_id": session_id},
        )
        seen: dict[int, float] = {}
        for document, score in results:
            turn_index = int(document.metadata["turn_index"])
            if turn_index not in seen or score > seen[turn_index]:
                seen[turn_index] = float(score)
        return sorted(
            [{"turn_index": ti, "score": score} for ti, score in seen.items()],
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

    def get_turn(self, session_xml: str, turn_index: int) -> str | None:
        return SessionTurnStore._get_turn_from_xml(session_xml, turn_index)

    @staticmethod
    def _get_turn_from_xml(session_xml: str, turn_index: int) -> str | None:
        for match in TURN_PATTERN.finditer(session_xml):
            if int(match.group(1)) == turn_index:
                return match.group(2).strip()
        return None

    def is_session_embedded(self, session_id: str) -> bool:
        results = self._chroma.get(where={"session_id": session_id}, limit=1)
        return len(results["ids"]) > 0

    def embedded_session_ids(self) -> set[str]:
        results = self._chroma.get(include=["metadatas"])
        return {
            metadata["session_id"]
            for metadata in results["metadatas"]
            if metadata is not None and "session_id" in metadata
        }


def _split_chunks(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, max(len(text), 1), chunk_size)]


def _chunk_id(session_id: str, turn_index: int, chunk_index: int) -> str:
    return f"{session_id}:turn:{turn_index}:chunk:{chunk_index}"
