from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib import request

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from consolidate_agent.config import Settings
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.types import KnowledgeRecord

if TYPE_CHECKING:
    from langchain_community.embeddings import DashScopeEmbeddings


class DashScopeOpenAICompatibleEmbeddings(Embeddings):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-v4",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"])]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_dashscope_embeddings(settings: Settings) -> Embeddings:
    if not settings.dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY is not configured. Cannot initialize DashScope embeddings.")
    try:
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(
            dashscope_api_key=settings.dashscope_api_key,
            model="text-embedding-v4",
        )
    except ImportError:
        print("DashScopeEmbeddings unavailable; using OpenAI-compatible embedding endpoint.", flush=True)
        return DashScopeOpenAICompatibleEmbeddings(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_api_base,
        )


class KnowledgeVectorStore(VectorStore):
    def __init__(self, store: KnowledgeStore, embeddings: Embeddings):
        self.store = store
        self._embeddings = embeddings

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def embed_knowledge_records(self, records: list[KnowledgeRecord]) -> None:
        missing = [
            record
            for record in records
            if record.evidence_count > 0 and self.store.get_knowledge_embedding(record.id) is None
        ]
        if not missing:
            return
        vectors = self.embeddings.embed_documents([_knowledge_record_text(record) for record in missing])
        for record, embedding in zip(missing, vectors, strict=True):
            self.store.save_knowledge_embedding(record.id, embedding)

    def similarity_search(self, query: str, k: int = 5, **kwargs: Any) -> list[Document]:
        records = self.similarity_search_knowledge_records(query, k=k)
        return [_knowledge_record_document(record) for record in records]

    def similarity_search_knowledge_records(self, query: str, k: int = 5) -> list[KnowledgeRecord]:
        records = self.store.list_admitted_knowledge_records()
        if not records:
            return []
        self.embed_knowledge_records(records)
        query_embedding = self.embeddings.embed_query(query)
        scored = []
        for record in records:
            embedding = self.store.get_knowledge_embedding(record.id)
            if embedding is None:
                continue
            scored.append((_cosine_similarity(query_embedding, embedding), record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:k]]

    def as_retriever_tool(self, name: str, description: str):
        try:
            from langchain.tools.retriever import create_retriever_tool
        except ModuleNotFoundError:
            from langchain_core.tools.retriever import create_retriever_tool

        return create_retriever_tool(self.as_retriever(), name, description)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ):
        raise NotImplementedError("KnowledgeVectorStore must be initialized with an existing KnowledgeStore.")


def search_knowledge(
    store: KnowledgeStore,
    embeddings: DashScopeEmbeddings,
    query_text: str,
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[dict]:
    query_embedding = embeddings.embed_query(query_text)
    records = store.list_verified_knowledge_records()
    if not records:
        return []

    record_map = {r.id: r for r in records}
    all_embeddings = store.load_all_knowledge_embeddings()

    scored = []
    for record_id, embedding in all_embeddings.items():
        record = record_map.get(record_id)
        if record is None:
            continue
        score = _cosine_similarity(query_embedding, embedding)
        if score >= threshold:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "record": record,
            "similarity_score": score,
        }
        for score, record in scored[:top_k]
    ]


def _knowledge_record_text(record: KnowledgeRecord) -> str:
    return "\n".join(
        [
            f"Title: {record.title}",
            f"Insight: {record.insight}",
            f"Applicability: {record.applicability}",
        ]
    )


def _knowledge_record_document(record: KnowledgeRecord) -> Document:
    return Document(
        page_content=_knowledge_record_text(record),
        metadata={
            "record_id": record.id,
            "title": record.title,
            "scope": record.scope.value,
        },
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vector = np.asarray(left, dtype=np.float32)
    right_vector = np.asarray(right, dtype=np.float32)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator == 0:
        return 0.0
    score = float(np.dot(left_vector, right_vector) / denominator)
    return max(-1.0, min(1.0, score))
