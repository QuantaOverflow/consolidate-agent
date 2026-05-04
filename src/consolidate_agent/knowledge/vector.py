from __future__ import annotations

import json
from typing import Any
from urllib import request

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from consolidate_agent.config import Settings
from consolidate_agent.knowledge.store import KnowledgeStore
from consolidate_agent.types import CanonicalKnowledge, KnowledgeRecord, MechanismTag

SIMILARITY_THRESHOLD = 0.3
KNOWLEDGE_RELATEDNESS_THRESHOLD = 0.7


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

    def embed_canonicals(self, canonicals: list[CanonicalKnowledge]) -> None:
        missing = [canonical for canonical in canonicals if self.store.get_canonical_embedding(canonical.canonical_id) is None]
        if not missing:
            return
        vectors = self.embeddings.embed_documents([_canonical_text(canonical) for canonical in missing])
        for canonical, embedding in zip(missing, vectors, strict=True):
            self.store.save_canonical_embedding(canonical.canonical_id, embedding)

    def embed_tags(self, tags: list[MechanismTag]) -> None:
        missing = [tag for tag in tags if self.store.get_tag_embedding(tag.tag_id) is None]
        if not missing:
            return
        vectors = self.embeddings.embed_documents([_tag_text(tag) for tag in missing])
        for tag, embedding in zip(missing, vectors, strict=True):
            self.store.save_tag_embedding(tag.tag_id, embedding)

    def embed_knowledge_records(self, records: list[KnowledgeRecord]) -> None:
        missing = [record for record in records if self.store.get_knowledge_embedding(record.id) is None]
        if not missing:
            return
        vectors = self.embeddings.embed_documents([_knowledge_record_text(record) for record in missing])
        for record, embedding in zip(missing, vectors, strict=True):
            self.store.save_knowledge_embedding(record.id, embedding)

    def find_best_tag(self, canonical: CanonicalKnowledge, tags: list[MechanismTag]) -> tuple[MechanismTag | None, float]:
        if not tags:
            return None, 0.0
        canonical_embedding = self.store.get_canonical_embedding(canonical.canonical_id)
        if canonical_embedding is None:
            canonical_embedding = self.embeddings.embed_query(_canonical_text(canonical))
            self.store.save_canonical_embedding(canonical.canonical_id, canonical_embedding)

        best_tag = None
        best_score = 0.0
        for tag in tags:
            tag_embedding = self.store.get_tag_embedding(tag.tag_id)
            if tag_embedding is None:
                tag_embedding = self.embeddings.embed_query(_tag_text(tag))
                self.store.save_tag_embedding(tag.tag_id, tag_embedding)
            score = _cosine_similarity(canonical_embedding, tag_embedding)
            if score > best_score:
                best_score = score
                best_tag = tag
        return (best_tag, best_score) if best_score >= SIMILARITY_THRESHOLD else (None, 0.0)

    def similarity_search(self, query: str, k: int = 5, **kwargs: Any) -> list[Document]:
        canonicals = self.similarity_search_canonicals(query, k=k)
        return [_canonical_document(canonical) for canonical in canonicals]

    def similarity_search_canonicals(self, query: str, k: int = 5) -> list[CanonicalKnowledge]:
        canonicals = self.store.list_active_canonicals()
        if not canonicals:
            return []
        self.embed_canonicals(canonicals)
        query_embedding = self.embeddings.embed_query(query)
        scored = []
        for canonical in canonicals:
            embedding = self.store.get_canonical_embedding(canonical.canonical_id)
            if embedding is None:
                continue
            scored.append((_cosine_similarity(query_embedding, embedding), canonical))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [canonical for _, canonical in scored[:k]]

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


def find_related_knowledge(
    store: KnowledgeStore,
    embeddings: Embeddings,
    canonical_id: str,
    threshold: float = KNOWLEDGE_RELATEDNESS_THRESHOLD,
    top_k: int = 3,
) -> list[dict]:
    _ = embeddings
    canonical_embedding = store.get_canonical_embedding(canonical_id)
    if canonical_embedding is None:
        return []

    records = store.list_all_knowledge_records()
    if not records:
        return []

    scored = []
    for record in records:
        embedding = store.get_knowledge_embedding(record.id)
        if embedding is None:
            continue
        score = _cosine_similarity(canonical_embedding, embedding)
        if score >= threshold:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "record_id": record.id,
            "title": record.title,
            "insight": record.insight,
            "applicability": record.applicability,
            "scope": record.scope.value,
            "similarity_score": score,
        }
        for score, record in scored[:top_k]
    ]


def _canonical_text(canonical: CanonicalKnowledge) -> str:
    return "\n".join(
        [
            f"Title: {canonical.title}",
            f"Category: {canonical.category.value}",
            f"Summary: {canonical.summary}",
            f"Preventive rule: {canonical.preventive_rule}",
            f"Scope: {canonical.scope.value}",
        ]
    )


def _tag_text(tag: MechanismTag) -> str:
    return "\n".join(
        [
            f"Name: {tag.name}",
            f"Definition: {tag.definition}",
            f"Positive examples: {'; '.join(tag.positive_examples)}",
            f"Negative examples: {'; '.join(tag.negative_examples)}",
        ]
    )


def _knowledge_record_text(record: KnowledgeRecord) -> str:
    return "\n".join(
        [
            f"Title: {record.title}",
            f"Insight: {record.insight}",
            f"Applicability: {record.applicability}",
        ]
    )


def _canonical_document(canonical: CanonicalKnowledge) -> Document:
    return Document(
        page_content=_canonical_text(canonical),
        metadata={
            "canonical_id": canonical.canonical_id,
            "title": canonical.title,
            "category": canonical.category.value,
            "scope": canonical.scope.value,
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
