from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings

from consolidate_agent.knowledge.session_turn_store import (
    _CHUNK_SIZE,
    _normalize_distances,
    _split_chunks,
    SessionTurnStore,
)

SESSION_XML = """<session>
<turn index="1" started_at="t1"><user>hello</user><assistant>world</assistant></turn>
<turn index="2" started_at="t2"><user>foo</user><bash>ls</bash><bash_result>bar</bash_result></turn>
</session>"""


def test_split_chunks_no_data_loss():
    text = "B" * (_CHUNK_SIZE * 3 + 500)
    assert "".join(_split_chunks(text, _CHUNK_SIZE)) == text


def test_get_turn_returns_full_content_for_long_turn():
    long_content = "X" * 20000
    xml = f'<session><turn index="1" started_at="t">{long_content}</turn></session>'
    result = SessionTurnStore._get_turn_from_xml(xml, 1)
    assert len(result) == 20000


def test_get_turn_correct_index():
    result = SessionTurnStore._get_turn_from_xml(SESSION_XML, 2)
    assert "<user>foo</user>" in result
    assert "<bash>ls</bash>" in result


def test_get_turn_not_found():
    assert SessionTurnStore._get_turn_from_xml(SESSION_XML, 99) is None


class FakeEmbeddings(Embeddings):
    def __init__(self):
        self.last_query: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.last_query = text
        return [0.1] * 10


class FakeDocument:
    def __init__(self, turn_index: int):
        self.metadata = {"turn_index": turn_index}


class FakeChroma:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, query: str, k: int, filter: dict):
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.results

    def similarity_search_with_relevance_scores(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("search_turns should use raw distance scores")


def test_embed_session_uses_summary_when_summarizer_provided(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    store = SessionTurnStore(
        chroma_path=tmp_path / "chroma",
        embeddings=embeddings,
        summarizer=lambda x: "摘要内容",
    )
    store.embed_session("s1", SESSION_XML)
    docs = store._chroma.get(include=["documents"])["documents"]
    assert all(d == "摘要内容" for d in docs)


def test_embed_session_falls_back_to_raw_when_no_summarizer(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    store = SessionTurnStore(
        chroma_path=tmp_path / "chroma",
        embeddings=embeddings,
    )
    store.embed_session("s1", SESSION_XML)
    docs = store._chroma.get(include=["documents"])["documents"]
    assert any("hello" in d or "foo" in d for d in docs)


def test_search_turns_translates_query_when_translator_provided(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    store = SessionTurnStore(
        chroma_path=tmp_path / "chroma",
        embeddings=embeddings,
        translator=lambda q: "中文查询",
    )
    store.embed_session("s1", SESSION_XML)
    store.search_turns("s1", "English query")
    assert embeddings.last_query == "中文查询"


def test_normalize_distances_maps_raw_distance_to_relative_score() -> None:
    assert _normalize_distances([2.0, 4.0, 6.0]) == [1.0, 0.5, 0.0]
    assert _normalize_distances([3.0, 3.0]) == [1.0, 1.0]
    assert _normalize_distances([]) == []


def test_search_turns_uses_raw_distance_and_normalizes_scores(tmp_path: Path) -> None:
    store = SessionTurnStore(tmp_path / "chroma", FakeEmbeddings())
    store._chroma = FakeChroma(
        [
            (FakeDocument(1), 2.0),
            (FakeDocument(2), 4.0),
            (FakeDocument(1), 6.0),
        ]
    )

    results = store.search_turns("s1", "query", top_k=2)

    assert results == [{"turn_index": 1, "score": 1.0}, {"turn_index": 2, "score": 0.5}]
    assert store._chroma.calls == [{"query": "query", "k": 6, "filter": {"session_id": "s1"}}]
