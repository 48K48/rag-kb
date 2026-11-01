"""混合检索的单元测试（不依赖 Ollama / Chroma / DeepSeek）。"""

from app.rag.bm25 import BM25Index, tokenize
from app.rag.retriever import Retriever
from app.rag.types import Chunk, RetrievedChunk


class FakeStore:
    """只实现 Retriever 用到的 search / get_all_chunks / version。"""

    def __init__(self, dense_results: list[RetrievedChunk], chunks: list[Chunk] | None = None):
        self._dense_results = dense_results
        self._chunks = chunks or []
        self.version = 1

    def search(self, query: str, top_k: int = 5, source: str | None = None):
        return self._dense_results[:top_k]

    def get_all_chunks(self) -> list[Chunk]:
        return self._chunks


class FakeBM25:
    """假的 BM25 索引：直接返回预设结果。"""

    def __init__(self, results: list[RetrievedChunk]):
        self._results = results

    def search(self, query: str, top_k: int = 5, source: str | None = None):
        return self._results[:top_k]


def test_tokenize_keeps_keywords_and_drops_stopwords():
    """中文分词应保留有意义的词，过滤“的”这类停用词。"""
    tokens = tokenize("Chroma 的 metadata 过滤")
    assert "chroma" in tokens
    assert "metadata" in tokens
    assert "过滤" in tokens
    assert "的" not in tokens


def test_bm25_search_ranks_keyword_match_first():
    """BM25 应把同时包含查询词的块排在第一。"""
    chunks = [
        Chunk("a", "Chroma 支持 metadata 过滤和持久化", {"source": "a.md"}),
        Chunk("b", "RAG 的流程是切分、向量化、检索、生成", {"source": "b.md"}),
        Chunk("c", "LangGraph 用状态图编排 Agent", {"source": "c.md"}),
    ]
    index = BM25Index(chunks)

    results = index.search("metadata 过滤", top_k=2)

    assert results[0].id == "a"
    # BM25-only 结果用 -1.0 标记，表示没有 cosine 距离。
    assert results[0].distance == -1.0
    assert results[0].metadata["retrieval"] == "bm25"


def test_rrf_fusion_prefers_chunk_in_both_lists():
    """两路都命中的块应该排在只命中一路的块前面。"""
    dense = [
        RetrievedChunk("a", "A", {"source": "a.md"}, 0.1),
        RetrievedChunk("b", "B", {"source": "b.md"}, 0.2),
    ]
    sparse = [
        RetrievedChunk("b", "B", {"source": "b.md"}, -1.0),
        RetrievedChunk("c", "C", {"source": "c.md"}, -1.0),
    ]

    retriever = Retriever(
        FakeStore(dense), top_k=3, max_distance=0.6, hybrid_enabled=True
    )
    # 直接注入假 BM25，跳过从 Chroma 建索引的过程。
    retriever._bm25 = FakeBM25(sparse)
    retriever._bm25_version = 1

    results = retriever.retrieve("query", top_k=3)

    # b 在两路都出现，RRF 分数最高；a 和 c 各只出现一路。
    assert [item.id for item in results] == ["b", "a", "c"]
    assert results[0].metadata["retrieval"] == "hybrid"


def test_hybrid_disabled_returns_dense_only():
    """关闭混合检索时，只返回通过距离阈值的向量结果。"""
    dense = [
        RetrievedChunk("a", "A", {"source": "a.md"}, 0.1),
        RetrievedChunk("b", "B", {"source": "b.md"}, 0.9),
    ]
    retriever = Retriever(
        FakeStore(dense), top_k=5, max_distance=0.6, hybrid_enabled=False
    )

    results = retriever.retrieve("query")

    # b 的距离 0.9 超过阈值，被过滤。
    assert [item.id for item in results] == ["a"]


def test_hybrid_can_return_bm25_only_result():
    """向量没有召回时，BM25 命中的结果仍然应该返回。"""
    sparse = [RetrievedChunk("c", "C", {"source": "c.md"}, -1.0)]
    retriever = Retriever(
        FakeStore([]), top_k=3, max_distance=0.6, hybrid_enabled=True
    )
    retriever._bm25 = FakeBM25(sparse)
    retriever._bm25_version = 1

    results = retriever.retrieve("query")

    assert [item.id for item in results] == ["c"]
    assert results[0].metadata["retrieval"] == "bm25"
