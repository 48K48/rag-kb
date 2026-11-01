"""ChromaStore 的单元测试（使用真实 Chroma + 假 embedding）。"""

from app.rag.store import ChromaStore
from app.rag.types import Chunk
from tests.fakes import FakeEmbedder


def test_add_search_list_and_delete(tmp_path):
    """验证写入、检索、列出来源、按来源删除的完整链路。

    tmp_path 是 pytest 提供的临时目录 fixture：
        每个测试拿到一个独立目录，测试之间不会互相污染，
        测试结束后 pytest 会自动清理（Windows 上可能保留，但不影响）。
    """
    store = ChromaStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_collection",
        embedder=FakeEmbedder(),
    )
    chunks = [
        Chunk(id="a", text="Agent 由模型和工具组成", metadata={"source": "a.md"}),
        Chunk(id="b", text="RAG 先检索再生成", metadata={"source": "b.md"}),
    ]
    store.add_chunks(chunks)

    # 写入后数量正确。
    assert store.count() == 2
    # 来源去重并排序。
    assert store.list_sources() == ["a.md", "b.md"]

    # 检索返回 top_k 条；FakeEmbedder 不保证语义排序，
    # 所以这里只验证「数量」和「距离非负」，不验证具体顺序。
    results = store.search("Agent", top_k=2)
    assert len(results) == 2
    assert all(result.distance >= 0 for result in results)

    # 文档统计按来源聚合块数和字符数。
    stats = store.list_document_stats()
    assert [item["source"] for item in stats] == ["a.md", "b.md"]
    assert stats[0]["chunks"] == 1
    assert stats[0]["chars"] == len("Agent 由模型和工具组成")
    assert store.get_document_stats("b.md")["chunks"] == 1
    assert store.get_document_stats("missing.md") is None

    # 按来源删除只影响该来源的块。
    store.delete_source("a.md")
    assert store.count() == 1
    assert store.list_sources() == ["b.md"]
