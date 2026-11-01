"""切分逻辑的单元测试（不依赖 Chroma / Ollama / DeepSeek）。"""

from app.rag.loader import LoadedDoc
from app.rag.splitter import split_documents


def test_split_preserves_metadata_and_unique_ids():
    """切分后必须保留来源 metadata，且每个块 id 唯一。

    为什么 id 唯一很重要：
        Chroma 用 id 做 upsert；
        如果两个块 id 相同，后写入的会覆盖前一个，导致丢数据。
    """
    docs = [LoadedDoc(text="这是第一段。" * 100, metadata={"source": "test.md"})]
    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "test.md" for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_empty_text_returns_no_chunks():
    """空文档不应该产生空块。

    空块会污染向量库：检索时可能命中一个没有任何内容的块，
    既浪费 token，又会让模型无法回答。
    """
    docs = [LoadedDoc(text="", metadata={"source": "empty.md"})]
    assert split_documents(docs) == []
