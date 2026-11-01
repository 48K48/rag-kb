"""只读知识库接口测试（不依赖 Ollama / Chroma / DeepSeek）。"""

import pytest
from fastapi.testclient import TestClient

from app.deps import get_retriever, get_store
from app.main import app
from app.rag.types import RetrievedChunk


class FakeRetriever:
    """按 source 过滤的假检索器，便于验证参数透传。"""

    def retrieve(self, question: str, top_k: int | None = None, source: str | None = None):
        if source and source != "a.md":
            return []
        return [
            RetrievedChunk(
                id="a.md:0:0",
                text="AI Agent 由模型、工具、记忆和规划四部分组成。",
                metadata={"source": "a.md", "retrieval": "dense"},
                distance=0.12,
            )
        ]


class FakeStore:
    def list_document_stats(self):
        return [
            {
                "source": "a.md",
                "chunks": 2,
                "pages": [],
                "chars": 120,
                "min_chunk_chars": 50,
                "max_chunk_chars": 70,
            }
        ]

    def get_document_stats(self, source: str):
        return next(
            (item for item in self.list_document_stats() if item["source"] == source),
            None,
        )

    def count(self) -> int:
        return 2


@pytest.fixture
def client():
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_store] = lambda: FakeStore()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_retriever, None)
    app.dependency_overrides.pop(get_store, None)


def test_retrieve_returns_numbered_results(client):
    response = client.post(
        "/api/kb/retrieve", json={"query": "AI Agent", "top_k": 5}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["message"].startswith("检索到")
    item = data["results"][0]
    assert item["index"] == 1
    assert item["source"] == "a.md"
    assert item["snippet"] == item["content"][:200]
    assert item["retrieval"] == "dense"


def test_retrieve_empty_result_has_message(client):
    response = client.post(
        "/api/kb/retrieve",
        json={"query": "不存在的问题", "source": "other.md"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["results"] == []
    assert "没有找到相关内容" in data["message"]


def test_list_documents_and_detail(client):
    listing = client.get("/api/kb/documents")
    assert listing.status_code == 200
    assert listing.json()["count"] == 1

    detail = client.get("/api/kb/documents/a.md")
    assert detail.status_code == 200
    assert detail.json()["found"] is True
    assert detail.json()["document"]["chunks"] == 2

    missing = client.get("/api/kb/documents/missing.md")
    assert missing.status_code == 200
    assert missing.json()["found"] is False


def test_stats(client):
    response = client.get("/api/kb/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["documents"] == 1
    assert data["chunks"] == 2
    assert data["collection"] == "knowledge_base"
