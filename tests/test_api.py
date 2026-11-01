"""
FastAPI 接口测试（不调用真实 LLM / Ollama / Chroma）。

关键技巧：dependency_overrides
    app/deps.py 里的 get_chain / get_store 是通过 Depends 注入的。
    测试时用 app.dependency_overrides 把它们替换成 Fake 对象，
    这样接口代码不变，但外部依赖被隔离，测试又快又稳定。

注意：
    这里覆盖的是「函数对象」本身；
    路由里必须写 Depends(get_chain)，覆盖才会生效。
"""

import pytest
from fastapi.testclient import TestClient

from app.deps import get_chain, get_store
from app.main import app
from app.models import ChatResponse, Source


class FakeChain:
    """假的问答链：不调用 DeepSeek，直接返回固定答案。"""

    def answer(self, question: str, top_k=None, source=None) -> ChatResponse:
        return ChatResponse(
            answer="测试答案 [1]",
            sources=[
                Source(
                    index=1,
                    source="test.md",
                    chunk_id="test.md:0:0",
                    snippet="测试片段",
                    distance=0.1,
                )
            ],
            latency_ms=1,
        )


class FakeStore:
    """假的向量库：只实现接口用到的 count / list_sources。"""

    def count(self) -> int:
        return 0

    def list_sources(self) -> list[str]:
        return []


@pytest.fixture
def client():
    """每个测试独立设置依赖覆盖，结束后清理，避免跨文件污染。"""
    app.dependency_overrides[get_chain] = lambda: FakeChain()
    app.dependency_overrides[get_store] = lambda: FakeStore()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_chain, None)
    app.dependency_overrides.pop(get_store, None)


def test_health(client):
    """健康检查返回 200 和 status=ok。"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # MCP 配置应出现在健康检查里，便于排查工具来源。
    assert "agent_tool_source" in data
    assert "mcp_server_url" in data


def test_chat(client):
    """问答接口返回 FakeChain 的固定答案和引用。"""
    response = client.post("/api/chat", json={"question": "测试问题"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "测试答案 [1]"
    assert data["sources"][0]["source"] == "test.md"


def test_documents(client):
    """文档列表接口返回 FakeStore 的空列表。"""
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert response.json() == {"count": 0, "sources": []}
