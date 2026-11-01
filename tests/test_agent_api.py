"""Agent API 测试（用 Fake Agent 替换真实模型，不联网）。"""

import pytest
from fastapi.testclient import TestClient

from app.deps import get_agent_service
from app.main import app
from app.models import AgentResponse, Source, ToolCallInfo


class FakeAgentService:
    """假的 Agent 服务：返回固定答案和来源。"""

    async def answer(self, question: str, history) -> AgentResponse:
        return AgentResponse(
            answer="Agent 测试答案 [1]",
            sources=[
                Source(
                    index=1,
                    source="test.md",
                    chunk_id="test.md:0:0",
                    snippet="测试片段",
                    distance=0.1,
                )
            ],
            tool_calls=[
                ToolCallInfo(
                    name="search_knowledge_base", args={"query": question}
                )
            ],
            used_rag=True,
            tool_source="mcp",
            latency_ms=1,
        )

    async def stream_answer(self, question: str, history):
        yield {"event": "status", "data": {"message": "分析中"}}
        yield {
            "event": "tool_call",
            "data": {"name": "search_knowledge_base", "args": {"query": question}},
        }
        yield {
            "event": "sources",
            "data": {
                "sources": [
                    {
                        "index": 1,
                        "source": "test.md",
                        "page": None,
                        "chunk_id": "test.md:0:0",
                        "snippet": "测试片段",
                        "distance": 0.1,
                    }
                ]
            },
        }
        yield {"event": "answer", "data": {"text": "Agent 测试答案 [1]"}}
        yield {
            "event": "done",
            "data": {
                "answer": "Agent 测试答案 [1]",
                "sources": [
                    {
                        "index": 1,
                        "source": "test.md",
                        "page": None,
                        "chunk_id": "test.md:0:0",
                        "snippet": "测试片段",
                        "distance": 0.1,
                    }
                ],
                "tool_calls": [
                    {"name": "search_knowledge_base", "args": {"query": question}}
                ],
                "used_rag": True,
                "refused": False,
                "tool_source": "mcp",
            },
        }


@pytest.fixture
def client():
    """每个测试独立设置依赖覆盖，结束后清理，避免跨文件污染。"""
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_agent_service, None)


def test_agent_chat(client):
    """非流式 Agent 接口返回答案、来源、工具调用和工具来源。"""
    response = client.post(
        "/api/agent/chat",
        json={
            "question": "测试问题",
            "history": [{"role": "user", "content": "上一轮问题"}],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Agent 测试答案 [1]"
    assert data["sources"][0]["source"] == "test.md"
    assert data["tool_calls"][0]["name"] == "search_knowledge_base"
    assert data["used_rag"] is True
    assert data["tool_source"] == "mcp"


def test_agent_chat_stream(client):
    """流式 Agent 接口应依次输出 status/tool_call/sources/answer/done。"""
    with client.stream(
        "POST", "/api/agent/chat/stream", json={"question": "测试问题"}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: status" in body
    assert "event: tool_call" in body
    assert "event: sources" in body
    assert "event: answer" in body
    assert "event: done" in body
    assert "Agent 测试答案" in body
    assert '"tool_source": "mcp"' in body
