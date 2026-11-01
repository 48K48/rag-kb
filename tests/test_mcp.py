"""FastMCP 服务与 API 客户端测试（不依赖真实 HTTP / Chroma / Ollama）。"""

import asyncio
import json

import httpx
import pytest
from mcp.server.transport_security import TransportSecurityMiddleware
from mcp.shared.memory import create_connected_server_and_client_session

import app.mcp.server as server_module
from app.mcp.api_client import KbApiClient, KbApiError
from app.models import (
    DocumentStats,
    KbDocumentDetailResponse,
    KbDocumentListResponse,
    KbStatsResponse,
    RetrieveResponse,
    RetrievedItem,
)


def test_api_client_retrieve():
    """客户端应发送正确请求并解析响应。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/kb/retrieve"
        payload = json.loads(request.content)
        assert payload == {"query": "AI Agent", "top_k": 3, "source": "a.md"}
        return httpx.Response(
            200,
            json={
                "query": "AI Agent",
                "count": 1,
                "message": "检索到 1 个片段",
                "results": [
                    {
                        "index": 1,
                        "source": "a.md",
                        "page": None,
                        "chunk_id": "a:0",
                        "snippet": "片段",
                        "content": "片段内容",
                        "distance": 0.1,
                        "retrieval": "dense",
                    }
                ],
            },
        )

    client = KbApiClient(
        "http://test", transport=httpx.MockTransport(handler)
    )
    result = client.retrieve("AI Agent", top_k=3, source="a.md")

    assert isinstance(result, RetrieveResponse)
    assert result.results[0].source == "a.md"


def test_api_client_document_info_encodes_source():
    """文件名里的空格/中文应正确编码到路径。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/kb/documents/agent 笔记.md"
        return httpx.Response(
            200,
            json={
                "found": True,
                "document": {
                    "source": "agent 笔记.md",
                    "chunks": 1,
                    "pages": [],
                    "chars": 10,
                    "min_chunk_chars": 10,
                    "max_chunk_chars": 10,
                },
                "message": "",
            },
        )

    client = KbApiClient(
        "http://test", transport=httpx.MockTransport(handler)
    )
    result = client.document_info("agent 笔记.md")

    assert result.found is True


def test_api_client_wraps_http_error():
    """HTTP 错误应转成 KbApiError，方便 MCP 工具统一处理。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = KbApiClient(
        "http://test", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(KbApiError, match="500"):
        client.stats()


class FakeApiClient:
    """给 MCP 工具用的假知识库客户端。"""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def retrieve(self, query: str, top_k: int = 5, source: str | None = None):
        if self.fail:
            raise KbApiError("无法连接知识库 API")
        return RetrieveResponse(
            query=query,
            count=1,
            message="检索到 1 个片段",
            results=[
                RetrievedItem(
                    index=1,
                    source="a.md",
                    page=None,
                    chunk_id="a:0",
                    snippet="片段",
                    content="片段内容",
                    distance=0.1,
                    retrieval="dense",
                )
            ],
        )

    def list_documents(self):
        return KbDocumentListResponse(
            count=1,
            documents=[
                DocumentStats(
                    source="a.md",
                    chunks=1,
                    pages=[],
                    chars=10,
                    min_chunk_chars=10,
                    max_chunk_chars=10,
                )
            ],
        )

    def document_info(self, source: str):
        if source == "missing.md":
            return KbDocumentDetailResponse(found=False, message="未找到文档")
        return KbDocumentDetailResponse(
            found=True,
            document=DocumentStats(
                source=source,
                chunks=1,
                pages=[],
                chars=10,
                min_chunk_chars=10,
                max_chunk_chars=10,
            ),
        )

    def stats(self):
        return KbStatsResponse(
            documents=1,
            chunks=1,
            collection="knowledge_base",
            embedding_model="qwen3-embedding:0.6b",
            hybrid_enabled=False,
            top_k=5,
            max_distance=0.6,
        )


@pytest.fixture
def fake_api(monkeypatch):
    fake = FakeApiClient()
    monkeypatch.setattr(server_module, "get_api_client", lambda: fake)
    return fake


def _run(coro):
    """在同步测试里运行一次性异步流程。"""
    return asyncio.run(coro)


def test_transport_security_allows_compose_service_name():
    """Docker 内用 mcp:8001 访问时必须通过 Host/Origin 校验。"""
    middleware = TransportSecurityMiddleware(
        server_module.MCP_TRANSPORT_SECURITY
    )

    assert middleware._validate_host("mcp:8001")
    assert middleware._validate_host("127.0.0.1:8001")
    assert not middleware._validate_host("evil.example:8001")
    assert middleware._validate_origin("http://mcp:8001")
    assert not middleware._validate_origin("http://evil.example:8001")


def test_mcp_lists_expected_tools():
    """MCP 服务应暴露 4 个只读工具，且检索参数 schema 正确。"""

    async def run():
        async with create_connected_server_and_client_session(
            server_module.mcp
        ) as session:
            return await session.list_tools()

    result = _run(run())
    tools = {tool.name: tool for tool in result.tools}

    assert set(tools) == {
        "search_knowledge_base",
        "list_knowledge_base_documents",
        "get_document_info",
        "knowledge_base_stats",
    }
    schema = tools["search_knowledge_base"].inputSchema
    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query", "top_k", "source"}
    assert tools["get_document_info"].inputSchema["required"] == ["source"]


def test_mcp_search_returns_structured_content(fake_api):
    """检索工具的 structuredContent 应包含带编号的结果。"""

    async def run():
        async with create_connected_server_and_client_session(
            server_module.mcp
        ) as session:
            return await session.call_tool(
                "search_knowledge_base", {"query": "AI Agent"}
            )

    result = _run(run())

    assert result.isError is False
    assert result.structuredContent["count"] == 1
    assert result.structuredContent["results"][0]["index"] == 1
    assert result.structuredContent["results"][0]["source"] == "a.md"


def test_mcp_document_info_not_found(fake_api):
    """文档不存在时应返回 found=false，而不是工具错误。"""

    async def run():
        async with create_connected_server_and_client_session(
            server_module.mcp
        ) as session:
            return await session.call_tool(
                "get_document_info", {"source": "missing.md"}
            )

    result = _run(run())

    assert result.isError is False
    assert result.structuredContent["found"] is False


def test_mcp_tool_error(monkeypatch):
    """API 不可达时应返回 isError=true，让 Agent 能自行处理。"""
    monkeypatch.setattr(
        server_module, "get_api_client", lambda: FakeApiClient(fail=True)
    )

    async def run():
        async with create_connected_server_and_client_session(
            server_module.mcp
        ) as session:
            return await session.call_tool(
                "search_knowledge_base", {"query": "AI Agent"}
            )

    result = _run(run())

    assert result.isError is True
    assert "无法连接知识库 API" in result.content[0].text
