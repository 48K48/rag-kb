"""Agent 的 MCP 工具加载与回退逻辑测试（不调用真实模型/MCP 服务）。"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.agent import KnowledgeAgent


class FakeRetriever:
    def retrieve(self, question, top_k=None, source=None):
        return []


class FakeStore:
    def list_sources(self):
        return []


class DummyTool:
    name = "search_knowledge_base"


def make_agent(tool_source: str) -> KnowledgeAgent:
    return KnowledgeAgent(
        retriever=FakeRetriever(),
        store=FakeStore(),
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        tool_source=tool_source,
    )


def test_local_mode_uses_builtin_tools_without_mcp():
    """local 模式不应尝试连接 MCP。"""
    agent = make_agent("local")

    with patch("app.agent.agent.load_mcp_tools") as loader:
        tools, source = asyncio.run(agent._load_tools())

    loader.assert_not_called()
    assert source == "local"
    assert {tool.name for tool in tools} == {
        "search_knowledge_base",
        "list_knowledge_base_documents",
        "get_document_info",
        "knowledge_base_stats",
    }


def test_mcp_mode_loads_tools_from_mcp():
    """mcp 模式应返回 MCP 工具并标记来源。"""
    agent = make_agent("mcp")

    with patch(
        "app.agent.agent.load_mcp_tools",
        new=AsyncMock(return_value=[DummyTool()]),
    ):
        tools, source = asyncio.run(agent._load_tools())

    assert source == "mcp"
    assert [tool.name for tool in tools] == ["search_knowledge_base"]


def test_auto_mode_falls_back_to_local_when_mcp_fails():
    """auto 模式连接失败时应回退本地工具。"""
    agent = make_agent("auto")

    with patch(
        "app.agent.agent.load_mcp_tools",
        new=AsyncMock(side_effect=RuntimeError("MCP down")),
    ):
        tools, source = asyncio.run(agent._load_tools())

    assert source == "local"
    assert {tool.name for tool in tools} == {
        "search_knowledge_base",
        "list_knowledge_base_documents",
        "get_document_info",
        "knowledge_base_stats",
    }


def test_mcp_mode_raises_when_mcp_fails():
    """mcp 模式必须暴露错误，不能静默回退。"""
    agent = make_agent("mcp")

    with patch(
        "app.agent.agent.load_mcp_tools",
        new=AsyncMock(side_effect=RuntimeError("MCP down")),
    ):
        with pytest.raises(RuntimeError, match="MCP 工具加载失败"):
            asyncio.run(agent._load_tools())
