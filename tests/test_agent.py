"""Agent 工具与消息解析的单元测试（不调用真实 DeepSeek）。"""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.agent import AgentTrace, build_agent_messages, parse_sources
from app.agent.tools import build_tools
from app.models import AgentMessage
from app.rag.types import RetrievedChunk


class FakeRetriever:
    """只实现工具用到的 retrieve。"""

    def retrieve(self, question: str, top_k: int | None = None, source: str | None = None):
        return [
            RetrievedChunk(
                id="agent笔记.md:0:0",
                text="AI Agent 由模型、工具、记忆和规划四部分组成。",
                metadata={"source": "agent笔记.md"},
                distance=0.12,
            )
        ]


class FakeStore:
    """只实现工具用到的文档统计方法。"""

    def list_sources(self) -> list[str]:
        return ["agent笔记.md", "rag笔记.md"]

    def list_document_stats(self) -> list[dict]:
        return [
            {
                "source": "agent笔记.md",
                "chunks": 1,
                "pages": [],
                "chars": 100,
                "min_chunk_chars": 100,
                "max_chunk_chars": 100,
            }
        ]

    def get_document_stats(self, source: str):
        return next(
            (
                item
                for item in self.list_document_stats()
                if item["source"] == source
            ),
            None,
        )

    def count(self) -> int:
        return 1


def test_search_tool_returns_numbered_context_and_artifact():
    """检索工具应同时返回给模型看的文本和给程序用的 artifact。"""
    tools = build_tools(FakeRetriever(), FakeStore())
    search = next(item for item in tools if item.name == "search_knowledge_base")

    content, artifact = search.func("AI Agent", 5)

    assert "[1]" in content
    assert "agent笔记.md" in content
    assert artifact[0]["index"] == 1
    assert artifact[0]["source"] == "agent笔记.md"
    assert artifact[0]["chunk_id"] == "agent笔记.md:0:0"


def test_list_documents_tool_returns_sources():
    """文档列表工具应返回去重后的文件名。"""
    tools = build_tools(FakeRetriever(), FakeStore())
    list_tool = next(
        item for item in tools if item.name == "list_knowledge_base_documents"
    )

    text = list_tool.func()

    assert "agent笔记.md" in text
    assert "rag笔记.md" in text


def test_get_document_info_tool_returns_json():
    """本地文档详情工具应返回结构化 JSON，找不到时 found=false。"""
    tools = build_tools(FakeRetriever(), FakeStore())
    info_tool = next(
        item for item in tools if item.name == "get_document_info"
    )

    found = json.loads(info_tool.func("agent笔记.md"))
    missing = json.loads(info_tool.func("missing.md"))

    assert found["found"] is True
    assert found["document"]["chunks"] == 1
    assert missing["found"] is False


def test_knowledge_base_stats_tool_returns_json():
    """本地知识库状态工具应返回文档数和块数。"""
    tools = build_tools(FakeRetriever(), FakeStore())
    stats_tool = next(
        item for item in tools if item.name == "knowledge_base_stats"
    )

    data = json.loads(stats_tool.func())

    assert data["documents"] == 1
    assert data["chunks"] == 1


def test_trace_extracts_sources_and_final_answer():
    """AgentTrace 应从 ToolMessage.artifact 取来源，从最后一条 AIMessage 取答案。"""
    trace = AgentTrace()
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "AI Agent"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="[1] 来源：agent笔记.md",
            name="search_knowledge_base",
            tool_call_id="call_1",
            artifact=[
                {
                    "index": 1,
                    "source": "agent笔记.md",
                    "page": None,
                    "chunk_id": "agent笔记.md:0:0",
                    "snippet": "AI Agent 由模型、工具、记忆和规划组成。",
                    "distance": 0.12,
                }
            ],
        ),
        AIMessage(content="AI Agent 由模型、工具、记忆和规划组成。[1]"),
    ]

    events = []
    for message in messages:
        events.extend(trace.add(message))

    assert [event["event"] for event in events] == [
        "tool_call",
        "tool_result",
        "sources",
    ]
    assert trace.used_rag is True
    assert trace.sources[0].source == "agent笔记.md"
    assert trace.tool_calls[0].name == "search_knowledge_base"
    assert trace.answer.endswith("[1]")
    assert trace.refused is False


def test_trace_dedupes_same_tool_call():
    """同一条 tool call 在多个 update 里出现时只记录一次。"""
    trace = AgentTrace()
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {"query": "x"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )

    trace.add(message)
    trace.add(message)

    assert len(trace.tool_calls) == 1


def test_history_trimming_keeps_recent_turns():
    """历史只保留最近 N 轮，再追加当前问题。"""
    history = []
    for i in range(3):
        history.append(AgentMessage(role="user", content=f"问题{i}"))
        history.append(AgentMessage(role="assistant", content=f"回答{i}"))

    messages = build_agent_messages("当前问题", history, max_turns=2)

    # 2 轮 = 4 条历史 + 1 条当前问题。
    assert len(messages) == 5
    assert messages[-1] == {"role": "user", "content": "当前问题"}
    assert messages[0]["content"] == "问题1"


def test_parse_sources_ignores_invalid_items():
    """单条来源格式异常时跳过，不影响其他来源。"""
    sources = parse_sources(
        [
            {
                "index": 1,
                "source": "a.md",
                "page": None,
                "chunk_id": "a",
                "snippet": "x",
                "distance": 0.1,
            },
            {"bad": "item"},
        ]
    )

    assert len(sources) == 1
    assert sources[0].source == "a.md"


def test_parse_sources_accepts_mcp_structured_content():
    """MCP 工具的 structuredContent 应能转成 Source。"""
    artifact = {
        "structured_content": {
            "results": [
                {
                    "index": 1,
                    "source": "a.md",
                    "page": None,
                    "chunk_id": "a:0",
                    "snippet": "x",
                    "content": "完整内容",
                    "distance": 0.1,
                    "retrieval": "dense",
                }
            ]
        }
    }

    sources = parse_sources(artifact)

    assert len(sources) == 1
    assert sources[0].source == "a.md"
    assert sources[0].snippet == "x"


def test_parse_sources_falls_back_to_content_for_snippet():
    """MCP 结果没有 snippet 时，用 content 前 200 字兜底。"""
    artifact = {
        "structured_content": {
            "results": [
                {
                    "index": 1,
                    "source": "a.md",
                    "chunk_id": "a:0",
                    "content": "长内容" * 100,
                    "distance": 0.1,
                }
            ]
        }
    }

    sources = parse_sources(artifact)

    assert len(sources) == 1
    assert len(sources[0].snippet) == 200


def test_missing_api_key_raises_readable_error():
    """没有配置 Key 时应在调用阶段抛出可读错误，而不是启动阶段崩溃。"""
    from app.agent.agent import KnowledgeAgent

    agent = KnowledgeAgent(
        retriever=FakeRetriever(),
        store=FakeStore(),
        api_key="",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        asyncio.run(agent.answer("测试问题", []))
