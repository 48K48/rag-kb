"""Agent 核心：用 LangChain create_agent 把 DeepSeek 和工具串起来。

与旧的 RagChain 的区别：
    RagChain 是固定流水线，每次都先检索再生成；
    KnowledgeAgent 把检索包装成工具，由模型自己决定：
        - 直接回答（闲聊、改写、总结）；
        - 调用 search_knowledge_base；
        - 调用 list_knowledge_base_documents；
        - 多轮工具调用后再回答。

为什么 streaming=False：
    LangChain + DeepSeek 的流式 tool call 存在参数分片问题，
    可能把空参数 {} 提前传给工具。关闭模型流式后 tool call 解析更稳定；
    Agent 层面仍然通过 stream_mode="updates" 实时推送工具调用状态。
"""

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.agent.mcp_client import load_mcp_tools
from app.agent.prompts import AGENT_SYSTEM_PROMPT, REFUSAL_MARKER
from app.agent.tools import build_tools
from app.models import AgentMessage, AgentResponse, Source, ToolCallInfo
from app.rag.retriever import Retriever
from app.rag.store import ChromaStore

logger = logging.getLogger(__name__)


def content_to_text(content) -> str:
    """把 LangChain 消息的 content 统一转成字符串。

    DeepSeek 通过 ChatOpenAI 返回的通常是 str，
    但其他兼容模型可能返回 [{"type": "text", "text": "..."}] 形式。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content) if content is not None else ""


def build_agent_messages(
    question: str, history: list[AgentMessage], max_turns: int
) -> list[dict]:
    """把客户端历史裁剪成最近 N 轮，再追加当前问题。

    一轮 = 用户消息 + 助手消息，所以最多取 max_turns * 2 条。
    """
    recent = history[-max_turns * 2 :] if max_turns > 0 else []
    messages = [{"role": item.role, "content": item.content} for item in recent]
    messages.append({"role": "user", "content": question})
    return messages


def _artifact_items(artifact) -> list:
    """从本地工具或 MCP 工具的 artifact 中取出结果列表。

    两种来源的形状不同：
        本地工具：            [{"index": 1, ...}, ...]
        MCP 适配器：          {"structured_content": {"results": [...]}}
    这里统一成 list，后续按同一套规则解析。
    """
    if isinstance(artifact, list):
        return artifact

    if isinstance(artifact, dict):
        payload = artifact.get("structured_content")
        if isinstance(payload, dict):
            results = payload.get("results") or payload.get("sources")
            if isinstance(results, list):
                return results
        # 兼容没有 structured_content 包装的直接 dict。
        results = artifact.get("results") or artifact.get("sources")
        if isinstance(results, list):
            return results

    return []


def parse_sources(artifact) -> list[Source]:
    """把 ToolMessage.artifact 转成 Source 列表。

    兼容三种形状：
        1) 本地工具返回的 list[dict]；
        2) MCP 适配器返回的 {"structured_content": {"results": [...]}}；
        3) 已经构造好的 Source 对象。
    """
    sources: list[Source] = []
    for item in _artifact_items(artifact):
        if isinstance(item, Source):
            sources.append(item)
            continue
        if not isinstance(item, dict):
            continue

        data = dict(item)
        # MCP 结果用 content 存完整原文，Source 只需要 snippet。
        if "snippet" not in data and "content" in data:
            data["snippet"] = str(data["content"])[:200]
        # 忽略 retrieval 等 Source 不认识的字段。
        allowed = {"index", "source", "page", "chunk_id", "snippet", "distance"}
        data = {key: value for key, value in data.items() if key in allowed}

        try:
            sources.append(Source(**data))
        except Exception:
            # 单条来源格式异常不应该影响整个回答。
            continue
    return sources


@dataclass
class AgentTrace:
    """从 Agent 消息流里累积答案、来源和工具调用。"""

    answer: str = ""
    sources: list[Source] = field(default_factory=list)
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    used_rag: bool = False
    usage: dict | None = None
    _seen_calls: set[str] = field(default_factory=set)

    def add(self, message: BaseMessage) -> list[dict]:
        """处理一条消息，返回需要发给前端的 SSE 事件。"""
        events: list[dict] = []

        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                # 同一个 tool call 可能在多个 update 里出现，按 id 去重。
                call_id = call.get("id") or ""
                key = call_id or f"{call.get('name')}:{call.get('args')}"
                if key in self._seen_calls:
                    continue
                self._seen_calls.add(key)

                name = call.get("name", "unknown")
                args = call.get("args") or {}
                if not isinstance(args, dict):
                    args = {"raw": args}

                self.tool_calls.append(ToolCallInfo(name=name, args=args))
                if name == "search_knowledge_base":
                    self.used_rag = True
                events.append(
                    {"event": "tool_call", "data": {"name": name, "args": args}}
                )

            # 只有不带 tool_calls 的 AIMessage 才是最终回答。
            if message.content and not message.tool_calls:
                self.answer = content_to_text(message.content)
                if message.usage_metadata:
                    self.usage = {
                        "prompt_tokens": message.usage_metadata.get("input_tokens"),
                        "completion_tokens": message.usage_metadata.get(
                            "output_tokens"
                        ),
                        "total_tokens": message.usage_metadata.get("total_tokens"),
                    }

        elif isinstance(message, ToolMessage):
            name = message.name or "tool"
            if name == "search_knowledge_base":
                self.used_rag = True
                # 多次检索时只保留最后一次非空结果，保证编号与最终回答一致。
                current_sources = parse_sources(message.artifact)
                if current_sources:
                    self.sources = current_sources
                events.append(
                    {
                        "event": "tool_result",
                        "data": {
                            "name": name,
                            "summary": f"检索到 {len(current_sources)} 个片段",
                        },
                    }
                )
                if current_sources:
                    events.append(
                        {
                            "event": "sources",
                            "data": {
                                "sources": [
                                    source.model_dump() for source in current_sources
                                ]
                            },
                        }
                    )
            else:
                events.append(
                    {
                        "event": "tool_result",
                        "data": {"name": name, "summary": "工具执行完成"},
                    }
                )

        return events

    @property
    def refused(self) -> bool:
        """模型是否输出了固定拒答话术。"""
        return REFUSAL_MARKER in self.answer


class KnowledgeAgent:
    """Agent 服务：懒加载 create_agent，提供普通和流式两种调用。"""

    def __init__(
        self,
        retriever: Retriever,
        store: ChromaStore,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_history_turns: int = 6,
        recursion_limit: int = 12,
        tool_source: str = "auto",
    ) -> None:
        self._retriever = retriever
        self._store = store
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_history_turns = max_history_turns
        self._recursion_limit = recursion_limit
        # local / mcp / auto，由 app.config 校验取值。
        self._tool_source_mode = tool_source
        self._tool_source: str | None = None

        # 第一次真正调用时才创建 Agent：
        # 没配 Key 时服务仍能启动，/api/health 也能正常返回。
        self._agent = None

    async def _load_tools(self) -> tuple[list[BaseTool], str]:
        """按配置加载工具，返回 (工具列表, 实际来源)。

        auto：先试 MCP，失败后回退本地工具；
        mcp： 失败直接抛错，暴露部署问题；
        local：完全走改造前的本地工具。
        """
        if self._tool_source_mode == "local":
            return build_tools(self._retriever, self._store), "local"

        try:
            tools = await load_mcp_tools()
            return tools, "mcp"
        except Exception as error:  # noqa: BLE001
            if self._tool_source_mode == "mcp":
                raise RuntimeError(f"MCP 工具加载失败：{error}") from error
            logger.warning("MCP 工具加载失败，回退本地工具：%s", error)
            return build_tools(self._retriever, self._store), "local"

    async def _ensure_agent(self):
        """确保 Agent 已创建；没有 Key 时抛出可读错误。

        并发首次请求可能重复创建 Agent，但创建是幂等的，
        低并发场景下不值得引入跨事件循环的锁。
        """
        if not self._api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY 或 AGENT_API_KEY，请在 .env 中设置后再试。"
            )

        if self._agent is None:
            tools, source = await self._load_tools()
            llm = ChatOpenAI(
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                # 关闭模型流式，规避 DeepSeek 流式 tool call 参数分片问题。
                streaming=False,
                # deepseek-flash 默认开启 thinking；关闭后工具调用更稳定，
                # 也避免多轮时 reasoning_content 回传不完整导致 400。
                extra_body={"thinking": {"type": "disabled"}},
            )
            self._agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=AGENT_SYSTEM_PROMPT,
                name="knowledge_agent",
            )
            self._tool_source = source
        return self._agent

    def _config(self) -> dict:
        """LangGraph 的运行配置：限制工具循环次数。"""
        return {"recursion_limit": self._recursion_limit}

    async def answer(
        self, question: str, history: list[AgentMessage]
    ) -> AgentResponse:
        """非流式问答：一次性返回答案、引用和工具调用记录。"""
        start = time.perf_counter()
        agent = await self._ensure_agent()

        result = await agent.ainvoke(
            {
                "messages": build_agent_messages(
                    question, history, self._max_history_turns
                )
            },
            config=self._config(),
        )

        trace = AgentTrace()
        for message in result["messages"]:
            trace.add(message)

        return AgentResponse(
            answer=trace.answer or "未能生成回答，请重试。",
            sources=trace.sources,
            tool_calls=trace.tool_calls,
            used_rag=trace.used_rag,
            refused=trace.refused,
            tool_source=self._tool_source,
            usage=trace.usage,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async def stream_answer(
        self, question: str, history: list[AgentMessage]
    ) -> AsyncIterator[dict]:
        """流式问答：按 LangGraph 节点推送事件。

        事件顺序通常是：
            status -> tool_call -> tool_result -> sources -> answer -> done
        如果没有调用工具，则直接 status -> answer -> done。
        """
        start = time.perf_counter()

        try:
            agent = await self._ensure_agent()
        except RuntimeError as error:
            yield {"event": "error", "data": {"message": str(error)}}
            return

        trace = AgentTrace()
        yield {"event": "status", "data": {"message": "Agent 正在分析问题…"}}

        try:
            async for update in agent.astream(
                {
                    "messages": build_agent_messages(
                        question, history, self._max_history_turns
                    )
                },
                config=self._config(),
                stream_mode="updates",
            ):
                for payload in update.values():
                    if not isinstance(payload, dict):
                        continue
                    for message in payload.get("messages") or []:
                        for event in trace.add(message):
                            yield event
        except Exception as error:
            logger.exception("Agent 流式调用失败")
            yield {"event": "error", "data": {"message": f"Agent 调用失败：{error}"}}
            return

        if not trace.answer:
            yield {
                "event": "error",
                "data": {"message": "Agent 没有生成有效回答，请重试。"},
            }
            return

        yield {"event": "answer", "data": {"text": trace.answer}}
        yield {
            "event": "done",
            "data": {
                "answer": trace.answer,
                "sources": [source.model_dump() for source in trace.sources],
                "tool_calls": [call.model_dump() for call in trace.tool_calls],
                "used_rag": trace.used_rag,
                "refused": trace.refused,
                "tool_source": self._tool_source,
                "usage": trace.usage,
                "latency_ms": int((time.perf_counter() - start) * 1000),
            },
        }
