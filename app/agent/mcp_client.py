"""通过 langchain-mcp-adapters 加载 FastMCP 工具。

为什么单独包一层：
    1. Agent 只依赖「异步加载一批 LangChain 工具」这个能力，
       不关心 MCP 传输细节；
    2. 测试时可以替换 load_mcp_tools，验证回退逻辑；
    3. 以后要接多个 MCP 服务，只改 connections 字典。

注意：
    - 适配器 0.3.2 生成的工具只有 coroutine，Agent 必须用 ainvoke/astream；
    - get_tools() 每次工具调用会新建一个 MCP 会话（HTTP 模式下只是新建连接）。
"""

import asyncio
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings

logger = logging.getLogger(__name__)


def build_mcp_client() -> MultiServerMCPClient:
    """按配置创建 MCP 客户端。"""
    return MultiServerMCPClient(
        {
            settings.mcp_server_name: {
                "transport": "streamable_http",
                "url": settings.mcp_server_url,
                # HTTP 请求超时；检索包含 embedding，不能设得太短。
                "timeout": settings.mcp_request_timeout,
            }
        }
    )


async def load_mcp_tools(
    timeout: float | None = None,
    retries: int | None = None,
) -> list[BaseTool]:
    """加载 MCP 工具，失败时按配置重试。

    Docker 启动时 MCP 可能比 API 晚几秒就绪，
    所以首次加载失败不应该立刻回退本地工具。
    """
    attempts = max(1, retries if retries is not None else settings.mcp_connect_retries)
    connect_timeout = timeout or settings.mcp_connect_timeout
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            client = build_mcp_client()
            tools = await asyncio.wait_for(
                client.get_tools(), timeout=connect_timeout
            )
            if not tools:
                raise RuntimeError("MCP 服务没有返回任何工具")
            logger.info(
                "已从 MCP 加载 %d 个工具：%s",
                len(tools),
                [tool.name for tool in tools],
            )
            return tools
        except Exception as error:  # noqa: BLE001
            last_error = error
            logger.warning(
                "MCP 工具加载失败（第 %d/%d 次）：%s",
                attempt,
                attempts,
                error,
            )
            if attempt < attempts:
                await asyncio.sleep(settings.mcp_retry_interval)

    raise RuntimeError(f"MCP 工具加载失败：{last_error}")
