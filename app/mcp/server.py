"""FastMCP 服务：把知识库只读能力暴露为 MCP 工具。

启动方式：
    # stdio，供 Claude Desktop / MCP Inspector 使用
    python -m app.mcp.server --transport stdio

    # Streamable HTTP，供 Agent 或远程客户端使用
    python -m app.mcp.server --transport streamable-http --host 0.0.0.0 --port 8001

设计要点：
    1. 只读：不暴露入库/删除，避免外部客户端改写知识库。
    2. 委托：工具实现调用 FastAPI /api/kb/*，MCP 进程不直接打开 Chroma。
    3. 名称稳定：工具名与本地 LangChain 工具一致，Agent 切换 local/mcp 时提示词不用改。
    4. 结构化输出：返回 Pydantic 模型，FastMCP 会同时生成 JSON 文本和 structuredContent，
       langchain-mcp-adapters 会把 structuredContent 放进 ToolMessage.artifact。
"""

import argparse
import logging
import sys
from typing import Annotated

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

from app.config import settings
from app.mcp.api_client import KbApiError, get_api_client
from app.models import (
    KbDocumentDetailResponse,
    KbDocumentListResponse,
    KbStatsResponse,
    RetrieveResponse,
)

logger = logging.getLogger(__name__)

MCP_INSTRUCTIONS = """你是 rag-kb 知识库的只读 MCP 服务。
1. 文档事实、概念、参数、步骤类问题，优先调用 search_knowledge_base。
2. 用户问“知识库里有哪些文档”时，调用 list_knowledge_base_documents。
3. 用户问某篇文档的块数、页数、规模时，调用 get_document_info。
4. 用户问知识库整体状态时，调用 knowledge_base_stats。
5. 引用检索结果时，使用 results[].index 对应的 [1][2] 编号。
6. 检索结果为空时，明确回答“知识库中没有找到相关内容”，不要编造。
7. 不要用训练数据里的知识冒充知识库内容。"""

# FastMCP 默认只允许 localhost 的 Host 头，Docker 内通过服务名 mcp:8001
# 访问会被 DNS rebinding 防护拒绝（421）。这里显式放行 compose 内部服务名。
MCP_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "127.0.0.1",
        "127.0.0.1:*",
        "localhost",
        "localhost:*",
        "[::1]:*",
        "mcp",
        "mcp:*",
        "rag-kb-mcp",
        "rag-kb-mcp:*",
        "host.docker.internal",
        "host.docker.internal:*",
    ],
    allowed_origins=[
        "http://127.0.0.1",
        "http://127.0.0.1:*",
        "http://localhost",
        "http://localhost:*",
        "http://[::1]:*",
        "http://mcp",
        "http://mcp:*",
        "http://rag-kb-mcp",
        "http://rag-kb-mcp:*",
        "http://host.docker.internal:*",
    ],
)


def _call(operation, *args, **kwargs):
    """执行 API 调用，把 KbApiError 转成 MCP 工具错误。"""
    try:
        return operation(*args, **kwargs)
    except KbApiError as error:
        raise ToolError(str(error)) from error


async def search_knowledge_base(
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="检索问题或关键词；尽量保留用户原话里的关键名词",
        ),
    ],
    top_k: Annotated[
        int,
        Field(ge=1, le=10, description="返回片段数，1-10，默认 5"),
    ] = 5,
    source: Annotated[
        str | None,
        Field(
            description="可选，只在这个文件名里检索；必须使用文档列表返回的完整文件名",
        ),
    ] = None,
) -> RetrieveResponse:
    """在本地知识库中检索与问题相关的原文片段。

    返回 JSON：results[].index 是引用编号，results[].content 是完整原文，
    results[].source/page 用于溯源，distance 越小越相关。
    没有命中时返回 count=0，并在 message 中给出固定提示。
    """
    query = (query or "").strip()
    if not query:
        raise ToolError("query 不能为空")

    source = source.strip() if source else None
    # httpx.Client 是同步的，放到线程池里执行，避免阻塞 MCP 事件循环。
    return await anyio.to_thread.run_sync(
        lambda: _call(
            get_api_client().retrieve,
            query=query,
            top_k=top_k,
            source=source,
        )
    )


async def list_knowledge_base_documents() -> KbDocumentListResponse:
    """列出知识库中已入库的所有文档，以及每篇文档的块数、页码和字数。"""
    return await anyio.to_thread.run_sync(
        lambda: _call(get_api_client().list_documents)
    )


async def get_document_info(
    source: Annotated[
        str,
        Field(
            min_length=1,
            max_length=255,
            description="完整文件名，例如 agent笔记.md",
        ),
    ],
) -> KbDocumentDetailResponse:
    """查询单篇文档的规模信息：块数、页码、总字数和单块长度范围。

    文档不存在时返回 found=false，不会抛出异常。
    """
    source = (source or "").strip()
    if not source:
        raise ToolError("source 不能为空")
    return await anyio.to_thread.run_sync(
        lambda: _call(get_api_client().document_info, source=source)
    )


async def knowledge_base_stats() -> KbStatsResponse:
    """返回知识库整体状态：文档数、块数、embedding 模型和混合检索开关。"""
    return await anyio.to_thread.run_sync(
        lambda: _call(get_api_client().stats)
    )


def build_server(host: str | None = None, port: int | None = None) -> FastMCP:
    """创建 FastMCP 实例并注册工具。

    单独封装是为了让 CLI 可以覆盖 host/port，同时让测试可以拿一个干净的实例。
    """
    server = FastMCP(
        "rag-kb",
        instructions=MCP_INSTRUCTIONS,
        host=host or settings.mcp_host,
        port=port or settings.mcp_port,
        transport_security=MCP_TRANSPORT_SECURITY,
    )
    for tool_fn in (
        search_knowledge_base,
        list_knowledge_base_documents,
        get_document_info,
        knowledge_base_stats,
    ):
        server.tool()(tool_fn)
    return server


# 模块级实例：测试和 MCP Inspector 直接 import 时使用默认配置。
mcp = build_server()


def main() -> None:
    """命令行入口：解析传输方式和监听地址。"""
    parser = argparse.ArgumentParser(description="rag-kb FastMCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="stdio 适合 Claude Desktop；streamable-http 适合 Agent/远程客户端",
    )
    parser.add_argument("--host", default=settings.mcp_host, help="HTTP 监听地址")
    parser.add_argument("--port", type=int, default=settings.mcp_port, help="HTTP 端口")
    args = parser.parse_args()

    # stdio 模式下日志必须写 stderr，否则会污染 JSON-RPC 输出。
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "启动 MCP 服务：transport=%s host=%s port=%s api=%s",
        args.transport,
        args.host,
        args.port,
        settings.kb_api_base_url,
    )

    server = build_server(host=args.host, port=args.port)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
