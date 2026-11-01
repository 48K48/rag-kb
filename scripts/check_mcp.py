"""MCP 冒烟测试：连接 FastMCP 服务，列出工具和 schema。

用法：
    python scripts/check_mcp.py
    python scripts/check_mcp.py --url http://127.0.0.1:8001/mcp
    python scripts/check_mcp.py --query "AI Agent 由哪几部分组成？"

前置条件：
    1. FastAPI 已启动（MCP 工具会调用 /api/kb/*）；
    2. MCP 服务已启动：
       python -m app.mcp.server --transport streamable-http --port 8001
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Windows 终端默认 GBK，中文输出会乱码；统一改成 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 允许直接运行 scripts/check_mcp.py；必须在 import app.* 之前执行。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402

from app.config import settings  # noqa: E402


def print_schema(tool) -> None:
    """打印工具参数 schema；兼容 dict 和 Pydantic 模型两种形式。"""
    schema = tool.args_schema
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    print(json.dumps(schema, ensure_ascii=False, indent=2))


def render_result(result) -> str:
    """把 LangChain 工具返回的内容块拼成可读文本。"""
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(result)


async def main() -> None:
    parser = argparse.ArgumentParser(description="检查 rag-kb MCP 服务")
    parser.add_argument(
        "--url",
        default=settings.mcp_server_url,
        help="MCP Streamable HTTP 地址",
    )
    parser.add_argument(
        "--query",
        default="",
        help="可选：调用一次 search_knowledge_base 验证链路",
    )
    args = parser.parse_args()

    client = MultiServerMCPClient(
        {
            settings.mcp_server_name: {
                "transport": "streamable_http",
                "url": args.url,
                "timeout": settings.mcp_request_timeout,
            }
        }
    )
    tools = await client.get_tools()

    if not tools:
        raise SystemExit("MCP 服务没有返回任何工具")

    print(f"发现 {len(tools)} 个 MCP 工具：\n")
    for tool in tools:
        print(f"## {tool.name}")
        print(tool.description or "（无描述）")
        print("参数：")
        print_schema(tool)
        print()

    if args.query:
        search = next(
            (tool for tool in tools if tool.name == "search_knowledge_base"),
            None,
        )
        if search is None:
            raise SystemExit("没有找到 search_knowledge_base 工具")
        result = await search.ainvoke({"query": args.query, "top_k": 3})
        print("检索冒烟结果：")
        print(render_result(result))


if __name__ == "__main__":
    asyncio.run(main())
