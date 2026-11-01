"""
FastAPI 入口。

启动方式：
    uvicorn app.main:app --reload

路由注册顺序很重要：
    1. 先注册 /api/* 接口；
    2. 最后把 "/" 挂载成静态文件目录。
    如果先 mount("/")，它会先匹配所有请求，/api 接口就永远进不来。
"""

from typing import Any

from fastapi import Depends, FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from app.api import agent, chat, ingest, kb
from app.config import settings
from app.deps import get_store
from app.rag.store import ChromaStore

# FastAPI 会自动根据 title/version/description 生成 /docs 文档。
app = FastAPI(
    title="Agent 知识库问答系统（Agent + RAG）",
    version="0.2.0",
    description="Agent 自主决定是否检索；RAG 作为工具，保留纯 RAG baseline",
)

# 注册业务路由：
#   agent  —— Agent 问答（主入口，Streamlit 使用）
#   chat   —— 固定 RAG 流水线（baseline）
#   ingest —— 文档入库
#   kb     —— 只读检索/文档统计（MCP 服务复用）
# 这些 router 内部都带了 /api 前缀。
app.include_router(agent.router)
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(kb.router)


@app.get("/api/health", tags=["health"], summary="健康检查")
def health(store: ChromaStore = Depends(get_store)) -> dict:
    """返回服务状态、库内块数、是否配置了模型 Key。

    这个接口不调用 Ollama / DeepSeek，
    所以即使模型服务没启动，也能看到基本状态。
    """
    return {
        "status": "ok",
        "collection_count": store.count(),
        "deepseek_configured": bool(settings.deepseek_api_key),
        "agent_configured": bool(settings.agent_api_key),
        "agent_model": settings.agent_model,
        "agent_base_url": settings.agent_base_url,
        "agent_tool_source": settings.agent_tool_source,
        "mcp_server_name": settings.mcp_server_name,
        "mcp_server_url": settings.mcp_server_url,
        "ollama_base_url": settings.ollama_base_url,
        "embedding_model": settings.embedding_model,
    }


def _to_binary_format(node: Any) -> None:
    """把 OpenAPI 里的 contentMediaType 文件标记改成 format: binary。

    背景：FastAPI 0.143 给 UploadFile 生成的 schema 是
        {"type": "string", "contentMediaType": "application/octet-stream"}
    这是 OpenAPI 3.1 的写法，但 Swagger UI 只认 "format": "binary"，
    因此 /docs 的上传接口只显示一个文本框，没有"选择文件"按钮。
    这里统一转换，保证 Swagger UI 能渲染文件选择框。
    """
    if isinstance(node, dict):
        if node.get("contentMediaType") == "application/octet-stream":
            node.pop("contentMediaType")
            node["format"] = "binary"
        for value in node.values():
            _to_binary_format(value)
    elif isinstance(node, list):
        for item in node:
            _to_binary_format(item)


def custom_openapi() -> dict[str, Any]:
    """生成并缓存 OpenAPI 描述，同时修正文件字段格式。"""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    _to_binary_format(schema)
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


# 静态页面兜底：访问 http://127.0.0.1:8000/ 时返回 app/static/index.html。
# html=True 表示把 index.html 作为目录的默认首页。
# 必须放在所有 API 路由之后，否则会抢走 /api 请求。
app.mount("/", StaticFiles(directory=str(settings.static_dir), html=True), name="static")
