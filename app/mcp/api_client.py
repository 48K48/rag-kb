"""MCP 服务访问 FastAPI 只读接口的客户端。

为什么不让 MCP 直接打开 Chroma：
    Chroma PersistentClient 会在进程内缓存集合和 HNSW 索引，
    API 新入库的文档在另一个 MCP 进程里可能看不到。
    让 FastAPI 成为唯一数据所有者，MCP 只通过 HTTP 读取，保证一致性。
"""

from functools import lru_cache
from urllib.parse import quote

import httpx

from app.config import settings
from app.models import (
    KbDocumentDetailResponse,
    KbDocumentListResponse,
    KbStatsResponse,
    RetrieveResponse,
)


class KbApiError(RuntimeError):
    """知识库 API 调用失败，由 MCP 工具转成可读的 ToolError。"""


class KbApiClient:
    """对 FastAPI /api/kb/* 的薄封装。"""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _request(
        self, method: str, path: str, **kwargs
    ) -> dict:
        """发请求并返回 JSON；把网络/HTTP 错误统一成 KbApiError。"""
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:300]
            raise KbApiError(
                f"知识库 API 返回 {error.response.status_code}：{detail}"
            ) from error
        except httpx.HTTPError as error:
            raise KbApiError(f"无法连接知识库 API：{error}") from error

    def retrieve(
        self, query: str, top_k: int = 5, source: str | None = None
    ) -> RetrieveResponse:
        payload = {"query": query, "top_k": top_k, "source": source}
        data = self._request("POST", "/api/kb/retrieve", json=payload)
        return RetrieveResponse.model_validate(data)

    def list_documents(self) -> KbDocumentListResponse:
        data = self._request("GET", "/api/kb/documents")
        return KbDocumentListResponse.model_validate(data)

    def document_info(self, source: str) -> KbDocumentDetailResponse:
        # 文件名可能包含中文、空格等字符，显式编码，避免路径解析错误。
        path = f"/api/kb/documents/{quote(source, safe='')}"
        data = self._request("GET", path)
        return KbDocumentDetailResponse.model_validate(data)

    def stats(self) -> KbStatsResponse:
        data = self._request("GET", "/api/kb/stats")
        return KbStatsResponse.model_validate(data)


@lru_cache
def get_api_client() -> KbApiClient:
    """进程内单例：httpx.Client 本身是线程安全的。"""
    return KbApiClient(settings.kb_api_base_url, settings.kb_api_timeout)
