"""只读知识库接口：供 MCP 服务和外部客户端复用。

为什么单独一组接口：
    MCP 服务运行在独立进程/容器里。如果它直接打开 Chroma，
    会各自缓存 HNSW 索引，API 新入库的文档在 MCP 侧看不到
    （这也是 CLI 入库后必须重启服务的原因）。
    让 FastAPI 成为唯一数据所有者，MCP 通过 HTTP 读取，保证数据一致。

这组接口全部只读，不提供入库和删除。
"""

from fastapi import APIRouter, Depends

from app.config import settings
from app.deps import get_retriever, get_store
from app.models import (
    DocumentStats,
    KbDocumentDetailResponse,
    KbDocumentListResponse,
    KbStatsResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedItem,
)
from app.rag.retriever import Retriever
from app.rag.store import ChromaStore

router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])


@router.post("/retrieve", response_model=RetrieveResponse, summary="只读检索")
def retrieve(
    request: RetrieveRequest,
    retriever: Retriever = Depends(get_retriever),
) -> RetrieveResponse:
    """检索知识库，返回带编号、原文和来源的结果。

    MCP 的 search_knowledge_base 工具直接复用这个接口，
    因此 REST 和 MCP 两边的检索参数、阈值和排序完全一致。
    """
    chunks = retriever.retrieve(
        request.query, top_k=request.top_k, source=request.source
    )

    results = [
        RetrievedItem(
            index=index,
            source=chunk.metadata.get("source", "未知"),
            page=chunk.metadata.get("page"),
            chunk_id=chunk.id,
            snippet=chunk.text[:200],
            content=chunk.text,
            distance=round(float(chunk.distance), 4),
            retrieval=chunk.metadata.get("retrieval", "dense"),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]

    return RetrieveResponse(
        query=request.query,
        count=len(results),
        message=(
            f"检索到 {len(results)} 个片段"
            if results
            else "知识库中没有找到相关内容。"
        ),
        results=results,
    )


@router.get(
    "/documents",
    response_model=KbDocumentListResponse,
    summary="列出文档及统计",
)
def list_documents(
    store: ChromaStore = Depends(get_store),
) -> KbDocumentListResponse:
    """返回每篇文档的块数、页码和字符数。"""
    documents = [
        DocumentStats(**item) for item in store.list_document_stats()
    ]
    return KbDocumentListResponse(count=len(documents), documents=documents)


@router.get(
    "/documents/{source}",
    response_model=KbDocumentDetailResponse,
    summary="单篇文档详情",
)
def document_detail(
    source: str,
    store: ChromaStore = Depends(get_store),
) -> KbDocumentDetailResponse:
    """按完整文件名查询单篇文档；不存在时返回 found=False。"""
    stats = store.get_document_stats(source)
    if stats is None:
        return KbDocumentDetailResponse(
            found=False,
            message=f"未找到文档：{source}",
        )
    return KbDocumentDetailResponse(
        found=True,
        document=DocumentStats(**stats),
    )


@router.get("/stats", response_model=KbStatsResponse, summary="知识库状态")
def stats(store: ChromaStore = Depends(get_store)) -> KbStatsResponse:
    """返回文档数、块数和当前检索配置。"""
    documents = store.list_document_stats()
    return KbStatsResponse(
        documents=len(documents),
        chunks=store.count(),
        collection=settings.collection_name,
        embedding_model=settings.embedding_model,
        hybrid_enabled=settings.hybrid_enabled,
        top_k=settings.top_k,
        max_distance=settings.max_distance,
    )
