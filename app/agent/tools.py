"""Agent 工具：把现有 RAG 能力包装成 LangChain Tool。

设计要点：
    1. 不重新实现检索，直接复用 app.rag.retriever.Retriever；
    2. search_knowledge_base 用 content_and_artifact：
       - content 给模型看，是带 [1][2] 编号的原文；
       - artifact 给程序用，是结构化的 sources，前端据此展示引用；
    3. list_knowledge_base_documents 让 Agent 能回答“知识库里有什么”。
"""

import json

from langchain_core.tools import BaseTool, tool

from app.config import settings
from app.rag.retriever import Retriever
from app.rag.store import ChromaStore
from app.rag.types import RetrievedChunk


def _format_location(chunk: RetrievedChunk) -> str:
    """把 metadata 里的来源和页码拼成人类可读的位置。"""
    source = chunk.metadata.get("source", "未知")
    page = chunk.metadata.get("page")
    return f"{source} 第{page}页" if page else str(source)


def _format_distance(distance: float) -> str:
    """BM25-only 结果是 -1.0，不能显示成假的 cosine 距离。"""
    return "BM25 召回" if distance < 0 else f"距离 {distance:.4f}"


def _to_source(chunk: RetrievedChunk, index: int) -> dict:
    """把 RetrievedChunk 转成 artifact 里的来源字典。"""
    return {
        "index": index,
        "source": chunk.metadata.get("source", "未知"),
        "page": chunk.metadata.get("page"),
        "chunk_id": chunk.id,
        "snippet": chunk.text[:200],
        "distance": round(float(chunk.distance), 4),
    }


def build_tools(retriever: Retriever, store: ChromaStore) -> list[BaseTool]:
    """创建 Agent 可用的工具列表。

    参数：
        retriever: 现有检索器，负责向量/BM25 召回和阈值过滤；
        store: Chroma 封装，负责列出已入库文档。
    """

    @tool(response_format="content_and_artifact")
    def search_knowledge_base(
        query: str, top_k: int = 5, source: str | None = None
    ) -> tuple[str, list[dict]]:
        """在本地知识库中检索与问题相关的原文片段。

        Args:
            query: 检索关键词或问题，尽量保留用户原话里的关键名词。
            top_k: 返回片段数，范围 1-10，默认 5。
            source: 可选，只在这个文件名里检索，例如 agent笔记.md。
        """
        # 模型可能传入过大/过小的值，这里统一收敛到安全范围。
        limit = max(1, min(int(top_k), 10))
        chunks = retriever.retrieve(query, top_k=limit, source=source)

        if not chunks:
            return "知识库中没有检索到相关内容。", []

        parts: list[str] = []
        sources: list[dict] = []
        for index, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[{index}] 来源：{_format_location(chunk)}"
                f"（{_format_distance(chunk.distance)}）\n{chunk.text}"
            )
            sources.append(_to_source(chunk, index))

        return "\n\n".join(parts), sources

    @tool
    def list_knowledge_base_documents() -> str:
        """列出知识库中已经入库的所有文档文件名。"""
        sources = store.list_sources()
        if not sources:
            return "知识库中还没有文档。"
        return "已入库文档：\n" + "\n".join(f"- {name}" for name in sources)

    @tool
    def get_document_info(source: str) -> str:
        """查询单篇文档的块数、页码、总字数和单块长度范围。

        source 必须是完整文件名，例如 agent笔记.md；文档不存在时返回 found=false。
        """
        stats = store.get_document_stats(source)
        if stats is None:
            return json.dumps(
                {"found": False, "message": f"未找到文档：{source}"},
                ensure_ascii=False,
            )
        return json.dumps(
            {"found": True, "document": stats},
            ensure_ascii=False,
        )

    @tool
    def knowledge_base_stats() -> str:
        """返回知识库整体状态：文档数、块数、embedding 模型和混合检索开关。"""
        documents = store.list_document_stats()
        payload = {
            "documents": len(documents),
            "chunks": store.count(),
            "collection": settings.collection_name,
            "embedding_model": settings.embedding_model,
            "hybrid_enabled": settings.hybrid_enabled,
            "top_k": settings.top_k,
            "max_distance": settings.max_distance,
        }
        return json.dumps(payload, ensure_ascii=False)

    return [
        search_knowledge_base,
        list_knowledge_base_documents,
        get_document_info,
        knowledge_base_stats,
    ]
