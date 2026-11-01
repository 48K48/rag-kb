"""
依赖装配：集中创建「单例」，供 FastAPI 的 Depends 注入。

为什么要用 lru_cache：
    - ChromaStore 初始化会打开数据库、创建集合；
    - OllamaEmbedder / OpenAI 客户端也只需要一个实例；
    - 用 lru_cache 后，每个进程只创建一次，避免每个请求都重复初始化。

为什么用函数而不是模块级变量：
    - 模块级变量在 import 时就执行，配置错误会导致整个服务启动失败；
    - 函数式装配可以延迟到第一次请求，也方便测试时用
      app.dependency_overrides 替换成 Fake 对象。
"""

from functools import lru_cache

from app.agent.agent import KnowledgeAgent
from app.config import settings
from app.rag.chain import RagChain
from app.rag.embedder import OllamaEmbedder
from app.rag.retriever import Retriever
from app.rag.store import ChromaStore


@lru_cache
def get_embedder() -> OllamaEmbedder:
    """创建 embedding 客户端（进程内单例）。"""
    return OllamaEmbedder(settings.embedding_model, settings.ollama_base_url)


@lru_cache
def get_store() -> ChromaStore:
    """创建向量库（进程内单例）。

    注意：这里只打开数据库，不会调用 Ollama，
    所以 /api/health 可以在 Ollama 没启动时也返回库内数量。
    """
    return ChromaStore(settings.chroma_dir, settings.collection_name, get_embedder())


@lru_cache
def get_retriever() -> Retriever:
    """创建检索器（进程内单例），使用配置里的检索和混合检索参数。"""
    return Retriever(
        store=get_store(),
        top_k=settings.top_k,
        max_distance=settings.max_distance,
        hybrid_enabled=settings.hybrid_enabled,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rrf_k=settings.rrf_k,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
    )


@lru_cache
def get_chain() -> RagChain:
    """创建 RAG 问答链（进程内单例）。

    没有配置 DEEPSEEK_API_KEY 时也能创建，
    真正调用 answer() 时才会抛出可读错误。
    """
    return RagChain(
        retriever=get_retriever(),
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
        max_context_chars=settings.max_context_chars,
    )


@lru_cache
def get_agent_service() -> KnowledgeAgent:
    """创建 Agent 服务（进程内单例）。

    与 RagChain 共用同一个 Retriever 和 ChromaStore，
    所以 Agent 工具检索到的数据与旧的 /api/chat 完全一致。
    """
    return KnowledgeAgent(
        retriever=get_retriever(),
        store=get_store(),
        api_key=settings.agent_api_key,
        base_url=settings.agent_base_url,
        model=settings.agent_model,
        temperature=settings.agent_temperature,
        max_tokens=settings.agent_max_tokens,
        max_history_turns=settings.agent_max_history_turns,
        recursion_limit=settings.agent_recursion_limit,
        tool_source=settings.agent_tool_source,
    )
