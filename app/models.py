"""
API 层的数据模型（Pydantic）。

这些模型负责三件事：
    1. 定义接口的输入/输出结构，FastAPI 会自动生成 /docs 文档；
    2. 自动校验参数（比如 top_k 必须在 1-20 之间），非法请求返回 422；
    3. 把内部数据序列化成 JSON 返回给前端。

注意：这里的模型只用于「对外接口」，
RAG 内部传递的 Chunk / RetrievedChunk 用的是轻量 dataclass（见 app/rag/types.py），
两者职责不同，不要混用。
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/chat 和 /api/chat/stream 的请求体。"""

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="用户问题；长度限制防止超长输入浪费 token",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="检索条数；太小容易漏答案，太大引入噪声",
    )
    source: str | None = Field(
        default=None,
        description="只检索指定来源文件（metadata 过滤）；不传表示全库检索",
    )


class Source(BaseModel):
    """一条引用来源，返回给前端展示「答案依据」。"""

    index: int              # 与答案里的 [1][2] 编号对应
    source: str             # 文件名，例如 agent笔记.md
    page: int | None = None # PDF 的页码；Markdown/TXT 为 None
    chunk_id: str           # 文本块在向量库里的唯一 id
    snippet: str            # 原文片段（截取前 200 字），方便人工核对
    distance: float         # cosine 距离，越小越相关


class AgentMessage(BaseModel):
    """多轮对话历史中的一条消息（只保存用户和助手文本）。

    Agent 的工具调用和工具结果由后端在单次请求内处理，
    不需要客户端回传，避免历史过长和格式不一致。
    """

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AgentChatRequest(BaseModel):
    """POST /api/agent/chat 和 /api/agent/chat/stream 的请求体。"""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="用户问题；比纯 RAG 接口稍长，给 Agent 更多上下文",
    )
    history: list[AgentMessage] = Field(
        default_factory=list,
        max_length=20,
        description="最近几轮对话，由客户端保存并回传；后端只使用最后 N 轮",
    )


class ToolCallInfo(BaseModel):
    """Agent 执行过的一次工具调用，返回给前端展示。"""

    name: str
    args: dict = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Agent 问答接口的响应体。"""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    # 本轮是否调用过知识库检索；False 表示 Agent 直接回答。
    used_rag: bool = False
    # 资料不足时的固定拒答标记。
    refused: bool = False
    # 本轮工具来自 MCP 还是进程内本地工具，便于前端展示和排查回退。
    tool_source: Literal["mcp", "local"] | None = None
    usage: dict | None = None
    latency_ms: int | None = None


class ChatResponse(BaseModel):
    """问答接口的响应体。"""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    # refused=True 表示：
    #   1) 检索阶段没有召回任何块，或
    #   2) 模型判断参考资料不足，返回了固定拒答话术。
    refused: bool = False
    # token 用量，用于成本统计；流式接口暂不返回。
    usage: dict | None = None
    # 服务端处理耗时（毫秒），用于性能对比。
    latency_ms: int | None = None


class IngestResponse(BaseModel):
    """POST /api/ingest 的响应体。"""

    files: int              # 本次上传的文件数
    chunks: int             # 本次切分出的文本块数
    collection_count: int   # 入库后向量库里的总块数


class DocumentListResponse(BaseModel):
    """GET /api/documents 的响应体。"""

    count: int
    sources: list[str]


# ---------- 只读知识库接口（/api/kb/*，供 MCP 服务复用） ----------


class RetrieveRequest(BaseModel):
    """POST /api/kb/retrieve 的请求体。"""

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="检索问题；保留关键名词，不要传空字符串",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回条数；太大会引入噪声和额外 token",
    )
    source: str | None = Field(
        default=None,
        description="只检索指定来源文件；必须是完整文件名",
    )


class RetrievedItem(BaseModel):
    """一条检索结果。

    index 同时用于答案里的 [1][2] 和前端来源面板编号；
    content 给模型看完整原文，snippet 给前端展示。
    """

    index: int
    source: str
    page: int | None = None
    chunk_id: str
    snippet: str
    content: str
    distance: float
    retrieval: str = "dense"


class RetrieveResponse(BaseModel):
    """POST /api/kb/retrieve 的响应体，也是 MCP 检索工具的返回值。"""

    query: str
    count: int
    message: str
    results: list[RetrievedItem] = Field(default_factory=list)


class DocumentStats(BaseModel):
    """单篇文档的统计信息。"""

    source: str
    chunks: int
    pages: list[int] = Field(default_factory=list)
    chars: int
    min_chunk_chars: int | None = None
    max_chunk_chars: int | None = None


class KbDocumentListResponse(BaseModel):
    """GET /api/kb/documents 的响应体。"""

    count: int
    documents: list[DocumentStats] = Field(default_factory=list)


class KbDocumentDetailResponse(BaseModel):
    """GET /api/kb/documents/{source} 的响应体。

    文档不存在时返回 found=False，而不是 404：
    MCP 工具把“没找到”当作正常结果交给模型，让模型继续换关键词或说明情况。
    """

    found: bool
    document: DocumentStats | None = None
    message: str = ""


class KbStatsResponse(BaseModel):
    """GET /api/kb/stats 的响应体。"""

    documents: int
    chunks: int
    collection: str
    embedding_model: str
    hybrid_enabled: bool
    top_k: int
    max_distance: float
