"""
问答接口：普通响应 + SSE 流式响应。

两个接口共用同一套 ChatRequest / RagChain：
    - POST /api/chat        一次性返回完整答案，适合脚本调用；
    - POST /api/chat/stream 逐字返回，适合浏览器聊天界面。

SSE（Server-Sent Events）的数据格式：
    event: 事件名
    data: JSON 字符串
    （空行）
前端按「空行」切分事件，再按 event 字段决定怎么处理。
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.deps import get_chain
from app.models import ChatRequest, ChatResponse
from app.rag.chain import RagChain

# prefix="/api"：这个路由下的所有路径都自动加 /api 前缀。
# tags=["chat"]：在 /docs 文档里分组显示。
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="知识库问答")
def chat(request: ChatRequest, chain: RagChain = Depends(get_chain)) -> ChatResponse:
    """非流式问答。

    FastAPI 会：
        1. 用 ChatRequest 校验请求体（非法参数返回 422）；
        2. 调用 get_chain() 注入 RagChain（测试时可被 dependency_overrides 替换）；
        3. 用 ChatResponse 序列化返回值。
    """
    try:
        return chain.answer(request.question, request.top_k, request.source)
    except RuntimeError as error:
        # 「未配置 DEEPSEEK_API_KEY」属于可预期错误，
        # 返回 503（服务暂不可用）比 500 更准确，前端也能给出友好提示。
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/chat/stream", summary="知识库问答（SSE 流式）")
def chat_stream(
    request: ChatRequest, chain: RagChain = Depends(get_chain)
) -> StreamingResponse:
    """流式问答。

    StreamingResponse 接收一个生成器：
        - 生成器每 yield 一段字符串，FastAPI 就立刻发给浏览器；
        - media_type="text/event-stream" 告诉浏览器这是 SSE。
    """

    def event_stream():
        try:
            # chain.stream_answer 产出的是 {"event": ..., "data": {...}} 字典，
            # 这里负责转成 SSE 的文本协议。
            for event in chain.stream_answer(
                request.question, request.top_k, request.source
            ):
                # ensure_ascii=False：保留中文，避免前端看到 \uXXXX。
                data = json.dumps(event["data"], ensure_ascii=False)
                # 注意末尾必须有两个换行：一个结束 data 行，一个结束事件。
                yield f"event: {event['event']}\ndata: {data}\n\n"
        except RuntimeError as error:
            # 例如没有配置 API Key：通过 SSE 的 error 事件告诉前端，
            # 而不是直接断开连接（断开后前端只能显示「网络错误」）。
            data = json.dumps({"message": str(error)}, ensure_ascii=False)
            yield f"event: error\ndata: {data}\n\n"
        except Exception as error:
            # 流式响应一旦开始，HTTP 状态码已经发出，无法再改成 500；
            # 所以兜底也走 SSE 的 error 事件。
            data = json.dumps({"message": f"服务异常：{error}"}, ensure_ascii=False)
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
