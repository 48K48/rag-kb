"""Agent 问答接口：普通响应 + SSE 流式响应。

与 app/api/chat.py 的区别：
    /api/chat 走固定的 RagChain（baseline）；
    /api/agent/chat 走 KnowledgeAgent，由模型决定是否调用 RAG 工具。
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.agent import KnowledgeAgent
from app.deps import get_agent_service
from app.models import AgentChatRequest, AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])


@router.post("/agent/chat", response_model=AgentResponse, summary="Agent 知识库问答")
async def agent_chat(
    request: AgentChatRequest,
    agent: KnowledgeAgent = Depends(get_agent_service),
) -> AgentResponse:
    """非流式 Agent 问答。

    Agent 可能：
        - 不调用工具，直接回答；
        - 调用一次或多次知识库检索；
        - 调用文档列表工具。
    返回的 tool_calls / used_rag / tool_source 可以让前端看到 Agent 做了什么。
    """
    try:
        return await agent.answer(request.question, request.history)
    except RuntimeError as error:
        # 未配置 Key 属于可预期错误，返回 503 比 500 更准确。
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("Agent 调用失败")
        raise HTTPException(
            status_code=502, detail=f"Agent 调用失败：{error}"
        ) from error


@router.post("/agent/chat/stream", summary="Agent 知识库问答（SSE 流式）")
async def agent_chat_stream(
    request: AgentChatRequest,
    agent: KnowledgeAgent = Depends(get_agent_service),
) -> StreamingResponse:
    """SSE 流式 Agent 问答。

    事件类型：
        status      Agent 开始处理
        tool_call   模型决定调用某个工具
        tool_result 工具执行完成
        sources     检索到的引用来源
        answer      完整答案
        done        结束，带 sources/tool_calls/usage/tool_source 等汇总
        error       可读错误信息
    """

    async def event_stream():
        try:
            async for event in agent.stream_answer(
                request.question, request.history
            ):
                data = json.dumps(event["data"], ensure_ascii=False)
                # SSE 事件以空行结束；data 必须是单行 JSON。
                yield f"event: {event['event']}\ndata: {data}\n\n"
        except Exception as error:
            # 流式响应一旦开始，HTTP 状态码已经发出，无法再改成 5xx；
            # 所以兜底也走 SSE 的 error 事件。
            logger.exception("Agent 流式调用失败")
            data = json.dumps({"message": f"服务异常：{error}"}, ensure_ascii=False)
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
