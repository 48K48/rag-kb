"""Streamlit 前端：通过 HTTP/SSE 调用 FastAPI 的 Agent 接口。

为什么前端不直接 import Agent：
    1. 前后端分离，Streamlit 可以单独部署/重启；
    2. FastAPI 仍是唯一的业务入口，/docs、测试和 curl 都能复用；
    3. Streamlit 进程不需要加载 Chroma、Ollama、LangChain，启动更轻。

启动：
    streamlit run app/frontend/streamlit_app.py

环境变量：
    API_BASE_URL  默认 http://127.0.0.1:8000
"""

import json
import os
from urllib.parse import quote

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = httpx.Timeout(180.0, connect=10.0)

st.set_page_config(
    page_title="知识库 Agent",
    page_icon="🤖",
    layout="centered",
)

# 只调整页面宽度和来源卡片，不依赖任何前端框架。
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 920px; }
      div[data-testid="stChatMessage"] { padding: 0.25rem 0; }
      .source-line { color: #2563eb; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    """GET 一个 JSON 接口；失败时返回 (None, 错误信息)。"""
    try:
        response = httpx.get(f"{API_BASE}{path}", timeout=10.0)
        response.raise_for_status()
        return response.json(), None
    except Exception as error:
        return None, str(error)


def iter_sse(response):
    """逐行解析 SSE 响应，产出 (event, data) 二元组。

    后端格式：
        event: token
        data: {"text": "..."}
        （空行）
    """
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in response.iter_lines():
        line = raw_line.rstrip("\r")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = {"raw": payload}
                yield event_name, data
            event_name, data_lines = "message", []
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    # 兼容服务器没有以空行结束最后一个事件的情况。
    if data_lines:
        payload = "\n".join(data_lines)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {"raw": payload}
        yield event_name, data


def render_sources(sources) -> None:
    """在助手消息下方展示引用来源。"""
    if not sources:
        return

    with st.expander(f"引用来源（{len(sources)}）", expanded=False):
        for source in sources:
            distance = float(source.get("distance") or 0)
            distance_text = (
                "BM25 召回" if distance < 0 else f"距离 {distance:.4f}"
            )
            page = f" · 第 {source['page']} 页" if source.get("page") else ""
            st.markdown(
                f"**[{source.get('index', '-')}] {source.get('source', '未知')}**"
                f"{page} · {distance_text}"
            )
            st.caption(source.get("snippet", ""))


def render_tool_calls(tool_calls) -> None:
    """展示 Agent 本轮调用过的工具，方便调试和解释行为。"""
    if not tool_calls:
        return

    with st.expander(f"工具调用（{len(tool_calls)}）", expanded=False):
        for call in tool_calls:
            args = json.dumps(call.get("args", {}), ensure_ascii=False)
            st.code(f"{call.get('name', 'tool')}({args})", language="text")


def render_assistant_message(message: dict) -> None:
    """渲染一条助手消息：正文 + 来源 + 工具调用。"""
    st.markdown(message.get("content", ""))
    if message.get("refused"):
        st.caption("知识库中没有找到相关内容，Agent 已拒答。")
    elif message.get("used_rag") is False:
        if message.get("tool_calls"):
            st.caption("本轮未调用检索工具（Agent 使用了其他工具）。")
        else:
            st.caption("本轮未调用知识库检索（Agent 直接回答）。")
    elif message.get("used_rag") and not message.get("sources"):
        st.caption("已调用知识库检索，但没有命中片段。")

    tool_source = message.get("tool_source")
    if tool_source:
        st.caption(f"工具来源：{'MCP' if tool_source == 'mcp' else '本地'}")

    render_tool_calls(message.get("tool_calls", []))
    render_sources(message.get("sources", []))


# ---------- 会话状态 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("知识库")
    health, health_error = api_get("/api/health")

    if health:
        st.success(f"API 正常 · {health.get('collection_count', 0)} 个文本块")
        st.caption(f"Agent 模型：{health.get('agent_model', '-')}")
        if not health.get("agent_configured"):
            st.warning("未配置 Agent API Key，请检查 .env")
    else:
        st.error(f"API 不可用：{health_error}")

    st.divider()
    st.subheader("上传文档")
    uploaded = st.file_uploader(
        "支持 md / txt / pdf",
        type=["md", "markdown", "txt", "pdf"],
        accept_multiple_files=True,
    )
    if st.button(
        "上传并入库",
        disabled=not uploaded,
        width="stretch",
    ):
        files = [
            (
                "files",
                (
                    item.name,
                    item.getvalue(),
                    item.type or "application/octet-stream",
                ),
            )
            for item in uploaded
        ]
        try:
            with st.spinner("正在切分并写入向量库…"):
                response = httpx.post(
                    f"{API_BASE}/api/ingest", files=files, timeout=120.0
                )
                response.raise_for_status()
                result = response.json()
            st.success(
                f"入库完成：{result['files']} 个文件，{result['chunks']} 个块"
            )
            st.rerun()
        except Exception as error:
            st.error(f"上传失败：{error}")

    st.divider()
    st.subheader("已入库文档")
    documents, documents_error = api_get("/api/documents")

    if documents:
        st.caption(f"共 {documents.get('count', 0)} 个文档")
        for source in documents.get("sources", []):
            name_col, action_col = st.columns([4, 1.5])
            name_col.write(source)
            if action_col.button("删除", key=f"delete-{source}", width="stretch"):
                try:
                    response = httpx.delete(
                        f"{API_BASE}/api/documents/{quote(source, safe='')}",
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    st.success(f"已删除：{source}")
                    st.rerun()
                except Exception as error:
                    st.error(f"删除失败：{error}")
    elif documents_error:
        st.error(documents_error)
    else:
        st.caption("还没有文档")

    st.divider()
    if st.button("清空对话", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# ---------- 主区域 ----------
st.title("知识库 Agent")
st.caption("Agent 自主决定是否检索知识库 · RAG 只是工具之一 · 支持多轮对话")

# 先渲染历史，再处理本次输入。
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])

prompt = st.chat_input(
    "问点什么，例如：AI Agent 由哪几部分组成？",
    max_chars=2000,
)

if prompt:
    # 历史只发最近 6 轮，避免请求体过大；后端还会再裁剪一次。
    # 单条内容截到 4000 字，和 AgentMessage 的校验上限保持一致。
    history = [
        {"role": item["role"], "content": item["content"][:4000]}
        for item in st.session_state.messages[-12:]
    ]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Agent 正在分析问题…", expanded=True)
        answer_box = st.empty()

        answer_text = ""
        sources: list[dict] = []
        tool_calls: list[dict] = []
        used_rag = False
        refused = False
        tool_source = None
        error_message: str | None = None

        try:
            with httpx.stream(
                "POST",
                f"{API_BASE}/api/agent/chat/stream",
                json={"question": prompt, "history": history},
                timeout=REQUEST_TIMEOUT,
            ) as response:
                if response.status_code != 200:
                    detail = response.read().decode("utf-8", errors="ignore")
                    error_message = f"HTTP {response.status_code}：{detail}"
                    status.update(label="请求失败", state="error")
                    st.error(error_message)
                else:
                    for event_name, data in iter_sse(response):
                        if event_name == "status":
                            status.write(data.get("message", "处理中…"))
                        elif event_name == "tool_call":
                            name = data.get("name", "tool")
                            args = json.dumps(
                                data.get("args", {}), ensure_ascii=False
                            )
                            tool_calls.append(
                                {"name": name, "args": data.get("args", {})}
                            )
                            status.write(f"🔧 调用工具 `{name}`：`{args}`")
                        elif event_name == "tool_result":
                            status.write(f"✅ {data.get('summary', '工具执行完成')}")
                        elif event_name == "sources":
                            sources = data.get("sources", [])
                        elif event_name == "answer":
                            answer_text = data.get("text", "")
                            answer_box.markdown(answer_text)
                        elif event_name == "token":
                            # 后端当前发送完整 answer；保留 token 事件兼容未来的逐字流式。
                            answer_text += data.get("text", "")
                            answer_box.markdown(answer_text)
                        elif event_name == "done":
                            sources = data.get("sources", sources)
                            tool_calls = data.get("tool_calls", tool_calls)
                            used_rag = bool(data.get("used_rag", used_rag))
                            refused = bool(data.get("refused", False))
                            tool_source = data.get("tool_source", tool_source)
                            if not answer_text:
                                answer_text = data.get("answer", "")
                                answer_box.markdown(answer_text)
                        elif event_name == "error":
                            error_message = data.get(
                                "message", "Agent 调用失败"
                            )
                            status.update(
                                label="Agent 调用失败", state="error"
                            )
                            st.error(error_message)
        except Exception as error:
            error_message = f"请求异常：{error}"
            status.update(label="请求失败", state="error")
            st.error(error_message)

        if error_message is None and not answer_text:
            error_message = "Agent 没有返回内容，请重试。"
            status.update(label="没有返回内容", state="error")
            st.warning(error_message)

        if error_message is None:
            if refused:
                status.update(
                    label="知识库中没有找到相关内容",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(label="完成", state="complete", expanded=False)

            render_sources(sources)
            if used_rag and not sources:
                st.caption("已调用知识库检索，但没有命中片段。")
            elif used_rag:
                st.caption("本轮已调用知识库检索。")
            elif tool_calls:
                st.caption("本轮未调用检索工具（Agent 使用了其他工具）。")
            else:
                st.caption("本轮未调用知识库检索（Agent 直接回答）。")

            if tool_source:
                st.caption(f"工具来源：{'MCP' if tool_source == 'mcp' else '本地'}")

    if error_message is None:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "tool_calls": tool_calls,
                "used_rag": used_rag,
                "refused": refused,
                "tool_source": tool_source,
            }
        )
