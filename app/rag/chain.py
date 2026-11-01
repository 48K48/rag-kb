"""
RAG 问答链：检索 -> 拼上下文 -> 调 DeepSeek -> 返回答案和引用。

这个文件是整个项目的核心，负责：
    1. 把用户问题交给 Retriever 检索；
    2. 把检索到的块拼成带编号的「参考资料」；
    3. 用 system prompt 约束模型「只能根据资料回答、必须标注引用」；
    4. 识别拒答（检索为空，或模型回答固定拒答话术）；
    5. 非流式 answer() 和流式 stream_answer() 两种输出。

为什么不用 LangChain 的 RetrievalQA：
    显式写出来每一步，才能讲清楚「上下文怎么拼、引用怎么来、
    拒答怎么判断」，面试时也更容易解释。
"""

import time

from openai import OpenAI

from app.models import ChatResponse, Source
from app.rag.retriever import Retriever
from app.rag.types import RetrievedChunk

# system prompt 是 RAG 防幻觉的第一道约束。
# 第 3 条很关键：明确告诉模型「资料不足就说不知道」，
# 否则模型倾向于用自身知识补全，导致答案看起来合理但无法溯源。
SYSTEM_PROMPT = """你是知识库问答助手。规则：
1. 只能根据「参考资料」回答，不得使用外部知识，不得编造。
2. 每个结论后用 [1]、[2] 这样的编号标注来源。
3. 如果参考资料不足以回答，直接回答“知识库中没有找到相关内容。”，不要猜测。
4. 用简体中文回答，简洁准确。"""

# 用固定话术识别「模型主动拒答」。
# 检索为空时我们也会直接返回这句话，所以两种拒答可以用同一个标记。
REFUSAL_MARKER = "知识库中没有找到相关内容"


class RagChain:
    """把 Retriever 和 DeepSeek 串起来的问答链。"""

    def __init__(
        self,
        retriever: Retriever,   #检索器
        api_key: str,
        model: str,
        base_url: str,
        max_context_chars: int = 6000,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.max_context_chars = max_context_chars

        # 没有 Key 时不创建客户端，等真正调用时再抛出明确错误。
        # 这样服务仍能启动，/api/health 也能正常返回（提示未配置 Key），
        # 而不是在 import 阶段就崩溃。
        self.client = (
            OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        )

    def _ensure_client(self) -> OpenAI:
        """确保 API Key 已配置；没有则抛出可读错误。"""
        if self.client is None:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 .env 中设置后再试。")
        return self.client

    def _retrieve(
        self, question: str, top_k: int | None, source: str | None
    ) -> list[RetrievedChunk]:
        """检索的薄封装，方便以后在这里加查询改写、多路召回等逻辑。"""
        return self.retriever.retrieve(question, top_k=top_k, source=source)

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """把检索结果拼成带编号的参考资料。

        格式示例：
            [1] 来源：agent笔记.md
            AI Agent 由模型、工具、记忆和规划四部分组成。

            [2] 来源：rag笔记.md 第2页
            RAG 的流程是……
        """
        parts: list[str] = []
        used_chars = 0

        for index, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "未知")
            page = chunk.metadata.get("page")
            location = f"{source} 第{page}页" if page else source

            # 控制上下文总长度：
            # 模型窗口有限，而且 token 是按量计费的。
            # 超出预算时截断当前块，并在下一轮直接退出。
            remain = self.max_context_chars - used_chars
            if remain <= 0:
                break
            text = chunk.text[:remain]

            parts.append(f"[{index}] 来源：{location}\n{text}")
            used_chars += len(text)

        return "\n\n".join(parts)

    @staticmethod
    def _to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
        """把内部 RetrievedChunk 转成 API 的 Source 模型。

        编号 index 必须和 prompt 里的 [1][2] 一致，
        所以这里用 enumerate(chunks, start=1)。
        """
        return [
            Source(
                index=index,
                source=chunk.metadata.get("source", "未知"),
                page=chunk.metadata.get("page"),
                chunk_id=chunk.id,
                # 只截前 200 字给前端展示，避免响应体过大。
                snippet=chunk.text[:200],
                distance=round(chunk.distance, 4),
            )
            for index, chunk in enumerate(chunks, start=1)
        ]

    def _messages(self, question: str, context: str) -> list[dict]:
        """构造 OpenAI 兼容的 messages 列表。"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"参考资料：\n{context}\n\n问题：{question}",
            },
        ]

    def answer(
        self, question: str, top_k: int | None = None, source: str | None = None
    ) -> ChatResponse:
        """非流式问答：一次性返回完整答案。"""
        start = time.perf_counter()
        chunks = self._retrieve(question, top_k, source)

        # 检索阶段直接拒答：没有块通过阈值，就没必要调用大模型，
        # 既避免幻觉，也节省 token。
        if not chunks:
            return ChatResponse(
                answer="知识库中没有找到相关内容。",
                sources=[],
                refused=True,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=self._messages(question, self._build_context(chunks)),
            # 低温度：RAG 需要稳定、可复现的答案，不需要发散创作。
            temperature=0.1,
        )
        answer = response.choices[0].message.content or ""

        # token 用量用于成本统计；不同兼容接口的字段可能略有差异。
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            answer=answer,
            sources=self._to_sources(chunks),
            # 两种拒答都算：
            #   1) 检索为空（上面已提前返回）；
            #   2) 模型判断资料不足，回答里包含固定拒答话术。
            refused=REFUSAL_MARKER in answer,
            usage=usage,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    def stream_answer(
        self, question: str, top_k: int | None = None, source: str | None = None
    ):
        """流式问答：生成一系列 SSE 事件。

        事件顺序：
            refused                        -> 检索为空，直接拒答
            sources -> token -> token ...  -> 先给来源，再逐字输出
            done                           -> 结束，带 refused 标记

        为什么先发 sources：
            前端可以先展示「依据」，用户不用等答案生成完才知道引用了哪些文档。
        """
        chunks = self._retrieve(question, top_k, source)
        if not chunks:
            yield {
                "event": "refused",
                "data": {"answer": "知识库中没有找到相关内容。"},
            }
            return

        client = self._ensure_client()

        # 先发来源事件；model_dump() 把 Pydantic 模型转成可 JSON 序列化的 dict。
        yield {
            "event": "sources",
            "data": {"sources": [s.model_dump() for s in self._to_sources(chunks)]},
        }

        stream = client.chat.completions.create(
            model=self.model,
            messages=self._messages(question, self._build_context(chunks)),
            temperature=0.1,
            stream=True,
        )

        # 一边转发 token，一边把内容累积起来，
        # 最后用于判断模型是否输出了拒答话术。
        answer_parts: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                answer_parts.append(delta)
                yield {"event": "token", "data": {"text": delta}}

        yield {
            "event": "done",
            "data": {"refused": REFUSAL_MARKER in "".join(answer_parts)},
        }
