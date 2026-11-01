"""
BM25 关键词检索（中文）。

为什么需要 BM25：
    向量检索擅长“语义相似”，但对“精确关键词”不敏感。
    例如用户问「metadata 过滤」，文档里也写着「metadata 过滤」，
    向量检索可能因为语义泛化把它排到后面；BM25 会直接因为词命中而排到前面。
    两路召回互补，再用 RRF 融合，通常比单路更稳。

为什么用 BM25Plus 而不是 BM25Okapi：
    BM25Okapi 在小语料上 IDF 可能为负，导致“越常见越不该出现”的反效果。
    BM25Plus 对 IDF 做了平滑，分数非负，适合当前这种只有几十个块的场景。

中文分词：
    BM25 是基于“词”的算法，中文必须先分词。
    用 jieba 切词，并过滤停用词和纯标点，否则「的、了、是」会稀释分数。
"""

import jieba
from rank_bm25 import BM25Plus

from app.rag.types import Chunk, RetrievedChunk

# 停用词：这些词几乎每篇文档都有，对区分相关性没有帮助。
# 只列最常见的，避免把有意义的词误删。
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "也", "都",
    "一个", "一种", "这个", "那个", "这些", "那些",
    "什么", "怎么", "如何", "为什么", "哪", "哪些", "哪个",
    "吗", "呢", "啊", "吧", "呀",
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on",
}


def tokenize(text: str) -> list[str]:
    """把文本切成 BM25 需要的词列表。

    处理步骤：
        1. jieba 分词（中文按词切，英文/数字保持原样）；
        2. 转小写，保证 "Metadata" 和 "metadata" 能匹配；
        3. 去掉首尾空白；
        4. 过滤停用词；
        5. 过滤纯标点/空白 token。
    """
    tokens: list[str] = []
    for token in jieba.cut(text):
        token = token.strip().lower()
        if not token or token in STOPWORDS:
            continue
        # 至少要包含一个字母、数字或汉字，过滤 "，。！？" 这类纯标点。
        if not any(char.isalnum() or "一" <= char <= "鿿" for char in token):
            continue
        tokens.append(token)
    return tokens


class BM25Index:
    """内存中的 BM25 索引。

    数据来源是 Chroma 里的全量文本块（由 Retriever 传入）。
    索引只在进程内缓存，不落盘；服务重启后第一次检索会自动重建。
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        """
        参数：
            chunks: 全量文本块；
            k1: 词频饱和参数，越大越看重词频；
            b: 文档长度归一化参数，0 表示不归一化，1 表示完全归一化。
        """
        self._chunks = chunks
        self._tokenized = [tokenize(chunk.text) for chunk in chunks]
        self._bm25 = (
            BM25Plus(self._tokenized, k1=k1, b=b) if self._tokenized else None
        )

    @property
    def size(self) -> int:
        """索引里的文档块数量（用于调试和测试）。"""
        return len(self._chunks)

    def search(
        self, query: str, top_k: int = 5, source: str | None = None
    ) -> list[RetrievedChunk]:
        """按 BM25 分数检索。

        参数：
            query: 用户问题；
            top_k: 返回条数；
            source: 可选，只在这个文件里检索（metadata 过滤）。

        返回：
            RetrievedChunk 列表，按 BM25 分数从高到低。
            distance 固定为 -1.0：这是一个标记值，表示该结果没有 cosine 距离，
            前端会显示为「BM25 召回」，而不是一个假的相似度。
        """
        if self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # 先过滤 source 和 0 分，再按分数降序排序。
        # 0 分表示查询词一个都没命中，放进结果只会增加噪声。
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if score > 0
            and (
                source is None
                or self._chunks[index].metadata.get("source") == source
            )
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)

        return [
            RetrievedChunk(
                id=self._chunks[index].id,
                text=self._chunks[index].text,
                # 在 metadata 里标记检索方式，便于调试和前端展示。
                metadata={**self._chunks[index].metadata, "retrieval": "bm25"},
                distance=-1.0,
            )
            for index, _score in candidates[:top_k]
        ]
