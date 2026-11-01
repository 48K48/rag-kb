"""
检索层：向量召回 + BM25 召回 + RRF 融合 + 距离阈值。

两种模式：
    1. 纯向量（HYBRID_ENABLED=false）：
        只用 Chroma 向量检索，按 MAX_DISTANCE 过滤。
        这是 baseline，行为和加混合检索之前完全一致。
    2. 混合检索（HYBRID_ENABLED=true）：
        向量召回 top DENSE_TOP_K
        + BM25 召回 top SPARSE_TOP_K
        -> RRF 融合 -> 取 top_k。

为什么用 RRF（Reciprocal Rank Fusion）：
    向量分数（cosine 距离）和 BM25 分数（关键词得分）量纲不同，
    直接加权相加需要先归一化，而且权重很难调。
    RRF 只看「排名」，不看原始分数：
        score = 1/(k + rank_dense) + 1/(k + rank_sparse)
    两路都排得靠前的文档得分最高，实现简单、鲁棒，是混合检索的常用做法。

为什么需要阈值：
    向量库永远会返回 top_k 个「最相似」的结果，
    即使问题和知识库完全无关，也会有某个块排第一。
    不做过滤，模型会拿着不相关的资料强行回答，产生幻觉。
    注意：阈值只作用于向量召回；BM25 结果用「分数 > 0」判断是否命中关键词。
"""

from dataclasses import replace

from app.rag.bm25 import BM25Index
from app.rag.store import ChromaStore
from app.rag.types import RetrievedChunk


class Retriever:
    """向量检索 + BM25 检索 + RRF 融合。"""

    def __init__(
        self,
        store: ChromaStore, #向量库
        top_k: int = 5,     #召回相似度最高的k条
        max_distance: float = 0.6,  #最大向量距离
        hybrid_enabled: bool = False,    #混合检索开关
        dense_top_k: int = 10,          #稠密向量检索单独召回几条
        sparse_top_k: int = 10,         #稀疏向量检索单独召回几条
        rrf_k: int = 60,                #平滑常数
        bm25_k1: float = 1.5,      #控制 BM25 里面【词频的饱和速度】，只对稀疏 BM25 检索生效
        bm25_b: float = 0.75,      #控制文档长度惩罚
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.max_distance = max_distance

        # ---------- 混合检索参数 ----------
        self.hybrid_enabled = hybrid_enabled
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = max(1, rrf_k)  # 防止配置成 0 导致除零
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        # BM25 索引懒加载：第一次检索时才构建，避免服务启动变慢。
        self._bm25: BM25Index | None = None
        # 记录索引对应的数据版本；Chroma 写入/删除后 version 自增，
        # 下次检索发现版本不一致就重建，保证新文档能被 BM25 检索到。
        self._bm25_version = -1

    def _get_bm25(self) -> BM25Index | None:
        """获取 BM25 索引；必要时从 Chroma 全量块重建。"""
        if not self.hybrid_enabled:
            return None

        if self._bm25 is None or self._bm25_version != self.store.version:
            chunks = self.store.get_all_chunks()
            self._bm25 = BM25Index(chunks, k1=self.bm25_k1, b=self.bm25_b)
            self._bm25_version = self.store.version

        return self._bm25

    def retrieve(
        self, question: str, top_k: int | None = None, source: str | None = None
    ) -> list[RetrievedChunk]:
        """检索与问题相关的文本块。

        参数：
            question: 用户问题；
            top_k: 覆盖默认检索条数（接口层允许用户临时指定）；
            source: 只检索指定来源文件。

        返回：
            按相关性排序的文本块。
            - 纯向量模式：距离 <= max_distance 的结果；
            - 混合模式：RRF 融合后的 top_k，可能包含 distance=-1.0 的 BM25-only 结果。
            如果两路都没有结果，返回空列表，上层据此拒答。
        """
        final_k = top_k or self.top_k

        # 混合模式下向量多召回一些，给 RRF 更多候选；
        # 纯向量模式保持原来的 top_k，确保 baseline 行为不变。
        dense_k = max(self.dense_top_k, final_k) if self.hybrid_enabled else final_k
        dense_results = self.store.search(question, dense_k, source)

        # 距离阈值只过滤向量结果；BM25 没有 cosine 距离，不能套用这个阈值。
        dense_results = [
            item for item in dense_results if item.distance <= self.max_distance
        ]

        if not self.hybrid_enabled:
            return dense_results[:final_k]

        bm25 = self._get_bm25()
        sparse_results = (
            bm25.search(question, max(self.sparse_top_k, final_k), source)
            if bm25
            else []
        )

        if not dense_results and not sparse_results:
            return []

        return self._rrf_fuse(dense_results, sparse_results, final_k)

    def _rrf_fuse(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """用 RRF 融合两路召回结果。

        实现步骤：
            1. 用 id 去重，同一块在两路都出现时只保留一条；
            2. 每个块累加 1/(rrf_k + rank)，rank 从 1 开始；
            3. 按 RRF 分数降序，取 top_k；
            4. 在 metadata 里标记 retrieval=dense/bm25/hybrid，方便调试。
        """
        merged: dict[str, dict] = {}

        for rank, item in enumerate(dense_results, start=1):
            entry = merged.setdefault(
                item.id, {"chunk": item, "score": 0.0, "methods": set()}
            )
            entry["score"] += 1.0 / (self.rrf_k + rank)
            entry["methods"].add("dense")

        for rank, item in enumerate(sparse_results, start=1):
            entry = merged.setdefault(
                item.id, {"chunk": item, "score": 0.0, "methods": set()}
            )
            entry["score"] += 1.0 / (self.rrf_k + rank)
            entry["methods"].add("bm25")

        # 排序规则：
        #   1. RRF 分数高的在前；
        #   2. 分数相同时，向量距离小的在前（BM25-only 的距离是 -1，用 999 兜底）。
        ordered = sorted(
            merged.values(),
            key=lambda entry: (
                -entry["score"],
                entry["chunk"].distance
                if entry["chunk"].distance >= 0
                else 999.0,
            ),
        )[:top_k]

        results: list[RetrievedChunk] = []
        for entry in ordered:
            chunk: RetrievedChunk = entry["chunk"]
            # 两路都命中 = hybrid；只命中一路 = dense 或 bm25。
            method = (
                "hybrid"
                if len(entry["methods"]) == 2
                else next(iter(entry["methods"]))
            )
            # dataclasses.replace 生成一个新对象，不修改缓存里的原对象。
            results.append(
                replace(
                    chunk,
                    metadata={**chunk.metadata, "retrieval": method},
                )
            )

        return results
