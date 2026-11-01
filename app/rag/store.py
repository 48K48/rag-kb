"""
Chroma 持久化封装。

这一层是「向量库适配器」：
    业务代码只调用 add_chunks / search / list_sources / delete_source / count，
    不关心底层是 Chroma 还是 Milvus。
    以后要接 Milvus，只需要新增一个 MilvusStore 实现同样的方法，
    再把 app/deps.py 里的 get_store() 换掉即可，其他代码不动。

Chroma 的几个关键点：
    1. PersistentClient(path=...)：数据直接落盘，进程退出不丢；
    2. 集合名要求 3-512 个字符，只能用字母/数字/点/下划线/短横线，
       且首尾必须是字母或数字（用 "t" 这种名字会直接报错）；
    3. metadata={"hnsw:space": "cosine"}：用 cosine 距离，
       文本 embedding 一般都用 cosine，距离越小越相似；
    4. upsert 而不是 add：同一个 id 重复写入会覆盖，方便重复入库；
    5. query 返回的是「按查询分组」的二维列表，
       例如 ids = [["chunk_0", "chunk_1"]]，所以要取 [0]。
"""

from pathlib import Path

import chromadb

from app.rag.types import Chunk, RetrievedChunk


class ChromaStore:
    """Chroma 向量库的薄封装。"""

    def __init__(self, persist_dir: Path, collection_name: str, embedder) -> None:
        """
        参数：
            persist_dir: 向量库落盘目录，不存在时 Chroma 会自动创建；
            collection_name: 集合名，相当于关系库里的表；
            embedder: 提供 embed_documents / embed_query 的对象，
                      生产环境是 OllamaEmbedder，测试时是 FakeEmbedder。
        """
        self._embedder = embedder
        # version 用于让 BM25 索引知道数据有没有变化：
        # 每次写入/删除都自增，Retriever 发现版本变化就重建 BM25 索引。
        self._version = 0
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # cosine 距离：0 表示方向完全一致，2 表示完全相反。
            # 对文本 embedding，cosine 通常比 L2 更合适。
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def version(self) -> int:
        """数据版本号：写入/删除后自增，供 BM25 索引判断是否需要重建。"""
        return self._version

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """把文本块写入向量库（已存在则覆盖）。"""
        if not chunks:
            # 空列表直接返回，避免 Chroma 对空批次报错。
            return

        # 先批量算向量，再一次性写入，减少网络/调用开销。
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedder.embed_documents(texts)

        # upsert：同一份文档重复入库时覆盖旧块，不会产生重复数据。
        # 注意：这里没有用 add，因为 add 在 id 冲突时可能报错。
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
        )
        # 数据变了，通知 BM25 索引下次检索时重建。
        self._version += 1

    def search(
        self, query: str, top_k: int = 5, source: str | None = None
    ) -> list[RetrievedChunk]:
        """按问题检索最相似的 top_k 个块。

        参数：
            query: 用户问题；
            top_k: 返回条数；
            source: 可选，只在这个文件里检索（metadata 过滤）。

        返回：
            RetrievedChunk 列表，按距离从近到远排列。
        """
        # 空库直接返回，避免 Chroma 在 n_results 大于 0 但库为空时报警告。
        if self._collection.count() == 0:
            return []

        # 查询向量必须和入库向量用同一个 embedding 模型。
        query_args = {
            "query_embeddings": [self._embedder.embed_query(query)],
            "n_results": top_k,
            # 指定返回哪些字段；ids 默认一定会返回。
            "include": ["documents", "metadatas", "distances"],
        }
        if source:
            # where 是 Chroma 的 metadata 过滤语法，在向量检索前执行。
            # 例如 {"source": "agent笔记.md"} 只在指定文件里检索。
            query_args["where"] = {"source": source}

        result = self._collection.query(**query_args)

        # Chroma 的返回值是「按查询分组」的二维列表：
        #   result["ids"]       -> [["chunk_0", "chunk_1"]]
        #   result["documents"] -> [["文本0", "文本1"]]
        # 一次只查一个问题，所以统一取 [0]。
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            RetrievedChunk(
                id=ids[i],
                text=documents[i],
                # metadata 可能为 None（旧数据），兜底成空字典。
                metadata=metadatas[i] or {},
                distance=float(distances[i]),
            )
            for i in range(len(ids))
        ]

    def get_all_chunks(self) -> list[Chunk]:
        """读取库内全部文本块（BM25 建索引用）。

        注意：这是一次性全量读取，适合小规模知识库。
        如果文档量很大，应该换成 Elasticsearch/BM25 服务或分片索引，
        否则内存和启动时间都会成为瓶颈。
        """
        data = self._collection.get(include=["documents", "metadatas"])
        return [
            Chunk(
                id=chunk_id,
                text=document or "",
                metadata=metadata or {},
            )
            for chunk_id, document, metadata in zip(
                data["ids"], data["documents"], data["metadatas"]
            )
        ]

    def list_sources(self) -> list[str]:
        """返回库里出现过的所有来源文件名（去重、排序）。"""
        # include=["metadatas"] 只取 metadata，不取文本和向量，减少传输量。
        data = self._collection.get(include=["metadatas"])
        sources = {
            metadata.get("source")
            for metadata in data["metadatas"]
            if metadata and metadata.get("source")
        }
        return sorted(sources)

    def list_document_stats(self) -> list[dict]:
        """按来源聚合块数、页码和字符数。

        一次读取全量 documents + metadatas，适合小规模知识库；
        大语料应换成 SQL 聚合或专门的统计表，避免每次全量扫描。
        """
        data = self._collection.get(include=["documents", "metadatas"])
        groups: dict[str, dict] = {}

        for document, metadata in zip(data["documents"], data["metadatas"]):
            metadata = metadata or {}
            source = metadata.get("source", "未知")
            text = document or ""

            entry = groups.setdefault(
                source,
                {
                    "source": source,
                    "chunks": 0,
                    "pages": set(),
                    "chars": 0,
                    "min_chunk_chars": None,
                    "max_chunk_chars": None,
                },
            )
            entry["chunks"] += 1
            entry["chars"] += len(text)
            length = len(text)

            page = metadata.get("page")
            if page is not None:
                entry["pages"].add(int(page))
            if entry["min_chunk_chars"] is None or length < entry["min_chunk_chars"]:
                entry["min_chunk_chars"] = length
            if entry["max_chunk_chars"] is None or length > entry["max_chunk_chars"]:
                entry["max_chunk_chars"] = length

        result: list[dict] = []
        for entry in sorted(groups.values(), key=lambda item: item["source"]):
            # set 不能直接序列化成 JSON，转成排序后的列表。
            result.append({**entry, "pages": sorted(entry["pages"])})
        return result

    def get_document_stats(self, source: str) -> dict | None:
        """返回单篇文档的统计信息；文档不存在时返回 None。"""
        for item in self.list_document_stats():
            if item["source"] == source:
                return item
        return None

    def delete_source(self, source: str) -> None:
        """按来源文件名删除该文档的所有块。"""
        # where 删除：Chroma 会删除所有 metadata.source == source 的记录。
        self._collection.delete(where={"source": source})
        # 数据变了，通知 BM25 索引下次检索时重建。
        self._version += 1

    def count(self) -> int:
        """返回库内文本块总数（用于健康检查和入库结果展示）。"""
        return self._collection.count()
