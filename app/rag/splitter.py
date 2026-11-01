"""
文本切分：把长文档拆成适合检索的小块。

为什么用 RecursiveCharacterTextSplitter：
    它会按 separators 的顺序尝试切分：
        先按空行（段落）切 -> 再按换行 -> 再按中文句号/问号/感叹号
        -> 再按分号/逗号 -> 最后才按空格和字符切。
    这样能尽量保证「一个块是一个语义完整的片段」，
    而不是把一句话从中间截断。

关键参数：
    - chunk_size：每个块的最大字符数；
    - chunk_overlap：相邻块的重叠字符数，避免答案刚好被切在边界上。
    这两个参数没有万能值，必须用自己的评测集做对比实验。
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loader import LoadedDoc
from app.rag.types import Chunk


def split_documents(
    docs: list[LoadedDoc], chunk_size: int = 500, chunk_overlap: int = 100
) -> list[Chunk]:
    """把 LoadedDoc 列表切分成 Chunk 列表。

    参数：
        docs: load_file 返回的原始文档；
        chunk_size: 单块最大字符数；
        chunk_overlap: 相邻块重叠字符数，必须小于 chunk_size。

    返回：
        Chunk 列表，每个 Chunk 都带 source/page/chunk_index metadata。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 中文文档优先按段落和中文标点切；
        # 空字符串放在最后，保证任何文本都能被切开。
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        # 用字符数而不是 token 数：实现简单，和 MAX_CONTEXT_CHARS 的单位一致。
        length_function=len,
    )

    chunks: list[Chunk] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")

        for index, part in enumerate(splitter.split_text(doc.text)):
            text = part.strip()
            # 跳过空白块：切分器可能在段落之间产生空字符串。
            if not text:
                continue

            # id 的组成：来源 + 页码 + 块序号。
            # 同一份文档重复入库时，id 不变，upsert 会覆盖旧块，
            # 所以不会产生重复数据；文档更新后旧块也会被覆盖。
            chunk_id = f"{source}:{page or 0}:{index}"
            metadata = {**doc.metadata, "chunk_index": index}
            chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))

    return chunks
