"""
RAG 内部数据结构。

为什么用 dataclass 而不是 Pydantic：
    - 这些对象只在 Python 内部流转（加载 -> 切分 -> 存储 -> 检索），
      不需要序列化成 JSON，也不需要复杂的校验；
    - dataclass 更轻量，创建成本低；
    - 对外的 API 模型放在 app/models.py，职责分离。
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """一个待入库的文本块。"""

    id: str
    # id 由「来源文件名:页码:块序号」组成。
    # 重复入库时用 upsert，同一个 id 会覆盖旧数据，避免重复。
    text: str
    # metadata 会原样存进 Chroma，支持按 source/page 过滤。
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """检索返回的一个文本块（比 Chunk 多一个相似度距离）。"""

    id: str
    text: str
    metadata: dict
    # cosine 距离：0 表示完全一致，越大越不相关。
    # 注意不同 embedding 模型的分数尺度不同，阈值必须用评测集校准。
    distance: float
