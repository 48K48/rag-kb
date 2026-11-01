"""
文档加载：把磁盘上的文件读成「文本 + metadata」。

支持格式：
    - Markdown / TXT：整个文件作为一篇文档；
    - PDF：按页拆分，每一页作为一篇文档，并记录页码。

为什么 PDF 要按页拆：
    - pypdf 的 extract_text 是按页工作的；
    - 保留页码后，前端可以显示「第 N 页」，引用更可信；
    - 后续如果要做「按页过滤」，metadata 已经具备了。
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# 支持的扩展名统一放在这里，入库接口和 CLI 共用，避免两处写死不一致。
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class LoadedDoc:
    """加载后的一篇原始文档（还未切分）。"""

    text: str
    # metadata 至少包含 source（文件名）；PDF 还会包含 page（页码）。
    metadata: dict


def load_file(path: Path) -> list[LoadedDoc]:
    """加载单个文件，返回文档列表。

    参数：
        path: 文件路径（Path 对象）。

    返回：
        LoadedDoc 列表。Markdown/TXT 返回 1 条；PDF 每页返回 1 条。

    异常：
        文件类型不在 SUPPORTED_SUFFIXES 中时抛出 ValueError，
        由上层 API 转成 400 响应。
    """
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown", ".txt"}:
        # errors="ignore"：遇到无法解码的字符时跳过，避免一个坏文件导致整批入库失败。
        text = path.read_text(encoding="utf-8", errors="ignore")
        # 只保留文件名，不保留完整路径：
        # 1) metadata 更干净；2) 避免把本机路径暴露给前端。
        return [LoadedDoc(text=text, metadata={"source": path.name})]

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        docs: list[LoadedDoc] = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # 扫描版 PDF 没有文本层，extract_text 会返回空字符串；
            # 这里跳过空页，避免产生空 chunk。扫描件需要 OCR，暂不支持。
            if text.strip():
                docs.append(
                    LoadedDoc(
                        text=text,
                        metadata={"source": path.name, "page": page_no},
                    )
                )
        return docs

    raise ValueError(f"不支持的文件类型：{suffix}（支持 {sorted(SUPPORTED_SUFFIXES)}）")
