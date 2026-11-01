"""
文档入库接口：上传 -> 保存 -> 加载 -> 切分 -> 写入 Chroma。

为什么要把上传的文件保存到 data/raw：
    1. 方便复现：出问题时可以直接看到原始文件；
    2. 方便重跑：改了 CHUNK_SIZE 后，不需要重新上传，直接跑 scripts/ingest.py；
    3. 方便排查：向量检索效果差时，先确认原始文档是否正确。

安全注意：
    - 只取 Path(filename).name，防止 ../../evil.md 这类路径穿越；
    - 限制文件大小，防止误传超大文件；
    - 只允许白名单扩展名。
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import settings
from app.deps import get_store
from app.models import DocumentListResponse, IngestResponse
from app.rag.loader import SUPPORTED_SUFFIXES, load_file
from app.rag.splitter import split_documents
from app.rag.store import ChromaStore

router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse, summary="上传文档并入库")
async def ingest(
    # File(...) 表示这是 multipart/form-data 的文件字段；
    # list[UploadFile] 支持一次上传多个文件。
    files: list[UploadFile] = File(..., description="支持 md/txt/pdf"),
    store: ChromaStore = Depends(get_store),
) -> IngestResponse:
    """上传一个或多个文档，切分后写入向量库。

    返回本次处理的文件数、切块数，以及入库后的总块数。
    """
    if not files:
        raise HTTPException(status_code=400, detail="没有收到文件")

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for upload in files:
        # Path(...).name 只保留文件名，去掉任何目录部分，防止路径穿越。
        filename = Path(upload.filename or "").name
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型：{filename}")

        # await upload.read()：UploadFile 是异步文件对象，必须 await。
        content = await upload.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"文件过大：{filename}")

        target = settings.raw_dir / filename
        target.write_bytes(content)
        saved.append(target)

    # 加载 + 切分 + 入库。
    # 注意：同一份文档重复上传时，chunk id 不变，upsert 会覆盖旧块。
    docs = []
    for path in saved:
        docs.extend(load_file(path))

    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    store.add_chunks(chunks)

    return IngestResponse(
        files=len(saved),
        chunks=len(chunks),
        collection_count=store.count(),
    )


@router.get("/documents", response_model=DocumentListResponse, summary="列出已入库文档")
def list_documents(store: ChromaStore = Depends(get_store)) -> DocumentListResponse:
    """返回库里出现过的所有来源文件名（去重）。"""
    sources = store.list_sources()
    return DocumentListResponse(count=len(sources), sources=sources)


@router.delete("/documents/{source}", summary="按来源删除文档")
def delete_document(source: str, store: ChromaStore = Depends(get_store)) -> dict:
    """删除指定来源的所有块。

    source 是文件名，例如 agent笔记.md。
    如果文件名里有特殊字符，前端需要做 URL 编码。
    """
    store.delete_source(source)
    return {"deleted": source, "collection_count": store.count()}
