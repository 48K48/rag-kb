"""
批量入库 CLI：python scripts/ingest.py [文档目录]

不传目录时默认处理 data/raw。
适合：
    - 第一次批量导入自己的文档；
    - 改了 CHUNK_SIZE / CHUNK_OVERLAP 后重新入库做对比实验。

为什么脚本里要 sys.path.insert：
    直接运行 python scripts/ingest.py 时，Python 只会把 scripts/ 加入 sys.path，
    项目根目录不在其中，from app.xxx 会失败。
    把项目根目录插到最前面，就能像在根目录运行一样导入 app 包。
"""

import sys
from pathlib import Path

# Windows 终端默认 GBK，中文输出会乱码；统一改成 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 把项目根目录加入模块搜索路径（必须在 import app.* 之前执行）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.deps import get_store  # noqa: E402
from app.rag.loader import SUPPORTED_SUFFIXES, load_file  # noqa: E402
from app.rag.splitter import split_documents  # noqa: E402


def main() -> None:
    """扫描目录、加载文档、切分并写入向量库。"""
    # 支持命令行传入目录；不传则用配置里的 data/raw。
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.raw_dir
    if not raw_dir.exists():
        raise SystemExit(f"目录不存在：{raw_dir}")

    # rglob("*") 递归查找子目录；只保留白名单扩展名。
    files = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise SystemExit(f"目录里没有支持的文档：{raw_dir}")

    # 逐个文件加载；PDF 会按页展开成多条 LoadedDoc。
    docs = []
    for path in files:
        docs.extend(load_file(path))

    # 切分后写入 Chroma。
    # 重复执行时 chunk id 不变，upsert 会覆盖旧块，所以是幂等的。
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    store = get_store()
    store.add_chunks(chunks)

    # 打印结果，方便确认入库是否符合预期。
    print(f"文件数：{len(files)}，切块数：{len(chunks)}，库内总数：{store.count()}")
    for source in store.list_sources():
        print(f"  - {source}")


if __name__ == "__main__":
    main()
