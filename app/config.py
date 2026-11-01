"""
配置中心：把「可调参数」和「代码」分开。

配置优先级（从高到低）：
    1. 系统环境变量（例如 Docker 的 -e、PowerShell 的 $env:XXX）
    2. 项目根目录的 .env 文件
    3. 下面 Settings 里的默认值

这样设计的好处：
    - 本地开发把密钥放在 .env，不会提交到 Git；
    - 做评测时可以用环境变量临时覆盖 CHUNK_SIZE / MAX_DISTANCE，
      不用改代码，方便做 A/B 对比；
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# BASE_DIR 指向项目根目录（app/config.py 的上两级）
# 所有默认路径都基于它，保证从任何工作目录启动都能找到文件。
BASE_DIR = Path(__file__).resolve().parent.parent

# load_dotenv 默认不会覆盖已经存在的系统环境变量，
# 所以「环境变量 > .env」的优先级是由它保证的。
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量；没配置时返回默认值。

    单独封装是为了让类型转换只写一次，
    并且配置写错时能在启动阶段立刻报错，而不是运行到一半才失败。
    """
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量；用法同 _env_int。"""
    return float(os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量。

    接受 1/true/yes/on（不区分大小写）为 True，
    其他值一律按 False 处理，避免 "False" 被 bool("False") 误判成 True。
    """
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """全局配置。

    frozen=True：配置创建后不允许修改，
    避免运行过程中某个模块偷偷改掉全局参数，导致行为不一致。
    """

    # ---------- 路径 ----------
    base_dir: Path = BASE_DIR
    raw_dir: Path = BASE_DIR / "data" / "raw"          # 原始文档目录
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma")))
    static_dir: Path = BASE_DIR / "app" / "static"     # 前端页面目录

    # ---------- 向量库 ----------
    # 集合名相当于关系库里的表名；Chroma 要求 3-512 个字符，
    # 只能用字母、数字、点、下划线、短横线，且首尾必须是字母或数字。
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_base")

    # ---------- 大模型（DeepSeek 兼容 OpenAI SDK） ----------
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    # deepseek-chat / deepseek-reasoner 已进入退役计划，默认改用 deepseek-flash。
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # ---------- Agent ----------
    # Agent 默认用 deepseek-flash；也可以单独配置成别的 OpenAI 兼容模型。
    # 空字符串也回退到 DeepSeek 配置，避免 .env 里留空导致启动失败。
    agent_model: str = os.getenv("AGENT_MODEL", "") or "deepseek-flash"
    agent_base_url: str = (
        os.getenv("AGENT_BASE_URL", "")
        or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    agent_api_key: str = os.getenv("AGENT_API_KEY", "") or os.getenv(
        "DEEPSEEK_API_KEY", ""
    )
    agent_temperature: float = _env_float("AGENT_TEMPERATURE", 0.1)
    # 限制单次回答的 token 数，避免 Agent 输出过长导致成本和延迟失控。
    agent_max_tokens: int = _env_int("AGENT_MAX_TOKENS", 2048)
    # 多轮对话只发送最近 N 轮（一轮 = 用户 + 助手两条消息）。
    agent_max_history_turns: int = _env_int("AGENT_MAX_HISTORY_TURNS", 6)
    # LangGraph 的最大递归步数；每次“模型 -> 工具 -> 模型”大约消耗 2 步。
    agent_recursion_limit: int = _env_int("AGENT_RECURSION_LIMIT", 12)

    # ---------- Agent 工具来源 ----------
    # local：只用进程内 LangChain 工具（改造前的行为）
    # mcp：  通过 FastMCP 服务加载工具，连接失败直接报错
    # auto： 先试 MCP，失败后回退 local（默认）
    agent_tool_source: str = (
        os.getenv("AGENT_TOOL_SOURCE", "auto").strip().lower()
    )
    mcp_server_name: str = os.getenv("MCP_SERVER_NAME", "rag-kb")
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")
    # 加载工具列表的超时（秒）；每次工具调用另有 HTTP 超时。
    mcp_connect_timeout: float = _env_float("MCP_CONNECT_TIMEOUT", 3.0)
    # Docker 启动时 MCP 可能还没就绪，首次加载失败后重试。
    mcp_connect_retries: int = _env_int("MCP_CONNECT_RETRIES", 2)
    mcp_retry_interval: float = _env_float("MCP_RETRY_INTERVAL", 1.0)
    # 单次 MCP 工具调用的 HTTP 超时；检索包含 embedding，不能设得太短。
    mcp_request_timeout: float = _env_float("MCP_REQUEST_TIMEOUT", 30.0)

    # ---------- MCP 服务自身 ----------
    # MCP 服务通过 KB_API_BASE_URL 调用 FastAPI 的只读接口，
    # 不直接打开 Chroma，避免多进程缓存导致数据不一致。
    mcp_host: str = os.getenv("MCP_HOST", "127.0.0.1")
    mcp_port: int = _env_int("MCP_PORT", 8001)
    kb_api_base_url: str = os.getenv("KB_API_BASE_URL", "http://127.0.0.1:8000")
    kb_api_timeout: float = _env_float("KB_API_TIMEOUT", 30.0)

    # ---------- 本地向量模型 ----------
    # Ollama 默认监听 11434；模型必须先 ollama pull 到本地。
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

    # ---------- 检索参数 ----------
    top_k: int = _env_int("TOP_K", 5)
    # cosine 距离越小越相似；超过这个值视为「不相关」，直接拒答。
    # 0.6 是当前语料下的经验值，换模型/换文档后必须用评测集重新校准。
    max_distance: float = _env_float("MAX_DISTANCE", 0.6)

    # ---------- 混合检索参数 ----------
    # False：纯向量（baseline，行为和加 BM25 之前一致）
    # True： 向量召回 + BM25 召回，用 RRF 融合
    hybrid_enabled: bool = _env_bool("HYBRID_ENABLED", False)
    # 两路各召回多少条，再融合；一般设为最终 top_k 的 2 倍左右。
    dense_top_k: int = _env_int("DENSE_TOP_K", 10)
    sparse_top_k: int = _env_int("SPARSE_TOP_K", 10)
    # RRF 的平滑常数，常用 60；越大越弱化“单路第一名”的优势。
    rrf_k: int = _env_int("RRF_K", 60)
    # BM25 参数：k1 控制词频饱和，b 控制文档长度归一化。
    bm25_k1: float = _env_float("BM25_K1", 1.5)
    bm25_b: float = _env_float("BM25_B", 0.75)

    # ---------- 切分参数 ----------
    # 块太大：检索不精准、浪费 token；块太小：语义不完整。
    # 这两个参数是第 1 个练习的变量。
    chunk_size: int = _env_int("CHUNK_SIZE", 500)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 100)

    # ---------- 生成参数 ----------
    # 限制送入模型的上下文字符数，防止 token 成本失控。
    max_context_chars: int = _env_int("MAX_CONTEXT_CHARS", 6000)
    # 单个上传文件大小上限（MB），避免误传超大文件拖垮服务。
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 20)

    def __post_init__(self) -> None:
        """启动阶段校验枚举值，避免运行到一半才发现配置写错。"""
        if self.agent_tool_source not in {"local", "mcp", "auto"}:
            raise ValueError(
                "AGENT_TOOL_SOURCE 只能是 local / mcp / auto，"
                f"当前值：{self.agent_tool_source}"
            )


# 模块级单例：其他模块直接 from app.config import settings 即可。
settings = Settings()
