"""
Embedding 封装。

为什么单独包一层：
    1. 业务代码只依赖 embed_documents / embed_query 两个方法，
       不直接依赖 LangChain 或 Ollama 的具体实现；
    2. 测试时可以注入 FakeEmbedder（见 tests/fakes.py），
       不需要启动 Ollama，测试速度快且结果稳定；
    3. 以后换 embedding 服务（例如 DashScope、OpenAI），只改这个文件。

关键原则：
    文档入库和用户提问必须使用「同一个模型」和「同一套维度」，
    否则向量空间不一致，检索结果会完全错误。
"""

from langchain_ollama import OllamaEmbeddings


class OllamaEmbedder:
    """基于本地 Ollama 的 embedding 客户端。"""

    def __init__(self, model: str, base_url: str) -> None:
        # base_url 指向 Ollama 服务，默认 http://127.0.0.1:11434。
        # model 例如 qwen3-embedding:0.6b，输出 1024 维向量。
        self._client = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量把文档文本转成向量（入库时用）。

        返回 shape = [文本数, 向量维度]。
        Ollama 会按 batch 处理，比逐条调用快。
        """
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """把用户问题转成向量（查询时用）。

        必须和 embed_documents 使用同一个模型，
        否则问题向量和文档向量不在同一个语义空间。
        """
        return self._client.embed_query(text)
