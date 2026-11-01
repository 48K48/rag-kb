"""
测试用的假 embedding。

为什么需要它：
    - 真实 embedding 需要启动 Ollama、加载模型，测试会变慢且依赖外部服务；
    - FakeEmbedder 不联网、结果确定，适合验证「存储/检索/接口」逻辑，
      而不是验证 embedding 模型本身的质量。

实现方式：
    把每个字符的 Unicode 码映射到 16 维向量的一维，再做归一化。
    这不是一个语义模型，只保证：
        1. 相同文本得到相同向量；
        2. 向量是固定 16 维，Chroma 可以正常建集合；
        3. 测试不依赖任何外部服务。
"""


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（入库时用）。"""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """单条向量化（查询时用）。"""
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        """把文本映射成 16 维单位向量。"""
        vector = [0.0] * 16
        for index, char in enumerate(text):
            # 用字符码做简单哈希，保证同一字符总是落在同一维。
            vector[index % 16] += (ord(char) % 89 + 1) / 90

        # 归一化：避免向量长度影响 cosine 距离。
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]
