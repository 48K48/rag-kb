"""测试包。

运行：在项目根目录执行 pytest -q
测试不依赖 Ollama / DeepSeek / 真实 Chroma 数据：
    - 切分测试用纯文本；
    - 存储测试用 FakeEmbedder + 临时目录；
    - 接口测试用 dependency_overrides 注入 FakeChain / FakeStore。
"""
