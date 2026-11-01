"""RAG 核心模块。

数据流：
    loader -> splitter -> embedder -> store -> retriever -> chain
    加载      切分        向量化       存储      检索        生成
"""
