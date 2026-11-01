"""RAG 知识库问答系统。

分层结构：
    app/api/     接口层：处理 HTTP 请求/响应、SSE
    app/rag/     核心层：加载、切分、向量化、存储、检索、生成
    app/deps.py  装配层：创建单例，供 FastAPI 依赖注入
    app/config.py 配置层：环境变量与默认值
"""
