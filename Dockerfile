# ============================================================
# RAG 知识库问答系统 · 容器镜像
#
# 构建：
#   docker build -t rag-kb .
#
# 运行（Ollama 跑在宿主机）：
#   docker run -d -p 8000:8000 ^
#     -e DEEPSEEK_API_KEY=你的key ^
#     -v %cd%/data/chroma:/app/data/chroma ^
#     --name rag-kb rag-kb
#
# Streamlit 前端使用同一个镜像，只覆盖启动命令：
#   docker run -d -p 8501:8501 ^
#     -e API_BASE_URL=http://host.docker.internal:8000 ^
#     --name rag-kb-ui rag-kb ^
#     streamlit run app/frontend/streamlit_app.py --server.address=0.0.0.0
#
# 说明：
#   - 镜像里不包含 Ollama，embedding 请求发往宿主机；
#   - data/chroma 挂载到宿主机，容器重建后向量库不丢；
#   - .dockerignore 已排除 .env、.venv、缓存等无关文件。
# ============================================================

# 用 slim 版本减小镜像体积；Python 3.13 与项目依赖兼容。
FROM python:3.13-slim

# PYTHONUNBUFFERED=1：日志立即输出，方便 docker logs 查看
# PYTHONUTF8=1 + LANG=C.UTF-8：保证中文文件名和中文输出正常
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    LANG=C.UTF-8

# 容器内的工作目录，后续 COPY / RUN / CMD 都相对于它
WORKDIR /app

# 先单独复制依赖文件：
# 依赖没变时可以复用 Docker 层缓存，不必每次重新 pip install。
COPY requirements.txt .

# 允许构建时指定 PyPI 源：国内网络可用清华/阿里云/华为云镜像加速。
# 用法：docker compose build --build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements.txt

# 复制应用代码、脚本、评测集和示例文档。
# data/chroma 不复制：它是运行时生成的向量库，应该用 volume 挂载。
COPY app ./app
COPY scripts ./scripts
COPY eval ./eval
COPY data/raw ./data/raw

# 预先创建向量库目录，避免首次启动时因目录不存在而报错
RUN mkdir -p /app/data/chroma

# Ollama 跑在宿主机：
#   - Windows / macOS 用 host.docker.internal
#   - Linux 需要在 docker run 时加 --add-host=host.docker.internal:host-gateway
# 如果 Ollama 在另一台机器，启动时用 -e OLLAMA_BASE_URL=... 覆盖。
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# 声明容器监听端口（只是文档作用，实际映射靠 docker run -p）
# 8000 = FastAPI，8501 = Streamlit
EXPOSE 8000 8501

# 必须监听 0.0.0.0，宿主机才能访问；
# 监听 127.0.0.1 的话只有容器内部能访问。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
