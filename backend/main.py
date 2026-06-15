"""
FastAPI 启动入口
================
配置 FastAPI 实例，开启 CORS，挂载路由，初始化 Agent。
启动: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 将 backend 目录加入 Python 路径
BACKEND_DIR: Path = Path(__file__).parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.siliconflow import SiliconFlowClient
from clients.qdrant_client import QdrantClientWrapper
from core.agent_flow import AgenticRAG
from api.chat import router as chat_router, set_agent

import logging

# ============================================================
# 日志配置
# ============================================================
log_level: str = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# 加载环境变量
# ============================================================
load_dotenv()

# ============================================================
# 全局实例
# ============================================================
sf_client: SiliconFlowClient = None  # type: ignore
qdrant: QdrantClientWrapper = None  # type: ignore
agent: AgenticRAG = None  # type: ignore


# ============================================================
# 应用生命周期管理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI 应用生命周期管理器。
    在启动时初始化所有组件，关闭时清理资源。
    """
    global sf_client, qdrant, agent

    logger.info("=" * 60)
    logger.info("高品質 RAG 知識庫系统 启动中...")
    logger.info("=" * 60)

    # 检查环境变量
    api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key or api_key == "sk-your-api-key-here":
        logger.error(
            "未配置有效的 SILICONFLOW_API_KEY！"
            "请在 .env 文件中设置你的 API Key。"
            "获取地址: https://cloud.siliconflow.cn/account/ak"
        )
        raise RuntimeError("缺少 SILICONFLOW_API_KEY 环境变量")

    try:
        # 初始化硅基流动 API 客户端
        logger.info("初始化硅基流动 API 客户端...")
        sf_client = SiliconFlowClient(api_key=api_key)
        logger.info("硅基流动 API 客户端初始化成功。")

        # 初始化 Qdrant 客户端 (容错：Qdrant 未就绪时仅警告)
        logger.info("初始化 Qdrant 向量数据库连接...")
        try:
            qdrant = QdrantClientWrapper(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                collection_name=os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base"),
            )
            info = qdrant.get_collection_info()
            logger.info(
                f"Qdrant 连接成功。集合: {info['name']}, "
                f"向量数: {info.get('vectors_count', 'N/A')}"
            )
        except Exception as qdrant_err:
            logger.warning(
                f"Qdrant 连接失败: {qdrant_err}。"
                f"请运行 'docker compose up -d' 启动 Qdrant。"
                f"服务将以降级模式运行（/api/chat 暂时不可用）。"
            )
            qdrant = None  # 标记为未连接

        # 初始化 Agent（仅当 Qdrant 可用时）
        if qdrant is not None:
            logger.info("初始化 Agentic RAG 引擎...")
            agent = AgenticRAG(
                sf_client=sf_client,
                qdrant=qdrant,
                retrieval_top_k=int(os.getenv("MAX_RETRIEVAL_TOP_K", "20")),
                rerank_top_k=int(os.getenv("MAX_RERANK_TOP_K", "5")),
                max_reflection_rounds=int(os.getenv("MAX_REFLECTION_ROUNDS", "1")),
            )
            set_agent(agent)
            logger.info("Agentic RAG 引擎初始化成功。")
        else:
            logger.warning("跳过 Agent 初始化（Qdrant 不可用）。")

    except Exception as e:
        logger.error(f"启动失败: {e}", exc_info=True)
        raise

    logger.info("=" * 60)
    logger.info("  ✅ 系统就绪！")
    logger.info(f"  🌐 聊天页面  -> http://localhost:8000")
    logger.info(f"  📖 API 文档   -> http://localhost:8000/docs")
    logger.info("=" * 60)

    yield  # 应用运行中

    # 关闭时清理
    logger.info("系统关闭，清理资源...")
    if qdrant:
        qdrant.close()
    logger.info("资源清理完成。")


# ============================================================
# FastAPI 应用实例
# ============================================================
app: FastAPI = FastAPI(
    title="高品質 RAG 知識庫系統",
    description="""
一个基于 Agentic RAG 模式的知识库问答系统。

## 核心特性
- **Agentic RAG**: Planner -> Retriever -> Reflector -> Generator 四阶段闭环
- **零成本 API**: 基于硅基流动免费 API
- **流式响应**: 支持 SSE 实时输出
- **智能推理**: <thinking> 标签内草稿分析，正文引用标注

## 使用方式
```bash
# 启动服务
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# API 文档
http://localhost:8000/docs

# 对话请求
curl -X POST http://localhost:8000/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{"query": "什么是 RAG？", "stream": true}'
```
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ============================================================
# CORS 配置
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Accel-Buffering"],
)

# ============================================================
# 挂载路由
# ============================================================
app.include_router(chat_router)


# ============================================================
# 静态文件 (前端页面)
# ============================================================
WEB_FRONTEND_DIR: Path = BACKEND_DIR.parent / "web_frontend"
if WEB_FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_FRONTEND_DIR)), name="static")


# ============================================================
# 根路由 -> 返回前端页面
# ============================================================
@app.get("/")
async def root() -> FileResponse:
    """根路由，返回 RAG 聊天前端页面。"""
    index_path = WEB_FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return FileResponse(str(index_path))  # will 404 if not exist


# ============================================================
# 直接运行入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=log_level.lower(),
    )
