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
from typing import AsyncIterator, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BACKEND_DIR: Path = Path(__file__).parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.siliconflow import SiliconFlowClient
from clients.qdrant_client import QdrantClientWrapper
from core.agent_flow import AgenticRAG
from api.chat import router as chat_router

import logging

log_level: str = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("=" * 60)
    logger.info("高品質 RAG 知識庫系统 启动中...")
    logger.info("=" * 60)

    api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key or api_key == "sk-your-api-key-here":
        logger.error(
            "未配置有效的 SILICONFLOW_API_KEY！"
            "请在 .env 文件中设置你的 API Key。"
            "获取地址: https://cloud.siliconflow.cn/account/ak"
        )
        raise RuntimeError("缺少 SILICONFLOW_API_KEY 环境变量")

    try:
        logger.info("初始化硅基流动 API 客户端...")
        sf_client = SiliconFlowClient(api_key=api_key)
        app.state.sf_client = sf_client
        logger.info("硅基流动 API 客户端初始化成功。")

        logger.info("初始化 Qdrant 向量数据库连接...")
        try:
            qdrant = QdrantClientWrapper(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                collection_name=os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base"),
                api_key=os.getenv("QDRANT_API_KEY"),
            )
            info = qdrant.get_collection_info()
            logger.info(
                f"Qdrant 连接成功。集合: {info['name']}, "
                f"向量数: {info.get('vectors_count', 'N/A')}"
            )
            app.state.qdrant = qdrant
        except Exception as qdrant_err:
            logger.warning(
                f"Qdrant 连接失败: {qdrant_err}。"
                f"请运行 'docker compose up -d' 启动 Qdrant。"
                f"服务将以降级模式运行（/api/chat 暂时不可用）。"
            )
            app.state.qdrant = None

        if app.state.qdrant is not None:
            logger.info("初始化 Agentic RAG 引擎...")
            agent = AgenticRAG(
                sf_client=sf_client,
                qdrant=qdrant,
                retrieval_top_k=int(os.getenv("MAX_RETRIEVAL_TOP_K", "20")),
                rerank_top_k=int(os.getenv("MAX_RERANK_TOP_K", "5")),
                max_reflection_rounds=int(os.getenv("MAX_REFLECTION_ROUNDS", "1")),
            )
            app.state.agent = agent
            logger.info("Agentic RAG 引擎初始化成功。")
        else:
            app.state.agent = None
            logger.warning("跳过 Agent 初始化（Qdrant 不可用）。")

    except Exception as e:
        logger.error(f"启动失败: {e}", exc_info=True)
        raise

    logger.info("=" * 60)
    logger.info("  ✅ 系统就绪！")
    logger.info(f"  🌐 聊天页面  -> http://localhost:8000")
    logger.info(f"  📖 API 文档   -> http://localhost:8000/docs")
    logger.info("=" * 60)

    yield

    logger.info("系统关闭，清理资源...")
    qdrant = getattr(app.state, "qdrant", None)
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

cors_origins_str: str = os.getenv("CORS_ORIGINS", "*")
cors_origins: List[str] = [o.strip() for o in cors_origins_str.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
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
