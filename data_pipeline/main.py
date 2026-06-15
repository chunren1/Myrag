"""
离线数据管道 - 执行入口
========================
从原始 Markdown 文件到向量化入库的完整流水线。

流程:
  1. Markdown 智能切分
  2. Doc2Query LLM 知识增强
  3. bge-m3 向量化
  4. 数据写入 Qdrant

用法:
  python -m data_pipeline.main --dir ./workspace/raw_docs
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List

# ⚠️ 必须在导入任何依赖环境变量的模块之前加载 .env！
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=str(_env_path))

# 将 data_pipeline 和 backend 加入 Python 路径
PIPELINE_DIR: Path = Path(__file__).parent
BACKEND_DIR: Path = Path(__file__).parent.parent / "backend"
for d in (str(PIPELINE_DIR), str(BACKEND_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

from clients.siliconflow import SiliconFlowClient
from clients.qdrant_client import QdrantClientWrapper
from markdown_splitter import MarkdownSplitter
from llm_enrichment import enrich_chunks
from embedder import embed_chunks
from qdrant_uploader import upload_chunks_to_qdrant

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
# 主流水线
# ============================================================
async def run_pipeline(
    docs_dir: str,
    chunk_size: int = 2048,
    chunk_overlap: int = 256,
    query_count: int = 3,
    embed_batch_size: int = 20,
    upload_batch_size: int = 100,
) -> int:
    """
    执行完整的离线数据管道。

    Args:
        docs_dir: 原始 Markdown 文件目录。
        chunk_size: 每个 chunk 的最大字符数。
        chunk_overlap: chunk 之间的重叠字符数。
        query_count: 每个 chunk 生成 Doc2Query 的问题数。
        embed_batch_size: Embedding 批次大小。
        upload_batch_size: Qdrant 写入批次大小。

    Returns:
        成功写入的 chunk 总数。
    """
    import hashlib
    import json

    logger.info("=" * 60)
    logger.info("离线数据管道启动")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # 阶段 0: 增量更新 — 文件 Hash 比对
    # ----------------------------------------------------------
    hash_file: Path = Path(docs_dir).parent / "file_hashes.json"
    old_hashes: Dict[str, str] = {}
    if hash_file.exists():
        old_hashes = json.loads(hash_file.read_text(encoding="utf-8"))

    new_hashes: Dict[str, str] = {}
    docs_path: Path = Path(docs_dir)
    for f in docs_path.glob("**/*.md"):
        content: str = f.read_text(encoding="utf-8")
        new_hashes[str(f.relative_to(docs_path))] = hashlib.md5(content.encode()).hexdigest()

    if old_hashes and old_hashes == new_hashes:
        logger.info("所有文档均未变化，跳过管道执行。")
        return 0

    if old_hashes:
        changed = [k for k in new_hashes if new_hashes[k] != old_hashes.get(k, "")]
        added = [k for k in new_hashes if k not in old_hashes]
        removed = [k for k in old_hashes if k not in new_hashes]
        if changed:
            logger.info(f"检测到 {len(changed)} 个文件变更: {changed}")
        if added:
            logger.info(f"检测到 {len(added)} 个新增文件: {added}")
        if removed:
            logger.info(f"检测到 {len(removed)} 个已删除文件: {removed}")

    # 保存最新 Hash
    hash_file.write_text(json.dumps(new_hashes, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----------------------------------------------------------
    # 阶段 1: 检查环境
    # ----------------------------------------------------------
    api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key or api_key == "sk-your-api-key-here":
        logger.error(
            "未配置有效的 SILICONFLOW_API_KEY！"
            "请在 .env 文件中设置你的 API Key。"
        )
        return 0

    # ----------------------------------------------------------
    # 阶段 2: 初始化客户端
    # ----------------------------------------------------------
    logger.info("初始化 API 客户端...")
    sf_client: SiliconFlowClient = SiliconFlowClient()
    qdrant: QdrantClientWrapper = QdrantClientWrapper()

    # ----------------------------------------------------------
    # 阶段 3: Markdown 切分
    # ----------------------------------------------------------
    logger.info("阶段 1/4: 智能切分 Markdown 文档...")
    splitter: MarkdownSplitter = MarkdownSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: List[Dict[str, Any]] = splitter.split_directory(docs_dir)

    if not chunks:
        logger.warning("没有切分出任何 chunk，管道终止。")
        return 0

    logger.info(f"切分完成: 共 {len(chunks)} 个 chunk。")

    # ----------------------------------------------------------
    # 阶段 3: Doc2Query 知识增强
    # ----------------------------------------------------------
    logger.info("阶段 2/4: LLM 知识增强 (Doc2Query)...")
    chunks = await enrich_chunks(
        client=sf_client,
        chunks=chunks,
        query_count=query_count,
    )

    # ----------------------------------------------------------
    # 阶段 4: bge-m3 向量化
    # ----------------------------------------------------------
    logger.info("阶段 3/4: bge-m3 向量化...")
    chunks = await embed_chunks(
        client=sf_client,
        chunks=chunks,
        embed_batch_size=embed_batch_size,
    )

    # ----------------------------------------------------------
    # 阶段 5: 写入 Qdrant
    # ----------------------------------------------------------
    logger.info("阶段 4/4: 写入 Qdrant 向量数据库...")
    count: int = await upload_chunks_to_qdrant(
        chunks=chunks,
        qdrant=qdrant,
        batch_size=upload_batch_size,
    )

    # ----------------------------------------------------------
    # 完成
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"离线数据管道执行完成！成功写入 {count} 个向量。")
    logger.info("=" * 60)

    return count


# ============================================================
# CLI 入口
# ============================================================
def main() -> None:
    """命令行入口函数。"""
    parser = argparse.ArgumentParser(
        description="高品質 RAG 知识库系统 - 离线数据管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m data_pipeline.main --dir ./workspace/raw_docs
  python -m data_pipeline.main --dir ./workspace/raw_docs --chunk-size 1024 --query-count 5
        """,
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./workspace/raw_docs",
        help="原始 Markdown 文件目录 (默认: ./workspace/raw_docs)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "2048")),
        help="每个 chunk 的最大字符数 (默认: 2048)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "256")),
        help="chunk 重叠字符数 (默认: 256)",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=int(os.getenv("DOC2QUERY_COUNT", "3")),
        help="每个 chunk 生成 Doc2Query 的问题数 (默认: 3)",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=20,
        help="Embedding 批次大小 (默认: 20)",
    )
    parser.add_argument(
        "--upload-batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "100")),
        help="Qdrant 写入批次大小 (默认: 100)",
    )

    args = parser.parse_args()

    # 运行异步管道
    count: int = asyncio.run(run_pipeline(
        docs_dir=args.dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        query_count=args.query_count,
        embed_batch_size=args.embed_batch_size,
        upload_batch_size=args.upload_batch_size,
    ))

    if count == 0:
        print("\n⚠️  管道未写入任何数据，请检查日志。")
    else:
        print(f"\n✅ 管道执行成功！共写入 {count} 个向量到 Qdrant。")


if __name__ == "__main__":
    main()
