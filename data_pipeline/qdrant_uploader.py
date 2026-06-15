"""
Qdrant 数据入库模块
===================
将带有向量的 chunk 批量写入 Qdrant 向量数据库。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# 将 backend 加入 Python 路径
BACKEND_DIR: Path = Path(__file__).parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.qdrant_client import QdrantClientWrapper
import logging

logger: logging.Logger = logging.getLogger(__name__)

# 默认批次大小
DEFAULT_UPLOAD_BATCH_SIZE: int = 100


async def upload_chunks_to_qdrant(
    chunks: List[Dict[str, Any]],
    qdrant: QdrantClientWrapper,
    batch_size: int = DEFAULT_UPLOAD_BATCH_SIZE,
) -> int:
    """
    将带有向量的 chunk 批量写入 Qdrant。

    每个 chunk 的 payload 包含完整的元数据，方便后续检索过滤。

    Args:
        chunks: 带有 "chunk_id", "vector", "content", "metadata" 字段的 chunk 列表。
        qdrant: Qdrant 客户端封装。
        batch_size: 每批写入的大小，默认 100。

    Returns:
        成功写入的总数。
    """
    logger.info(f"开始写入 Qdrant，共 {len(chunks)} 个 chunk，批次大小: {batch_size}...")

    # 验证数据完整性
    valid_chunks: List[Dict[str, Any]] = []
    for c in chunks:
        if "vector" not in c or not c.get("vector"):
            logger.warning(f"Chunk '{c.get('chunk_id', 'unknown')}' 缺少向量，跳过。")
            continue
        valid_chunks.append(c)

    if not valid_chunks:
        logger.warning("没有有效的 chunk 可以写入！")
        return 0

    # 转换为 Qdrant 数据点格式
    points: List[Dict[str, Any]] = []
    for c in valid_chunks:
        points.append({
            "id": c["chunk_id"],
            "vector": c["vector"],
            "payload": {
                "content": c["content"],
                "source": c["metadata"].get("source", ""),
                "title_path": c["metadata"].get("title_path", ""),
                "h1": c["metadata"].get("h1", ""),
                "h2": c["metadata"].get("h2", ""),
                "h3": c["metadata"].get("h3", ""),
                "h4": c["metadata"].get("h4", ""),
                "chunk_index": c["metadata"].get("chunk_index", 0),
                "char_count": c["metadata"].get("char_count", 0),
                "predicted_queries": c["metadata"].get("predicted_queries", []),
            },
        })

    # 分批写入
    qdrant.upsert_batch(points=points, batch_size=batch_size)

    logger.info(f"Qdrant 写入完成，成功写入 {len(valid_chunks)} 个向量。")

    # 验证写入结果
    info = qdrant.get_collection_info()
    logger.info(f"集合状态: {info['points_count']} 个点, {info['vectors_count']} 个向量")

    return len(valid_chunks)
