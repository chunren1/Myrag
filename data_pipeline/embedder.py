"""
向量化模块 (Embedder)
====================
调用 bge-m3 模型将 chunk 文本转为 1024 维向量。
支持分批处理，避免单次请求过大导致 429 限流。
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

# 将 backend 加入 Python 路径
BACKEND_DIR: Path = Path(__file__).parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.siliconflow import SiliconFlowClient, EMBEDDING_MODEL
import logging

logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# Embedding 配置
# ============================================================
DEFAULT_EMBED_BATCH_SIZE: int = 20  # 每次 Embedding API 调用处理的最大文本数
DEFAULT_MAX_CHARS_PER_TEXT: int = 8192  # 超过此长度的文本将被截断


async def embed_chunks(
    client: SiliconFlowClient,
    chunks: List[Dict[str, Any]],
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    为 chunks 批量生成 Embedding 向量。

    对于每个 chunk:
    1. 取其 content 字段作为输入文本（已被 llm_enrichment 增强过）。
    2. 按 batch_size 分批调用 Embedding API。
    3. 将向量存入 chunk["vector"]。

    Args:
        client: 硅基流动 API 客户端。
        chunks: 增强后的 chunk 列表。
        embed_batch_size: 每次 API 调用处理的文本数，默认 20。

    Returns:
        带有 vector 字段的 chunk 列表 (原地修改并返回)。
    """
    logger.info(f"开始向量化，共 {len(chunks)} 个 chunk，批次大小: {embed_batch_size}...")

    # 准备所有待向量化的文本 (截断过长内容)
    texts: List[str] = [
        c["content"][:DEFAULT_MAX_CHARS_PER_TEXT] for c in chunks
    ]

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), embed_batch_size):
        batch_texts: List[str] = texts[i: i + embed_batch_size]

        # 调用 Embedding API，内置 429 重试
        batch_embeddings: List[List[float]] = await client.create_embeddings(
            texts=batch_texts,
            model=EMBEDDING_MODEL,
        )
        all_embeddings.extend(batch_embeddings)

        logger.info(f"Embedding 进度: {min(i + embed_batch_size, len(texts))}/{len(texts)}")

        # 批次间小延迟，进一步降低限流风险
        if i + embed_batch_size < len(texts):
            await asyncio.sleep(0.1)

    # 将向量赋值给 chunk
    if len(all_embeddings) != len(chunks):
        logger.error(
            f"向量数量 ({len(all_embeddings)}) 与 chunk 数量 ({len(chunks)}) 不匹配！"
        )
        raise ValueError("Embedding 结果数量不匹配")

    for idx, chunk in enumerate(chunks):
        chunk["vector"] = all_embeddings[idx]

    logger.info(f"向量化完成，共生成 {len(all_embeddings)} 个 {len(all_embeddings[0])} 维向量")
    return chunks
