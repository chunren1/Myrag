"""
检索器 (Retriever)
===================
两层检索架构：
  1. 粗排: 调用 Qdrant 向量检索，每子问题召回 Top K
  2. 精排: 调用 bge-reranker-v2-m3 重排序，合并去重后返回 Top N
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BACKEND_DIR: Path = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.siliconflow import SiliconFlowClient, EMBEDDING_MODEL
from clients.qdrant_client import QdrantClientWrapper
import logging

logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# 检索配置
# ============================================================
DEFAULT_RETRIEVAL_TOP_K: int = 20  # 粗排每子问题召回数
DEFAULT_RERANK_TOP_K: int = 5       # 精排最终返回数


class Retriever:
    """
    多阶段检索器: 粗排 (向量检索) -> 精排 (Rerank 模型)。

    流程:
    1. 对每个子问题做 Embedding 向量化。
    2. 在 Qdrant 中检索 Top K 个相似文档 (粗排)。
    3. 合并所有子问题的检索结果，去重。
    4. 调用 bge-reranker-v2-m3 对合并结果重排序 (精排)。
    5. 返回 Top N 个最相关文档。
    """

    def __init__(
        self,
        sf_client: SiliconFlowClient,
        qdrant: QdrantClientWrapper,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> None:
        """
        初始化检索器。

        Args:
            sf_client: 硅基流动 API 客户端 (用于 Embedding + Rerank)。
            qdrant: Qdrant 客户端封装。
            retrieval_top_k: 粗排时每个子问题召回的文档数。
            rerank_top_k: 精排后最终返回的文档数。
        """
        self.sf_client: SiliconFlowClient = sf_client
        self.qdrant: QdrantClientWrapper = qdrant
        self.retrieval_top_k: int = retrieval_top_k
        self.rerank_top_k: int = rerank_top_k

    # ----------------------------------------------------------
    # 主检索方法
    # ----------------------------------------------------------
    async def retrieve(
        self,
        sub_queries: List[str],
        return_raw: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        执行多阶段检索。

        Args:
            sub_queries: 拆解后的子问题列表。
            return_raw: 是否返回原始的粗排结果（跳过精排），调试用。

        Returns:
            精排后的文档列表，每个元素包含:
            {
                "doc_id": str (chunk_id),
                "content": str,
                "score": float (relevance_score),
                "metadata": dict (title_path, source, h1, h2...)
            }
        """
        if not sub_queries:
            logger.warning("子问题列表为空，无法检索。")
            return []

        logger.info(f"开始检索: {len(sub_queries)} 个子问题, "
                     f"粗排 Top {self.retrieval_top_k}, 精排 Top {self.rerank_top_k}")

        # ------------------------------------------------------
        # 阶段 1: 粗排 - 向量检索
        # ------------------------------------------------------
        all_candidates: Dict[str, Dict[str, Any]] = {}

        for idx, sub_query in enumerate(sub_queries):
            logger.info(f"检索子问题 [{idx + 1}/{len(sub_queries)}]: {sub_query[:60]}...")

            # 1a. 将子问题转为向量
            embeddings: List[List[float]] = await self.sf_client.create_embeddings(
                texts=[sub_query],
                model=EMBEDDING_MODEL,
            )
            query_vector: List[float] = embeddings[0]

            # 1b. Qdrant 向量检索
            results: List[Dict[str, Any]] = self.qdrant.search(
                query_vector=query_vector,
                limit=self.retrieval_top_k,
            )

            logger.info(f"  粗排召回 {len(results)} 条结果")

            # 1c. 合并去重 (以 chunk_id 为 key，保留最高分)
            for r in results:
                doc_id: str = str(r["id"])
                score: float = r["score"]

                if doc_id not in all_candidates or score < all_candidates[doc_id]["score"]:
                    all_candidates[doc_id] = {
                        "doc_id": doc_id,
                        "content": r["payload"].get("content", ""),
                        "score": score,
                        "metadata": {
                            "title_path": r["payload"].get("title_path", ""),
                            "source": r["payload"].get("source", ""),
                            "h1": r["payload"].get("h1", ""),
                            "h2": r["payload"].get("h2", ""),
                            "h3": r["payload"].get("h3", ""),
                            "chunk_index": r["payload"].get("chunk_index", 0),
                        },
                    }

        candidates: List[Dict[str, Any]] = list(all_candidates.values())
        logger.info(f"粗排合并去重后: {len(candidates)} 个候选文档")

        if not candidates:
            logger.warning("粗排未召回任何文档！")
            return []

        # 如果直接返回原始结果 (调试模式或候选数 <= rerank_top_k)
        if return_raw or len(candidates) <= self.rerank_top_k:
            candidates.sort(key=lambda x: x["score"])
            return candidates[:self.rerank_top_k]

        # ------------------------------------------------------
        # 阶段 2: 精排 - bge-reranker-v2-m3
        # ------------------------------------------------------
        return await self._rerank(
            query=sub_queries[0],  # 用第一个子问题作为主查询
            candidates=candidates,
        )

    # ----------------------------------------------------------
    # 精简检索 (单查询，跳过子问题拆解)
    # ----------------------------------------------------------
    async def retrieve_simple(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        单查询快速检索 (用于补充检索等场景)。

        Args:
            query: 查询文本。
            top_k: 返回文档数。

        Returns:
            精排后的文档列表。
        """
        return await self.retrieve(
            sub_queries=[query],
            return_raw=(top_k <= self.rerank_top_k),
        )

    # ----------------------------------------------------------
    # 精排方法 (内部)
    # ----------------------------------------------------------
    async def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        调用 bge-reranker-v2-m3 对候选文档进行精排。

        Args:
            query: 查询文本 (用第一个子问题)。
            candidates: 粗排候选文档列表。

        Returns:
            精排 Top N 文档列表。
        """
        logger.info(f"开始精排: {len(candidates)} 个候选文档 -> Top {self.rerank_top_k}")

        # 准备文档列表
        documents: List[str] = [c["content"] for c in candidates]

        try:
            # 调用 Rerank API
            rerank_results: List[Dict[str, Any]] = await self.sf_client.rerank(
                query=query,
                documents=documents,
                top_n=self.rerank_top_k,
            )

            # 构建结果
            ranked: List[Dict[str, Any]] = []
            for r in rerank_results:
                idx: int = r["index"]
                candidate = candidates[idx]
                ranked.append({
                    "doc_id": candidate["doc_id"],
                    "content": candidate["content"],
                    "score": r["relevance_score"],
                    "metadata": candidate["metadata"],
                })

            logger.info(
                f"精排完成: {len(ranked)} 条结果, "
                f"最高分: {ranked[0]['score']:.4f}" if ranked else "精排完成: 0 条结果"
            )

            return ranked

        except Exception as e:
            logger.warning(f"精排失败 ({e})，回退到粗排结果。")
            # 回退：按粗排分数降序取 Top N
            candidates.sort(key=lambda x: x["score"])
            return candidates[:self.rerank_top_k]
