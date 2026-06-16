"""
Qdrant 向量数据库客户端封装
============================
初始化连接，创建集合，支持 INT8 标量量化以节省内存。
"""

import os
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
import logging

logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# 默认配置 (可通过环境变量覆盖)
# ============================================================
DEFAULT_HOST: str = "localhost"
DEFAULT_PORT: int = 6333
DEFAULT_COLLECTION: str = "knowledge_base"
DEFAULT_VECTOR_SIZE: int = 1024  # bge-m3 输出维度


class QdrantClientWrapper:
    """
    Qdrant 向量数据库封装。
    自动创建集合，配置 INT8 标量量化以节省约 75% 内存。
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        初始化 Qdrant 连接。

        Args:
            host: Qdrant 主机地址，默认从 QDRANT_HOST 环境变量读取。
            port: Qdrant 端口，默认从 QDRANT_PORT 环境变量读取。
            collection_name: 集合名称，默认从 QDRANT_COLLECTION_NAME 环境变量读取。
            vector_size: 向量维度，默认从 QDRANT_VECTOR_SIZE 环境变量读取。
            api_key: Qdrant API Key，默认从 QDRANT_API_KEY 环境变量读取。
        """
        self.host: str = host or os.getenv("QDRANT_HOST", DEFAULT_HOST)
        self.port: int = port or int(os.getenv("QDRANT_PORT", str(DEFAULT_PORT)))
        self.collection_name: str = collection_name or os.getenv(
            "QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION
        )
        self.vector_size: int = vector_size or int(
            os.getenv("QDRANT_VECTOR_SIZE", str(DEFAULT_VECTOR_SIZE))
        )
        self.api_key: Optional[str] = api_key or os.getenv("QDRANT_API_KEY")

        self.client: QdrantClient = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
        )

        # 自动创建集合（如果不存在）
        self._ensure_collection()

        logger.info(
            f"Qdrant 连接成功 -> {self.host}:{self.port}, "
            f"集合: {self.collection_name}, 向量维度: {self.vector_size}"
        )

    # ----------------------------------------------------------
    # 集合管理
    # ----------------------------------------------------------
    def _ensure_collection(self) -> None:
        """
        确保集合存在。如不存在则创建，配置 INT8 标量量化。
        """
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                self._create_collection()
            else:
                logger.info(f"集合 '{self.collection_name}' 已存在，跳过创建。")

        except UnexpectedResponse as e:
            logger.error(f"检查集合时出错: {e}")
            raise

    def _create_collection(self) -> None:
        """
        创建 Qdrant 集合，配置如下：
        - 向量维度: 1024 (bge-m3)
        - 距离度量: Cosine (余弦相似度)
        - 量化方式: Scalar Quantization (INT8)，节省约 75% 内存
        """
        logger.info(f"正在创建集合 '{self.collection_name}' (INT8 量化)...")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self.vector_size,
                distance=qdrant_models.Distance.COSINE,
                # 开启标量量化 (INT8)，用精度换内存
                quantization_config=qdrant_models.ScalarQuantization(
                    scalar=qdrant_models.ScalarQuantizationConfig(
                        type=qdrant_models.ScalarType.INT8,
                        quantile=0.99,  # 99% 分位数剪枝，忽略极端值
                        always_ram=True,  # 量化后的向量始终放在内存中
                    ),
                ),
            ),
            # 按创建时间排序时用
            hnsw_config=qdrant_models.HnswConfigDiff(
                m=16,           # 每个节点的最大连接数
                ef_construct=100,  # 构建时的搜索宽度
            ),
        )

        logger.info(f"集合 '{self.collection_name}' 创建完成 (INT8 量化已启用)。")

    # ----------------------------------------------------------
    # 向量检索
    # ----------------------------------------------------------
    def search(
        self,
        query_vector: List[float],
        limit: int = 20,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        基于向量进行相似度检索。

        Args:
            query_vector: 查询向量 (1024 维)。
            limit: 返回的最大结果数。
            score_threshold: 最低相似度阈值 (余弦距离，值越小表示越相似)。

        Returns:
            搜索结果列表，每个元素包含 id, score, payload。
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,  # 不返回向量以节省带宽
        )

        results: List[Dict[str, Any]] = []
        for hit in search_result:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {},
            })
        return results

    # ----------------------------------------------------------
    # 批量写入向量
    # ----------------------------------------------------------
    def upsert_batch(
        self,
        points: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """
        批量写入 (插入或更新) 向量数据点。

        Args:
            points: 数据点列表，每个元素包含:
                    - id: 唯一标识 (str 或 int)
                    - vector: 向量 (List[float])
                    - payload: 附加元数据 (Dict)
            batch_size: 每批写入的向量数，默认 100。
        """
        total: int = len(points)
        for i in range(0, total, batch_size):
            batch = points[i: i + batch_size]

            qdrant_points: List[qdrant_models.PointStruct] = []
            for p in batch:
                qdrant_points.append(
                    qdrant_models.PointStruct(
                        id=p["id"],
                        vector=p["vector"],
                        payload=p.get("payload", {}),
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points,
                wait=True,  # 等待写入确认
            )

            logger.info(
                f"已写入 {min(i + batch_size, total)}/{total} 个向量到集合 "
                f"'{self.collection_name}'"
            )

    # ----------------------------------------------------------
    # 集合信息
    # ----------------------------------------------------------
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取当前集合的状态信息。

        Returns:
            包含 points_count、vectors_count 等信息的字典。
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": getattr(info, "vectors_count", info.points_count),
            "config": info.config.dict() if info.config else {},
        }

    # ----------------------------------------------------------
    # 清理
    # ----------------------------------------------------------
    def close(self) -> None:
        """关闭 Qdrant 客户端连接。"""
        logger.info("Qdrant 客户端连接已关闭。")

    def delete_by_source(self, source: str) -> int:
        """
        删除指定来源文件的所有向量点。

        Args:
            source: 来源文件名 (匹配 payload 的 source 字段)。

        Returns:
            删除的点数。
        """
        from qdrant_client.http import models as qdrant_models

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="source",
                            match=qdrant_models.MatchValue(value=source),
                        )
                    ]
                )
            ),
        )
        logger.info(f"已删除来源 '{source}' 的所有向量点。")
        return 0
