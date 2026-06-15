"""
🌟 Agent 核心编排器
===================
完整的 Agentic RAG 闭环:
  Planner (问题拆解) -> Retriever (多阶段检索) -> Reflector (反思) -> Generator (最终生成)

流程:
  1. 接收用户原始问题
  2. Planner: 拆解为 2-3 个子问题
  3. Retriever: 逐子问题检索 + 合并 + Rerank 精排
  4. Reflector: 评估上下文充分性，必要时补充检索（最多 1 轮）
  5. Generator: 基于最终上下文生成引用标注的深度报告
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

BACKEND_DIR: Path = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.siliconflow import SiliconFlowClient, GENERATOR_MODEL
from clients.qdrant_client import QdrantClientWrapper
from core.query_decomposer import decompose_query
from core.retriever import Retriever
from core.context_reflector import ContextReflector
from core.prompt_builder import build_generator_prompt
import logging

logger: logging.Logger = logging.getLogger(__name__)


class AgenticRAG:
    """
    Agentic RAG 完整编排器。

    使用示例:
        sf_client = SiliconFlowClient()
        qdrant = QdrantClientWrapper()
        agent = AgenticRAG(sf_client, qdrant)

        # 非流式
        answer = await agent.run("Transformer 相比 RNN 有哪些优势？")

        # 流式
        async for chunk in agent.run_stream("什么是注意力机制？"):
            yield chunk
    """

    def __init__(
        self,
        sf_client: SiliconFlowClient,
        qdrant: QdrantClientWrapper,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
        max_reflection_rounds: int = 1,
    ) -> None:
        """
        初始化 Agent。

        Args:
            sf_client: 硅基流动 API 客户端。
            qdrant: Qdrant 客户端封装。
            retrieval_top_k: 粗排每子问题召回数。
            rerank_top_k: 精排最终返回数。
            max_reflection_rounds: 最大反思轮次。
        """
        self.sf_client: SiliconFlowClient = sf_client
        self.qdrant: QdrantClientWrapper = qdrant

        # 链路追踪日志文件
        self._trace_file: Path = BACKEND_DIR.parent / "workspace" / "logs" / "agent_trace.jsonl"
        self._trace_file.parent.mkdir(parents=True, exist_ok=True)

        # 初始化各模块
        self.retriever: Retriever = Retriever(
            sf_client=sf_client,
            qdrant=qdrant,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
        )
        self.reflector: ContextReflector = ContextReflector(
            sf_client=sf_client,
            max_reflection_rounds=max_reflection_rounds,
        )

    def _trace(self, step: str, data: Dict[str, Any]) -> None:
        """写入一条链路追踪记录到 agent_trace.jsonl。"""
        try:
            record: Dict[str, Any] = {
                "ts": datetime.now().isoformat(),
                "step": step,
                **data,
            }
            with open(self._trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 追踪失败不影响主流程

    # ----------------------------------------------------------
    # 非流式运行 (返回完整文本)
    # ----------------------------------------------------------
    async def run(self, query: str) -> Dict[str, Any]:
        """
        执行完整的 Agentic RAG 流程，返回完整答案。

        Args:
            query: 用户原始问题。

        Returns:
            {
                "answer": str (完整回答),
                "sub_queries": List[str] (拆解后的子问题),
                "retrieved_docs": List[Dict] (最终使用的文档),
                "reflection_logs": List[Dict] (反思日志),
            }
        """
        trace_id: str = datetime.now().strftime("%Y%m%d%H%M%S")
        logger.info(f"Agent 启动 [{trace_id}], 查询: '{query[:80]}...'")
        self._trace("start", {"trace_id": trace_id, "query": query})

        # ------------------------------------------------------
        # 步骤 1: Planner - 问题拆解
        # ------------------------------------------------------
        sub_queries: List[str] = await decompose_query(
            client=self.sf_client,
            query=query,
        )
        logger.info(f"步骤 1/4 完成: 拆解为 {len(sub_queries)} 个子问题")
        self._trace("planner", {"sub_queries": sub_queries})

        # ------------------------------------------------------
        # 步骤 2: Retriever - 多阶段检索
        # ------------------------------------------------------
        retrieved_docs: List[Dict[str, Any]] = await self.retriever.retrieve(
            sub_queries=sub_queries,
        )
        logger.info(f"步骤 2/4 完成: 检索到 {len(retrieved_docs)} 篇文档")
        self._trace("retriever", {
            "doc_count": len(retrieved_docs),
            "doc_ids": [d["doc_id"] for d in retrieved_docs],
            "top_scores": [round(d["score"], 4) for d in retrieved_docs[:5]],
        })

        # ------------------------------------------------------
        # 步骤 3: Reflector - 上下文反思
        # ------------------------------------------------------
        reflection_result: Dict[str, Any] = await self.reflector.reflect(
            original_query=query,
            retrieved_docs=retrieved_docs,
            retriever=self.retriever,
        )
        final_docs: List[Dict[str, Any]] = reflection_result["docs"]
        reflection_logs: List[Dict[str, Any]] = reflection_result["reflection_logs"]
        logger.info(
            f"步骤 3/4 完成: 反思 {reflection_result['reflection_count']} 次, "
            f"最终 {len(final_docs)} 篇文档"
        )
        self._trace("reflector", {
            "reflection_count": reflection_result["reflection_count"],
            "is_sufficient": reflection_result.get("is_sufficient"),
            "logs": reflection_logs,
        })

        # ------------------------------------------------------
        # 步骤 4: Generator - 最终生成
        # ------------------------------------------------------
        prompts: Dict[str, str] = build_generator_prompt(
            query=query,
            retrieved_docs=final_docs,
            reflection_logs=reflection_logs,
        )

        response: str = await self.sf_client.call_llm(
            model=GENERATOR_MODEL,
            system_prompt=prompts["system"],
            user_prompt=prompts["user"],
            temperature=0.7,
            top_p=0.9,
            max_tokens=8192,  # Qwen3-8B 支持长上下文，充分输出详尽报告
        )

        logger.info(f"步骤 4/4 完成: 生成长度 {len(response)} 字符的答案")
        self._trace("generator", {
            "answer_length": len(response),
            "final_doc_count": len(final_docs),
        })

        return {
            "answer": response,
            "sub_queries": sub_queries,
            "retrieved_docs": final_docs,
            "reflection_logs": reflection_logs,
        }

    # ----------------------------------------------------------
    # 流式运行 (返回异步生成器)
    # ----------------------------------------------------------
    async def run_stream(self, query: str) -> AsyncIterator[str]:
        """
        执行 Agentic RAG 流程，通过异步生成器流式返回答案。

        流程与非流式相同，只有步骤 4 使用 call_llm_stream 逐块输出。

        Args:
            query: 用户原始问题。

        Yields:
            Generator 输出的文本块。
        """
        logger.info(f"Agent (流式) 启动, 查询: '{query[:80]}...'")

        # 步骤 1: Planner
        sub_queries: List[str] = await decompose_query(
            client=self.sf_client,
            query=query,
        )

        # 步骤 2: Retriever
        retrieved_docs: List[Dict[str, Any]] = await self.retriever.retrieve(
            sub_queries=sub_queries,
        )

        # 步骤 3: Reflector
        reflection_result: Dict[str, Any] = await self.reflector.reflect(
            original_query=query,
            retrieved_docs=retrieved_docs,
            retriever=self.retriever,
        )
        final_docs: List[Dict[str, Any]] = reflection_result["docs"]
        reflection_logs: List[Dict[str, Any]] = reflection_result["reflection_logs"]

        # 步骤 4: Generator (流式)
        prompts: Dict[str, str] = build_generator_prompt(
            query=query,
            retrieved_docs=final_docs,
            reflection_logs=reflection_logs,
        )

        async for chunk in self.sf_client.call_llm_stream(
            model=GENERATOR_MODEL,
            system_prompt=prompts["system"],
            user_prompt=prompts["user"],
            temperature=0.7,
            top_p=0.9,
            max_tokens=8192,  # Qwen3-8B 支持长上下文，充分输出详尽报告
        ):
            yield chunk

    # ----------------------------------------------------------
    # 简化版: 不拆解问题，直接检索 + 生成
    # ----------------------------------------------------------
    async def run_simple(self, query: str) -> Dict[str, Any]:
        """
        简化流程: 跳过问题拆解和反思，直接检索 + 生成。
        用于简单查询或性能敏感场景。

        Args:
            query: 用户问题。

        Returns:
            {"answer": str, "retrieved_docs": List[Dict]}
        """
        docs: List[Dict[str, Any]] = await self.retriever.retrieve_simple(
            query=query,
            top_k=5,
        )

        prompts: Dict[str, str] = build_generator_prompt(
            query=query,
            retrieved_docs=docs,
        )

        response: str = await self.sf_client.call_llm(
            model=GENERATOR_MODEL,
            system_prompt=prompts["system"],
            user_prompt=prompts["user"],
            temperature=0.7,
            top_p=0.9,
            max_tokens=8192,  # Qwen3-8B 支持长上下文，充分输出详尽报告
        )

        return {
            "answer": response,
            "sub_queries": [query],
            "retrieved_docs": docs,
            "reflection_logs": [],
        }
