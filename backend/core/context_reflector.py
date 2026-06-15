"""
上下文反思器 (Reflector)
========================
评估检索到的上下文是否足以回答问题。
如果不足，生成补充搜索词进行额外检索（最多 1 次反思）。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_DIR: Path = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.siliconflow import SiliconFlowClient, REFLECTOR_MODEL
import logging

logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# 反思 System Prompt
# ============================================================
REFLECTION_SYSTEM_PROMPT: str = """你是一个严格的检索质量评估专家。你的任务是评估检索到的文档是否能支撑准确回答用户问题。

## 评估标准
1. **充分性**: 检索到的文档是否包含回答问题的关键信息？
2. **完整性**: 是否需要补充其他角度或细节？
3. **可靠性**: 文档来源是否可信？内容是否准确？

## 输出格式
严格返回一个 JSON 对象:
```json
{
  "is_sufficient": true或false,
  "confidence": 0.0到1.0之间的分数,
  "reasoning": "你的判断依据（简短）",
  "supplementary_query": "如果不足，生成一个补充搜索关键词或问题；如果足够，设为空字符串"
}
```

请评估以下检索结果。"""


class ContextReflector:
    """
    上下文反思器。

    流程:
    1. 将检索结果和用户问题发给 Qwen LLM 评估。
    2. 如果 LLM 判断上下文不足：
       - 使用 LLM 生成的补充搜索词进行第 2 次检索。
       - 将新结果合并到原有上下文。
       - 最多反思 1 次（防止死循环和限流）。
    3. 返回最终上下文集合。
    """

    def __init__(
        self,
        sf_client: SiliconFlowClient,
        max_reflection_rounds: int = 1,
    ) -> None:
        """
        初始化反思器。

        Args:
            sf_client: 硅基流动 API 客户端。
            max_reflection_rounds: 最大反思轮次，默认 1。
        """
        self.sf_client: SiliconFlowClient = sf_client
        self.max_reflection_rounds: int = max_reflection_rounds

    # ----------------------------------------------------------
    # 主反思方法
    # ----------------------------------------------------------
    async def reflect(
        self,
        original_query: str,
        retrieved_docs: List[Dict[str, Any]],
        retriever: Any,  # Retriever 实例，用于补充检索
    ) -> Dict[str, Any]:
        """
        反思检索上下文是否充分，必要时触发补充检索。

        Args:
            original_query: 用户原始问题。
            retrieved_docs: 当前检索到的文档列表。
            retriever: Retriever 实例（用于补充检索）。

        Returns:
            {
                "docs": List[Dict] (最终文档列表),
                "reflection_count": int (反思次数),
                "is_sufficient": bool (最终是否充分),
                "reflection_logs": List[Dict] (反思日志)
            }
        """
        reflection_count: int = 0
        all_docs: List[Dict[str, Any]] = list(retrieved_docs)
        reflection_logs: List[Dict[str, Any]] = []
        seen_ids: set = set()  # 用于去重已检索过的文档

        for d in all_docs:
            seen_ids.add(d.get("doc_id", ""))

        for round_idx in range(self.max_reflection_rounds):
            # 构建反思提示
            context_summary: str = self._build_context_summary(all_docs[:10])

            user_prompt: str = (
                f"## 用户问题\n{original_query}\n\n"
                f"## 检索到的文档 (摘要)\n{context_summary}\n\n"
                f"请评估这些文档是否能充分回答用户问题。"
            )

            try:
                result: Any = await self.sf_client.call_llm_json(
                    model=REFLECTOR_MODEL,
                    system_prompt=REFLECTION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    top_p=0.8,
                    max_tokens=512,
                )

                is_sufficient: bool = result.get("is_sufficient", True)
                confidence: float = result.get("confidence", 0.5)
                reasoning: str = result.get("reasoning", "")
                supplementary_query: str = result.get("supplementary_query", "")

                log_entry: Dict[str, Any] = {
                    "round": round_idx + 1,
                    "is_sufficient": is_sufficient,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "supplementary_query": supplementary_query,
                }
                reflection_logs.append(log_entry)

                logger.info(
                    f"反思 第{round_idx + 1}轮: "
                    f"is_sufficient={is_sufficient}, "
                    f"confidence={confidence:.2f}"
                )

                # 如果上下文足够 或 补充查询为空，结束反思
                if is_sufficient or not supplementary_query.strip():
                    break

                # 上下文不足，执行补充检索
                reflection_count += 1
                logger.info(f"上下文不足，补充检索: '{supplementary_query}'")

                new_docs: List[Dict[str, Any]] = await retriever.retrieve_simple(
                    query=supplementary_query,
                    top_k=5,
                )

                # 去重合并
                added_count: int = 0
                for doc in new_docs:
                    doc_id: str = doc.get("doc_id", "")
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                        added_count += 1

                logger.info(f"补充检索返回 {len(new_docs)} 条，新增 {added_count} 条")

            except Exception as e:
                logger.warning(f"反思评估失败 ({e})，跳过本轮。")
                reflection_logs.append({
                    "round": round_idx + 1,
                    "is_sufficient": True,
                    "confidence": 0.0,
                    "reasoning": f"反思评估异常: {e}",
                    "supplementary_query": "",
                })
                break

        logger.info(
            f"反思完成: {reflection_count} 次补充检索, "
            f"最终共 {len(all_docs)} 条文档"
        )

        return {
            "docs": all_docs,
            "reflection_count": reflection_count,
            "is_sufficient": is_sufficient if 'is_sufficient' in dir() else True,
            "reflection_logs": reflection_logs,
        }

    # ----------------------------------------------------------
    # 构建上下文摘要
    # ----------------------------------------------------------
    @staticmethod
    def _build_context_summary(
        docs: List[Dict[str, Any]],
        max_chars_per_doc: int = 500,
    ) -> str:
        """
        将文档列表构建为用于 LLM 评估的摘要文本。

        Args:
            docs: 文档列表。
            max_chars_per_doc: 每篇文档最多保留的字符数。

        Returns:
            格式化的摘要文本。
        """
        lines: List[str] = []
        for idx, doc in enumerate(docs):
            title: str = doc.get("metadata", {}).get("title_path", "未知标题")
            content: str = doc.get("content", "")
            snippet: str = content[:max_chars_per_doc].replace("\n", " ")

            lines.append(
                f"[Doc {idx + 1}] 标题: {title}\n"
                f"内容摘要: {snippet}...\n"
            )

        return "\n".join(lines)
