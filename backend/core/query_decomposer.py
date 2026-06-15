"""
问题拆解器 (Planner)
====================
将用户输入的复杂问题拆解为 2-3 个原子性子问题，
以提升检索召回率和精准度。
"""

import sys
from pathlib import Path
from typing import Any, List, Optional

# 将 backend 加入 Python 路径
BACKEND_DIR: Path = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.siliconflow import SiliconFlowClient, PLANNER_MODEL
import logging

logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# 问题拆解 System Prompt
# ============================================================
QUERY_DECOMPOSITION_SYSTEM: str = """你是一个专业的查询分析与拆解助手。你的任务是将用户提出的复杂问题拆解为 2~3 个更简单、更原子的子问题。

## 拆解原则
1. **保持语义完整性**：每个子问题必须是完整、独立的提问。
2. **从简到深**：先提取核心事实查询，再扩展为分析性/比较性问题。
3. **去冗余**：子问题之间不应有大量重叠。
4. **忠实于原问题**：不要添加原问题没有涉及的领域或假设。
5. **覆盖全面**：子问题应共同覆盖原问题的所有关键方面。

## 输出格式
严格返回一个 JSON 对象，格式如下：
```json
{
  "original_query": "用户原始问题",
  "sub_queries": ["子问题1", "子问题2", "子问题3"],
  "reasoning": "简要说明拆解逻辑（一句话）"
}
```

## 示例
用户问题: "Transformer 相比 RNN 有哪些优势？在 NLP 中的应用现状如何？"
输出:
```json
{
  "original_query": "Transformer 相比 RNN 有哪些优势？在 NLP 中的应用现状如何？",
  "sub_queries": [
    "Transformer 架构相比 RNN 的核心技术优势有哪些？",
    "Transformer 在 NLP 领域的主要应用有哪些？",
    "当前 NLP 领域使用 Transformer 的最新进展和成果是什么？"
  ],
  "reasoning": "先拆解架构对比，再拆解应用场景，最后补充现状概览"
}
```

现在请拆解以下用户问题。"""


async def decompose_query(
    client: SiliconFlowClient,
    query: str,
    max_sub_queries: int = 3,
) -> List[str]:
    """
    将用户问题拆解为多个子问题。

    如果原问题很简单（如短句提问），可能只返回 1 个子问题（原问题本身）。

    Args:
        client: 硅基流动 API 客户端。
        query: 用户的原始问题。
        max_sub_queries: 最大子问题数量，默认 3。

    Returns:
        子问题列表，至少包含原问题本身。
    """
    # 如果问题很短（少于 15 字），认为是简单问题，直接返回原问题
    if len(query.strip()) <= 15:
        logger.info(f"问题较短 ({len(query)} 字)，跳过拆解。")
        return [query.strip()]

    try:
        user_prompt: str = f"用户问题: {query}"

        result: Any = await client.call_llm_json(
            model=PLANNER_MODEL,
            system_prompt=QUERY_DECOMPOSITION_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.3,
            top_p=0.8,
            max_tokens=1024,
        )

        # 解析结果
        sub_queries: List[str] = []
        if isinstance(result, dict):
            sub_queries = result.get("sub_queries", [])
        elif isinstance(result, list):
            sub_queries = result

        # 确保 sub_queries 是字符串列表
        sub_queries = [str(q) for q in sub_queries[:max_sub_queries]]

        # 如果拆解失败或结果为空，使用原问题
        if not sub_queries:
            logger.warning("问题拆解返回空列表，使用原问题。")
            return [query.strip()]

        # 记录拆解结果
        reasoning: str = ""
        if isinstance(result, dict):
            reasoning = result.get("reasoning", "")
        logger.info(
            f"问题拆解完成: {len(sub_queries)} 个子问题"
            + (f" (原因: {reasoning})" if reasoning else "")
        )

        return sub_queries

    except Exception as e:
        logger.warning(f"问题拆解失败 ({e})，使用原问题。")
        return [query.strip()]
