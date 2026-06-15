"""
LLM 知识增强模块 (Doc2Query)
============================
调用 Qwen LLM 为每个 chunk 生成 N 个预测问题 (Doc2Query)，
将生成的问题与原文拼接，提升后续检索的召回率。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 将 backend 加入 Python 路径，以便导入 siliconflow 客户端
BACKEND_DIR: Path = Path(__file__).parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.siliconflow import SiliconFlowClient, PLANNER_MODEL
import logging

logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# Doc2Query 配置
# ============================================================
DEFAULT_QUERY_COUNT: int = 3  # 每个 chunk 生成的问题数

# Doc2Query 的 System Prompt
DOC2QUERY_SYSTEM_PROMPT: str = """你是一个专业的检索增强(RAG)系统助手。你的任务是为给定的文档片段生成高质量的检索问题。

## 要求
1. 仔细阅读文档片段，理解其核心内容和关键信息。
2. 生成 {query_count} 个可能被用户提出的检索问题，这些问题应该：
   - 与该文档片段高度相关
   - 覆盖文档中的关键知识点
   - 使用多样化的提问方式（什么、如何、为什么、哪些等）
   - 语言自然流畅，模拟真实用户的提问习惯
3. 严格以 JSON 数组格式返回，不要包含任何其他内容。

## 返回格式示例
```json
["问题1", "问题2", "问题3"]
```

现在请为以下文档片段生成 {query_count} 个检索问题："""


async def generate_queries_for_chunk(
    client: SiliconFlowClient,
    chunk_content: str,
    query_count: int = DEFAULT_QUERY_COUNT,
) -> List[str]:
    """
    为单个 chunk 生成 Doc2Query 预测问题。

    Args:
        client: 硅基流动 API 客户端。
        chunk_content: chunk 正文。
        query_count: 生成的问题数量，默认 3。

    Returns:
        预测问题列表，如 ["什么是...？", "如何...？", "为什么...？"]
    """
    # 构建用户提示词
    user_prompt: str = f"文档片段:\n```\n{chunk_content[:3000]}\n```"  # 限制输入长度

    try:
        # 调用 Qwen 生成问题 (使用 call_llm_json 省去解析步骤)
        result = await client.call_llm_json(
            model=PLANNER_MODEL,
            system_prompt=DOC2QUERY_SYSTEM_PROMPT.format(query_count=query_count),
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=1024,
        )

        # 结果可能是 dict 包装的数组，也可能是直接数组
        if isinstance(result, list):
            questions: List[str] = [str(q) for q in result[:query_count]]
        elif isinstance(result, dict):
            # 有些模型返回 {"questions": [...]}
            for key in ("questions", "queries", "results"):
                if key in result:
                    questions = [str(q) for q in result[key][:query_count]]
                    break
            else:
                # 取 dict 的第一个 list 值
                for val in result.values():
                    if isinstance(val, list):
                        questions = [str(v) for v in val[:query_count]]
                        break
                else:
                    questions = []
        else:
            questions = []

        logger.info(f"Doc2Query 生成完成: {len(questions)} 个问题")
        return questions

    except Exception as e:
        logger.warning(f"Doc2Query 生成失败 ({e})，将使用空列表。")
        return []


async def enrich_chunks(
    client: SiliconFlowClient,
    chunks: List[Dict[str, Any]],
    query_count: int = DEFAULT_QUERY_COUNT,
) -> List[Dict[str, Any]]:
    """
    批量为 chunks 生成 Doc2Query 并拼接到正文后。

    对于每个 chunk:
    1. 调用 LLM 生成 3 个预测问题。
    2. 将问题追加到 content 末尾（用于 Embedding）。
    3. 将原始问题列表存入 metadata["predicted_queries"]。

    Args:
        client: 硅基流动 API 客户端。
        chunks: chunk 列表 (来自 markdown_splitter.py)。
        query_count: 每个 chunk 生成的问题数。

    Returns:
        增强后的 chunk 列表 (原地修改并返回)。
    """
    logger.info(f"开始 Doc2Query 增强，共 {len(chunks)} 个 chunk...")

    for idx, chunk in enumerate(chunks):
        content: str = chunk["content"]
        questions: List[str] = await generate_queries_for_chunk(
            client=client,
            chunk_content=content,
            query_count=query_count,
        )

        if questions:
            # 将问题拼接到原文末尾（Embedding 时可以捕获语义关联）
            chunk["content"] = content + "\n\n" + "\n".join(f"- {q}" for q in questions)
            chunk["metadata"]["predicted_queries"] = questions
        else:
            chunk["metadata"]["predicted_queries"] = []

        # 每处理 10 个打印一次进度
        if (idx + 1) % 10 == 0 or (idx + 1) == len(chunks):
            logger.info(f"Doc2Query 进度: {idx + 1}/{len(chunks)}")

    logger.info(f"Doc2Query 增强完成，共处理 {len(chunks)} 个 chunk")
    return chunks
