"""
提示词工程模块
==============
构建 Generator 的系统提示词。
要求模型先在 <thinking> 标签内分析资料，再输出带引用的深度报告。
"""

from typing import Any, Dict, List


# ============================================================
# Generator 系统提示词
# ============================================================
GENERATOR_SYSTEM_PROMPT: str = """你是一个专业的知识库问答助手。请基于检索到的参考资料，给出准确、简洁的回答。

## 格式要求
1. 先在 <thinking> 标签内做简短分析，然后输出正文
2. 正文直接回答要点，简洁明了，不要标注文档来源
3. 资料不足时如实说明，不要编造

## 示例

<thinking>
从参考资料可知，规范要求驼峰命名、禁止魔法值等。
</thinking>

编码规范主要包括：命名使用驼峰法，类名大写开头、方法名小写开头。禁止使用魔法值，所有常量须统一定义。
"""

# 注意: GENERATOR_SYSTEM_PROMPT 末尾故意留有不完整的格式示例，
# 实际使用时会拼接参考资料


def build_generator_prompt(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    reflection_logs: List[Dict[str, Any]] = None,
    max_context_chars: int = 12000,  # 约 6000 tokens (中文环境下)
) -> Dict[str, str]:
    """
    构建 Generator 的 system_prompt 和 user_prompt。
    自动截断过长上下文，优先保留高相关度文档。

    Args:
        query: 用户的原始问题。
        retrieved_docs: 检索+反思后的最终文档列表。
        reflection_logs: 反思日志（可选，用于生成器的上下文感知）。
        max_context_chars: 参考资料的字符数上限（约 token 数 = chars / 2）。

    Returns:
        {"system": str, "user": str} - 分别对应 system 和 user 角色的提示词。
    """
    if reflection_logs is None:
        reflection_logs = []

    # ----------------------------------------------------------
    # 构建参考资料部分（按相关度从高到低，截断溢出内容）
    # ----------------------------------------------------------
    reference_parts: List[str] = []
    total_chars: int = 0
    truncated: bool = False

    for idx, doc in enumerate(retrieved_docs):
        title: str = doc.get("metadata", {}).get("title_path", "未知来源")
        content: str = doc.get("content", "")
        relevance: float = doc.get("score", 0.0)

        block: str = (
            f"### [参考文档 {idx + 1}]\n"
            f"- 来源: {title}\n"
            f"- 相关度: {relevance:.4f}\n"
            f"- 内容:\n{content}\n"
        )

        if total_chars + len(block) > max_context_chars:
            remaining: int = max(max_context_chars - total_chars - 150, 200)
            truncated_content: str = content[:remaining] + "\n...(内容过长已截断)"
            block = (
                f"### [参考文档 {idx + 1}]\n"
                f"- 来源: {title}\n"
                f"- 相关度: {relevance:.4f}\n"
                f"- 内容:\n{truncated_content}\n"
            )
            reference_parts.append(block)
            truncated = True
            break

        reference_parts.append(block)
        total_chars += len(block)

    if truncated:
        import logging
        logging.getLogger(__name__).info(
            f"上下文截断: 保留 {len(reference_parts)} 篇文档, 共 {total_chars} 字符"
        )

    if not reference_parts:
        references: str = "（未检索到任何参考资料，请如实告知用户无法回答，并给出建议。）"
    else:
        references: str = "\n---\n".join(reference_parts)

    # ----------------------------------------------------------
    # 构建反思反馈（如果有）
    # ----------------------------------------------------------
    reflection_note: str = ""
    if reflection_logs:
        reflection_note = "\n\n## 检索质量说明\n"
        for log in reflection_logs:
            reflection_note += (
                f"- 第{log['round']}轮反思: "
                f"{'充分' if log.get('is_sufficient') else '进行了补充检索'}"
            )
            if log.get("supplementary_query"):
                reflection_note += f" (补充查询: {log['supplementary_query']})"
            reflection_note += "\n"

    # ----------------------------------------------------------
    # 构建最终消息
    # ----------------------------------------------------------
    system_prompt: str = (
        GENERATOR_SYSTEM_PROMPT
        + "\n## 参考资料\n"
        + references
        + reflection_note
        + "\n\n**注意**: 回答中不要标注文档来源。"
    )

    user_prompt: str = f"请基于上面的参考资料回答以下问题：\n\n{query}"

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


def build_planner_prompt(query: str) -> Dict[str, str]:
    """
    构建 Planner (问题拆解器) 的提示词。
    （实际拆解逻辑在 query_decomposer.py 中，此处为备用工具函数。）

    Args:
        query: 用户原始问题。

    Returns:
        {"system": str, "user": str}
    """
    return {
        "system": "你是一个查询规划专家。请将复杂问题拆解为 2~3 个原子性子问题，以 JSON 数组格式返回。",
        "user": f"请拆解以下问题: {query}",
    }
