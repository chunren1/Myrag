"""
对话 API 接口
=============
提供 `/api/chat` 端点，支持 SSE (Server-Sent Events) 流式响应。
内置状态机过滤 <thinking>...</thinking> 标签内的内容。
"""

import asyncio
import json
import sys
from pathlib import Path
from enum import Enum, auto
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 将 backend 加入 Python 路径
BACKEND_DIR: Path = Path(__file__).parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from clients.siliconflow import SiliconFlowClient
from clients.qdrant_client import QdrantClientWrapper
from core.agent_flow import AgenticRAG
import logging

logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# 路由初始化
# ============================================================
router: APIRouter = APIRouter(prefix="/api", tags=["chat"])

# 全局 Agent 实例 (在 main.py 启动时注入)
_agent: AgenticRAG = None  # type: ignore


def set_agent(agent: AgenticRAG) -> None:
    """设置全局 Agent 实例（由 main.py 在启动时调用）。"""
    global _agent
    _agent = agent


# ============================================================
# Pydantic 模型
# ============================================================
class ChatRequest(BaseModel):
    """对话请求模型。"""
    query: str = Field(..., description="用户输入的问题", min_length=1, max_length=5000)
    stream: bool = Field(default=True, description="是否使用流式响应 (SSE)")
    mode: str = Field(
        default="agentic",
        description="运行模式: 'agentic' (完整 Agent) 或 'simple' (简单检索)",
        pattern="^(agentic|simple)$",
    )


class ChatResponse(BaseModel):
    """非流式对话响应模型。"""
    answer: str = Field(..., description="Agent 的完整回答")
    sub_queries: List[str] = Field(default_factory=list, description="拆解后的子问题")
    retrieved_doc_count: int = Field(default=0, description="检索到的文档数量")
    reflection_logs: List[Dict[str, Any]] = Field(default_factory=list, description="反思日志")


# ============================================================
# 状态机: 过滤 <thinking> 标签内的内容
# ============================================================
class ThinkingFilter:
    """
    流式 <thinking> 标签过滤器。
    在流式输出中实时去除 <thinking>...</thinking> 草稿内容。
    支持标签跨 chunk 边界。
    """
    _OPEN = "<thinking>"
    _CLOSE = "</thinking>"

    def __init__(self) -> None:
        self._inside: bool = False   # 当前是否在 thinking 标签内
        self._buf: str = ""          # 残留缓冲区

    def feed(self, chunk: str) -> str:
        """输入一块文本，返回过滤后的安全文本。"""
        self._buf += chunk
        out: str = ""

        while self._buf:
            if not self._inside:
                idx: int = self._buf.find(self._OPEN)
                if idx == -1:
                    # 没有完整 opening tag，保留末尾可能的部分匹配
                    keep = self._partial_end(self._buf, self._OPEN)
                    if keep:
                        out += self._buf[: -len(keep)]
                        self._buf = keep
                        return out
                    out += self._buf
                    self._buf = ""
                    return out
                # 找到了 opening tag
                out += self._buf[:idx]
                self._buf = self._buf[idx + len(self._OPEN):]
                self._inside = True
            else:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    # 没有完整 closing tag，保留末尾部分匹配
                    keep = self._partial_end(self._buf, self._CLOSE)
                    if keep:
                        self._buf = keep
                    else:
                        self._buf = ""
                    return out
                # 找到了 closing tag
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._inside = False

        return out

    def flush(self) -> str:
        """流结束，吐出缓冲区中非 thinking 的残留。"""
        if not self._inside and self._buf:
            r = self._buf
            self._buf = ""
            return r
        self._buf = ""
        return ""

    @staticmethod
    def _partial_end(text: str, tag: str) -> str:
        """检查 text 末尾是否包含 tag 的部分前缀，返回匹配的后缀部分。"""
        for n in range(len(tag) - 1, 0, -1):
            prefix = tag[:n]
            if text.endswith(prefix):
                return text[-n:]
        return ""


# ============================================================
# 智能模式选择
# ============================================================
_SIMPLE_TRIGGERS = ["是什么", "什么是", "定义", "多少", "哪位", "哪个", "什么时候", "在哪里"]


def _smart_mode(query: str, user_mode: str) -> str:
    """短问题或简单查询自动降级为 simple 模式以提高响应速度。"""
    if user_mode == "simple":
        return "simple"
    q = query.strip()
    # 短问题：15 字以内
    if len(q) <= 15:
        return "simple"
    # 含简单疑问词且不超过一个问号
    if q.count("?") + q.count("？") <= 1:
        for w in _SIMPLE_TRIGGERS:
            if w in q[:30]:
                return "simple"
    return "agentic"


# ============================================================
# API 端点
# ============================================================

@router.post("/chat", response_model=None)
async def chat_endpoint(req: ChatRequest) -> Any:
    """
    对话接口，支持 SSE 流式和非流式两种模式。

    ### 流式模式 (stream=True，默认)
    返回格式: text/event-stream (SSE)
    每条消息格式: data: {"type": "...", "content": "..."}\n\n
    类型:
      - "thinking_start": 开始思考 (仅通知)
      - "thinking_end": 结束思考 (仅通知)
      - "text": 正文内容
      - "done": 流结束
      - "error": 错误信息

    ### 非流式模式 (stream=False)
    返回 JSON 格式的完整回答。
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent 服务尚未初始化")

    # 智能模式选择：短问题/陈述句自动走 simple 以提高速度
    actual_mode: str = _smart_mode(req.query, req.mode)
    if actual_mode != req.mode:
        logger.info(f"智能加速: {req.mode} -> {actual_mode} (问题较短或为简单查询)")

    logger.info(f"收到对话请求: '{req.query[:80]}...' (stream={req.stream}, mode={actual_mode})")

    if req.stream:
        # === 流式响应 (SSE) ===
        return StreamingResponse(
            content=_stream_chat(req, actual_mode),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            },
        )
    else:
        # === 非流式响应 ===
        try:
            if actual_mode == "simple":
                result = await _agent.run_simple(req.query)
            else:
                result = await _agent.run(req.query)

            # 对非流式回答也应用 thinking 过滤
            filtered_answer: str = _filter_thinking_from_text(result["answer"])

            return ChatResponse(
                answer=filtered_answer,
                sub_queries=result["sub_queries"],
                retrieved_doc_count=len(result["retrieved_docs"]),
                reflection_logs=result["reflection_logs"],
            )
        except Exception as e:
            logger.error(f"对话处理失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")


def _filter_thinking_from_text(text: str) -> str:
    """
    使用 ThinkingFilter 清洗完整文本中的 <thinking> 标签。

    Args:
        text: 原始回答文本。

    Returns:
        清洗后的纯净正文。
    """
    import re

    # 方法1: 正则快速移除 <thinking>...</thinking> 及其代码块包裹
    cleaned: str = text
    # 移除 ```xml\n<thinking>...</thinking>\n``` 格式
    cleaned = re.sub(r'```(?:xml)?\s*\n?<thinking>.*?</thinking>\s*\n?```', '', cleaned, flags=re.DOTALL)
    # 移除普通 <thinking>...</thinking>
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


async def _stream_chat(req: ChatRequest, mode: str) -> Any:
    """
    流式对话生成器。

    Args:
        req: 对话请求。
        mode: 实际使用的模式。

    Yields:
        SSE 格式的数据块。
    """
    filter_obj: ThinkingFilter = ThinkingFilter()
    first_chunk_sent: bool = False

    try:
        # 运行 Agent (流式)
        if mode == "simple":
            result = await _agent.run_simple(req.query)
            answer_text: str = result["answer"]

            # 非流式模式下也模拟 SSE 输出（一次性模拟流式）
            # 过滤 thinking 标签
            filtered: str = ""
            for i in range(0, len(answer_text), 100):  # 100 字符一批
                chunk: str = answer_text[i: i + 100]
                filtered += filter_obj.feed(chunk)
                if filtered:
                    yield f"data: {json.dumps({'type': 'text', 'content': filtered}, ensure_ascii=False)}\n\n"
                    filtered = ""

            remaining: str = filter_obj.flush()
            if remaining:
                yield f"data: {json.dumps({'type': 'text', 'content': remaining}, ensure_ascii=False)}\n\n"

            yield "data: {\"type\": \"done\"}\n\n"
            return

        # Agentic 模式: 真正的流式
        async for chunk in _agent.run_stream(req.query):
            # 过滤 <thinking> 标签（状态机）
            filtered: str = filter_obj.feed(chunk)

            if filtered:
                # 兜底：再用正则做一次安全清洗
                safe: str = _filter_thinking_from_text(filtered)
                if safe:
                    yield f"data: {json.dumps({'type': 'text', 'content': safe}, ensure_ascii=False)}\n\n"
                    first_chunk_sent = True

        # 刷新缓冲区
        remaining: str = filter_obj.flush()
        if remaining:
            remaining = _filter_thinking_from_text(remaining)
            if remaining:
                yield f"data: {json.dumps({'type': 'text', 'content': remaining}, ensure_ascii=False)}\n\n"

        # 发送完成信号
        yield "data: {\"type\": \"done\"}\n\n"

    except Exception as e:
        logger.error(f"流式对话失败: {e}", exc_info=True)
        error_msg: str = json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)
        # 确保错误消息在过滤状态下也能发出
        yield f"data: {error_msg}\n\n"


# ============================================================
# 健康检查
# ============================================================
@router.get("/health")
async def health_check() -> Dict[str, str]:
    """健康检查端点。"""
    return {"status": "ok", "service": "high-quality-rag"}
