"""
对话 API 接口
=============
提供 `/api/chat` 端点，支持 SSE (Server-Sent Events) 流式响应。
内置状态机过滤 <thinking>...</thinking> 标签内的内容。
"""

import json
import re
from enum import Enum, auto
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dependencies import get_agent
from core.agent_flow import AgenticRAG
import logging

logger: logging.Logger = logging.getLogger(__name__)

router: APIRouter = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户输入的问题", min_length=1, max_length=5000)
    stream: bool = Field(default=True, description="是否使用流式响应 (SSE)")
    mode: str = Field(
        default="agentic",
        description="运行模式: 'agentic' (完整 Agent) 或 'simple' (简单检索)",
        pattern="^(agentic|simple)$",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的完整回答")
    sub_queries: List[str] = Field(default_factory=list, description="拆解后的子问题")
    retrieved_doc_count: int = Field(default=0, description="检索到的文档数量")
    reflection_logs: List[Dict[str, Any]] = Field(default_factory=list, description="反思日志")


class ThinkingFilter:
    _OPEN = "<thinking>"
    _CLOSE = "</thinking>"

    def __init__(self) -> None:
        self._inside: bool = False
        self._buf: str = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: str = ""

        while self._buf:
            if not self._inside:
                idx: int = self._buf.find(self._OPEN)
                if idx == -1:
                    keep = self._partial_end(self._buf, self._OPEN)
                    if keep:
                        out += self._buf[: -len(keep)]
                        self._buf = keep
                        return out
                    out += self._buf
                    self._buf = ""
                    return out
                out += self._buf[:idx]
                self._buf = self._buf[idx + len(self._OPEN):]
                self._inside = True
            else:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    keep = self._partial_end(self._buf, self._CLOSE)
                    if keep:
                        self._buf = keep
                    else:
                        self._buf = ""
                    return out
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._inside = False

        return out

    def flush(self) -> str:
        if not self._inside and self._buf:
            r = self._buf
            self._buf = ""
            return r
        self._buf = ""
        return ""

    @staticmethod
    def _partial_end(text: str, tag: str) -> str:
        for n in range(len(tag) - 1, 0, -1):
            prefix = tag[:n]
            if text.endswith(prefix):
                return text[-n:]
        return ""


_SIMPLE_TRIGGERS = ["是什么", "什么是", "定义", "多少", "哪位", "哪个", "什么时候", "在哪里"]


def _smart_mode(query: str, user_mode: str) -> str:
    if user_mode == "simple":
        return "simple"
    q = query.strip()
    if len(q) <= 15:
        return "simple"
    if q.count("?") + q.count("？") <= 1:
        for w in _SIMPLE_TRIGGERS:
            if w in q[:30]:
                return "simple"
    return "agentic"


@router.post("/chat", response_model=None)
async def chat_endpoint(
    req: ChatRequest,
    agent: AgenticRAG = Depends(get_agent),
) -> Any:
    actual_mode: str = _smart_mode(req.query, req.mode)
    if actual_mode != req.mode:
        logger.info(f"智能加速: {req.mode} -> {actual_mode} (问题较短或为简单查询)")

    logger.info(f"收到对话请求: '{req.query[:80]}...' (stream={req.stream}, mode={actual_mode})")

    if req.stream:
        return StreamingResponse(
            content=_stream_chat(req, actual_mode, agent),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            if actual_mode == "simple":
                result = await agent.run_simple(req.query)
            else:
                result = await agent.run(req.query)

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
    cleaned: str = text
    cleaned = re.sub(r'```(?:xml)?\s*\n?<thinking>.*?</thinking>\s*\n?```', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


async def _stream_chat(req: ChatRequest, mode: str, agent: AgenticRAG) -> Any:
    filter_obj: ThinkingFilter = ThinkingFilter()

    try:
        if mode == "simple":
            result = await agent.run_simple(req.query)
            answer_text: str = result["answer"]

            filtered: str = ""
            for i in range(0, len(answer_text), 100):
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

        async for chunk in agent.run_stream(req.query):
            filtered: str = filter_obj.feed(chunk)

            if filtered:
                safe: str = _filter_thinking_from_text(filtered)
                if safe:
                    yield f"data: {json.dumps({'type': 'text', 'content': safe}, ensure_ascii=False)}\n\n"

        remaining: str = filter_obj.flush()
        if remaining:
            remaining = _filter_thinking_from_text(remaining)
            if remaining:
                yield f"data: {json.dumps({'type': 'text', 'content': remaining}, ensure_ascii=False)}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"

    except Exception as e:
        logger.error(f"流式对话失败: {e}", exc_info=True)
        error_msg: str = json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)
        yield f"data: {error_msg}\n\n"


@router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "high-quality-rag"}
