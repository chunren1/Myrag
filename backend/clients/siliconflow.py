"""
硅基流动 (SiliconFlow) API 客户端封装
=======================================
使用 openai 兼容接口，内置 429 限流重试机制（指数退避 + 随机抖动）。
"""

import os
import random
import time
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)
from httpx import HTTPStatusError, ReadTimeout, ConnectError
import logging

logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# 模型名称常量
# ============================================================
EMBEDDING_MODEL: str = "BAAI/bge-m3"              # Embedding 向量化模型
RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"     # Rerank 精排模型
PLANNER_MODEL: str = "Qwen/Qwen3.5-4B"            # 问题拆解（轻量快响应，擅长 JSON）
REFLECTOR_MODEL: str = "Qwen/Qwen3.5-4B"          # 上下文反思（同上）
GENERATOR_MODEL: str = "Qwen/Qwen3-8B"            # 最终生成（大模型保证输出质量）

# API 配置
BASE_URL: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")

# 最大重试次数
MAX_RETRY_ATTEMPTS: int = 5


# ============================================================
# 429 限流判断函数
# ============================================================
def _is_rate_limit_error(exception: BaseException) -> bool:
    """判断异常是否为 HTTP 429 Rate Limit 错误。"""
    if isinstance(exception, HTTPStatusError):
        return exception.response.status_code == 429
    # openai 库会将 HTTP 429 包装为特定异常，检查状态码属性
    status_code: Optional[int] = getattr(exception, "status_code", None)
    if status_code is not None:
        return status_code == 429
    # 兜底：检查异常消息中是否包含 429 关键词
    error_msg: str = str(exception).lower()
    return "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg


# ============================================================
# 带随机抖动 (Jitter) 的指数退避等待策略
# ============================================================
def _with_jitter(retry_state: Any) -> float:
    """
    计算带随机抖动的等待时间。
    公式: min(2^x + random(0, 1), 60)  秒
    其中 x = 尝试次数 (1-indexed)
    最大等待不超过 60 秒。
    """
    attempt: int = retry_state.attempt_number
    base_wait: float = 2.0 ** attempt  # 指数退避: 2, 4, 8, 16, 32 秒
    jitter: float = random.uniform(0, 1.0)  # 随机抖动 0~1 秒
    wait_time: float = min(base_wait + jitter, 60.0)
    return wait_time


# ============================================================
# 硅基流动 API 客户端
# ============================================================
class SiliconFlowClient:
    """
    封装硅基流动 API 的异步客户端。
    使用 openai 兼容接口，内置 tenacity 重试机制处理 429 限流。
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        """
        初始化客户端。

        Args:
            api_key: 硅基流动 API Key，若不传则从环境变量 SILICONFLOW_API_KEY 读取。
            base_url: API 基础 URL，若不传则从环境变量 SILICONFLOW_BASE_URL 读取。
        """
        self.api_key: str = api_key or API_KEY
        if not self.api_key or self.api_key == "sk-your-api-key-here":
            raise ValueError(
                "请设置有效的 SILICONFLOW_API_KEY 环境变量，"
                "或直接传入 api_key 参数。"
                "获取地址: https://cloud.siliconflow.cn/account/ak"
            )

        self.base_url: str = base_url or BASE_URL
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120.0,          # 请求超时 120 秒
            max_retries=0,          # 禁用 openai 自带重试，由 tenacity 统一管理
        )

    # ----------------------------------------------------------
    # 统一的 LLM 请求方法（内置 429 重试）
    # ----------------------------------------------------------
    async def _chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Any:
        """
        发送聊天补全请求，内置带随机抖动的指数退避重试。

        Args:
            model: 模型名称。
            messages: 消息列表。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            top_p: 核采样参数。
            response_format: 响应格式 (如 {"type": "json_object"} 启用 JSON 模式)。
            stream: 是否流式输出。

        Returns:
            API 原始响应对象。
        """

        @retry(
            retry=retry_if_exception_type(Exception),
            retry_error_callback=lambda retry_state: None,
            wait=_with_jitter,
            stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG),
            reraise=True,
        )
        async def _do_request() -> Any:
            """实际执行 API 请求的内部函数。"""
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream,
                }
                if top_p is not None:
                    kwargs["top_p"] = top_p
                if response_format is not None:
                    kwargs["response_format"] = response_format

                response = await self.client.chat.completions.create(**kwargs)
                return response

            except Exception as e:
                # 判断是否为 429 错误，是则触发 tenacity 重试
                if _is_rate_limit_error(e):
                    logger.warning(
                        f"遇到 HTTP 429 限流错误 (模型: {model})，"
                        f"将自动重试..."
                    )
                raise e

        return await _do_request()

    # ----------------------------------------------------------
    # 通用 LLM 调用 (非流式，返回完整文本)
    # ----------------------------------------------------------
    async def call_llm(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: Optional[float] = None,
    ) -> str:
        """
        调用 LLM，返回完整文本。

        Args:
            model: 模型名称。
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            top_p: 核采样参数。

        Returns:
            模型生成的完整文本。
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self._chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
        )

        content: str = response.choices[0].message.content or ""
        return content

    # ----------------------------------------------------------
    # 强制 JSON 输出调用
    # ----------------------------------------------------------
    async def call_llm_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        top_p: Optional[float] = None,
    ) -> Any:
        """
        调用 LLM 并强制返回 JSON 格式。

        使用 response_format={"type": "json_object"} 强制模型输出合法 JSON。
        会自动清理响应的 markdown 代码块标记，并尝试解析为 JSON。

        Args:
            model: 模型名称 (推荐 Qwen 系列，JSON 支持较好)。
            system_prompt: 系统提示词 (需明确要求输出 JSON)。
            user_prompt: 用户提示词。
            temperature: 采样温度 (JSON 模式建议较低温度)。
            max_tokens: 最大生成 token 数。
            top_p: 核采样参数。

        Returns:
            解析后的 Python 对象 (dict / list / 等)。
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self._chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format={"type": "json_object"},
            stream=False,
        )

        raw_content: str = response.choices[0].message.content or "{}"
        # 清理可能的 markdown 代码块包裹
        cleaned: str = self._clean_json_response(raw_content)
        return json.loads(cleaned)

    # ----------------------------------------------------------
    # 流式输出调用 (返回异步生成器)
    # ----------------------------------------------------------
    async def call_llm_stream(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        调用 LLM 并以异步生成器形式逐块返回文本。

        用法:
            async for chunk in client.call_llm_stream(...):
                yield chunk

        Args:
            model: 模型名称。
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。
            temperature: 采样温度。
            max_tokens: 最大生成 token 数。
            top_p: 核采样参数。

        Yields:
            每次返回一段增量文本 (delta)。
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        stream = await self._chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ----------------------------------------------------------
    # Embedding 向量化
    # ----------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=_with_jitter,
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def create_embeddings(
        self,
        texts: List[str],
        model: str = EMBEDDING_MODEL,
    ) -> List[List[float]]:
        """
        调用 Embedding API 将文本列表转为向量。

        硅基流动的 bge-m3 模型输出 1024 维向量。

        Args:
            texts: 待向量化的文本列表。
            model: 嵌入模型名称，默认 BAAI/bge-m3。

        Returns:
            向量列表，每个向量为 1024 维浮点数列表。
        """
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=texts,
            )
            embeddings: List[List[float]] = [d.embedding for d in response.data]
            return embeddings

        except Exception as e:
            if _is_rate_limit_error(e):
                logger.warning("Embedding API 遇到 429 限流，正在重试...")
            raise e

    # ----------------------------------------------------------
    # Rerank 精排 (通过 httpx 直接调用 /v1/rerank 接口)
    # ----------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=_with_jitter,
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: str = RERANK_MODEL,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        调用 Rerank API 对文档进行重排序。

        注意：openai 库不直接支持 /v1/rerank 接口，
        因此此处使用底层 httpx 直接发送 HTTP 请求。

        Args:
            query: 查询文本。
            documents: 待排序的文档列表。
            model: 重排序模型，默认 BAAI/bge-reranker-v2-m3。
            top_n: 返回 Top N 个结果。

        Returns:
            排序结果列表，每个元素包含: {"index": int, "document": str, "relevance_score": float}
        """
        import httpx

        url: str = f"{self.base_url}/rerank"
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(url, json=payload, headers=headers)

            if response.status_code == 429:
                logger.warning("Rerank API 遇到 429 限流，正在重试...")
                raise HTTPStatusError(
                    "429 Rate Limit",
                    request=response.request,
                    response=response,
                )

            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            return result.get("results", [])

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------
    @staticmethod
    def _clean_json_response(raw: str) -> str:
        """
        清理 LLM JSON 响应中的 markdown 代码块标记。

        例如:
            ```json\n{...}\n``` -> {...}
        """
        text: str = raw.strip()
        # 移除开头的 ```json 或 ```
        if text.startswith("```"):
            # 找到第一个换行符后的内容
            first_newline: int = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            else:
                text = text[3:]
        # 移除结尾的 ```
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
