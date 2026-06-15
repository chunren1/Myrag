"""
智能 Markdown 文档切分器
========================
使用 langchain 的 MarkdownHeaderTextSplitter 按标题层级切分，
保留标题路径到 metadata 中供后续检索时使用。
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import logging

logger: logging.Logger = logging.getLogger(__name__)

# ============================================================
# 切分配置
# ============================================================
# Markdown 标题层级定义 (用于保留结构)
HEADERS_TO_SPLIT_ON: List[tuple] = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

# 默认切分参数
DEFAULT_CHUNK_SIZE: int = 2048
DEFAULT_CHUNK_OVERLAP: int = 256


class MarkdownSplitter:
    """
    智能 Markdown 切分器。

    分两步处理：
    1. 按标题层级 (h1~h4) 初步切分，保留段落结构。
    2. 对超过 chunk_size 的大段落，使用 RecursiveCharacterTextSplitter 二次切分。
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        headers_to_split_on: Optional[List[tuple]] = None,
    ) -> None:
        """
        初始化切分器。

        Args:
            chunk_size: 每个 chunk 的最大字符数。
            chunk_overlap: chunk 之间的重叠字符数。
            headers_to_split_on: 标题层级定义，默认 h1~h4。
        """
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.headers: List[tuple] = headers_to_split_on or HEADERS_TO_SPLIT_ON

        # LangChain 标题切分器
        self.header_splitter: MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers,
            strip_headers=True,  # 从内容中移除标题行
        )

        # 字符级递归切分器 (用于二次切分大段落)
        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",    # 优先按空行切
                "\n",      # 再按换行切
                "。",      # 中文句号
                ".",       # 英文句号
                "；",      # 中文分号
                ";",       # 英文分号
                "，",      # 中文逗号
                ",",       # 英文逗号
                " ",       # 空格兜底
            ],
            length_function=len,
        )

    # ----------------------------------------------------------
    # 主切分方法
    # ----------------------------------------------------------
    def split_document(
        self,
        file_path: str,
        source_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        切分单个 Markdown 文件为 chunk 列表。

        Args:
            file_path: Markdown 文件的绝对路径。
            source_name: 来源名称 (文件名)，若不传则自动提取。

        Returns:
            chunk 列表，每个元素为:
            {
                "chunk_id": str (UUID 格式),
                "content": str (chunk 正文),
                "metadata": {
                    "source": str,
                    "title_path": str (如 "主标题 > 二级标题 > 三级标题"),
                    "h1": str, "h2": str, ...
                    "chunk_index": int,
                    "char_count": int,
                }
            }
        """
        filepath_obj: Path = Path(file_path)
        if not filepath_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if source_name is None:
            source_name = filepath_obj.name

        # 读取原始 Markdown 文本
        raw_text: str = filepath_obj.read_text(encoding="utf-8")

        logger.info(f"开始切分文件: {source_name}, 原始大小: {len(raw_text)} 字符")

        # 第一步：按标题层级切分
        header_splits = self.header_splitter.split_text(raw_text)

        chunks: List[Dict[str, Any]] = []

        for header_doc in header_splits:
            content: str = header_doc.page_content.strip()
            if not content:
                continue  # 跳过空段落

            metadata: Dict[str, Any] = dict(header_doc.metadata)
            # 构建标题路径字符串
            title_parts: List[str] = []
            for level in ["h1", "h2", "h3", "h4"]:
                if metadata.get(level):
                    title_parts.append(metadata[level])
            title_path: str = " > ".join(title_parts) if title_parts else "未分类"

            # 第二步：如果此段内容超过 chunk_size，进行二次切分
            if len(content) > self.chunk_size:
                sub_chunks = self.text_splitter.split_text(content)
                for idx, sub_content in enumerate(sub_chunks):
                    chunk_id: str = self._generate_chunk_id(
                        source_name, title_path, len(chunks), content
                    )
                    chunks.append({
                        "chunk_id": chunk_id,
                        "content": sub_content.strip(),
                        "metadata": {
                            "source": source_name,
                            "title_path": title_path,
                            **metadata,
                            "chunk_index": idx,
                            "char_count": len(sub_content.strip()),
                        },
                    })
            else:
                chunk_id = self._generate_chunk_id(
                    source_name, title_path, len(chunks), content
                )
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": {
                        "source": source_name,
                        "title_path": title_path,
                        **metadata,
                        "chunk_index": 0,
                        "char_count": len(content),
                    },
                })

        logger.info(
            f"文件 '{source_name}' 切分完成: "
            f"共 {len(chunks)} 个 chunk, "
            f"平均大小: {sum(c['metadata']['char_count'] for c in chunks) // max(len(chunks), 1)} 字符"
        )

        return chunks

    # ----------------------------------------------------------
    # 批量切分目录
    # ----------------------------------------------------------
    def split_directory(
        self,
        directory: str,
        file_extensions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量切分目录下所有 Markdown 文件。

        Args:
            directory: 目录路径。
            file_extensions: 允许的文件后缀，默认 ['.md', '.mdx']。

        Returns:
            所有文件切分后的 chunk 汇总列表。
        """
        if file_extensions is None:
            file_extensions = [".md", ".mdx"]

        dir_path: Path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"目录不存在: {directory}")

        all_chunks: List[Dict[str, Any]] = []
        for ext in file_extensions:
            for file_path in dir_path.glob(f"**/*{ext}"):
                try:
                    chunks = self.split_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"切分文件 '{file_path}' 失败: {e}")

        logger.info(
            f"目录 '{directory}' 批量切分完成: "
            f"共处理 {len(all_chunks)} 个 chunk"
        )
        return all_chunks

    # ----------------------------------------------------------
    # 工具方法: 生成唯一 Chunk ID
    # ----------------------------------------------------------
    @staticmethod
    def _generate_chunk_id(
        source: str,
        title_path: str,
        index: int,
        content: str,
    ) -> str:
        """
        基于内容和元数据生成稳定的 chunk_id (MD5 哈希)。

        Args:
            source: 来源文件名。
            title_path: 标题路径。
            index: chunk 序号。
            content: chunk 内容。

        Returns:
            32 位十六进制字符串作为唯一 ID。
        """
        raw: str = f"{source}|{title_path}|{index}|{content[:200]}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
