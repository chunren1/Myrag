# Dify 平台集成说明

## 概述

本目录预留给 [Dify](https://dify.ai) 平台的配置和导出文件。

Dify 是一个开源的 LLM 应用开发平台，可用于：
- 可视化编排 RAG 工作流
- 构建对话机器人
- 管理 Prompt 模板

## 与本项目的关系

本项目的 `backend/` 是一个独立的 FastAPI 服务，Dify 可作为**可选前端**通过 API 调用本服务：

1. **在 Dify 中配置自定义工具**：将 `/api/chat` 注册为自定义 API 工具
2. **在 Dify 工作流中调用**：使用 HTTP 请求节点调用本服务的对话接口
3. **利用 Dify 的对话管理**：Dify 负责会话历史、用户管理等前端功能

## 接入步骤

1. 确保本项目的 FastAPI 服务已启动 (`uvicorn backend.main:app`)
2. 在 Dify 后台 -> 工具 -> 自定义工具 中创建新工具
3. 配置 OpenAPI Schema（访问 `http://localhost:8000/openapi.json` 获取）
4. 在工作流中使用该工具节点

## 注意事项

- 当前版本 Dify 目录仅作为预留，本项目核心 RAG 逻辑完全独立运行
- 如需直接在 Dify 内使用硅基流动模型，请在 Dify 的模型供应商中配置 SiliconFlow
