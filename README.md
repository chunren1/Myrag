# 🧠 High-Quality RAG

> 基于 Agentic RAG 架构的零成本知识库问答系统，硅基流动免费 API + 本地轻量部署。

## 架构

```
用户提问 → Planner(拆解) → Retriever(检索+精排) → Reflector(反思) → Generator(生成)
                                                                        ↓
                                                                  SSE 流式输出
```

## 技术栈

| 组件 | 模型/技术 |
|------|----------|
| Embedding | BAAI/bge-m3 (1024维) |
| Rerank | BAAI/bge-reranker-v2-m3 |
| Planner/Reflector | Qwen3.5-4B |
| Generator | Qwen3-8B |
| 向量库 | Qdrant (INT8 量化) |
| 后端 | FastAPI + SSE |
| API | 硅基流动 (免费) |

## 快速开始

```powershell
# 1. 安装依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 配置 .env（填入你的硅基流动 API Key）
# SILICONFLOW_API_KEY=sk-xxx

# 3. 启动 Qdrant
docker compose up -d

# 4. 导入文档
# 把 .md 文件放入 workspace/raw_docs/ 然后：
npm run pipeline

# 5. 启动服务
npm start
# 打开 http://localhost:8000
```

## npm 命令

| 命令 | 说明 |
|------|------|
| `npm start` | 启动 Qdrant + 后端 |
| `npm run pipeline` | 导入文档到知识库 |
| `npm run stop` | 停止 Qdrant |

## API

| 端点 | 说明 |
|------|------|
| `GET /` | 聊天页面 |
| `POST /api/chat` | 对话接口 (支持 SSE 流式) |
| `GET /docs` | Swagger 文档 |

请求示例：
```json
{
  "query": "Java 开发规范是什么？",
  "stream": true,
  "mode": "agentic"
}
```

## 项目结构

```
backend/              # FastAPI 服务
  core/                # Agent 闭环核心
  clients/             # API 客户端封装
  api/                 # 路由 & SSE
data_pipeline/         # 离线数据管道
web_frontend/          # 聊天界面
workspace/             # 文档 & 日志
```

## 特性

- 零成本：全部使用硅基流动免费 API
- Agentic RAG：规划→检索→反思→生成 四阶段闭环
- 流式输出：SSE 实时推送，自动过滤草稿内容
- 智能路由：短问题自动加速
- 防限流：指数退避 + 随机抖动重试
